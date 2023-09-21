#include <iostream>
#include <string>
#include <random>
#include <time.h>
#include <chrono>
#include <mpi.h>
#include <CL/cl2.hpp>
#include <stdlib.h>


using namespace std;
using namespace std::chrono;


const int N = 16;

void generateRandomArray(int* array, int size)
{
	// Random Number Generation
    // random_device rd;
	// the above implementation of std::random_device did not appear so random. I used the current time as seed (below) instead to give better random values.

    mt19937 e2(time(nullptr));

    // Initialize array with non-zero values; if we skip this, there will be a zero value appearing in the sorted array, which 'steals' the spot of a number 
	// from the initial array; we initialize with any number outside of the random number generator, here the range was set between -40000000 and 40000000.
	// so we can use numbers under the lower linit or over the upper limit from line 33
	for (int i = 0; i < size; i++) {
		array[i] = -999; // Set to a large negative value (guaranteed to be outside the range)
	}

	// Initialize array with random values
    uniform_int_distribution<int> dist(-100, 100);

	// Generate random values between -40000000 and 40000000 and assign them to A[i]
	// the number from line 29 will be overwritten
    for (int i = 0; i < size; i++) {
        int random_value = dist(e2);
        array[i] = random_value; // Assign the generated random value to array[i]
    }


}

void displayArray(int array[], int arraySize)
{
    for (int i = 0; i < arraySize; i++)
    {
        cout << array[i] << "  ";
    }
    cout << endl;
}

int partition(int array3[], int low, int high)
{
    int pivot = array3[low];
    int i = low, j = high;

    while (true)
    {
        while (array3[i] < pivot)
            i++;

        while (array3[j] > pivot)
            j--;

        if (i >= j)
            return j;

        int temp = array3[i];
        array3[i] = array3[j];
        array3[j] = temp;

        i++;
        j--;
    }
}

void quicksort(int array3[], int low, int high)
{
    if (low < high)
    {
        int partitionIndex = partition(array3, low, high);
        quicksort(array3, low, partitionIndex);
        quicksort(array3, partitionIndex + 1, high);
    }
}

void merge(int array[], int left[], int right[], int leftSize, int rightSize)
{
    int i = 0, j = 0, k = 0;
    
    while (i < leftSize && j < rightSize)
    {
        if (left[i] <= right[j])
        {
            array[k] = left[i];
            i++;
        }
        else
        {
            array[k] = right[j];
            j++;
        }
        k++;
    }

    while (i < leftSize)
    {
        array[k] = left[i];
        i++;
        k++;
    }

    while (j < rightSize)
    {
        array[k] = right[j];
        j++;
        k++;
    }
}

int main(int argc, char** argv)
{
    // MPI
    int num_procs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int local_size = N / num_procs;

    // Instead of raw arrays, which automatically manage memory and provide better safety
    vector<int> localA(local_size);
    vector<int> A(N); // Allocate A on all processes

    if (rank == 0)
    {
        generateRandomArray(A.data(), N);
        cout << "UNSORTED" << endl;
        displayArray(A.data(), N);
    }

    // Scatter input data to all processes
    MPI_Scatter(A.data(), local_size, MPI_INT, localA.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // MPI Initialization
      // Add a barrier to ensure all processes have scattered data before quicksort
    MPI_Barrier(MPI_COMM_WORLD);

    auto start = high_resolution_clock::now();

    // get available OpenCL platforms
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    // get available OpenCL devices (use the first platform)
    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    //
    if (devices.empty()) {
        platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
    }
    // create OpenCL context for the device
    cl::Context context(devices);
    // create OpenCL command queue for the device
    cl::CommandQueue queue(context, devices[0]);
    // create OpenCL memory buffer for array A - local
    cl::Buffer buffer_localA(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * local_size, localA.data());


    // kernel function #1       QUICKSORT + PARTITION together
    const char* quicksortKernelSource = R"(
            __kernel void quicksort_partition(__global int* array, int low, int high)
            {
                // declare an integer array 'stack' to serve as a stack data structure for keeping track of subproblems; top is initialized to -1, indicating that the stack is initially empty
                int stack[32];
                int top = -1;

                // low and high indices of the initial problem are pushed onto the stack. This effectively initializes the stack with the full range of indices to sort
                stack[++top] = low;
                stack[++top] = high;

                // do until stack is empty
                while (top >= 0)
                {
                    high = stack[top--];
                    low = stack[top--];

                    // this loop is responsible for partitioning the subarray into two smaller subarrays
                    while (low < high)
                    {
                        // FROM the initial pivot function:
                        int pivot = array[low];

                        // left and right bounds in the subarray
                        int i = low, j = high;

                        while (i < j)
                        {
                            while (array[i] < pivot)
                                i++;

                            while (array[j] > pivot)
                                j--;

                            if (i <= j)
                            {
                                int temp = array[i];
                                array[i] = array[j];
                                array[j] = temp;
                                i++;
                                j--;
                            }
                        }

                        // this logic decides which subarray to push onto the stack for further processing. It chooses the smaller subarray and pushes 
                        // its indices onto the stack. This ensures that smaller subproblems are processed first, which is a common strategy in quicksort 
                        // to limit the maximum stack depth.
                        if (j - low < high - i)
                        {
                            stack[++top] = i;
                            stack[++top] = high;
                            high = j;
                        }
                        else
                        {
                            stack[++top] = low;
                            stack[++top] = j;
                            low = i;
                        }
                    }
                }
            }
        )";
    // kernel function #2           MERGE
    const char* mergeKernelSource = R"(
        __kernel void merge(__global int array[], __global int left[], __global int right[], int leftSize, int rightSize)
        {
            int i = 0;
            int j = 0;
            int k = 0;
                
            while (i < leftSize && j < rightSize)
            {
                if (left[i] <= right[j])
                    array[k++] = left[i++];
                else
                    array[k++] = right[j++];
            }

            while (i < leftSize)
                array[k++] = left[i++];

            while (j < rightSize)
                array[k++] = right[j++];
        }
    )";

    // Load & compile the OpenCL program
    cl::Program program(context, quicksortKernelSource);
    cl::Program program2(context, mergeKernelSource);

    program.build(devices);
    program2.build(devices);

    // create OpenCL kernels
    cl::Kernel kernel(program, "quicksort_partition");
    cl::Kernel kernel2(program2, "merge");

    // Perform quicksort on localA ; before it was :    quicksort(localA.data(), 0, local_size - 1); Now, instead we have lines 265 - 274
    // Set the kernel arguments
    kernel.setArg(0, buffer_localA);
    kernel.setArg(1, 0);
    kernel.setArg(2, local_size - 1);

    // Enqueue the kernel for execution
    queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(local_size));
    queue.finish();

    // Read the sorted data back from the buffer
    queue.enqueueReadBuffer(buffer_localA, CL_TRUE, 0, sizeof(int) * local_size, localA.data());

    // Add a barrier to ensure all processes have completed quicksort before merging
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather the sorted portions back to process 0
    MPI_Gather(localA.data(), local_size, MPI_INT, A.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "After QuickSort (Before Merge):" << endl;
        displayArray(A.data(), N); // Print the sorted array before the merge

        // Merge sorted portions
        // merge function can merge 2 subarrays at a time, but when we run with 6, 8 12 etc processes we need to run merge few times 
        // with each iteration 'step' counter is doubled
        // the loop continues until step is equal to or greater than the number of processes
        // for example when N=16, np = 4 then we would get 4 subarrays of 4 items. We would run merge on first 2 subarrays, then merge
        // on other 2 subarrays , then merge on the two subarrays that we got from both merges. 
        //  ** So we assume we need 3 merge function calls **
        // we start at :
        // first run  : (step = 1; 1<4 ; step = 2)  ----------------run
        // second run : (step = 2; 2< 4; step = 2*2) ---------------run
        // third run  : (step = 4; 4<4; ...) ------------------exit at false
        for (int step = 1; step < num_procs; step *= 2)
        {
            // iterate through number of processes
            // continuing example above
            // first run  : with step = 1 :  (i = 0; 0<4 ; i = 0 + 2*1), then (i = 2; 2 < 4 ; i = 2+ 2*1) , then (i= 4; 4<4 ; ..) -> exit at false
            // second run : with step = 2 :  (i= 0; i< 4; i = 0 + 2 * 2) run, then (i=4; 4 < 4; ...) -> exit at false
            // third run  : with step = 4  never happens
            // we see from 1st and 2nd run we run second for loop 3 times
            for (int i = 0; i < num_procs; i += 2 * step)
            {
                // in first outer for loop run this will be run once
                // determine the sizes of subarrays
                int leftSize = std::min(local_size * step, N - i * local_size);            // so for 1st step at index 0 (very first run) we get min(4 * 1, 16 - 0 * 4 ) = min (4,16) = 4 meaning during 1st run we have 4 elements
                int rightSize = std::min(local_size * step, N - (i + step) * local_size);  // here: min( 4 * 1, 16 - (0+1) * 4) = min (4,12) = 4 , so during 1st run we still have 4 elements in the other subarray

                //  merging only occurs if there are pairs of subarrays to merge
                // so based on the example of N=16, np = 4 we would require 3 merges, so we would stop at i+step = 4;
                // we started at step = 1 and i=0, so we would 
                // first run  : i + step = 0 + 1 = 1 , then i + step = 2 + 1 = 3  (2 x merge clause run)
                // second run : i + step = 0 + 2 = 2 (1 x merge clause run)
                // third step : i + step = 0 + 4 = 4 (abort)
                // so before we exit the first for loop we will run the below if clause 3 times, which confirms our assumption from line 293

                if (i + step < num_procs)
                {
                    // create OpenCL buffers for the left and right subarrays to be merged, as well as a buffer to store the result of the merge
                    cl::Buffer left_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * leftSize, A.data() + i * local_size);
                    cl::Buffer right_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * rightSize, A.data() + (i + step) * local_size);
                    cl::Buffer result_buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * (leftSize + rightSize));

                    // set merge kernel arguments
                    kernel2.setArg(0, result_buffer);
                    kernel2.setArg(1, left_buffer);
                    kernel2.setArg(2, right_buffer);
                    kernel2.setArg(3, leftSize);
                    kernel2.setArg(4, rightSize);

                    // enqueue for execution
                    queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(leftSize, rightSize));
                    queue.finish();

                    // Copy the merged data back to the original array on host memory 
                    queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0, sizeof(int) * (leftSize + rightSize), A.data() + i * local_size);
                
                    // Release OpenCL resources
                    clReleaseMemObject(left_buffer());
                    clReleaseMemObject(right_buffer());
                    clReleaseMemObject(result_buffer());
                }
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);

        cout << "Quicksort and merge applied" << endl;
        // Display sorted array or any other required output
        displayArray(A.data(), N);
        cout << "MPI + OpenCL execution time: " << duration.count() << " microseconds" << endl;

        // Release OpenCL resources
        queue.finish();
        clReleaseKernel(kernel());
        clReleaseKernel(kernel2());
        clReleaseMemObject(buffer_localA());
        //clReleaseProgram(program());
        //clReleaseProgram(program2());
        //clReleaseCommandQueue(queue());
        //clReleaseContext(context());

    }

    MPI_Finalize();

    return 0;
}
