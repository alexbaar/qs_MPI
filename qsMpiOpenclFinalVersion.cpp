#include <iostream>
#include <string>
#include <random>
#include <time.h>
#include <chrono>
#include <mpi.h>
#include <CL/cl2.hpp>

using namespace std;
using namespace std::chrono;
const int N = 16;

void generateRandomArray(int* array, int size)
{
    // Random Number Generation
    mt19937 e2(time(nullptr));

    // Initialize array with non-zero values
    for (int i = 0; i < size; i++) {
        array[i] = -999; // Set to a large negative value (guaranteed to be outside the range)
    }

    // Initialize array with random values
    uniform_int_distribution<int> dist(-100, 100);

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

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::vector<cl::Device> devices;
    platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

    if (devices.empty()) {
        platforms[0].getDevices(CL_DEVICE_TYPE_CPU, &devices);
    }

    cl::Context context(devices);

    cl::CommandQueue queue(context, devices[0]);

    cl::Buffer buffer_localA(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, sizeof(int) * local_size, localA.data());

    const char* quicksortKernelSource = R"(
            __kernel void quicksort(__global int array[], int low, int high)
            {
                int i = get_global_id(0);
                int j = high;
                int pivot = array[low];

                while (true)
                {
                    while (array[i] < pivot)
                        i++;

                    while (array[j] > pivot)
                        j--;

                    if (i >= j)
                        break;

                    int temp = array[i];
                    array[i] = array[j];
                    array[j] = temp;

                    i++;
                    j--;
                }

                if (low < high)
                {
                    int partitionIndex = partition(array, low, high);
                    quicksort(array, low, partitionIndex);
                    quicksort(array, partitionIndex + 1, high);
                }
            }
        )";

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

    cl::Program program(context, quicksortKernelSource);
    cl::Program program2(context, mergeKernelSource);

    program.build(devices);
    program2.build(devices);

    cl::Kernel kernel(program, "quicksort");
    cl::Kernel kernel2(program2, "merge");

    // Perform quicksort on localA
    quicksort(localA.data(), 0, local_size - 1);

    // Add a barrier to ensure all processes have completed quicksort before merging
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather the sorted portions back to process 0
    MPI_Gather(localA.data(), local_size, MPI_INT, A.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        cout << "After QuickSort (Before Merge):" << endl;
        displayArray(A.data(), N); // Print the sorted array before the merge

        // Merge sorted portions
        for (int step = 1; step < num_procs; step *= 2)
        {
            for (int i = 0; i < num_procs; i += 2 * step)
            {
                int leftSize = std::min(local_size * step, N - i * local_size);
                int rightSize = std::min(local_size * step, N - (i + step) * local_size);

                if (i + step < num_procs)
                {
                    cl::Buffer left_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * leftSize, A.data() + i * local_size);
                    cl::Buffer right_buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * rightSize, A.data() + (i + step) * local_size);
                    cl::Buffer result_buffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * (leftSize + rightSize));

                    kernel2.setArg(0, result_buffer);
                    kernel2.setArg(1, left_buffer);
                    kernel2.setArg(2, right_buffer);
                    kernel2.setArg(3, leftSize);
                    kernel2.setArg(4, rightSize);

                    queue.enqueueNDRangeKernel(kernel2, cl::NullRange, cl::NDRange(leftSize, rightSize));
                    queue.finish();

                    // Copy the merged data back to the original array
                    queue.enqueueReadBuffer(result_buffer, CL_TRUE, 0, sizeof(int) * (leftSize + rightSize), A.data() + i * local_size);
                }
            }
        }

        auto end = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(end - start);

        cout << "Quicksort and merge applied" << endl;
        // Display sorted array or any other required output
        displayArray(A.data(), N);
        cout << "MPI + OpenCL execution time: " << duration.count() << " microseconds" << endl;
    }

    MPI_Finalize();

    return 0;
}