#include <iostream>
#include <string>
#include <random>
#include <time.h>
#include <chrono>
#include <mpi.h>
#include <cstdlib>

using namespace std;
using namespace std::chrono;
const int N = 500000;

void generateRandomArray(int* array, int size)
{
	// Random Number Generation
    // random_device rd;
	// the above implementation of std::random_device did not appear so random. I used the current time as seed (below) instead to give better random values.

    mt19937 e2(time(nullptr));

    // Initialize array with non-zero values; if we skip this, there will be a zero value appearing in the sorted array, which 'steals' the spot of a number 
	// from the initial array; we initialize with any number outside of the random number generator, here the range was set between -40000000 and 40000000.
	// so we can use numbers under the lower linit or over the upper limit from line
	for (int i = 0; i < size; i++) {
		array[i] = -999; // Set to a large negative value (guaranteed to be outside the range)
	}

	// Initialize array with random values
    uniform_int_distribution<int> dist(-100, 100);

	// Generate random values between -40000000 and 40000000 and assign them to A[i]
	// the number from line 66 will be overwritten
    for (int i = 0; i < size; i++) {
        int random_value = dist(e2);
        array[i] = random_value; // Assign the generated random value to array[i]
    }


}

void displayArray(int array3[], int arraySize)
{
    for (int i = 0; i < arraySize; i++)
    {
        cout << array3[i] << "  ";
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
        
        swap(array3[i], array3[j]);
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


// Merge step #1

//      this function takes 5 arguments:
// 1. int array[] - this will hold the entire, final, sorted & merged array
// 2. int left[], int right[] - the subarrays that are already sorted; if we run the program with 2 processes we will get 2 subarrays,
//                              4 processes = 4 subarrays, 8 processes = 8 subarrays etc. The important note is that array size N
//                              must be dividable by the processes number: N=100, np = 4 (OK), but N=100, np = 7 will not work !!
// 3. int leftSize, int rightSize - how many numbers each set has. Because of the condition from point 2 above, each set must have
//                                  the same number of elements for comparison
void merge(int array[], int left[], int right[], int leftSize, int rightSize)
{
// counters: left, right, final array; initialized to the address of first items at index 0
    int i,j,j = 0;

    //  use a temporary array for merging and copy the merged results back into A, because without that this function may overwrite elements
    // I tried and without temp array, it overwrites, so:
    vector<int> temp(leftSize + rightSize);

    // while within the bounds of each subarray; it might happen that we will end up comparing left subarray with one last item
    // against right that still has for instance 3 elements left. That is ok, subarrays should only start with even number of items, later it does not matter
    // && condition means we will have to perform comparison of both subarrays to their last item
    while (i < leftSize && j < rightSize)
    {
        // compare first pair of indices (at index 0) from both subarrays
        // if the 1st item from subarray left is smaller or equal, then it has precedence to be put in the temp/final array 
        if (left[i] <= right[j])
            //  temp[k] is set to left[i], and both k and i are incremented, moving the comparison and insertion to the 
            //  next elements in the left and temp arrays; the right index place though is not incremented, meaning that again
            //  we will compare against it
            temp[k++] = left[i++];
        else
            // the opposite of above; here the right subarray's index is incremented, the left one stays in same place
            temp[k++] = right[j++];
    }

    // those while loops just make sure we have hopped over each item in the subarrays; we could skip this step
    while (i < leftSize)
        temp[k++] = left[i++];

    while (j < rightSize)
        temp[k++] = right[j++];

    // Copy merged elements from temp array back to the original array
    for (int x = 0; x < leftSize + rightSize; x++)
    {
        array[x] = temp[x];
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

    // instead of raw arrays, which automatically manages memory and provides better safety in case memory allocation fails
    vector<int> localA(local_size);
    vector<int> A(N); // Allocate A on all processes

    if (rank == 0)
    {
        generateRandomArray(A.data(), N); 
        cout << "UNSORTED" << endl;
        displayArray(A.data(), N);

    }
    // Data distribution : scatter input data (that was broken into chunks) to all processes
    MPI_Scatter(A.data(), local_size, MPI_INT, localA.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform QS on the array; record the time taken by the process
    auto start = high_resolution_clock::now();
    quicksort(localA.data(), 0, local_size - 1);
   
    // Gather the sorted portions back to process 0; each chunk is sorted, but not the whole array. The final step is merge, see below
    MPI_Gather(localA.data(), local_size, MPI_INT, A.data(), local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        // Merge step #2
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
                // so before we exit the first for loop we will run the below if clause 3 times, which confirms our assumption from line 174
                if (i + step < num_procs)
                {
                    // void merge(int array[], int left[], int right[], int leftSize, int rightSize)

                    merge(
                    // A.data() -> returns an pointer pointing to the first element in the array object 
                    // in line 147 array A was created with size N so here = 16                                   1st merge run             2nd merge run           3rd merge run
                    A.data() + i * local_size,  // int array[]                        start index of array:          0 * 4 = 0                2 * 4 = 8                0 * 4 = 0
                    A.data() + i * local_size,  // int left []                        start index of left array:     0 * 4 = 0                2 * 4 = 8                0 * 4 = 0
                    A.data() + (i + step) * local_size, // int right[]                start index of right arr :     (0+1) * 4 = 5            (2+1) * 4 = 12          (0+2) * 4 = 8          
                    leftSize, 
                    rightSize);                                                 
                }
            }
        }
        // record end time (Quicksort on subarrays + merge them into 1 big array
        auto end = high_resolution_clock::now();
        // Measure time and display sorted array

        auto duration = duration_cast<microseconds>(end - start);

        cout << "Quicksort applied" << endl;
        displayArray(A.data(), N);
        cout << "MPI execution time: " << duration.count() << " microseconds" << endl;
    }


    MPI_Finalize();

    return 0;
}
