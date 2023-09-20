#include <iostream>
#include <string>
#include <random>
#include <time.h>
#include <chrono>
#include <mpi.h>
#include <cstdlib>

using namespace std;
using namespace std::chrono;

const long long N = 10;

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


int main(int argc, char** argv)
{
    // MPI
    int num_procs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const int local_size = N / num_procs;

    int* localA = new int[local_size];
    int* A = new int[N]; // Allocate A on all processes

    if (rank == 0)
    {
        generateRandomArray(A, N); 
        cout << "UNSORTED" << endl;
        displayArray(A, N);

    }
    // Scatter input data to all processes
    MPI_Scatter(A, local_size, MPI_INT, localA, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform QS on the array; record the time taken by the process
    auto start = high_resolution_clock::now();
    quicksort(localA, 0, local_size - 1);
    auto end = high_resolution_clock::now();

   
    // Gather the sorted portions back to process 0
    MPI_Gather(localA, local_size, MPI_INT, A, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {


        // Measure time and display sorted array

        auto duration = duration_cast<microseconds>(end - start);

        cout << "Quicksort applied" << endl;
        displayArray(A, N);
        cout << "MPI execution time: " << duration.count() << " microseconds" << endl;
    }

    delete[] localA;
    if (rank == 0)
    {
        delete[] A;
    }

    MPI_Finalize();

    return 0;
}