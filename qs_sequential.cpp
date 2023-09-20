#include <iostream>
#include <string>
#include <random>
#include <time.h>
#include <chrono>

using namespace std;
using namespace std::chrono;

const long long N = 500000;

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

int main()
{
	// Random Number Generation
    // random_device rd;
	// the above implementation of std::random_device did not appear so random. I used the current time as seed (below) instead to give better random values.

    mt19937 e2(time(nullptr));
	int A[N];

	// Initialize array with non-zero values; if we skip this, there will be a zero value appearing in the sorted array, which 'steals' the spot of a number 
	// from the initial array; we initialize with any number outside of the random number generator, here the range was set between -40000000 and 40000000.
	// so we can use numbers under the lower linit or over the upper limit from line
	for (int i = 0; i < N; i++) {
		A[i] = -999; // Set to a large negative value (guaranteed to be outside the range)
	}

	// Initialize array with random values
	uniform_int_distribution<int> dist(-100, 100); 

	// Generate random values between -100 and 100 and assign them to A[i]
	// the number from line 66 will be overwritten
	for (int i = 0; i < N; i++) {
		int random_value = dist(e2);
		A[i] = random_value; // Assign the generated random value to A[i]
	}

	// show unsorted array, populated with random values
    cout << "UNSORTED" << endl;
    //displayArray(A, N);

    cout << "Quicksort applied" << endl;

	// perform QS on the array; record the time taken by the process
	auto start = high_resolution_clock::now();
    quicksort(A, 0, N-1);
	auto end = high_resolution_clock::now();
	// chose microseconds 
	// 1 second = 1 000 000 microseconds
    auto duration = duration_cast<microseconds>(end - start);

	// Show sorted array and print time taken by QS operation
    //displayArray(A, N);
	// Print execution time
    cout << "Sequential execution time: " << duration.count() << " microseconds" << endl;
	
    return 0;
}
