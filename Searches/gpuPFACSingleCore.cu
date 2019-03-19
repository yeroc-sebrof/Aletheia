#include <array>
#include <algorithm>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

// define the types between unix and windows

#ifdef _WIN32

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#endif // _WIN32

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#define DEBUG true

using thrust::host_vector;
using thrust::device_vector;
using thrust::device_ptr;
using thrust::device_pointer_cast;

using std::thread;
using std::pair;
using std::vector;
using std::array;
using std::string;


// using size_t is too inconsistent between OS
#define tablePointerType uint16_t

#define chunkSize 20000

// This could be changed to 257 to allow for wildcards after setup to allow for this
#define rowSize (int)256

// Setup //
host_vector<tablePointerType> pfacLookupCreate(vector<string> patterns)
{
	/*
	The PFAC table can be described as:
		a starting row that will be sent to upon error (which all rows have as default values)
		rows for when the end of the pattern is found
		row for starting the search from - how does the program know where the starting row is at runtime?
		all of the redirections from each value given
	*/

	//sort the string list provided first by size then by character
	// https://stackoverflow.com/questions/45865239/how-do-i-sort-string-of-arrays-based-on-length-then-alphabetic-order
	sort(patterns.begin(), patterns.end(),
		[](const std::string& lhs, const std::string& rhs) {
		return lhs.size() == rhs.size() ?
			lhs < rhs : lhs.size() < rhs.size(); });

	// The malleable table that will be returned as a '2D array'
	host_vector<tablePointerType> table((2 + patterns.size()) * rowSize);
	// This has been initialising as 0 making resizing easy

	size_t currentPlace;

	// for each pattern
	for (int i = 0; i < patterns.size(); ++i)
	{
		currentPlace = patterns.size() + 1;

		// for each character bar last
		for (int j = 0; j < patterns[i].size() - 1; ++j)
		{
			// if the current place doesnt point anywhere
			if (table[(currentPlace * rowSize) + patterns[i][j]] == 0)
			{
				// make a new row in match table to point at
				table.resize(table.size() + rowSize);
				//std::fill(table.end() - rowSize, table.end(), 0);

				// point at it at current place
				table[(currentPlace * rowSize) + patterns[i][j]] = (tablePointerType)((table.size() / rowSize) - 1);
			}

			// follow where the pointer say's is the next space
			currentPlace = table[(currentPlace * rowSize) + patterns[i][j]];

		}

		// final char
		table[(currentPlace * rowSize) + patterns[i][patterns[i].size() - 1]] = (tablePointerType)(i + 1); // point to complete pattern

	}

	return table;
}

// Execution //
__global__ void pfacSearch(int* needleFound, bool* results, char* haystack, int haysize, tablePointerType startingRow, device_ptr<tablePointerType> pfacMatching)
{
	// Stride will be worked out next
	for (size_t i=0; i < haysize; ++i)
	{
		// if the current character is pointing to another character in the starting row
		if (pfacMatching[(rowSize * startingRow) + haystack[i]] != 0)
		{
			// init as the place of the 2nd char in the pattern if the above shows as non-zero
			int currentPlace = pfacMatching[(rowSize * startingRow) + haystack[i]];

			// check pfac row currentPlace for the haystack char i+j
			for (int j = 1; pfacMatching[(currentPlace * rowSize) + haystack[i + j]] != 0; ++j)
			{
				// current place becomes the next character in the pattern
				currentPlace = pfacMatching[(currentPlace * rowSize) + haystack[i + j]];

				// If we have reached the end of a pattern
				if (currentPlace < startingRow)
				{
					// where a prefix pattern exists of a bigger pattern this will push multiple times
					results[i] = true; // using this type it is a non issue however
					atomicAdd(needleFound, 1); // Was having issues with just needleFound++
				}
			}
		}
	}

	return;
}

cudaError_t cudaManager(char haystack[], int haysize, host_vector<int> pfacTable, int startingRow, vector<size_t>& foundPatterns)
{
	// To be returned on any issue
	cudaError_t cudaStatus;

	// Where we are making use of the bit-shifty method of storage we want to know how many bytes are needing alloc'd
	//int hayBits = haysize;
	//if (haysize % 8)
	//	hayBits += (8-(haysize % 8));

	// This value will be used to triage if a pattern was even found
	int needlesFound = 0;
	int* cuda_needlesFound = 0;

	// Pointer to allocate cuda memory to the cuda device
	char* cuda_haystack = 0;
	bool* cuda_resultArray = 0; // TODO HAYBITS
	device_vector<tablePointerType> cuda_pfacTable;

	// buffer to bring the foundPatterns back to for a quick re-search
	bool buffer[chunkSize];

	// Choosing the first GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	/* Add a method to expand the users choice regarding this. There should also be a method regarding the
	classification of acceptable hardware*/
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed! CUDA capable hardware is required for this process?\n");
		goto Error;
	}

	// Allocate memory that CUDA can access //
	// haystack -- in
	cudaStatus = cudaMalloc((void**)&cuda_haystack, haysize * sizeof(char));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!\n");
		goto Error;
	}
	
	// This line should copy the contents of the host_vector into the device -- in
	// Thrust lib handles the back end of this and the block below is here in the even the structures in use change
	cuda_pfacTable = pfacTable;

	// patterns found -- out
	cudaStatus = cudaMalloc((void**)&cuda_resultArray, haysize); // TODO HAYBITS
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 1 failed!\n");
		goto Error;
	}

	// 
	cudaStatus = cudaMalloc((void**)&cuda_needlesFound, sizeof(int));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc 2 failed!\n");
		goto Error;
	}

	// Copy input from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(cuda_haystack, haystack, haysize * sizeof(char), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to device failed!\n");
		goto Error;
	}
	
	//// Run on Cuda
	pfacSearch <<<2, 4>>> (cuda_needlesFound, cuda_resultArray, cuda_haystack, haysize, startingRow, device_pointer_cast(&cuda_pfacTable[0]));

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	//// Wait for GPU to finish here
	cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	////Housekeeping
	cudaStatus = cudaMemcpy(&needlesFound, cuda_needlesFound, sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy to Host 1 failed!\n");
			goto Error;
	}

	if (needlesFound == 0)
	{
		fprintf(stderr, "There were no Needles found!\n");
		goto Error;
	}
	else
		fprintf(stdout, "Needles found == %i\n", needlesFound);

	cudaStatus = cudaMemcpy(buffer, cuda_resultArray, haysize, cudaMemcpyDeviceToHost); // TODO HAYBITS
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy to Host 2 failed!\n");
		goto Error;
	}

	// This can be done later on a separate CPU thread. Multiple buffer system will also be required for the loop
	for (int i = 0; i < haysize; ++i)
	{
		if (buffer[i])
			foundPatterns.push_back(i);
	}

Error:
	cudaFree(cuda_haystack);
	cudaFree(cuda_resultArray);
	return cudaStatus;
};

int main()
{
	vector<string> patterns = { "any", "three", "words", "rod", "word" };
	host_vector<tablePointerType> pfacMatching = pfacLookupCreate(patterns);
	vector<size_t> results;

	char haystack[] = "i don't know; do I go with any three words that go along with those words I said?";

	// CUDA Manager
	//pair<tablePointerType, vector<tablePointerType>> pfacMatching = { 0, 0 };
	//pfacMatching.first = (tablePointerType)(patterns.size() + 1);
	cudaManager(haystack, sizeof(haystack), pfacMatching, patterns.size()+1, results);
	//// End the thread managment. This could be looped between fetching new chunks

	if (results.size())
	{
		printf("\n\nPatterns start at chars: ");

		for (int i = 0; i < results.size(); ++i)
			printf("%i ", (int)results[i]);
	}
	else
	{
		printf("\nNo Matches");
		return 1;
	}

	return 0;
}