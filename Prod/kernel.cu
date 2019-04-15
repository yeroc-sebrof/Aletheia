#include "../../fileChunkingClass/fileHandler.h"

#include <algorithm>

// define the types between unix and windows
#ifdef _WIN32

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// using size_t is too inconsistent between OS
#define tablePointerType uint16_t
#define hayCountType 

#elif // _WIN32

// fix this to have a different unix compat type
#define tablePointerType uint16_t

#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#define DEBUG true

using thrust::host_vector;
using thrust::device_vector;
using thrust::device_ptr;
using thrust::device_pointer_cast;

using std::vector;
using std::string;
using std::thread;
using std::pair;

#define chunkSize (unsigned long int)768*KB

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

	if (patterns.size() > (1 << 15))
	{
		cerr << "Too many elements there will be an issue with generating with the current pointer type";
		exit(4);
	}

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

#if DEBUG
	cout << endl << "PFAC SIZE - " << table.size() << " x " << rowSize << endl;
#endif // DEBUG


	return table;
}

// Execution //
__global__ void pfacSearch(unsigned int* needleFound, bool* results, char* haystack, unsigned long int haysize, tablePointerType startingRow, device_ptr<tablePointerType> pfacMatching)
{
	// Stride will be worked out next
	unsigned long int stride = blockDim.x * gridDim.x;
	unsigned long int currentPlace;
	unsigned short int j;

	for (unsigned long int i = threadIdx.x + (blockIdx.x * blockDim.x); i < haysize; i += stride)
	{
		// if the current character is pointing to another character in the starting row
		if (pfacMatching[(rowSize * startingRow) + haystack[i]] != 0)
		{
			// init as the place of the 2nd char in the pattern if the above shows as non-zero
			currentPlace = pfacMatching[(rowSize * startingRow) + haystack[i]];

			// check pfac row currentPlace for the haystack char i+j
			for (j = 1; pfacMatching[(currentPlace * rowSize) + haystack[i + j]] != 0; ++j)
			{
				// current place becomes the next character in the pattern
				currentPlace = pfacMatching[(currentPlace * rowSize) + haystack[i + j]];

				// If we have reached the end of a pattern
				if (currentPlace < startingRow)
				{
					// where a prefix pattern exists of a bigger pattern this will push multiple times
					results[i] = true; // using this type it is a non issue however
					atomicAdd(needleFound, 1);
				}
			}
		}
	}

	return;
}

void resultParse(vector<unsigned int>* foundPatterns, bool* buffer, unsigned long int chunkNo)
{ // Plan was there to make this even easier using bools without padding using bitshifting methods
	
	for (unsigned int i = 0; i < chunkSize; ++i)
	{
		if (buffer[i])
		{
			foundPatterns->push_back(i + (chunkNo * chunkSize));
		}
	}
	
	return;
}

cudaError_t cudaManager(fileHandler& chunkManager, host_vector<tablePointerType> pfacTable, tablePointerType startingRow, vector<unsigned int>& foundPatterns)
{
	// Local Variables
	thread processResults;

	unsigned int needlesFound = 0; // triage if the search of starting bytes is even required

	/// Haybits
	bool cpu_resultArrayBuffer[chunkSize]; // Buffer to hold the returning bit array from the GPU
	
	// Start the chunk fetching
	chunkManager.readFirstChunk();
	

	// Cuda Management device property management //
	cudaError_t cudaStatus;
	cudaDeviceProp prop;

	device_vector<tablePointerType> cuda_pfacTable;  // Vector that GPU can access -- Thrust Lib

	// Choosing the First GPU to run on. Multiple GPU's is out of scope/future work
	cudaStatus = cudaSetDevice(0); // Further choice behind which GPU should also be implemented for the user
	if (cudaStatus != cudaSuccess) {
		cerr << endl << "cudaSetDevice failed! CUDA capable hardware is required for this process?" << endl;  goto Error;
	}

	// Querying the device for properties
	cudaStatus = cudaGetDeviceProperties(&prop, 0);
	if (cudaStatus != cudaSuccess) {
		cerr << endl << "GetDeviceProperties failed!" << endl;  goto Error;
	}
	

	// Start GPU Mallocs //
	char* cuda_haystack = 0; // Array of Char's containing chunks of haystack
	bool* cuda_resultArray = 0; // Array of Boolean Values indicating found pattern

	unsigned int* cuda_needlesFound = 0; // Single Unsigned int

	// Copy of the PFAC table
	cuda_pfacTable = pfacTable; // Populating GPU Vector

	cudaStatus = cudaMalloc((void**)&cuda_haystack, chunkSize); // Assigning haystack to memory. Char's fit within chunkSize
	if (cudaStatus != cudaSuccess) {
		cerr << endl << "cudaMalloc cuda_haystack failed!" << endl;  goto Error;
	}

	/// Haybits
	cudaStatus = cudaMalloc((void**)&cuda_resultArray, chunkSize); // Assigning result array. Bool == 1 byte
	if (cudaStatus != cudaSuccess) {
		cerr << endl << "cudaMalloc resultArray failed!" << endl;  goto Error;
	}

	cudaStatus = cudaMalloc((void**)&cuda_needlesFound, sizeof(unsigned int)); // Malloc one unsigned int
	if (cudaStatus != cudaSuccess) {
		cerr << endl << "cudaMalloc resultArray failed!" << endl;  goto Error;
	}

	while (chunkManager.getCurrChunkNo() < chunkManager.getTotalChunks()-1) // Loop through all of the available chunks
	{
		// Copy haystack into the GPU //
		chunkManager.waitForRead(); // Make sure the chunk is read in

		cudaStatus = cudaMemcpy(cuda_haystack, chunkManager.buffer, chunkSize * sizeof(char), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			cerr << endl << "cudaMemcpy of chunkManager.buffer to device failed!" << endl;  goto Error;
		}

		chunkManager.asyncReadNextChunk();
#if DEBUG
		cout << endl << chunkManager.getCurrChunkNo() << " Complete of " << chunkManager.getTotalChunks() << endl;
#endif // DEBUG
		

		// Run Search of current chunk on Device //
		// Types										// u long				bool			 char[]			u long		tablePoint	thrust::device_ptr
		pfacSearch <<< 16, prop.maxThreadsPerBlock >>> (cuda_needlesFound, cuda_resultArray, cuda_haystack, chunkSize, startingRow, device_pointer_cast(&cuda_pfacTable[0]));

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			cerr << endl << "addKernel launch failed: " << cudaGetErrorString(cudaStatus) << endl;  goto Error;
		}


		// Wait for GPU to finish before restarting with the next chunk //
		cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			cerr << endl << "cudaDeviceSynchronize returned error code " << cudaStatus << " after launching addKernel!" << endl;  goto Error;
		}


		// Gather Results // 
		cudaStatus = cudaMemcpy(&needlesFound, cuda_needlesFound, sizeof(unsigned int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			cerr << endl << "cudaMemcpy to Host of cuda_needlesFound failed!" << endl;  goto Error;
		}

		if (needlesFound == 0)
		{
			cout << endl << "There were no Needles found in chunk ";
		}
		else
		{
			cout << endl << "Needles found equals " << needlesFound << endl;
			
			cudaStatus = cudaMemcpy(cpu_resultArrayBuffer, cuda_resultArray, chunkSize, cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				cerr << endl << "cudaMemcpy to Host of cuda_resultArray failed!" << endl;  goto Error;
			}

			if (processResults.joinable())
				processResults.join();

			processResults = thread(resultParse, &foundPatterns, cpu_resultArrayBuffer, chunkManager.getCurrChunkNo());
		}


		// Reset the GPU values //
		cudaStatus = cudaMemset(cuda_needlesFound, 0 , sizeof(unsigned int));
		if (cudaStatus != cudaSuccess) {
			cerr << endl << "cudaMemset of cuda_needlesFound failed!" << endl;  goto Error;
		}

		cudaStatus = cudaMemset(cuda_resultArray, 0, chunkSize);
		if (cudaStatus != cudaSuccess) {
			cerr << endl << "cudaMemset of cuda_resultArray failed!" << endl;  goto Error;
		}
	}

Error:
	if (processResults.joinable())
		processResults.join();

	cudaFree(cuda_haystack);
	cudaFree(cuda_resultArray);
	cudaFree(cuda_needlesFound);
	return cudaStatus;
};

int main()
{
	// Generate the PFAC table on CPU before we start
	vector<string> patterns = { "any", "three", "that", "rod", "word" };

	if (patterns.size() > (1 << 13))
	{
		cerr << endl << endl << "There was an issue regarding the number of patterns" << endl << "The implementation may not be able to function correctly with this. tablePointerType may require being increased. If this error has arose then hardware should have advanced to also be able to handle it";
		exit(1);
	}

	host_vector<tablePointerType> pfacMatching = pfacLookupCreate(patterns);
	vector<unsigned int> results;

	fileHandler chunkManager("200MBWordlist.test", chunkSize);

	tablePointerType startingRow = patterns.size() + 1;

	// CUDA Manager device manager
	cudaManager(chunkManager, pfacMatching, startingRow, results);

	if (results.size())
	{
		cout << endl << results.size() << " Patterns Found";
		//for (int i = 0; i < results.size(); ++i)
		//{
		//	cout << endl << "Found Patterns in chunk " << results[i].first << endl;
			//for (int j = 0; j < results[i].second.size(); ++j)
				//cout << results[i].second[j] << " ";
		//}

		//for (int j = 0; j < results.size(); ++j)
		//	cout << results[j] << " ";

	}
	else
	{
		printf("\nNo Matches");
	}

	return 0;
}