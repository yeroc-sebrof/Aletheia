#include "fileChunkingClass/fileHandler.h"

#include <chrono>
#include <algorithm>

using std::vector;
using std::string;
using std::thread;
using std::pair;
using std::mutex;
using std::unique_lock;

// Timings
using namespace std::chrono;

// This could be changed to 257 to allow for wildcards after setup to allow for this
#define rowSize (int)256
#define chunkSize 200*MB

#define tablePointerType uint16_t
#define hayCountType unsigned int

std::ostream& operator < (std::ostream& os, const std::basic_string<uchar>& str) {
	for (auto ch : str)
		os << static_cast<char>(ch);
	return os;
};

// Setup //
vector<tablePointerType> pfacLookupCreate(vector<ustring>& patterns)
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

	//sort the string list provided first by size then by character -- This was modified slightly to work with ustring
	// https://stackoverflow.com/questions/45865239/how-do-i-sort-string-of-arrays-based-on-length-then-alphabetic-order
	sort(patterns.begin(), patterns.end(),
		[](const ustring& lhs, const ustring& rhs) {
		return lhs.size() == rhs.size() ?
			lhs < rhs : lhs.size() < rhs.size(); });

	// The malleable table that will be returned as a '2D array'
	vector<tablePointerType> table((2 + patterns.size()) * rowSize);
	// This has been initialising as 0 making resizing easy

	unsigned long int currentPlace;

	// for each pattern
	for (unsigned int i = 0; i < patterns.size(); ++i)
	{
		currentPlace = patterns.size() + 1;

		// for each character bar last
		for (unsigned int j = 0; j < patterns[i].size() - 1; ++j)
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
	cout << endl << "PFAC SIZE - " << table.size() << endl;
#endif // DEBUG


	return table;
}

int pfacSearch(mutex* result, vector<size_t>* results, unsigned int tid, unsigned int totalthreads, uchar* haystack, unsigned long int haysize, pair<tablePointerType, vector<tablePointerType>> pfacMatching)
{
	// Stride will be worked out next
	for (size_t i = tid; i < haysize; i += totalthreads)
	{
		// if the current character is pointing to another character in the starting row
		if (pfacMatching.second[(rowSize * pfacMatching.first) + haystack[i]] != 0)
		{
			// init as the place of the 2nd char in the pattern if the above shows as non-zero
			int currentPlace = pfacMatching.second[(rowSize * pfacMatching.first) + haystack[i]];

			// check pfac row currentPlace for the haystack char i+j
			for (int j = 1; pfacMatching.second[(currentPlace * rowSize) + haystack[i + j]] != 0; ++j)
			{
				// current place becomes the next character in the pattern
				currentPlace = pfacMatching.second[(currentPlace * rowSize)+ haystack[i + j]];

				// If we have reached the end of a pattern
				if (currentPlace < pfacMatching.first)
				{
					// The benefit in pfacSearch for CPU is in the CPU's ability to add results directly to the vector
					unique_lock<mutex> unq(*result);
					results->push_back(i);
				}
			}
		}
	}

	return 0;
}

int main(int argc, char *argv[])
{
	steady_clock::time_point t0 = std::chrono::steady_clock::now();

	vector<ustring> patterns = {
		//{ 71, 73, 70, 56, 55, 97 }, { 0, 59 },	//gif
		//{ 71, 73, 70, 56, 57, 97 }, { 0, 0, 59 },	//gif
		//{ 255, 216, 255, 224, 0, 16 }, { 255, 217 },	//jpg
		//{ 80, 78, 71 }, { 255, 252, 253, 254 },	//png
		//{ 0, 0, 1, 186 }, { 0, 0, 1, 185 },	//mpg
		//{ 0, 0, 1, 179 }, { 0, 0, 1, 183 },	//mpg
		//{ 208, 207, 17, 224, 161, 177, 26, 225, 0, 0 }, { 208, 207, 17, 224, 161, 177, 26, 225, 0, 0 },	//doc
		//{ 60, 104, 116, 109, 108 }, { 60, 47, 104, 116, 109, 108, 62 },	//htm
		{ 37, 80, 68, 70 }, { 37, 69, 79, 70, 13 },	//pdf
		{ 37, 80, 68, 70 }, { 37, 69, 79, 70, 10 },	//pdf
		//{ 80, 75, 3, 4 }, { 60, 172 },	//zip
		//{ 79, 103, 103, 83, 0, 2 }, { 79, 103, 103, 83, 0, 2 }	//ogg
		//{'t', 'e', 's', 't'} }; // This one still works for the wordlist testing
	};

	//vector<string> patterns = { "anything", "three", "that'll", "roderick", "wordly" };

	if (patterns.size() > (1 << 13))
	{
		cerr << endl << endl
			<< "There was an issue regarding the number of patterns" << endl
			<< "The implementation may not be able to function correctly with this. tablePointerType may require being increased." << endl
			<< " If this error has arose then hardware should have advanced to also be able to handle it";
		exit(1);
	}

	pair<tablePointerType, vector<tablePointerType>> pfacMatching = { 0, pfacLookupCreate(patterns) };

	steady_clock::time_point t1 = std::chrono::steady_clock::now();

	fileHandler chunkManager(argv[1], chunkSize, patterns.back().size());
	//fileHandler chunkManager("200MBWordlist.test", 200 * MB, patterns.back().size());
	
	pfacMatching.first = (tablePointerType)(patterns.size() + 1);

	mutex resultMut;
	vector<size_t> results;

	vector<thread> searches;


	while (chunkManager.getCurrChunkNo() < chunkManager.getTotalChunks() - 1) // Loop through all of the available chunks
	{
		chunkManager.waitForRead();

		//for (int i = 0; i < thread::hardware_concurrency(); i++)
			//searches.push_back(thread(pfacSearch, &resultMut, &results, i, thread::hardware_concurrency(), chunkManager.buffer, chunkSize, pfacMatching));

		//thread 0, total threads 1
		searches.push_back(thread(pfacSearch, &resultMut, &results, 0, 1, chunkManager.buffer, chunkSize, pfacMatching));

		//for (int i = 0; i < thread::hardware_concurrency(); i++)
		//{
			if (searches.back().joinable())
			{
				searches.back().join();
			}
		//}

		cout << endl << chunkManager.getCurrChunkNo() + 1 << " Complete of " << chunkManager.getTotalChunks() << endl;

		// Can't asyncRead because of no secondary buffer like with GPU mem
		chunkManager.asyncReadNextChunk();
	}

	chunkManager.waitForRead();

	for (int i = 0; i < thread::hardware_concurrency(); i++)
		searches.push_back(thread(pfacSearch, &resultMut, &results, i, thread::hardware_concurrency(), chunkManager.buffer, chunkSize, pfacMatching));

	// thread 0, total threads 1
	//searches.push_back(thread(pfacSearch, &resultMut, &results, 0, 1, chunkManager.buffer, chunkManager.remainder ? chunkManager.remainder : chunkSize, pfacMatching));

	for (int i = 0; i < thread::hardware_concurrency(); i++)
	{
		if (searches.back().joinable())
		{
			searches.back().join();
		}
	}

	steady_clock::time_point t2 = std::chrono::steady_clock::now();

	if (results.size())
	{
		cout << endl << results.size() << " Pattern(s) Found" << endl;
		//for (int i = 0; i < results.size(); ++i)
		//{
		//	cout << endl << "Found Patterns in chunk " << results[i].first << endl;
			//for (int j = 0; j < results[i].second.size(); ++j)
				//cout << results[i].second[j] << " ";
		//}

		for (int j = 0; j < results.size(); ++j)
			cout << results[j] << " ";
	}
	else
	{
		printf("\nNo Matches");
	}

	auto PFACGEN = duration_cast<nanoseconds> (t1 - t0).count();
	auto PFACSEARCH = duration_cast<milliseconds> (t2 - t1).count();

	cout << endl << endl << "Time for PFAC Table: " << PFACGEN << "ns" << endl;
	cout << "Time for search: " << PFACSEARCH << "ms" << endl;

	return 0;
}
