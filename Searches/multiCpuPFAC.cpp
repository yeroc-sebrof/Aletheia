#include <cstdio>
#include <vector>
#include <string>
#include <thread>
#include <algorithm>
#include <mutex>

using std::thread;
using std::mutex;
using std::unique_lock;
using std::pair;
using std::vector;
using std::string;

// using size_t here would be too inconsistent
#define tablePointerType uint16_t

// This could be changed to 257 to allow for wildcards after setup to allow for this
#define rowSize (int)256

#define totalThreads std::thread::hardware_concurrency()

vector<tablePointerType> pfacLookupCreate(vector<string> patterns)
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
	vector<tablePointerType> table((2 + patterns.size()) * rowSize);
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

void pfacSearch(size_t tid, mutex* mut, vector<size_t>* results, char* haystack, int haysize, pair<tablePointerType, vector<tablePointerType>> pfacMatching)
{
	// Stride will be worked out next
	for (size_t i = tid; i < haysize; i += totalThreads)
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
				currentPlace = pfacMatching.second[(currentPlace * rowSize) + haystack[i + j]];

				// If we have reached the end of a pattern
				if (currentPlace < pfacMatching.first)
				{
					// where a prefix pattern exists of a bigger pattern this will push multiple times
					unique_lock<mutex> unq(*mut);
					results->push_back(i);
				}
			}
		}
	}

	return;
}

int main()
{
	vector<string> patterns = { "any", "three", "words", "rod", "word" };
	vector<thread> test;
	pair<tablePointerType, vector<tablePointerType>> pfacMatching = { 0, pfacLookupCreate(patterns) };
	vector<size_t> results;
	mutex mut;

	pfacMatching.first = (tablePointerType)(patterns.size() + 1);
	char haystack[] = "i don't know; do I go with any three words that go along with those words I said?";

	// Start the thread managment
	size_t i;

	for (i = 0; i < totalThreads; ++i)
		test.push_back(thread(pfacSearch, i, &mut, &results, haystack, sizeof(haystack), pfacMatching));

	for (i=0; i < totalThreads; i++)
		test[i].join();
	// End the thread managment. This could be looped between fetching new chunks

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