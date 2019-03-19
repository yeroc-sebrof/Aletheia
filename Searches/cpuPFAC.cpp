#include <array>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

using std::pair;
using std::vector;
using std::string;
using std::array;

//int** convertVector(vector<array<int, 256>> vectIn)
//{
//	
//
//	return;
//}

vector<array<int, 256>> pfacLookupCreate(vector<string> patterns)
{
	/*
	The PFAC table can be described as:
		a starting row that will be sent to upon error (which all rows have as default values)
		rows for when the end of the pattern is found
		row for starting the search from - how does the program know where the starting row is at runtime?
		all of the redirections from each value given
	*/

	// A new array of 256 int's to be a new row
	array<int, 256> newRow = { 0 };

	//sort the string list provided first by size then by character
	// https://stackoverflow.com/questions/45865239/how-do-i-sort-string-of-arrays-based-on-length-then-alphabetic-order
	sort(patterns.begin(), patterns.end(),
		[](const std::string& lhs, const std::string& rhs) {
		return lhs.size() == rhs.size() ?
			lhs < rhs : lhs.size() < rhs.size(); });

	// The malleable table that will be returned as a 2D array
	vector<array<int, 256>> table;

	// Fail row
	table.push_back(newRow);

	// pushing pattern found rows
	for (size_t i = 0; i < patterns.size(); ++i)
		table.push_back(newRow);

	// The row where pattern matching begins
	table.push_back(newRow);

	int currentPlace;

	// for each pattern
	for (size_t i = 0; i < patterns.size(); ++i)
	{
		currentPlace = patterns.size() + 1;

		// for each character bar last
		for (size_t j = 0; j < patterns[i].size() - 1; ++j)
		{
			// if the current place doesnt point anywhere
			if (table[currentPlace][patterns[i][j]] == 0)
			{
				// make a new row in match table to point at
				table.push_back(newRow);

				// point at it at current place
				table[currentPlace][patterns[i][j]] = table.size() - 1;
			}

			// follow where the pointer say's is the next space
			currentPlace = table[currentPlace][patterns[i][j]];

		}

		// final char
		table[currentPlace][patterns[i][patterns[i].size() - 1]] = i + 1; // point to complete pattern

	}

	return table;
}

vector<int> pfacSearch(char* haystack, int minPatSize, pair<int&, vector<array<int, 256>>*> pfacMatching)
{
	// Return an array of ints where the first element is how many results have been discovered followed by the results of said result
	//int fail[] = { 0, 0 };
	// This will be returned to for opimisation


	// Final results holding var init
	vector<int> results;

	// This could be passed in later
	size_t haysize = strlen(haystack);

	int currentPlace;

	// Stride will be worked out next
	for (size_t i = 0; i < haysize - minPatSize; ++i)
	{
		// if the current character is pointing to another character in the starting row
		if (pfacMatching.second->at(pfacMatching.first)[haystack[i]] != 0)
		{
			// init as the place of the 2nd char in the pattern if the above shows as non-zero
			currentPlace = pfacMatching.second->at(pfacMatching.first)[haystack[i]];

			// check pfac row currentPlace for the haystack char i+j
			for (int j = 1; pfacMatching.second->at(currentPlace)[haystack[i + j]] != 0; ++j) /// TODO fix the fact this will look out of bounds when searching for long patterns
			{
				// current place becomes the next character in the pattern
				currentPlace = pfacMatching.second->at(currentPlace)[haystack[i + j]];

				// If we have reached the end of a pattern
				if (currentPlace < pfacMatching.first)
					// where a prefix pattern exists of a bigger pattern this will push multiple times
					results.push_back(i);
			}
		}
	}

	return results;
}

int main()
{

	vector<string> patterns = { "any", "three", "words", "rod", "word" };

	//int** lookupTable = convertVector(
	vector<array<int, 256>> jumpTable = pfacLookupCreate(patterns);
	//)

	int startingRow = patterns.size() + 1;

	pair<int&, vector<array<int, 256>>*> pfacMatching = { startingRow, &jumpTable };

	char haystack[] = "i don't know; do I go with any three words that go along with those words I said? ro";

	vector<int> results = pfacSearch(haystack, patterns.front().size(), pfacMatching);

	if (results.size())
	{
		printf("\n\nPatterns start at chars: ");

		for (size_t i = 0; i < results.size(); ++i)
			printf("%d ", results[i]);
	}
	else
	{
		printf("\nNo Matches");
		return 1;
	}

	return 0;
}