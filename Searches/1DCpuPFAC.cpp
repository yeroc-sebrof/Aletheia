#include <array>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>

using std::pair;
using std::vector;
using std::string;
using std::array;

#define tablePointerType uint16_t

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

	// The malleable table that will be returned as a 2D array
	vector<tablePointerType> table((2 + patterns.size()) * 256);
	// This has been initialising as 0 so we good for now. Suppose it's cause this type cant be negative

	size_t currentPlace;

	// for each pattern
	for (int i = 0; i < patterns.size(); ++i)
	{
		currentPlace = patterns.size() + 1;

		// for each character bar last
		for (int j = 0; j < patterns[i].size() - 1; ++j)
		{
			// if the current place doesnt point anywhere
			if (table[(currentPlace * 256) + patterns[i][j]] == 0)
			{
				// make a new row in match table to point at
				table.resize(table.size() + 256);
				//std::fill(table.end() - 256, table.end(), 0);

				// point at it at current place
				table[(currentPlace * 256) + patterns[i][j]] = (tablePointerType)((table.size()/256)-1);
			}

			// follow where the pointer say's is the next space
			currentPlace = table[(currentPlace * 256) + patterns[i][j]];

		}

		// final char
		table[(currentPlace * 256) + patterns[i][patterns[i].size() - 1]] = (tablePointerType)(i + 1); // point to complete pattern

	}

	return table;
}

vector<size_t> pfacSearch(char* haystack, int haysize, pair<tablePointerType, vector<tablePointerType>> pfacMatching)
{
	// Final results holding var init
	vector<size_t> results;

	// Stride will be worked out next
	for (size_t i = 0; i < haysize; ++i)
	{
		// if the current character is pointing to another character in the starting row
		if (pfacMatching.second[(256 * pfacMatching.first) + haystack[i]] != 0)
		{
			// init as the place of the 2nd char in the pattern if the above shows as non-zero
			int currentPlace = pfacMatching.second[(256 * pfacMatching.first) + haystack[i]];

			// check pfac row currentPlace for the haystack char i+j
			for (int j = 1; pfacMatching.second[(currentPlace * 256) + haystack[i + j]] != 0; ++j)
			{
				// current place becomes the next character in the pattern
				currentPlace = pfacMatching.second[(currentPlace * 256)+ haystack[i + j]];

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

	pair<tablePointerType, vector<tablePointerType>> pfacMatching = { 0, pfacLookupCreate(patterns) };

	pfacMatching.first = (tablePointerType)(patterns.size() + 1);

	char haystack[] = "i don't know; do I go with any three words that go along with those words I said?";

	vector<size_t> results = pfacSearch(haystack, sizeof(haystack), pfacMatching);

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