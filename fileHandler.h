#pragma once
#include <cstdio>
#include <math.h>
#include <iostream>
#include <vector>
#include <string>

#define endl (string)"\n"

#define KB (int)1024
#define MB (int)1024*1024

#define DEBUG true
#define CLIARGS false

using std::string;
using std::cout;
using std::cerr;
using std::vector;

class fileHandler
{
	FILE * fileToCarve; // File pointer
	size_t chunkSize;
	string fileName;

	long currChunk=0;
	double totalChunks;
	long fSize; // File size variable

public:
	char *buffer;

	fileHandler(string filename, size_t=20);
	virtual ~fileHandler();

	long checkFileSize();	// rerun the file size check
	void resetPointer();	// return the file pointer to zero
	void confirmFileSize(); // Will check if the file size is equal to the file size given at the start.
	// This should only change in the event the file has been changed as we use it

	void readNextChunk();	// reads currChunkNo into buffer

	double getTotalChunks();	// returns the file size / chunk size rounded up
	long getCurrChunkNo();	// returns the int value of ChunkNo

	bool setCurrChunkNo(long);	// sets the next chunk to be read (Chunks start at 0)
};

