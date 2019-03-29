#include "fileHandler.h"


//fileHandler::fileHandler(string fileName, size_t currChunkSize = 20MB)
fileHandler::fileHandler(string fileNameGiven, long currChunkSize)
{
	fileName = fileNameGiven;  // Set this for the debug messages in future
	cnkSize = currChunkSize; // By having a default value we can change it easier for testing

	// Open the file requested and since this is a C method it doesn't like strings just char arrays
#pragma warning(suppress : 4996)
	fileToCarve = fopen(fileName.c_str(), "rb");

	// Checking the file opened worked before setting buffers in the event an error occurs
	if (fileToCarve == NULL) // If file didn't open
	{
		fprintf(stderr, "File Error with: %s\nDoes this file exitst here?\n", fileName.c_str());

#if defined(_WIN32)
		system("cd");
#elif defined(__unix__)
		system("pwd");
#endif // _WIN32
		successful = false;
	}

	// This may need done again later so is kept to a function
	checkFileSize();

	// Setup buffers - Due to the use of C methods and the size of the file chunks
	// sticking to the methods demonstrated in the examples seemed the best approach
	// compared to trying to load the files into CPP char arrays.
	buffer = (char*)malloc(sizeof(char)*cnkSize);
	if (buffer == NULL)
	{
		fprintf(stderr, "Memory Allocation Error for chunks of size %d Bytes", cnkSize);
		successful = false;
	}

	// Good to have handy in a variable
	totalChunks = ceil(fSize / cnkSize);
	// Remainder is still a chunk. Just the GPU's problem of how to propogate that throughout

	return;
}

long fileHandler::checkFileSize()
{
	rewind(fileToCarve);
	fseek(fileToCarve, 0, SEEK_END);
	fSize = ftell(fileToCarve);

#if DEBUG == true
	fprintf(stderr, "%s is of size: %d Bytes\n\n", fileName.c_str(), fSize);
#endif 

	rewind(fileToCarve);
	currChunk = 0;
	return fSize;
}

void fileHandler::resetPointer()
{
	rewind(fileToCarve);
	currChunk = 0;
	return;
}

// Changes the value of successful to false in the event the file size changed. Also resets the pointer to the begining
void fileHandler::confirmFileSize()
{
	long fileSizeBefore = fSize;

	if (fileSizeBefore != checkFileSize())
	{
		fprintf(stderr, "The file has been altered in some way. This is not the file from before; exiting");
		free(buffer); // To be safe
		successful = false;
	}
}

void fileHandler::readNextChunk()
{
	fread(buffer, cnkSize, 1, fileToCarve);
	return;
}

double fileHandler::getTotalChunks()
{
	return totalChunks;
}

long fileHandler::getCurrChunkNo()
{
	return currChunk;
}

bool fileHandler::setCurrChunkNo(long newChunkNo)
{
	if (newChunkNo >= getTotalChunks())
	{
		return false;
	}

	// Set the file cursor to the newChunkNo chunks into the file
	if (EXIT_SUCCESS == fseek(fileToCarve, newChunkNo*cnkSize, SEEK_SET))
	{
		currChunk = newChunkNo;
	}
	else { // If the file was not repointed correctly

		fprintf(stderr, "There was an issue repointing to position %d the pointer was reset to the start as a result and execution has continued\n", newChunkNo);
		resetPointer();
		return false;
	}

	return true;
}


fileHandler::~fileHandler()
{
	free(buffer);
	return;
}