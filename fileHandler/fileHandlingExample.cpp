#include "fileHandler.h"

int main(int argc, char** argv)
{
#if DEBUG == true
	cout << "DEBUG: We are currently in path" << endl << "DEBUG: ";

	#ifdef _WIN32
		system("cd");
	#endif //_WIN32

	#ifdef __unix__
		system("pwd");
	#endif //__unix__

		cout << endl << endl;
#endif // DEBUG

	// Init the file handler class
	fileHandler test("TestFile.test");

	// Looped the chunk reading for some examples
	for (int i = 0; i < 3; i++)
	{
		test.readNextChunk();
		cout << "Chunk No " << i << ":" << test.buffer << endl;
	}
	
	// reseting the pointer to the start of the file
	test.resetPointer();

	// setting the chunk to one in
	if (!test.setCurrChunkNo(1))
	{
		cerr << "That didn't work as planned";
		return 1;
	}

	// reading that chunk and writing to console
	test.readNextChunk();
	cout << "Chunk No 1:" << test.buffer << endl;

	return 0;
}
