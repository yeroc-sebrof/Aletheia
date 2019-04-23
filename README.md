# Aletheia
This repository holds the files related to a CUDA enabled file header/footer search writen in CPP11. This code is cross-platform compatible and has been tested on Windows 10, Ubuntu 18.04 and Fedora 29.

Not all of the files in this repository are required for the running of the final compiled program, there is many sections of code that remain for a point of comparison that were created before the introduction of a git repository. In order to compile the Aletheia program a sister repository [fileChunkerClass](https://github.com/yeroc-sebrof/fileChunkingClass) must also be installed.

The following code will compiler this code on a Unix Operating system:

	$ git clone https://github.com/yeroc-sebrof/fileChunkingClass
	$ git clone https://github.com/yeroc-sebrof/Aletheia
	$ cd fileChunkingClass; git checkout C-method; cd ..
	$ nvcc Aletheia/Prod/kernel.cu fileChunkingClass/fileHandler.cpp -o aletheia.run

In the event Scalpel releases a new set of patterns -- OR you wish to load in different patterns that are commented out in the scalpel configuration -- the python script [scalpelConfigReJig.py](https://github.com/yeroc-sebrof/Aletheia/ConfigMaker/scalpelConfigReJig.py) will help. This script should ensure that the pattern meets with Aletheia's requirements. One large requirement includes that Aletheia is **unable** to search for patterns containing Wildcards.
