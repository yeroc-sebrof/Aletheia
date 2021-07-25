current_dir := $(shell pwd)

default: build cleanDocker

docker:
	echo $(current_dir)
	docker run --name alethia -v $(current_dir):/build -d --rm nvidia/cuda:11.4.0-devel-ubuntu18.04 sleep 600

build: docker kernel.cu fileChunkingClass/fileHandler.h fileChunkingClass/fileHandler.cpp
	docker exec alethia nvcc /build/kernel.cu /build/fileChunkingClass/fileHandler.cpp -o /build/Alethia.run

cleanDocker:
	docker kill alethia
