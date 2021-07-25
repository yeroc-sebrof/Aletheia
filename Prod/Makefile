current_dir := $(shell pwd)

default: build cleanDocker

docker:
	echo $(current_dir)
	docker run --name alethia -v $(current_dir):/build -d --rm nvidia/cuda:11.4.0-devel-ubuntu18.04 sleep 600

build: docker kernel.cu fileChunkingClass/fileHandler.h fileChunkingClass/fileHandler.cpp
	docker exec alethia nvcc /build/kernel.cu /build/fileChunkingClass/fileHandler.cpp -o /build/Alethia.run

test: build
	docker exec alethia apt-get update
	docker exec alethia apt-get install -y git nano wget unzip foremost
	docker exec alethia wget https://netix.dl.sourceforge.net/project/dftt/Test%20Images/11_%20Basic%20Data%20Carving%20%231/11-carve-fat.zip -O /test.zip
	docker exec alethia unzip /test.zip
	docker exec alethia foremost -w -i /11-carve-fat/11-carve-fat.dd -o /build/foremost -T
	docker exec alethia /build/Alethia.run /11-carve-fat/11-carve-fat.dd

cleanDocker:
	docker kill alethia
