current_dir := $(shell pwd)

default: cpuPFAC multiCpuPFAC gpuPFACSingleCore gpuPFACMultiBlock cleanDocker

cpuPFAC: cpuPFAC.cpp
	g++ cpuPFAC.cpp -o cpuPFAC.run

multiCpuPFAC: multiCpuPFAC.cpp
	g++ -pthread multiCpuPFAC.cpp -o multiCpuPFAC.run

docker:
	echo $(current_dir)
	docker run --name alethia -v $(current_dir):/build -d --rm nvidia/cuda:11.4.0-devel-ubuntu18.04 sleep 600

gpuPFACSingleCore: docker gpuPFACSingleCore.cu
	docker exec alethia nvcc /build/gpuPFACSingleCore.cu -o /build/gpuPFACSingleCore.run

gpuPFACMultiBlock: docker gpuPFACMultiBlock.cu
	docker exec alethia nvcc /build/gpuPFACMultiBlock.cu -o /build/gpuPFACMultiBlock.run

cleanDocker:
	docker kill alethia
