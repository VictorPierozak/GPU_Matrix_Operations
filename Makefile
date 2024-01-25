CUDACOMP = nvcc
CCOMP = gcc
LOADER = nvcc
SRC = ./src

environment_test: $(SRC)/cuda_test.cu
	$(CUDACOMP) $(SRC)/cuda_test.cu -o cuda_test
	./cuda_test