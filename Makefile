CUDACOMP = nvcc
CCOMP = gcc
LOADER = nvcc

SRC = ./src
INC = -I ./inc
BIN = ./bin

environment_test: $(SRC)/cuda_test.cu
	$(CUDACOMP) $(SRC)/cuda_test.cu -o cuda_test
	./cuda_test

transpose_test: $(BIN)/matrix_operations.o $(SRC)/transpose_test.cu
	$(CUDACOMP) -dc $(SRC)/transpose_test.cu -o $(BIN)/transpose_test.o 
	$(CUDACOMP) $(BIN)/transpose_test.o $(BIN)/matrix_operations.o -o transpose_test

matrix_operations:
	$(CUDACOMP) -dc $(SRC)/matrix_operations.cu -o $(BIN)/matrix_operations.o