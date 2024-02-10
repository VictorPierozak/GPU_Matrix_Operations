CUDACOMP = nvcc
CCOMP = gcc
LOADER = nvcc

ROOT = /home/wiktor/Desktop/gpu_matrix_operations

SRC = $(ROOT)/src
INC = -I $(ROOT)/inc
OBJ = $(ROOT)/obj
BIN = $(ROOT)/bin

environment_test: $(SRC)/cuda_test.cu
	$(CUDACOMP) $(SRC)/cuda_test.cu -o $(BIN)/cuda_test
	./cuda_test

transpose_test: $(OBJ)/matrix_operations.o $(SRC)/transpose_test.cu
	$(CUDACOMP) -dc $(SRC)/transpose_test.cu -o $(OBJ)/transpose_test.o 
	$(CUDACOMP) $(OBJ)/transpose_test.o $(OBJ)/matrix_operations.o -o $(BIN)/transpose_test

matrix_operations:
	$(CUDACOMP) -dc $(SRC)/matrix_operations.cu -o $(OBJ)/matrix_operations.o