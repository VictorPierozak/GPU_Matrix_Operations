CUDACOMP = nvcc
CCOMP = gcc
LOADER = nvcc

ROOT = /home/wiktor/Desktop/gpu_matrix_operations

SRC = $(ROOT)/src
INC = $(ROOT)/inc
OBJ = $(ROOT)/obj
BIN = $(ROOT)/bin

environment_test: $(SRC)/cuda_test.cu
	$(CUDACOMP) $(SRC)/cuda_test.cu -o $(BIN)/cuda_test
	./cuda_test

transpose_test: $(OBJ)/matrix_operations.o $(SRC)/transpose_test.cu
	$(CUDACOMP) -dc $(SRC)/transpose_test.cu -o $(OBJ)/transpose_test.o 
	$(CUDACOMP) $(OBJ)/transpose_test.o $(OBJ)/matrix_operations.o -o $(BIN)/transpose_test

addition_test: $(OBJ)/matrix_operations.o $(SRC)/addition_test.cu
	$(CUDACOMP) -dc -Xcompiler -fopenmp -I$(INC) $(SRC)/addition_test.cu -o $(OBJ)/addition_test.o 
	$(CUDACOMP) -lgomp $(OBJ)/addition_test.o $(OBJ)/matrix_operations.o -o $(BIN)/addition_test
	$(BIN)/addition_test	

multiplication_test: $(OBJ)/matrix_operations.o $(SRC)/multiplication_test.cu
	$(CUDACOMP) -dc -Xcompiler -fopenmp -I$(INC) $(SRC)/multiplication_test.cu -o $(OBJ)/multiplication_test.o 
	$(CUDACOMP) -lgomp $(OBJ)/multiplication_test.o $(OBJ)/matrix_operations.o -o $(BIN)/multiplication_test
	$(BIN)/multiplication_test

array_reduction_test:  $(OBJ)/matrix_operations.o $(SRC)/array_reduction_test.cu
	$(CUDACOMP) -dc -Xcompiler -fopenmp -I$(INC) $(SRC)/array_reduction_test.cu -o $(OBJ)/array_reduction_test.o 
	$(CUDACOMP) -lgomp $(OBJ)/array_reduction_test.o $(OBJ)/matrix_operations.o -o $(BIN)/array_reduction_test
	$(BIN)/array_reduction_test
	
matrix_operations.o: 
	$(CUDACOMP) -dc $(SRC)/matrix_operations.cu -o $(OBJ)/matrix_operations.o