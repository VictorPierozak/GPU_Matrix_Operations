#include <stdio.h>
#include"matrix_operations.cuh"

#define rows 500
#define cols 1024
#define OMP_THREADS 12

void multiply(float** result, float* A, m_int Arows, m_int Acols, float* B, m_int Brows, m_int Bcols);

int main()
{
    float* A = (float*) malloc(sizeof(float)*rows*cols);    
    float* B = (float*) malloc(sizeof(float)*rows*cols);
    size_t N = rows*cols;
    #pragma omp parallel for num_threads(OMP_THREADS)
    for(size_t i = 0; i < N; i++)
    {
        A[i] = i % 1000;
        B[i] = -1.0*(float)(i%1000);
    }
    float* C = NULL;
    float* C_test = NULL;
    multiply(&C_test, A, rows, cols, B, cols, rows);

    dim3 blockSize = {32, 32, 1};
   
    printf("\nMultiplication - start\n");
    multiply_tiled(&C, A, rows, cols, B, cols, rows, blockSize);
    printf("\nMultiplication - end\n");
    int test = 1;
     for(size_t i = 0; i < rows*rows; i += 1)
    {
        if(abs(C[i] - C_test[i]) > 10e-6)
        {
            printf("%d ", i);
            test = 0;
            break;
        }
    }
    if(test) printf("\nMultiplication succesed!\n");
    else printf("\nFailure!\n");
    free(A);
    free(B);
    free(C);
    free(C_test);
    return 0;
}

void multiply(float** result, float* A, m_int Arows, m_int Acols, float* B, m_int Brows, m_int Bcols)
{
     if (Acols != Brows) {
    fprintf(stderr, "Error: Incompatible matrix dimensions for multiplication\n");
    return;
    }
     *result = (float*) malloc(sizeof(float)*Arows*Bcols); 

    // Iterate over rows of the result matrix
    #pragma omp parallel for num_threads(OMP_THREADS)
    for (int i = 0; i < Arows; ++i) {
    for (int j = 0; j < Bcols; ++j) {
      float sum = 0.0;
      for (int k = 0; k < Acols; ++k) {
        sum += A[i * Acols + k] * B[k * Bcols + j];
      }
      (*result)[i * Bcols + j] = sum;
    }
  }
}