#include <stdio.h>
#include"matrix_operations.cuh"

#define rows 10e4 
#define cols 10e3
#define OMP_THREADS 12

//template void add<32, 32, 8>(float* dest, float* A, float *B, m_int r, m_int c);

int main()
{
    float* A = (float*) malloc(sizeof(float)*rows*cols);    
    float* B = (float*) malloc(sizeof(float)*rows*cols);
    size_t N = rows*cols;
    #pragma omp parallel for num_threads(OMP_THREADS)
    for(size_t i = 0; i < N; i++)
    {
        A[i] = i;
        B[i] = -1.0*(float)i;
    }
    float* C = (float*) malloc(sizeof(float)*rows*cols);
    dim3 blockSize = {32, 32, 1};
   
    printf("\nAddition - start\n");
    add(C, A, B, rows, cols, blockSize);
    printf("\nAddition - end\n");
    int test = 1;
     for(size_t i = 0; i < rows*cols; i += 1)
    {
        if(C[i] != 0)
        {
            test = 0;
            break;
        }
    }
    if(test) printf("\nAddition succesed!\n");
    else printf("\nFailure!\n");
    free(A);
    free(B);
    free(C);
    cudaDeviceReset();
    return 0;
}