#include<stdio.h>
#include"../inc/matrix_operations.cuh"

#define NX 900
#define NY 1000

#define BDIMX 32
#define BDIMY 16

int main()
{
    size_t matrix_size = NX*NY*sizeof(float);
    float* matrix_H = (float*) malloc(matrix_size);
    for(int y = 0; y < NY; y++)
        for(int x = 0; x < NX; x++)
        {
            matrix_H[y*NX + x] = y*NX + x;
        }
    float * matrix_D = NULL;
    cudaMalloc(&matrix_D, matrix_size);
    cudaMemcpy(matrix_D, matrix_H, matrix_size, cudaMemcpyHostToDevice);

    float* out_H = (float*) malloc(matrix_size);
    float* out_D = NULL;
    cudaMalloc(&out_D, matrix_size);

    dim3 blockSize, gridSize;

    blockSize = {BDIMX, BDIMY, 1};
    gridSize = {(NX + BDIMX - 1)/BDIMX, (NY + BDIMY -1)/BDIMY, 1};

    transpose<<<gridSize, blockSize, calcSharedMemorySize(blockSize, matrix_H)>>>(
        matrix_D, out_D, NX, NY, calcPadding(blockSize));
    
    cudaMemcpy(out_H, out_D, matrix_size, cudaMemcpyDeviceToHost);

    int flag = 1;
     for(int y = 0; y < NY; y++)
        for(int x = 0; x < NX; x++)
        {
            if(out_H[x*NY + y] != matrix_H[y*NX + x])
            {
                flag = 0;
                break;
            }
        }
    
    if(flag == 0) printf("\nOperation failed!\n");
    else printf("\nMatrix has been transposed!\n");

    cudaFree(out_D);
    cudaFree(matrix_D);
    free(out_H);
    free(matrix_H);
    cudaDeviceReset();
}