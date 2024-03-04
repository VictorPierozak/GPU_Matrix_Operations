#include<stdio.h>
#include<chrono>
#include"../inc/matrix_operations.cuh"

#define NX 512
#define NY 1024

#define BDIMX 32
#define BDIMY 32

int main()
{
    size_t matrix_size = NX*NY*sizeof(float);
    float* matrix_H = (float*) malloc(matrix_size);
    float* matrix_H_trans = (float*) malloc(matrix_size);
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

    //transpose_padding<<<gridSize, blockSize, (blockSize.x*2 + 1) * blockSize.y * sizeof(float)>>>(
      //  matrix_D, out_D, NX, NY, 1);
    
     auto start_cuda = std::chrono::high_resolution_clock::now();
    transpose<<<gridSize, blockSize, (blockSize.x+1) * blockSize.y * sizeof(float)>>>(
        matrix_D, out_D, NX, NY);
        
    cudaDeviceSynchronize();
      auto end_cuda = std::chrono::high_resolution_clock::now();
  double kernel_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_cuda - start_cuda).count();

    printf("CUDA implementation time: %.2f ms\n", kernel_time_ms);

    cudaMemcpy(out_H, out_D, matrix_size, cudaMemcpyDeviceToHost);

    for (int j = 0; j < NY; j++)
    for (int i = 0; i < NX; i++)
      matrix_H_trans[i * NY + j] = matrix_H[j * NX + i];

    int flag = 1;
    for(int i = 0; i < NX*NY; i++)
        if( matrix_H_trans[i] != out_H[i])
        {
            flag = 0;
            break;
        }
    
    if(flag == 0) printf("\nOperation failed!\n");
    else printf("\nMatrix has been transposed!\n");

    cudaFree(out_D);
    cudaFree(matrix_D);
    free(out_H);
    free(matrix_H);
}