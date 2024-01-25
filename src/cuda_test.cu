#include<stdio.h>

__global__ void kernel(int* a_device)
{
   int idx = threadIdx.x;
   a_device[idx] = idx*2;
}
int main()
{
    int* a = (int*) malloc(4*sizeof(int));
    for(int i = 0; i < 4;  i++)
    {
        a[i] = 0;
    }
    int* a_device;
    cudaMalloc(&a_device, 4*sizeof(int));
    cudaMemcpy(a_device, a, sizeof(int)*4, cudaMemcpyHostToDevice);
    kernel<<<1,4>>>(a_device);
    cudaMemcpy(a, a_device, sizeof(int)*4, cudaMemcpyDeviceToHost);

    int isSetted = 1;
    for(int i = 0; i < 4; i++)
        if(a[i] != i*2) isSetted = 0;

    if(isSetted == 1) printf("\nCUDA ENVIRONMENT IS SETTED UP\n");
    else printf("\nERROR - CUDA ENVIRONMENT IS NOT SETTED UP\n");    

    free(a);
    cudaFree(a_device);
    return 0;
}