#include"../inc/matrix_operations.cuh"

#define STREAMS 8

__constant__ size_t cN;

// // //

__global__ void transpose(float* in, float* out, unsigned int nx, unsigned int ny, unsigned int padding)
{
    extern __shared__ float tile[];

    unsigned int in_idx, out_idx;

    unsigned int ix = blockIdx.x*blockDim.x*2 + threadIdx.x;
    unsigned int iy = blockIdx.y*blockDim.y + threadIdx.y;

    in_idx = iy*nx + ix;

    unsigned int block_idx, block_row, block_col;
    block_idx = threadIdx.y * blockDim.x + threadIdx.x;
    block_row = block_idx/blockDim.y;
    block_col = block_idx%blockDim.y;    

    unsigned int ox = blockIdx.y*blockDim.y + block_col;
    unsigned int oy = blockIdx.x*blockDim.x*2 + block_row;

    out_idx =  oy* ny + ox;

    if(ix + blockDim.x < nx && iy < ny)
    {
        unsigned int row_idx = threadIdx.y * (blockDim.x *2 + padding) + threadIdx.x;
        tile[row_idx] = in[in_idx];
        tile[row_idx+blockDim.x] = in[in_idx + blockDim.x];

        __syncthreads();

        unsigned int col_idx = block_col * (blockDim.x *2 + padding) + block_row;
        out[out_idx] = tile[col_idx];
        out[out_idx+ny*blockDim.x] = tile[col_idx + blockDim.x];
    }
}

//

void add(float* dest, float* A, float *B, size_t r, size_t c, dim3 blockSize)
{
    size_t N = r*c;
    dim3 gridSize = {(N+blockSize.x-1)/blockSize.x, (N+blockSize.y-1)/blockSize.y, 1};
    cudaStream_t* streams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*STREAMS);
    for(int i = 0; i < STREAMS; i++)
        cudaStreamCreate(&streams[i]);
    size_t chunk = (N + STREAMS - 1)/STREAMS;

    float* A_D, *B_D, *dest_D;
    cudaMalloc(&A_D, sizeof(float)*N);
    cudaMalloc(&B_D, sizeof(float)*N);
    cudaMalloc(&dest_D, sizeof(float)*N);
    cudaMemcpyToSymbol(cN, &N, sizeof(size_t));
    for(int i = 0; i < STREAMS; i++ )
    {
        size_t offset = i*chunk;
        cudaMemcpyAsync(&A_D[offset], &A[offset], sizeof(float)*chunk, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&B_D[offset], &B[offset], sizeof(float)*chunk, cudaMemcpyHostToDevice, streams[i]);
        add<<<blockSize, gridSize,0,streams[i]>>>(dest_D, A_D, B_D);
        cudaMemcpyAsync(&dest[offset], &dest_D[offset], sizeof(float)*chunk,cudaMemcpyDeviceToHost, streams[i]);
    }
    for(int i = 0; i < STREAMS; i++)
        cudaStreamDestroy(streams[i]);
    cudaFree(A_D);
    cudaFree(B_D);
    cudaFree(dest_D);
}

__global__ void add(float* dest, float* A, float* B)
{
    size_t idx = blockIdx.y * gridDim.x + blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
    if(idx < cN) dest[idx] = A[idx] + B[idx];
}