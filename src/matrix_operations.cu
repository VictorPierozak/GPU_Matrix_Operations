#include"../inc/matrix_operations.cuh"

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

