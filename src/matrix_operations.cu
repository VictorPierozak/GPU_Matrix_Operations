#include"../inc/matrix_operations.cuh"

__global__ void transpose(float* in, float* out, unsigned int nx, unsigned int ny)
{
    extern __shared__ float tile[];

    unsigned int ix, iy, ti, to;
    ix = blockIdx.x*blockDim.x + threadIdx.x;
    iy = blockIdx.y*blockDim.y + threadIdx.y;

    ti = iy*nx + ix;

    unsigned int bidx, irow, icol;
    bidx = threadIdx.y*blockDim.x + threadIdx.x;
    irow = bidx/blockDim.y;
    icol = bidx%blockDim.y;    

    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;

    to =  iy * ny + ix;

    if(ix < nx && iy < ny)
    {
        tile[bidx] = in[ti];

        __syncthreads();

        out[to] = tile[icol * blockDim.x + irow ];
    }
}

