#include"../inc/matrix_operations.cuh"

#define GET_IDX blockIdx.y * gridDim.x + blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x

__global__ void transpose(double* in, double* out, m_int nx, m_int ny)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    if(x < nx && y < ny)
    tile[threadIdx.y][threadIdx.x] = in[y*width + x];

    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;
    width = gridDim.y * blockDim.y;

    if( x < ny && y < nx)
    out[y*width + x] = tile[threadIdx.x][threadIdx.y];
}


__global__ void add(double* dest, double* A, double* B, m_int rows, m_int cols)
{
    m_int idx = blockIdx.y * gridDim.x + blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
    if(idx < rows * cols) dest[idx] = A[idx] + B[idx];
}

__global__ void multiply(double* dest, double* A, double *B, m_int Arows, m_int Acols, m_int Brows, m_int Bcols)
{
    __shared__ float A_shared[TILE_DIM][TILE_DIM];
    __shared__ float B_shared[TILE_DIM][TILE_DIM];

    m_int row = blockDim.y * blockIdx.y + threadIdx.y;
    m_int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    float c0 = 0.0f;

    for(m_int k = 0; k < (Brows + TILE_DIM - 1)/TILE_DIM; k++)
    {
        if(k*TILE_DIM + threadIdx.x < Acols && row < Arows)
            A_shared[threadIdx.y][threadIdx.x] = A[row * Acols + k*TILE_DIM + threadIdx.x];
        else A_shared[threadIdx.y][threadIdx.x] = 0;
        if(k*TILE_DIM + threadIdx.y < Brows && col < Bcols)
            B_shared[threadIdx.y][threadIdx.x] = B[(k*TILE_DIM + threadIdx.y)*Bcols + col];
        else B_shared[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();
        for(int i = 0; i < TILE_DIM; i++)
            c0 = __fmaf_rn(A_shared[threadIdx.y][i], B_shared[i][threadIdx.x], c0);
        __syncthreads();
    }

    if(row < Arows && col < Bcols)
    dest[row*Bcols + col] = c0;
}

void __global__ multiply(double *A, double factor, m_int rows, m_int cols)
{
    m_int idx = GET_IDX;
    if(idx < rows*cols) A[idx] = A[idx]*factor;
}

double array_sum(double* A_device, m_int size, dim3 blockSize, dim3 gridSize)
{
    array_sum<<<gridSize, blockSize>>>(A_device, size);
    cudaDeviceSynchronize();
    double partial = 0.0f;
    double result = 0;
    for(m_int i = 0; i < gridSize.x; i++)
    {
        cudaMemcpy(&partial, &A_device[i*blockSize.x], sizeof(double), cudaMemcpyDeviceToHost);
        result += partial;
    }

    return result;
}

__global__ void array_sum(double* vector, m_int size)
{
    m_int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a0 = 0.0f;
    
    for(m_int stride = blockDim.x>>1; stride >= 32; stride >>= 1)
    {
        if(threadIdx.x < stride && idx + stride < size) vector[idx] += vector[idx + stride];
        __syncthreads();
    }

    a0 =  vector[idx];

    if(threadIdx.x < 32)
    {
        a0 += __shfl_down_sync(-1, a0, 16);
        a0 += __shfl_down_sync(-1, a0, 8);
        a0 += __shfl_down_sync(-1, a0, 4);
        a0 += __shfl_down_sync(-1, a0, 2);
        a0 += __shfl_down_sync(-1, a0, 1);
    }

    if(threadIdx.x == 0) vector[idx] = a0;
}
