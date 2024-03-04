#include"../inc/matrix_operations.cuh"
#include<stdio.h>
#define GET_IDX blockIdx.y * gridDim.x + blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x

#define TILE_SIZE 32
#define STREAMS 8
#define BDIMX 32
#define BDIMY 32

__constant__ m_int cSize;
__constant__ float cFactor;

__constant__ m_int cRowsA;
__constant__ m_int cColsA;
__constant__ m_int cRowsB;
__constant__ m_int cColsB;

__global__ void transpose(float* in, float* out, m_int nx, m_int ny)
{
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;

    if(x < nx && y < ny)
    tile[threadIdx.y * blockDim.x + threadIdx.x] = in[y*width + x];

    __syncthreads();

    x = blockIdx.y * blockDim.x + threadIdx.x;
    y = blockIdx.x * blockDim.y + threadIdx.y;
    width = gridDim.y * blockDim.y;

    if( x < ny && y < nx)
    out[y*width + x] = tile[threadIdx.x * blockDim.x + threadIdx.y];
}


//

void add(float* dest, float* A, float *B, m_int r, m_int c, dim3 blockSize)
{
    m_int N = r*c;
    dim3 gridSize = {(N+blockSize.x-1)/blockSize.x, (N+blockSize.y-1)/blockSize.y, 1};
    cudaStream_t* streams = (cudaStream_t*) malloc(sizeof(cudaStream_t)*STREAMS);
    for(int i = 0; i < STREAMS; i++)
        cudaStreamCreate(&streams[i]);
    m_int chunk = (N + STREAMS - 1)/STREAMS;

    float* A_D, *B_D, *dest_D;
    cudaMalloc(&A_D, sizeof(float)*N);
    cudaMalloc(&B_D, sizeof(float)*N);
    cudaMalloc(&dest_D, sizeof(float)*N);
    cudaMemcpyToSymbol(cSize, &N, sizeof(m_int));
    for(int i = 0; i < STREAMS; i++ )
    {
        m_int offset = i*chunk;
        cudaMemcpyAsync(&A_D[offset], &A[offset], sizeof(float)*chunk, cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&B_D[offset], &B[offset], sizeof(float)*chunk, cudaMemcpyHostToDevice, streams[i]);
        add<<<blockSize, gridSize,0,streams[i]>>>(dest_D, A_D, B_D);
        cudaMemcpyAsync(&dest[offset], &dest_D[offset], sizeof(float)*chunk,cudaMemcpyDeviceToHost, streams[i]);
        cudaStreamDestroy(streams[i]);
    }
        
    cudaFree(A_D);
    cudaFree(B_D);
    cudaFree(dest_D);
}

__global__ void add(float* dest, float* A, float* B)
{
    m_int idx = blockIdx.y * gridDim.x + blockIdx.x + threadIdx.y * blockDim.x + threadIdx.x;
    if(idx < cSize) dest[idx] = A[idx] + B[idx];
}

//



void multiply(float** result, float* A, m_int Arows, m_int Acols, float* B, m_int Brows, m_int Bcols, dim3 blockSize)
{
    if(Brows != Acols) return;
    m_int Asize = Arows*Acols;
    m_int Bsize = Brows*Bcols;
    m_int Csize = Arows*Bcols;
    dim3 gridSize = setGridSize(blockSize, Arows, Bcols);
    *result = (float*) malloc(sizeof(float)*Csize);

    float* A_device, *B_device, *C_device;
    cudaMalloc(&A_device, sizeof(float)*Asize);
    cudaMalloc(&B_device, sizeof(float)*Bsize);
    cudaMalloc(&C_device, sizeof(float)*Csize);
    
    cudaMemcpyToSymbol(cRowsA, &Arows, sizeof(m_int));
    cudaMemcpyToSymbol(cColsA, &Acols, sizeof(m_int));
    cudaMemcpyToSymbol(cRowsB, &Brows, sizeof(m_int));
    cudaMemcpyToSymbol(cColsB, &Bcols, sizeof(m_int));

    cudaMemcpy(A_device, A, sizeof(float)*Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, sizeof(float)*Bsize, cudaMemcpyHostToDevice);
    
    multiply_tiled<<<gridSize, blockSize>>>(C_device, A_device, B_device);
    
    cudaMemcpy(*result, C_device, sizeof(float)*Csize, cudaMemcpyDeviceToHost);

    cudaFree(C_device);
    cudaFree(A_device);
    cudaFree(B_device);
}

void multiply_tiled(float** result, float* A, m_int Arows, m_int Acols, float* B, m_int Brows, m_int Bcols, dim3 blockSize)
{
    if(Brows != Acols) return;
    m_int Asize = Arows*Acols;
    m_int Bsize = Brows*Bcols;
    m_int Csize = Arows*Bcols;
    dim3 gridSize = setGridSize(blockSize, Arows, Bcols);
    *result = (float*) malloc(sizeof(float)*Csize);

    float* A_device, *B_device, *C_device;
    cudaMalloc(&A_device, sizeof(float)*Asize);
    cudaMalloc(&B_device, sizeof(float)*Bsize);
    cudaMalloc(&C_device, sizeof(float)*Csize);
    
    cudaMemcpyToSymbol(cRowsA, &Arows, sizeof(m_int));
    cudaMemcpyToSymbol(cColsA, &Acols, sizeof(m_int));
    cudaMemcpyToSymbol(cRowsB, &Brows, sizeof(m_int));
    cudaMemcpyToSymbol(cColsB, &Bcols, sizeof(m_int));

    cudaMemcpy(A_device, A, sizeof(float)*Asize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_device, B, sizeof(float)*Bsize, cudaMemcpyHostToDevice);
    
    multiply_tiled<<<gridSize, blockSize>>>(C_device, A_device, B_device);
    
    cudaMemcpy(*result, C_device, sizeof(float)*Csize, cudaMemcpyDeviceToHost);

    cudaFree(C_device);
    cudaFree(A_device);
    cudaFree(B_device);
}

__global__ void multiply(float* dest, float* A, float* B)
{
    m_int row = blockDim.y * blockIdx.y + threadIdx.y;
    m_int col = blockDim.x * blockIdx.x + threadIdx.x;
    if( row >= cRowsA || col >= cColsB ) return;
    float c0 = 0.0f;
    for(m_int i = 0; i < cRowsB; i++)
        c0 = __fmaf_rn(A[row*cColsA + i], B[i*cColsB + col], c0);
    dest[row*cColsB + col] = c0;
}

__global__ void multiply_tiled(float* dest, float* A, float* B)
{
    __shared__ float A_shared[TILE_SIZE][TILE_SIZE];
    __shared__ float B_shared[TILE_SIZE][TILE_SIZE];

    m_int row = blockDim.y * blockIdx.y + threadIdx.y;
    m_int col = blockDim.x * blockIdx.x + threadIdx.x;
    
    float c0 = 0.0f;

    for(m_int k = 0; k < (cRowsB + TILE_SIZE - 1)/TILE_SIZE; k++)
    {
        if(k*TILE_SIZE + threadIdx.x < cColsA && row < cRowsA)
            A_shared[threadIdx.y][threadIdx.x] = A[row * cColsA + k*TILE_SIZE + threadIdx.x];
        else A_shared[threadIdx.y][threadIdx.x] = 0;
        if(k*TILE_SIZE + threadIdx.y < cRowsB && col < cColsB)
            B_shared[threadIdx.y][threadIdx.x] = B[(k*TILE_SIZE + threadIdx.y)*cColsB + col];
        else B_shared[threadIdx.y][threadIdx.x] = 0;

        __syncthreads();
        for(int i = 0; i < TILE_SIZE; i++)
            c0 = __fmaf_rn(A_shared[threadIdx.y][i], B_shared[i][threadIdx.x], c0);
        __syncthreads();
    }

    if(row < cRowsA && col < cColsB)
    dest[row*cColsB + col] = c0;
}

void multiply_sc_inplace(float* A, m_int Arows, m_int Acols, float scalar, dim3 blockSize)
{
    dim3 gridSize = setGridSize(blockSize, Arows, Acols);

    cudaStream_t* streamsArray = (cudaStream_t*) malloc(sizeof(cudaStream_t)*STREAMS);
    for(m_int i = 0; i < STREAMS; i++)
    {
        cudaStreamCreate(&streamsArray[i]);
    }

    m_int Asize = Arows*Acols;
    float* A_device;
    cudaMalloc(&A_device, sizeof(float)*Asize);

    cudaMemcpyToSymbol(cFactor, &scalar, sizeof(float));
    cudaMemcpyToSymbol(cSize, &Asize, sizeof(m_int));

    m_int chunk = (Asize + STREAMS - 1)/STREAMS;
    for(m_int s = 0; s < STREAMS; s++)
    {
        m_int offset = s*chunk;
        m_int range = (offset + chunk < Asize) ? chunk: Asize - s*offset;
        cudaMemcpyAsync(&A[offset], &A_device[offset], sizeof(float)*range, cudaMemcpyHostToDevice, streamsArray[s]);
        multiply_sc_inplace<<<gridSize, blockSize>>>(A_device);
        cudaMemcpyAsync(&A_device[offset], &A[offset], sizeof(float)*range, cudaMemcpyDeviceToHost, streamsArray[s]);
        cudaStreamDestroy(streamsArray[s]);
    }

    cudaFree(A_device);
    free(streamsArray);
}

void __global__ multiply_sc_inplace(float* A)
{
    m_int idx = GET_IDX;
    if(idx < cSize) A[idx] = A[idx]*cFactor;
}

void array_sum(double* vector, m_int size, double *result, m_int blockSize)
{
    m_int gridSize =(size + blockSize - 1)/blockSize;
    double* vector_device = NULL;
    cudaMemcpyToSymbol(cSize, &size, sizeof(m_int));
    cudaMalloc(&vector_device, sizeof(double)*size);
    cudaMemcpy(vector_device, vector, sizeof(double)*size, cudaMemcpyHostToDevice);
    array_sum<<<gridSize, blockSize>>>(vector_device);
    cudaDeviceSynchronize();
    double partial = 0.0f;
    *result = 0;
    for(m_int i = 0; i < gridSize; i++)
    {
        cudaMemcpy(&partial, &vector_device[i*blockSize], sizeof(double), cudaMemcpyDeviceToHost);
        *result += partial;
    }

    cudaFree(vector_device);
}

__global__ void array_sum(double* vector)
{
    m_int idx = blockIdx.x * blockDim.x + threadIdx.x;
    double a0 = 0.0f;
    
    for(m_int stride = blockDim.x>>1; stride >= 32; stride >>= 1)
    {
        if(threadIdx.x < stride && idx + stride < cSize) vector[idx] += vector[idx + stride];
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
