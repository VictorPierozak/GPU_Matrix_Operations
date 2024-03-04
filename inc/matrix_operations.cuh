typedef int64_t m_int;

#define TILE_DIM 32

__global__ void add(double* dest, double* A, double *B, m_int rows, m_int cols);
__global__ void multiply(double* dest, double* A, double *B, m_int Arows, m_int Acols, m_int Brows, m_int Bcols);


__global__ void multiply(double *A, double factor, m_int rows, m_int cols);

// Transpose matrix. Block size should be 32 x 32 x 1
__global__ void transpose(double* dest, double * A, m_int rows, m_int cols);

__global__ void array_sum(double* vector, m_int size);
double array_sum(double* A_device, m_int size, dim3 blockSize, dim3 gridSize);