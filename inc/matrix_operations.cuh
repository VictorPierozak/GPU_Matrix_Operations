
#define SUCCESS 1
#define FAILURE -1

typedef int64_t m_int;

// Universal //
inline dim3 setGridSize(dim3 blockSize, m_int rows, m_int cols)
{
    dim3 gridSize = {(cols + blockSize.x - 1)/blockSize.x, (rows + blockSize.y - 1)/blockSize.y, 1};
    return gridSize;
}

// Transposition //

inline unsigned int calcPadding(dim3 blockSize)
{
    return (blockSize.x + blockSize.y - 1)/blockSize.y;
}

template<typename T>
inline int calcSharedMemorySize(dim3 blockSize, T* matrix)
{
    return (blockSize.x*2 + calcPadding(blockSize)) * blockSize.y * sizeof(T);
}

__global__ void transpose(float* in, float* out, unsigned int nx, unsigned int ny, unsigned int padding);

// Addition //

void add(float* dest, float* A, float *B, size_t r, size_t c, dim3 blockSize);
__global__ void add(float* dest, float* A, float* B);

// Mat x vec //
void multiply(float** result, float* A, m_int Arows, m_int Acols, float* B, m_int Brows, m_int Bcols, dim3 blockSize);
__global__ void multiply(float* dest, float* A, float* B);