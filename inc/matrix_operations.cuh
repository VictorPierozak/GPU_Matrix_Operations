
#define SUCCESS 1
#define FAILURE -1

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

    // Matrix multiplication //
__const__ size_t A_d1;
__const__ size_t A_d2;
__const__ size_t B_d1;
__const__ size_t B_d2;

void multiply(float* dest, float* A, float *B, size_t d1, size_t d2);
__global__ void multiply(float* dest, float* A, float* B);