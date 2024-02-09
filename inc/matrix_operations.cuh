
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