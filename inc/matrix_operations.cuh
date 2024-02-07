

template<typename T>
inline int calcSharedMemorySize(dim3 blockSize, T* matrix)
{
    return blockSize.x * blockSize.y * sizeof(T);
}
__global__ void transpose(float* in, float* out, unsigned int nx, unsigned int ny);