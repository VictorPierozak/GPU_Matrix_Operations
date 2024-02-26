#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include"matrix_operations.cuh"

// Assuming CUDA libraries are included and linked appropriately

const m_int VECTOR_SIZE = 1349; // Size of the vector (multiple of 32)

void initialize_vector(float* vector) {
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    vector[i] = i * 1.0f; // Assign sample values
  }
}

// Main function to test the host function and kernel
int main() {
  // Allocate memory for the vector on the host
  float* vector_h = new float[VECTOR_SIZE];

  // Initialize the vector with sample values
  initialize_vector(vector_h);
  printf("Vector size: %d\n", VECTOR_SIZE);

  // Result variable on the host
  float result = 0.0f;

  // Define block size (adjust as needed for your hardware)
  m_int blockSize(256);

  // **Measure CUDA kernel execution time**
  auto start_cuda = std::chrono::high_resolution_clock::now();
  // Call the host function to perform the sum on the device
  array_sum(vector_h, VECTOR_SIZE, &result, blockSize);
  auto end_cuda = std::chrono::high_resolution_clock::now();
  double kernel_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count();

  // **Measure CPU summation time**
  auto start_cpu = std::chrono::high_resolution_clock::now();
  float cpu_sum = 0.0f;
  for (int i = 0; i < VECTOR_SIZE; i++) 
  {
    cpu_sum += vector_h[i];
  }
  auto end_cpu = std::chrono::high_resolution_clock::now();
  double cpu_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();

  printf("CUDA kernel time: %.2f ms\n", kernel_time_ms);
  printf("CPU summation time: %.2f ms\n", cpu_time_ms);
  // Check if the result matches the expected sum
  if (abs(result - cpu_sum) < 1e-6) {
    printf("Results are correct!\n");    
  } else {
    printf("Incorrect results!\n");
  }

  // Free the allocated memory
  delete[] vector_h;

  return 0;
}
