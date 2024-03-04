#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <chrono>
#include"matrix_operations.cuh"

// Assuming CUDA libraries are included and linked appropriately

const m_int VECTOR_SIZE = 2<<24; // Size of the vector (multiple of 32)

void initialize_vector(double* vector) {
  for (int i = 0; i < VECTOR_SIZE; ++i) {
    vector[i] = i%1000; // Assign sample values
  }
}

// Main function to test the host function and kernel
int main() {
  // Allocate memory for the vector on the host
  double* vector_h = new double[VECTOR_SIZE];

  // Initialize the vector with sample values
  initialize_vector(vector_h);
  printf("Vector size: %d\n", VECTOR_SIZE);

  // Result variable on the host
  double result = 0.0f;

  // Define block size (adjust as needed for your hardware)
  m_int blockSize(256);

    // **Measure CPU summation time**
  auto start_cpu = std::chrono::high_resolution_clock::now();
  double cpu_sum = 0.0f;
  for (m_int i = 0; i < VECTOR_SIZE; i++) 
  {
    cpu_sum += vector_h[i];
  }
  auto end_cpu = std::chrono::high_resolution_clock::now();
  double cpu_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu).count();

  // **Measure CUDA kernel execution time**
  auto start_cuda = std::chrono::high_resolution_clock::now();
  // Call the host function to perform the sum on the device
  array_sum(vector_h, VECTOR_SIZE, &result, blockSize);
  auto end_cuda = std::chrono::high_resolution_clock::now();
  double kernel_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_cuda - start_cuda).count();

  printf("CUDA implementation time: %.2f ms\n", kernel_time_ms);
  printf("CPU summation time: %.2f ms\n", cpu_time_ms);
  // Check if the result matches the expected sum
  if (abs(result - cpu_sum) < 1e-6) {
    printf("Results are correct!\n");    
  } else {
    printf("Incorrect results!\n");
  }
  printf("GPU result: %f\n", result);
  printf("CPU result: %f\n", cpu_sum);
  printf( "Diffrence: %f\n", abs(result - cpu_sum));
  // Free the allocated memory
  delete[] vector_h;

  cudaDeviceReset();
  return 0;
}
