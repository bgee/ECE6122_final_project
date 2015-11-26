#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

// Define and implement the GPU addition function
// This version is a vector addition, with N threads
// and and N blocks
// Adding one a and b instance and storing in one c instance.
__global__ void add(int *a, int *b, int *c)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  c[index] = a[index] + b[index];
}

// Nmber of blocks
#define N (2048*2048)
#define THREADS_PER_BLOCK 512


int main()
{
  int *a, *b, *c; // host copies of a, b, c
  int *d_a, *d_b, *d_c; // device copies of a, b, c
  int size = N* sizeof(int);
  // Allocate space for device copies of a, b, c
  cudaMalloc((void **)&d_a, size);
  cudaMalloc((void **)&d_b, size);
  cudaMalloc((void **)&d_c, size);
  // Allocate memory for the host a, b, and c arrays
  a = (int*)malloc(size);
  b = (int*)malloc(size);
  c = (int*)malloc(size);
  // Store known values in the a and b arrays
  for (int i = 0; i < N; ++i)
    {
      a[i] = 10*i;
      b[i] = 20*i;
    }
  // Copy inputs to device
  cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
  // Launch add() kernel on GPU with N threads on 1 block
  add<<<N/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
  cudaError err = cudaGetLastError();  
  if(cudaSuccess != err)
  {
    std::cout << "cuda kernel error: " << cudaGetErrorString(err) << std::endl;
  }

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  // Print results removed...too big
  // for (int i = 0; i < N; ++i)
  //   {
  //     std::cout << "sum[" << i << "] is " << c[i] << std::endl;
  //   }
  std::cout << "Sum[0] is " << c[0] << std::endl;
  std::cout << "Sum[" << N - 1 << "] is " << c[N-1] << std::endl;
  // Cleanup
  free(a);
  free(b);
  free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
