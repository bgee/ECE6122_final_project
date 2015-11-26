#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <stdio.h>

// Define and implement the GPU addition function
// This version is a vector addition, with each block
// Adding one a and b instance and storing in one c instance.
__global__ void add(int *a, int *b, int *c)
{
  
  printf("abcabc%d\n", blockIdx.x);
  c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

// Nmber of blocks
#define N 512

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
  // Launch add() kernel on GPU with N blocks
  add<<<N,1>>>(d_a, d_b, d_c);

  // Copy result back to host
  cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
  // Print results
  for (int i = 0; i < N; ++i)
    {
      std::cout << "sum[" << i << "] is " << c[i] << std::endl;
    }
  // Cleanup
  free(a);
  free(b);
  free(c);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  return 0;
}
