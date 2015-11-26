/* 
 * File:   HelloCuda.cu
 * 
 * Created on June 24, 2012
 * 
 * Purpose:  Demonstrate 2D Blocks and Threads with Hello World
 * 
 * If it works, it was written by Brian Swenson.
 * Otherwise, I have no idea who wrote it.
 */

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>


using namespace std;

dim3 blocks(4, 4);
dim3 threads(8, 8);

__global__ void sayHelloCuda1D()
{
	int threadIdxX = threadIdx.x;
	int blockIdxX = blockIdx.x;
	
	//dimension of the block in threads
	int blockDimX = blockDim.x;

	//dimension of the grid in blocks
	int gridDimX = gridDim.x;
	
	//calculate a unique thread id, useful for things like accessing arrays
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	
	//printf("Hello thread %d! (threadIdx.x=%d, blockIdx.x=%d, blockDim.x=%d, gridDim.x=%d\n", id, threadIdxX, blockIdxX, blockDimX, gridDimX);
}

__global__ void sayHelloCuda2D()
{
	int threadIdxX = threadIdx.x;
	int threadIdxY = threadIdx.y;
	int blockIdxX = blockIdx.x;
	int blockIdxY = blockIdx.y;

	//dimension of the block in threads
	int blockDimX = blockDim.x;
	int blockDimY = blockDim.y;
	
	//dimension of the grid in blocks
	int gridDimX = gridDim.x;
	int gridDimY = gridDim.y;

	//calculate a unique thread id, useful for things like accessing arrays
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	int id = x + y * blockDim.x * gridDim.x;

	//printf("Hello thread %d! (threadIdx.x=%d threadIdx.y=%d, blockIdx.x=%d, blockIdx.y=%d, blockDim.x=%d, blockDim.y=%d, gridDim.x=%d, gridDim.y=%d)\n", id, threadIdxX, threadIdxY, blockIdxX, blockIdxY, blockDimX, blockDimY, gridDimX, gridDimY);
}

int main(int argc, char** argv)
{
	printf("1 block with 10 threads\n");
	sayHelloCuda1D<<<1, 10>>>();
	cudaDeviceSynchronize();
	printf("\n");
	
	printf("10 blocks each with 1 thread\n");
	sayHelloCuda1D<<<10, 1>>>();
	cudaDeviceSynchronize();
	printf("\n");
	
	printf("3 blocks each with 3 threads\n");
	sayHelloCuda1D<<<3, 3>>>();
	cudaDeviceSynchronize();
	printf("\n");

	dim3 blocks(3, 3);
	dim3 threads(2, 2);
	printf("3x3 blocks each with 2x2 threads\n");
	sayHelloCuda2D<<<blocks, threads>>>();
	cudaDeviceSynchronize();
	
  return 0;
  
}
