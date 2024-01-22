#include <iostream>
#include <cuda_runtime.h>


__global__ void Check(){
    //Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    printf("threadIdx:(%d, %d, %d) | blockIdx:(%d, %d, %d) | blockDim: (%d, %d, %d) | gridDim(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

}

int main(){
    int nElem = 6;

    dim3 dimBlock(3);
    dim3 dimGrid((nElem + dimBlock.x - 1)/dimBlock.x);

    printf("grid.x %d; grid.y %d; grid.z %d; \n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("block.x %d; block.y %d; block.z %d; \n", dimBlock.x, dimBlock.y, dimBlock.z);

    Check<<<dimGrid, dimBlock>>>();
    cudaDeviceReset();
}