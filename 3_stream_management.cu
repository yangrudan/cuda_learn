#include <iostream>
#include <cuda_runtime.h>


__global__ void myKernel()
{

}


int main() {
    //创建两个流
    cudaStream_t stream[2];
    for(int i = 0; i < 2; ++i){
        cudaStreamCreate(&stream[i]);
    }

    //每个流 host2device, kernel run, device2host
    for(int i = 0; i < 2; ++i){
        cudaMemcpyAsync(dev_input + i *size, hostPtr + i *size, size, cudaMemcpyHostToDevice, stream[i]);
    }
    for(int i = 0; i < 2; ++i){
        myKernel<<<100, 512>>>(dev_out + i * size, dev_input + i *size, size)
    }
    for(int i = 0; i < 2; ++i){
        cudaMemcpyAsync(dev_input + i *size, hostPtr + i *size, size, cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaThreadSynchronize();
}


