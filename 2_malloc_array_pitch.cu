#include <iostream>
#include <cuda_runtime.h>

unsigned long height = 10;
unsigned long width = 20;

__global__ void myKernel(float* devPtr, int pitch, int height,int width)
{
    for (int r = 0; r < height; ++r){
        float * row = (float *)((char*)devPtr,+ r * pitch);
        for (int c = 0; c < width; ++c){
            float element = row[c];
        }
    }
}


int main() {
//    float* devPtr;
//    cudaMalloc((void **)&devPtr, 256 * sizeof (float ));  //在线性内存分配一个256浮点元素的数组;

    // 分配2D数组建议使用cudaMallocPitch(), 保证访问行地址, 或拷贝2D数组到设备内存的其他区域的最佳性能
    //host code
    float* devPtr;
    size_t pitch;
    cudaMallocPitch((void **)&devPtr, &pitch, width * sizeof (float), height);
    myKernel<<<100, 512>>>(devPtr, pitch, 10, 10);

    //分配一个宽 x 高带有32-bit 浮点数的2D数组;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaArray* cuArray;
    cudaMallocArray(&cuArray, &channelDesc, width, height);
}


