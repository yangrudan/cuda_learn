#include "cuda_runtime.h"
#include "device_launch_parameters.h"  // threadIdx

#include <stdio.h>    // io
#include <time.h>     // time_t
#include <stdlib.h>  // rand
#include <memory.h>  //memset

#define CHECK(call)                                   \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
}


void checkResult(float* hostRef, float* deviceRef, const int N)
{
    double eps = 1.0E-8;
    int match = 1;
    for (int i = 0; i < N; i++)
    {
        if (hostRef[i] - deviceRef[i] > eps)
        {
            match = 0;
            printf("\nArrays do not match\n");
            printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], deviceRef[i], i);
            break;
        }
    }
    if (match)
        printf("Arrays match!\n");
}

void initialData(float* p, const int N)
{
    //generate different seed from random number
    time_t t;
    srand((unsigned int)time(&t));  // 生成种子

    for (int i = 0; i < N; i++)
    {
        p[i] = (float)(rand() & 0xFF) / 10.0f;  // 随机数
    }
}


__device__ void checkIndex(void) {
    printf("blockIdx: (%d, %d, %d) threadIdx: (%d, %d, %d) \n"
           "gridDim: (%d, %d, %d) blockDim: (%d, %d, %d) \n ==========================\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z
    );
}

// cpu
void sumArraysOnHost(float* a, float* b, float* c, const int N)
{
    for (int i = 0; i < N; i++)
    {
        c[i] = a[i] + b[i];
    }
}

// 设备端：去掉了循环
__global__ void sumArraysOnDevice(float* a, float* b, float* c, const int N)
{
//    checkIndex();
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


int main(void)
{
    int device = 0;
    cudaSetDevice(device);  // 设置显卡号

    // 1 分配内存
    // host memory
    int nElem = 32;
    size_t nBytes = nElem * sizeof(nElem);
    float* h_a, * h_b, * hostRef, *gpuRef;
    h_a = (float*)malloc(nBytes);
    h_b = (float*)malloc(nBytes);
    hostRef = (float*)malloc(nBytes); // 主机端求得的结果
    gpuRef = (float*)malloc(nBytes);  // 设备端拷回的数据
    // 初始化
    initialData(h_a, nElem);
    initialData(h_b, nElem);
    memset(hostRef, 0, nBytes);
    memset(hostRef, 0, nBytes);

    // device memory
    float* d_a, * d_b, * d_c;
    cudaMalloc((float**)&d_a, nBytes);
    cudaMalloc((float**)&d_b, nBytes);
    cudaMalloc((float**)&d_c, nBytes);

    // 2 transfer data from host to device
    cudaMemcpy(d_a, h_a, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, nBytes, cudaMemcpyHostToDevice);

    // 3 在主机端调用设备端核函数
    dim3 block(nElem);
    dim3 grid(nElem / block.x);
    sumArraysOnDevice<<<grid, block>>>(d_a, d_b, d_c, nElem);

    // 4 transfer data from device to host
    cudaMemcpy(gpuRef, d_c, nBytes, cudaMemcpyDeviceToHost);

    //确认下结果
    sumArraysOnHost(h_a, h_b, hostRef, nElem);
    checkResult(hostRef, gpuRef, nElem);

    // 5 释放内存
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(hostRef);
    free(gpuRef);

    return 0;
}