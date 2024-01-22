#include <iostream>
#include <cuda_runtime.h>



int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::cout << "deviceCount : " << deviceCount  << std::endl;

    int device;
    for(device = 0; device < deviceCount; ++device){
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);
        std::cout << "device canMapHostMemory: " << deviceProp.canMapHostMemory  << std::endl;
        std::cout << "device asyncEngineCount: " << deviceProp.asyncEngineCount  << std::endl;
        std::cout << "device canUseHostPointerForRegisteredMem: " << deviceProp.canUseHostPointerForRegisteredMem  << std::endl;
        std::cout << "device clockRate: " << deviceProp.clockRate  << std::endl;
        std::cout << "device hostNativeAtomicSupported: " << deviceProp.hostNativeAtomicSupported  << std::endl;
        std::cout << "device l2CacheSize: " << deviceProp.l2CacheSize  << std::endl;
        std::cout << "maxThreadsPerBlock is " << deviceProp.maxThreadsPerBlock << std::endl;
    }
}

//deviceCount : 1
//device canMapHostMemory: 1
//device asyncEngineCount: 7
//device canUseHostPointerForRegisteredMem: 1
//device clockRate: 1380000
//device hostNativeAtomicSupported: 0
//device l2CacheSize: 6291456
//maxThreadsPerBlock is 1024
//multiProcessorCount is 80