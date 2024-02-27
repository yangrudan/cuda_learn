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
        std::cout << "device name: " << deviceProp.name  << std::endl;
        std::cout << "device major: " << deviceProp.major  << std::endl;
        std::cout << "device minor: " << deviceProp.minor  << std::endl;
        std::cout << "device canMapHostMemory: " << deviceProp.canMapHostMemory  << std::endl;
        std::cout << "device asyncEngineCount: " << deviceProp.asyncEngineCount  << std::endl;
        std::cout << "device canUseHostPointerForRegisteredMem: " << deviceProp.canUseHostPointerForRegisteredMem  << std::endl;
        std::cout << "device clockRate: " << deviceProp.clockRate  << std::endl;
        std::cout << "device hostNativeAtomicSupported: " << deviceProp.hostNativeAtomicSupported  << std::endl;
        std::cout << "device l2CacheSize: " << deviceProp.l2CacheSize  << std::endl;
        std::cout << "maxThreadsPerBlock is " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "maxThreadsDim is " << deviceProp.maxThreadsDim[0] << " "<< deviceProp.maxThreadsDim[1]<< " " <<deviceProp.maxThreadsDim[2] <<std::endl;
        std::cout << "maxGridSize is " << deviceProp.maxGridSize[0] << " "<< deviceProp.maxGridSize[1]<< " " <<deviceProp.maxGridSize[2] <<std::endl;
        std::cout << "totalGlobalMem is " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "totalConstMem is " << deviceProp.totalConstMem << std::endl;
        std::cout << "sharedMemPerBlock is " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "regsPerBlock is " << deviceProp.regsPerBlock << std::endl;
        std::cout << "warpSize is " << deviceProp.warpSize << std::endl;
        std::cout << "memPitch is " << deviceProp.memPitch << std::endl;

    }
}

deviceCount : 4
device name: Tesla V100-PCIE-32GB
device major: 7
device minor: 0
device canMapHostMemory: 1
device asyncEngineCount: 7
device canUseHostPointerForRegisteredMem: 1
device clockRate: 1380000
device hostNativeAtomicSupported: 0
device l2CacheSize: 6291456
maxThreadsPerBlock is 1024
maxThreadsDim is 1024 1024 64
maxGridSize is 2147483647 65535 65535
totalGlobalMem is 34079637504
totalConstMem is 65536
sharedMemPerBlock is 49152
regsPerBlock is 65536
warpSize is 32
memPitch is 2147483647
device name: Tesla V100-PCIE-32GB
device major: 7
device minor: 0
device canMapHostMemory: 1
device asyncEngineCount: 7
device canUseHostPointerForRegisteredMem: 1
device clockRate: 1380000
device hostNativeAtomicSupported: 0
device l2CacheSize: 6291456
maxThreadsPerBlock is 1024
maxThreadsDim is 1024 1024 64
maxGridSize is 2147483647 65535 65535
totalGlobalMem is 34079637504
totalConstMem is 65536
sharedMemPerBlock is 49152
regsPerBlock is 65536
warpSize is 32
memPitch is 2147483647
device name: Tesla V100-PCIE-32GB
device major: 7
device minor: 0
device canMapHostMemory: 1
device asyncEngineCount: 7
device canUseHostPointerForRegisteredMem: 1
device clockRate: 1380000
device hostNativeAtomicSupported: 0
device l2CacheSize: 6291456
maxThreadsPerBlock is 1024
maxThreadsDim is 1024 1024 64
maxGridSize is 2147483647 65535 65535
totalGlobalMem is 34079637504
totalConstMem is 65536
sharedMemPerBlock is 49152
regsPerBlock is 65536
warpSize is 32
memPitch is 2147483647
device name: Tesla V100-PCIE-32GB
device major: 7
device minor: 0
device canMapHostMemory: 1
device asyncEngineCount: 7
device canUseHostPointerForRegisteredMem: 1
device clockRate: 1380000
device hostNativeAtomicSupported: 0
device l2CacheSize: 6291456
maxThreadsPerBlock is 1024
maxThreadsDim is 1024 1024 64
maxGridSize is 2147483647 65535 65535
totalGlobalMem is 34079637504
totalConstMem is 65536
sharedMemPerBlock is 49152
regsPerBlock is 65536
warpSize is 32
memPitch is 2147483647