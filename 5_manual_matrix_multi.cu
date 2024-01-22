#include <iostream>
#include <cuda_runtime.h>


#define BLOCK_SIZE 16

__global__ void Muld(float *, float *, int, int, float *);

void Mul(float *A, float *B, int hA, int wA, int wB, float *C){
    int size;

    float *Ad;
    size = hA * wA * sizeof (float);
    cudaMalloc((void **)&Ad, size);
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    float *Bd;
    size = wA * wB * sizeof(float);
    cudaMalloc((void **)&Bd, size);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);

    float *Cd;
    size = hA * wB * sizeof (float);
    cudaMalloc((void **)&Cd, size);

    dim3 dimBlock = (16, 16);
    dim3 dimGrid = (1, 1);

    printf("grid.x %d; grid.y %d; grid.z %d; \n", dimGrid.x, dimGrid.y, dimGrid.z);
    printf("block.x %d; block.y %d; block.z %d; \n", dimBlock.x, dimBlock.y, dimBlock.z);

    Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}

__global__ void Muld(float *A, float *B, int wA, int wB, float * C){
    //Block index
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //thread index
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    printf("bx %d| by %d | tx %d | ty %d", bx, by, tx ,ty);

    printf("threadIdx:(%d, %d, %d) | blockIdx:(%d, %d, %d) | blockDim: (%d, %d, %d) | gridDim(%d, %d, %d)\n", threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

    //Index of the first sub-matrix of A processed by the block
    int aBegin = wA * BLOCK_SIZE * by;

    //Index of the last sub-matrix of A processed by the block
    int aEnd = aBegin + wA -1;

    //Step size used to iterate through the sub-matrices of A
    int aStep = BLOCK_SIZE;

    //Index of the first sub-matrix of B processed by the block
    int bBegin = BLOCK_SIZE * bx;

    //Step size used to iterate through the sub-matrices of B
    int bStep = BLOCK_SIZE * wB;

    float Csub = 0;

    for (int a = aBegin, b = bBegin;
         a < aEnd;
         a += aStep, b+= bStep){
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        As[ty][tx] = A[a + wA*ty +tx];
        Bs[ty][tx] = B[a + wB*ty +tx];

        __syncthreads();

        int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;

        for(int k = 0; k<BLOCK_SIZE; ++k){
            Csub += As[ty][k] * Bs[k][tx];
            C[c + wB * ty + k] = Csub;
            __syncthreads();
        }


    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB*ty + tx] = Csub;
}

int main(){
    int N = 16;
    float A[N][N], B[N][N], C[N][N];

    // Initialize matrices A and B (you may use your own initialization logic)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = 1.0f;  // You can replace this with your initialization logic
            B[i][j] = 2.0f;  // You can replace this with your initialization logic
        }
    }
    Mul(*A, *B, N, N, N,*C);

    // Print the result (optional)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }
}