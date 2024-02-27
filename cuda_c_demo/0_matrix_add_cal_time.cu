#include <iostream>
#include <cuda_runtime.h>

const int N = 10;

// Kernel definition
__global__ void MatAdd(float A[N][N], float B[N][N], float C[N][N]) {
    int i = threadIdx.x;
    int j = threadIdx.y;
    C[i][j] = A[i][j] + B[i][j];
}

int main() {
    float A[N][N], B[N][N], C[N][N];

    // Initialize matrices A and B (you may use your own initialization logic)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = 1.0f;  // You can replace this with your initialization logic
            B[i][j] = 2.0f;  // You can replace this with your initialization logic
        }
    }

    float (*d_A)[N], (*d_B)[N], (*d_C)[N];

    // Allocate device memory
    cudaMalloc((void **)&d_A, N * N * sizeof(float));
    cudaMalloc((void **)&d_B, N * N * sizeof(float));
    cudaMalloc((void **)&d_C, N * N * sizeof(float));

    // Copy data from host to device
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start event
    cudaEventRecord(start);

    // Kernel invocation with one block of N x N x 1 threads
    dim3 threadsPerBlock(N, N);
    MatAdd<<<1, threadsPerBlock>>>(d_A, d_B, d_C);

    // Record the stop event
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time elapsed: " << milliseconds << " ms" << std::endl;

    // Copy result from device to host
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print the result (optional)
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << std::endl;
    }

    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
