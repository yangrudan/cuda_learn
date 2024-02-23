#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testError(void);
float testPerformance(
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
        dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat);

void cpuSgemm(
        float *a, float *b, float *c, const int M, const int N, const int K) {

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float psum = 0.0;
            for (int k = 0; k < K; k++) {
                psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
            }
            c[OFFSET(m, n, N)] = psum;
        }
    }
}


__global__ void naiveSgemm(
        float * __restrict__ a, float * __restrict__ b, float * __restrict__ c,
        const int M, const int N, const int K) {

    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    if (m < M && n < N) {
        float psum = 0.0;
#pragma unroll
        for (int k = 0; k < K; k++) {
            psum += a[OFFSET(m, k, K)] * b[OFFSET(k, n, N)];
        }
        c[OFFSET(m, n, N)] = psum;
    }
}

int main(void) {
    float max_error = testError();
    printf("Max Error = %f\n", max_error);

    printf("\nKernal = naiveSgemm\n");
    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    const int outer_repeat = 10, inner_repeat = 1;
    const int BM = 32, BN = 32;
    void (*gpuSgemm) (float *, float *, float *, const int, const int, const int) = naiveSgemm;
    const int TESTNUM = 15;

    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        dim3 blockDim(BN, BM);
        dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testPerformance(gpuSgemm, gridDim, blockDim, M, N, K, inner_repeat);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Gflops = ((double)M) * N * K * 2 / 1024 / 1024 / 1024 / avg_sec;

        printf("M N K = %6d %6d %6d, Time = %12.8lf %12.8lf %12.8lf s, AVG Performance = %10.4lf Gflops\n", M, N, K, min_sec, avg_sec, max_sec, avg_Gflops);
    }
    return 0;
}


float testError(void) {
    const int BM = 32, BN = 32;
    const int M = 512, N = 512, K = 512;
    dim3 blockDim(BN, BM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c, *h_d_c;
    h_a = (float *)malloc(size_a);
    h_b = (float *)malloc(size_b);
    h_c = (float *)malloc(size_c);
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);
    h_d_c = (float *)malloc(size_c);

    srand(time(0));
    for (int i = 0; i < M * K; i++)
        h_a[i] = rand() / float(RAND_MAX);
    for (int i = 0; i < K * N; i++)
        h_b[i] = rand() / float(RAND_MAX);
    cudaMemset(d_c, 15, size_c);

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);
    naiveSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaMemcpy(h_d_c, d_c, size_c, cudaMemcpyDeviceToHost);

    float max_error = 0.0;
    for (int i = 0; i < M * N; i++) {
        float this_error = abs(h_d_c[i] - h_c[i]);
        if (max_error != max_error || this_error != this_error) // nan
            max_error = -NAN;
        else
            max_error = max(max_error, this_error);
    }

    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_d_c);

    return max_error;
}


float testPerformance(
        void (*gpuSgemm) (float *, float *, float *, const int, const int, const int),
        dim3 gridDim, dim3 blockDim, const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++)
        gpuSgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, M, N, K);
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    float msec, sec;
    cudaEventElapsedTime(&msec, start, end);
    sec = msec / 1000.0 / repeat;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return sec;
}
/*
 Kernal = naiveSgemm
M N K =    128    128   1024, Time =   0.00009421   0.00009543   0.00010026 s, AVG Performance =   327.4748 Gflops
M N K =    192    192   1024, Time =   0.00009418   0.00009502   0.00009526 s, AVG Performance =   739.9695 Gflops
M N K =    256    256   1024, Time =   0.00009520   0.00009613   0.00009728 s, AVG Performance =  1300.2629 Gflops
M N K =    384    384   1024, Time =   0.00018224   0.00018328   0.00018432 s, AVG Performance =  1534.5105 Gflops
M N K =    512    512   1024, Time =   0.00035530   0.00035624   0.00035840 s, AVG Performance =  1403.5356 Gflops
M N K =    768    768   1024, Time =   0.00071066   0.00071661   0.00071885 s, AVG Performance =  1569.8960 Gflops
M N K =   1024   1024   1024, Time =   0.00115504   0.00115610   0.00115917 s, AVG Performance =  1729.9554 Gflops
M N K =   1536   1536   1024, Time =   0.00261424   0.00266404   0.00267981 s, AVG Performance =  1689.1649 Gflops
M N K =   2048   2048   1024, Time =   0.00426496   0.00450294   0.00466022 s, AVG Performance =  1776.6180 Gflops
M N K =   3072   3072   1024, Time =   0.00952627   0.00955464   0.00956723 s, AVG Performance =  1883.9021 Gflops
M N K =   4096   4096   1024, Time =   0.01682125   0.01690153   0.01704755 s, AVG Performance =  1893.3198 Gflops
M N K =   6144   6144   1024, Time =   0.03800678   0.03812086   0.03818086 s, AVG Performance =  1888.7298 Gflops
M N K =   8192   8192   1024, Time =   0.06766899   0.06783232   0.06795264 s, AVG Performance =  1887.0061 Gflops
M N K =  12288  12288   1024, Time =   0.15264767   0.15344159   0.15378432 s, AVG Performance =  1876.9357 Gflops
M N K =  16384  16384   1024, Time =   0.27192318   0.27260968   0.27276799 s, AVG Performance =  1878.1431 Gflops
 */