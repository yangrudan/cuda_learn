#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

float testCublasError(const int M, const int N, const int K);
float testCublasPerformance(const int M, const int N, const int K, const int repeat);

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



int main(void) {
    printf("\nKernal = cublas\n");
    const int outer_repeat = 10, inner_repeat = 1;
    {
        const int M = 512, N = 512, K = 512;
        float max_error = testCublasError(M, N, K);
        printf("Max Error = %f\n", max_error);
    }

    const int M_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int N_list[15] = {128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384};
    const int K_list[15] = {1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024};

    const int TESTNUM = 15;
    for (int i = 0; i < TESTNUM; i++) {
        const int M = M_list[i], N = N_list[i], K = K_list[i];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int j = 0; j < outer_repeat; j++) {
            double this_sec = testCublasPerformance(M, N, K, inner_repeat);
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


float testCublasError(const int M, const int N, const int K) {

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

    cpuSgemm(h_a, h_b, h_c, M, N, K);

    cudaMemcpy(d_a, h_a, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size_b, cudaMemcpyHostToDevice);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;
    // cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);

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


float testCublasPerformance(const int M, const int N, const int K, const int repeat) {

    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    float cublas_alpha = 1.0;
    float cublas_beta = 0;

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        //cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K, &cublas_alpha, d_a, K, d_b, N, &cublas_beta, d_c, M);
        cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &cublas_alpha, d_b, N, d_a, K, &cublas_beta, d_c, N);
    }
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
 * Kernal = cublas
Max Error = 0.000198
M N K =    128    128   1024, Time =   0.00003482   0.00004211   0.00008726 s, AVG Performance =   742.1816 Gflops
M N K =    192    192   1024, Time =   0.00003990   0.00004105   0.00004400 s, AVG Performance =  1712.7334 Gflops
M N K =    256    256   1024, Time =   0.00004605   0.00005222   0.00006144 s, AVG Performance =  2393.6822 Gflops
M N K =    384    384   1024, Time =   0.00006144   0.00006676   0.00007680 s, AVG Performance =  4212.7510 Gflops
M N K =    512    512   1024, Time =   0.00010342   0.00011469   0.00013824 s, AVG Performance =  4359.4108 Gflops
M N K =    768    768   1024, Time =   0.00015562   0.00016619   0.00018637 s, AVG Performance =  6769.5395 Gflops
M N K =   1024   1024   1024, Time =   0.00024986   0.00026655   0.00027853 s, AVG Performance =  7503.3616 Gflops
M N K =   1536   1536   1024, Time =   0.00047306   0.00048629   0.00050176 s, AVG Performance =  9253.7147 Gflops
M N K =   2048   2048   1024, Time =   0.00089088   0.00089743   0.00090522 s, AVG Performance =  8914.3402 Gflops
M N K =   3072   3072   1024, Time =   0.00169062   0.00169758   0.00170598 s, AVG Performance = 10603.3045 Gflops
M N K =   4096   4096   1024, Time =   0.00290509   0.00292249   0.00292963 s, AVG Performance = 10949.5566 Gflops
M N K =   6144   6144   1024, Time =   0.00647373   0.00651325   0.00653926 s, AVG Performance = 11054.3871 Gflops
M N K =   8192   8192   1024, Time =   0.01024205   0.01044910   0.01125171 s, AVG Performance = 12249.8581 Gflops
M N K =  12288  12288   1024, Time =   0.02238362   0.02248499   0.02280243 s, AVG Performance = 12808.5439 Gflops
M N K =  16384  16384   1024, Time =   0.03962470   0.03966710   0.03968717 s, AVG Performance = 12907.4231 Gflops

Process finished with exit code 0
 */
