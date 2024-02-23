#include <iostream>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))

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
    cpuSgemm(reinterpret_cast<float *>(A), reinterpret_cast<float *>(B), reinterpret_cast<float *>(C), N, N , N);

    std::cout << "C[0][0]:  " << C[0][0] << " " << std::endl;
}