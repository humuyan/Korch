#pragma once
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <chrono>
#include <iostream>

#define CONV_MODE CUDNN_CROSS_CORRELATION

namespace ch {
using namespace std::chrono;
}

class CUBLASGemm {
    const cudaDataType_t DATA_TYPE = CUDA_R_32F;
    const int warmup = 200, rounds = 200;

    int b, m, n, k;
    cublasOperation_t transa, transb;
    float *A, *B, *C;
    int lda, ldb, ldc;
    cublasHandle_t cublas;

public:
    CUBLASGemm(int b, int m, int n, int k, int transa, int transb, int tf32)
        : b(b),
          m(m),
          n(n),
          k(k),
          transa((cublasOperation_t) transa),
          transb((cublasOperation_t) transb) {
        cudaMalloc(&A, b * m * k * sizeof(float));
        cudaMalloc(&B, b * k * n * sizeof(float));
        cudaMalloc(&C, b * m * n * sizeof(float));
        lda = transa ? k : m;
        ldb = transb ? n : k;
        ldc = m;
        cublasCreate(&cublas);
        if (tf32) {
            cublasSetMathMode(cublas, CUBLAS_TF32_TENSOR_OP_MATH);
        }
    }
    ~CUBLASGemm() {
        cudaFree(A), cudaFree(B), cudaFree(C);
        cublasDestroy(cublas);
    }
    float profile_gemm() {
        const float alpha = 1.0, beta = 0.0;
        if (cublasGemmStridedBatchedEx(cublas,
                                       transa,
                                       transb,
                                       m,
                                       n,
                                       k,
                                       &alpha,
                                       A,
                                       DATA_TYPE,
                                       lda,
                                       m * k,
                                       B,
                                       DATA_TYPE,
                                       ldb,
                                       k * n,
                                       &beta,
                                       C,
                                       DATA_TYPE,
                                       ldc,
                                       m * n,
                                       b,
                                       DATA_TYPE,
                                       CUBLAS_GEMM_DEFAULT) !=
            CUBLAS_STATUS_SUCCESS) {
            return -1;
        }
        ch::time_point<ch::high_resolution_clock, ch::nanoseconds> begin, end;
        for (int i = 0; i < warmup; ++i) {
            cublasGemmStridedBatchedEx(cublas,
                                       transa,
                                       transb,
                                       m,
                                       n,
                                       k,
                                       &alpha,
                                       A,
                                       DATA_TYPE,
                                       lda,
                                       m * k,
                                       B,
                                       DATA_TYPE,
                                       ldb,
                                       k * n,
                                       &beta,
                                       C,
                                       DATA_TYPE,
                                       ldc,
                                       m * n,
                                       b,
                                       DATA_TYPE,
                                       CUBLAS_GEMM_DEFAULT);
        }
        cudaDeviceSynchronize();
        begin = ch::high_resolution_clock::now();
        for (int i = 0; i < rounds; ++i) {
            cublasGemmStridedBatchedEx(cublas,
                                       transa,
                                       transb,
                                       m,
                                       n,
                                       k,
                                       &alpha,
                                       A,
                                       DATA_TYPE,
                                       lda,
                                       m * k,
                                       B,
                                       DATA_TYPE,
                                       ldb,
                                       k * n,
                                       &beta,
                                       C,
                                       DATA_TYPE,
                                       ldc,
                                       m * n,
                                       b,
                                       DATA_TYPE,
                                       CUBLAS_GEMM_DEFAULT);
        }
        cudaDeviceSynchronize();
        end = ch::high_resolution_clock::now();
        return ch::duration_cast<ch::duration<double>>(end - begin).count() *
               1000 / rounds;
    }
};

float profile_gemm(
    int b, int m, int n, int k, int transa, int transb, int tf32) {
    return CUBLASGemm(b, m, n, k, transa, transb, tf32).profile_gemm();
}