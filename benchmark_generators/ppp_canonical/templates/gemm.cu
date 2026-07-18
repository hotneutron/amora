// AMORA PPP fp16 GEMM classification workload.
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    exit(1);
  }
}

static void check_cublas(cublasStatus_t status, const char* what) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "%s: cublas status %d\n", what, (int)status);
    exit(1);
  }
}

int main(int argc, char** argv) {
  int m = argc > 1 ? atoi(argv[1]) : 4096;
  int n = argc > 2 ? atoi(argv[2]) : 4096;
  int k = argc > 3 ? atoi(argv[3]) : 4096;
  __half* a = nullptr;
  __half* b = nullptr;
  __half* c = nullptr;
  check(cudaMalloc(&a, (size_t)m * k * sizeof(__half)), "cudaMalloc a");
  check(cudaMalloc(&b, (size_t)k * n * sizeof(__half)), "cudaMalloc b");
  check(cudaMalloc(&c, (size_t)m * n * sizeof(__half)), "cudaMalloc c");
  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);
#ifndef AMORA_GCOM_TRACE
  check_cublas(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &alpha, a, m, b, k, &beta, c, m), "warmup hgemm");
  check(cudaDeviceSynchronize(), "warmup sync");
#endif
  check_cublas(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &alpha, a, m, b, k, &beta, c, m), "measured hgemm");
  check(cudaDeviceSynchronize(), "measured sync");
  cublasDestroy(handle);
  cudaFree(a);
  cudaFree(b);
  cudaFree(c);
  return 0;
}
