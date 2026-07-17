// AMORA PPP GELU plus fp16 GEMM classification workload.
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

__global__ void amora_ppp_gelu_prepass(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    long elements) {
  long index = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  float value = __half2float(input[index]);
  float cubic = value * value * value;
  float tanh_input = 0.7978845608028654f * (value + 0.044715f * cubic);
  output[index] = __float2half(0.5f * value * (1.0f + tanhf(tanh_input)));
}

int main(int argc, char** argv) {
  int m = argc > 1 ? atoi(argv[1]) : 1024;
  int n = argc > 2 ? atoi(argv[2]) : 1024;
  int k = argc > 3 ? atoi(argv[3]) : 1024;
  __half* input = nullptr;
  __half* activated = nullptr;
  __half* weights = nullptr;
  __half* output = nullptr;
  check(cudaMalloc(&input, (size_t)m * k * sizeof(__half)), "cudaMalloc input");
  check(cudaMalloc(&activated, (size_t)m * k * sizeof(__half)), "cudaMalloc activated");
  check(cudaMalloc(&weights, (size_t)k * n * sizeof(__half)), "cudaMalloc weights");
  check(cudaMalloc(&output, (size_t)m * n * sizeof(__half)), "cudaMalloc output");
  check(cudaMemset(input, 1, (size_t)m * k * sizeof(__half)), "cudaMemset input");
  check(cudaMemset(weights, 2, (size_t)k * n * sizeof(__half)), "cudaMemset weights");

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);
  long elements = (long)m * k;
  int blocks = (int)((elements + 255) / 256);

  amora_ppp_gelu_prepass<<<blocks, 256>>>(input, activated, elements);
  check(cudaDeviceSynchronize(), "warmup prepass");
  check_cublas(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &alpha, activated, m, weights, k, &beta, output, m),
               "warmup hgemm");
  check(cudaDeviceSynchronize(), "warmup hgemm sync");

  amora_ppp_gelu_prepass<<<blocks, 256>>>(input, activated, elements);
  check(cudaDeviceSynchronize(), "measured prepass");
  check_cublas(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &alpha, activated, m, weights, k, &beta, output, m),
               "measured hgemm");
  check(cudaDeviceSynchronize(), "measured hgemm sync");

  cublasDestroy(handle);
  cudaFree(input);
  cudaFree(activated);
  cudaFree(weights);
  cudaFree(output);
  return 0;
}
