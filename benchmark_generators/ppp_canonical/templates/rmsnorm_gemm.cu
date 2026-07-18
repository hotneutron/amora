// AMORA PPP RMSNorm plus fp16 GEMM classification workload.
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

__global__ void amora_ppp_rmsnorm_prepass(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int rows,
    int hidden) {
  int row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float partial[256];
  float sum = 0.0f;
  for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
    float value = __half2float(input[row * (long)hidden + index]);
    sum += value * value;
  }
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }
  float scale = rsqrtf(partial[0] / max(hidden, 1) + 1.0e-6f);
  for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
    output[row * (long)hidden + index] =
        __float2half(__half2float(input[row * (long)hidden + index]) * scale);
  }
}

int main(int argc, char** argv) {
  int m = argc > 1 ? atoi(argv[1]) : 1024;
  int n = argc > 2 ? atoi(argv[2]) : 1024;
  int k = argc > 3 ? atoi(argv[3]) : 1024;
  __half* input = nullptr;
  __half* normalized = nullptr;
  __half* weights = nullptr;
  __half* output = nullptr;
  check(cudaMalloc(&input, (size_t)m * k * sizeof(__half)), "cudaMalloc input");
  check(cudaMalloc(&normalized, (size_t)m * k * sizeof(__half)), "cudaMalloc normalized");
  check(cudaMalloc(&weights, (size_t)k * n * sizeof(__half)), "cudaMalloc weights");
  check(cudaMalloc(&output, (size_t)m * n * sizeof(__half)), "cudaMalloc output");
  check(cudaMemset(input, 1, (size_t)m * k * sizeof(__half)), "cudaMemset input");
  check(cudaMemset(weights, 2, (size_t)k * n * sizeof(__half)), "cudaMemset weights");

  cublasHandle_t handle;
  check_cublas(cublasCreate(&handle), "cublasCreate");
  check_cublas(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH), "cublasSetMathMode");
  __half alpha = __float2half(1.0f);
  __half beta = __float2half(0.0f);

#ifndef AMORA_GCOM_TRACE
  amora_ppp_rmsnorm_prepass<<<m, 256>>>(input, normalized, m, k);
  check(cudaDeviceSynchronize(), "warmup prepass");
  check_cublas(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &alpha, normalized, m, weights, k, &beta, output, m),
               "warmup hgemm");
  check(cudaDeviceSynchronize(), "warmup hgemm sync");
#endif
  amora_ppp_rmsnorm_prepass<<<m, 256>>>(input, normalized, m, k);
  check(cudaDeviceSynchronize(), "measured prepass");
  check_cublas(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                           &alpha, normalized, m, weights, k, &beta, output, m),
               "measured hgemm");
  check(cudaDeviceSynchronize(), "measured hgemm sync");

  cublasDestroy(handle);
  cudaFree(input);
  cudaFree(normalized);
  cudaFree(weights);
  cudaFree(output);
  return 0;
}
