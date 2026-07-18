// AMORA PPP canonical GELU classification workload.
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

__global__ void amora_ppp_gelu_kernel(
    const __half* __restrict__ x,
    __half* __restrict__ y,
    long n) {
  long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  float value = __half2float(x[idx]);
  float cubic = value * value * value;
  float tanh_input = 0.7978845608028654f * (value + 0.044715f * cubic);
  y[idx] = __float2half(0.5f * value * (1.0f + tanhf(tanh_input)));
}

int main(int argc, char** argv) {
  long n = argc > 1 ? atol(argv[1]) : 1048576;
  const int threads = 256;
  long blocks = max(1L, (n + threads - 1) / threads);

  __half* x = nullptr;
  __half* y = nullptr;
  check(cudaMalloc(&x, (size_t)n * sizeof(__half)), "cudaMalloc x");
  check(cudaMalloc(&y, (size_t)n * sizeof(__half)), "cudaMalloc y");
  check(cudaMemset(x, 1, (size_t)n * sizeof(__half)), "cudaMemset x");

#ifndef AMORA_GCOM_TRACE
  amora_ppp_gelu_kernel<<<blocks, threads>>>(x, y, n);
  check(cudaDeviceSynchronize(), "warmup gelu");
#endif
  amora_ppp_gelu_kernel<<<blocks, threads>>>(x, y, n);
  check(cudaDeviceSynchronize(), "measured gelu");

  cudaFree(x);
  cudaFree(y);
  return 0;
}
