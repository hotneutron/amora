// AMORA PPP canonical RMSNorm classification workload.
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

__global__ void amora_ppp_rmsnorm_kernel(
    const __half* __restrict__ x,
    const __half* __restrict__ gamma,
    __half* __restrict__ y,
    long rows,
    int hidden) {
  long row = blockIdx.x;
  if (row >= rows) return;
  __shared__ float partial[256];
  float sum = 0.0f;
  for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
    float value = __half2float(x[row * (long)hidden + index]);
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
    float value = __half2float(x[row * (long)hidden + index]);
    y[row * (long)hidden + index] = __float2half(
        value * scale * __half2float(gamma[index]));
  }
}

int main(int argc, char** argv) {
  int batch = argc > 1 ? atoi(argv[1]) : 1;
  int rows_per_batch = argc > 2 ? atoi(argv[2]) : 512;
  int hidden = argc > 3 ? atoi(argv[3]) : 4096;
  long rows = (long)batch * rows_per_batch;
  const int threads = 256;

  __half* x = nullptr;
  __half* gamma = nullptr;
  __half* y = nullptr;
  check(cudaMalloc(&x, (size_t)rows * hidden * sizeof(__half)), "cudaMalloc x");
  check(cudaMalloc(&gamma, (size_t)hidden * sizeof(__half)), "cudaMalloc gamma");
  check(cudaMalloc(&y, (size_t)rows * hidden * sizeof(__half)), "cudaMalloc y");
  check(cudaMemset(x, 1, (size_t)rows * hidden * sizeof(__half)), "cudaMemset x");
  check(cudaMemset(gamma, 1, (size_t)hidden * sizeof(__half)), "cudaMemset gamma");

#ifndef AMORA_GCOM_TRACE
  amora_ppp_rmsnorm_kernel<<<rows, threads>>>(x, gamma, y, rows, hidden);
  check(cudaDeviceSynchronize(), "warmup rmsnorm");
#endif
  amora_ppp_rmsnorm_kernel<<<rows, threads>>>(x, gamma, y, rows, hidden);
  check(cudaDeviceSynchronize(), "measured rmsnorm");

  cudaFree(x);
  cudaFree(gamma);
  cudaFree(y);
  return 0;
}
