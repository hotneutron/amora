// AMORA PPP bounded MegaMoE classification workload.
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

__global__ void amora_ppp_megamoe_kernel(
    const unsigned char* __restrict__ x,
    const unsigned char* __restrict__ weights,
    __half* __restrict__ out,
    long tokens,
    int hidden,
    int experts,
    int top_k,
    int sample_hidden) {
  long route = blockIdx.x;
  long token = route / top_k;
  int expert = route % experts;
  if (token >= tokens) return;
  float accum = 0.0f;
  int length = min(sample_hidden, hidden);
  for (int index = threadIdx.x; index < length; index += blockDim.x) {
    accum += (float)x[token * (long)hidden + index] *
             (float)weights[expert * (long)hidden + index];
  }
  if (threadIdx.x < hidden) {
    out[token * (long)hidden + threadIdx.x] = __float2half(accum * 1.0e-4f);
  }
}

int main(int argc, char** argv) {
  int batch = argc > 1 ? atoi(argv[1]) : 1;
  int sequence = argc > 2 ? atoi(argv[2]) : 4096;
  int hidden = argc > 3 ? atoi(argv[3]) : 4096;
  int experts = argc > 4 ? atoi(argv[4]) : 128;
  int top_k = argc > 5 ? atoi(argv[5]) : 8;
  int sample_hidden = argc > 6 ? atoi(argv[6]) : 1024;
  top_k = max(top_k, 1);
  sample_hidden = max(sample_hidden, 1);
  long tokens = (long)batch * sequence;
  long routes = tokens * top_k;

  unsigned char* x = nullptr;
  unsigned char* weights = nullptr;
  __half* out = nullptr;
  check(cudaMalloc(&x, (size_t)tokens * hidden), "cudaMalloc x");
  check(cudaMalloc(&weights, (size_t)experts * hidden), "cudaMalloc weights");
  check(cudaMalloc(&out, (size_t)tokens * hidden * sizeof(__half)), "cudaMalloc out");
  check(cudaMemset(x, 3, (size_t)tokens * hidden), "cudaMemset x");
  check(cudaMemset(weights, 5, (size_t)experts * hidden), "cudaMemset weights");

  const int threads = 256;
#ifndef AMORA_GCOM_TRACE
  amora_ppp_megamoe_kernel<<<routes, threads>>>(
      x, weights, out, tokens, hidden, experts, top_k, sample_hidden);
  check(cudaDeviceSynchronize(), "warmup megamoe");
#endif
  amora_ppp_megamoe_kernel<<<routes, threads>>>(
      x, weights, out, tokens, hidden, experts, top_k, sample_hidden);
  check(cudaDeviceSynchronize(), "measured megamoe");

  cudaFree(x);
  cudaFree(weights);
  cudaFree(out);
  return 0;
}
