// Interconnect injection-rate probe.
//
// A multi-SM grid streams a cache-exceeding global buffer (float4 grid-stride
// reads), sweeping the number of resident blocks (grid = mp_count * {1,2,4,8})
// to vary the offered load on the memory interconnect. Per offered load we
// report aggregate achieved GB/s, timed with CUDA events. Timing-first; NCU DRAM
// byte counters can corroborate later.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define AMORA_CHECK(call)                                                      \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(_e));  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

typedef float4 vec_t;

extern "C" __global__ void
amora_icn_injection_rate(const vec_t *in, float *partial, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n; i += stride) {
    vec_t v = in[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == -1.f) partial[0] = acc;  // prevent DCE
}

int main(int argc, char **argv) {
  int mb = 512;
  int iters = 5;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--mb") == 0) mb = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
  }
  if (mb <= 0) mb = 512;
  if (iters <= 0) iters = 5;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  size_t bytes = (size_t)mb * 1024 * 1024;
  size_t n = bytes / sizeof(vec_t);

  vec_t *d_in = nullptr;
  float *d_p = nullptr;
  AMORA_CHECK(cudaMalloc(&d_in, bytes));
  AMORA_CHECK(cudaMalloc(&d_p, sizeof(float)));
  AMORA_CHECK(cudaMemset(d_in, 1, bytes));

  const int block = 256;
  const int blocks_per_sm[] = {1, 2, 4, 8};
  const int n_loads = (int)(sizeof(blocks_per_sm) / sizeof(blocks_per_sm[0]));

  cudaEvent_t a, b;
  AMORA_CHECK(cudaEventCreate(&a));
  AMORA_CHECK(cudaEventCreate(&b));

  // Warm-up.
  amora_icn_injection_rate<<<prop.multiProcessorCount, block>>>(d_in, d_p, n);
  AMORA_CHECK(cudaDeviceSynchronize());

  std::printf("{\"device_name\":\"%s\",\"sweep\":[", prop.name);

  for (int s = 0; s < n_loads; ++s) {
    int k = blocks_per_sm[s];
    int grid = prop.multiProcessorCount * k;
    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
      AMORA_CHECK(cudaEventRecord(a));
      amora_icn_injection_rate<<<grid, block>>>(d_in, d_p, n);
      AMORA_CHECK(cudaEventRecord(b));
      AMORA_CHECK(cudaEventSynchronize(b));
      float ms = 0.f;
      AMORA_CHECK(cudaEventElapsedTime(&ms, a, b));
      if (ms < best) best = ms;
    }
    double gbps = (double)bytes / (best / 1000.0) / 1e9;
    std::printf("%s{\"blocks_per_sm\":%d,\"gbps\":%.2f}", (s == 0 ? "" : ","), k, gbps);
  }

  std::printf("]}\n");

  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaFree(d_in);
  cudaFree(d_p);
  return 0;
}
