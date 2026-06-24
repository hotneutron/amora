// Dependent FP32 FMA latency probe.
//
// One thread runs a long dependent fma chain bracketed by clock64() reads.
// Cycles-per-fma is reported as the median across many launches.

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

#ifndef AMORA_CHAIN_LEN
#define AMORA_CHAIN_LEN 4096
#endif

extern "C" __global__ void
amora_baseline_fp32_dependent_chain(float *out, unsigned long long *cycles_out, float seed) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  float x = seed;
  unsigned long long start = clock64();
#pragma unroll 32
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
    x = fmaf(x, 1.000001f, 0.000001f);
  }
  unsigned long long stop = clock64();
  cycles_out[0] = stop - start;
  out[0] = x;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

int main(int argc, char **argv) {
  int repeats = 64;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 64;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  float *d_out = nullptr;
  unsigned long long *d_cycles = nullptr;
  AMORA_CHECK(cudaMalloc(&d_out, sizeof(float)));
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

  // Warm-up.
  amora_baseline_fp32_dependent_chain<<<1, 32>>>(d_out, d_cycles, 1.0f);
  AMORA_CHECK(cudaDeviceSynchronize());

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);
  for (int r = 0; r < repeats; ++r) {
    amora_baseline_fp32_dependent_chain<<<1, 32>>>(d_out, d_cycles, 1.0f + (float)r);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }

  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  unsigned long long median = samples[repeats / 2];
  unsigned long long minc = samples[0];
  unsigned long long maxc = samples[repeats - 1];
  double cycles_per_fma = (double)median / (double)AMORA_CHAIN_LEN;

  std::printf(
      "{\"device_name\":\"%s\",\"chain_length\":%d,\"repeats\":%d,"
      "\"cycles_min\":%llu,\"cycles_median\":%llu,\"cycles_max\":%llu,"
      "\"cycles_per_fma\":%.4f}\n",
      prop.name, AMORA_CHAIN_LEN, repeats,
      (unsigned long long)minc, (unsigned long long)median, (unsigned long long)maxc,
      cycles_per_fma);

  std::free(samples);
  cudaFree(d_out);
  cudaFree(d_cycles);
  return 0;
}
