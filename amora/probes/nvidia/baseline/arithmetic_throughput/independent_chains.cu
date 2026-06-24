// Independent FP32 FMA throughput probe.
//
// 4 independent FMA chains per thread; full warps and many CTAs are launched so
// the SM can ILP/TLP overlap them. Reports cycles-per-fma per-thread as the
// effective pipeline throughput.

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
#ifndef AMORA_INDEP_CHAINS
#define AMORA_INDEP_CHAINS 4
#endif

extern "C" __global__ void amora_baseline_fp32_independent_chains(
    float *out, unsigned long long *cycles_out, float seed) {
  float a = seed;
  float b = seed + 1.0f;
  float c = seed + 2.0f;
  float d = seed + 3.0f;
  unsigned long long start = clock64();
#pragma unroll 32
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
    a = fmaf(a, 1.000001f, 0.000001f);
    b = fmaf(b, 1.000002f, 0.000002f);
    c = fmaf(c, 1.000003f, 0.000003f);
    d = fmaf(d, 1.000004f, 0.000004f);
  }
  unsigned long long stop = clock64();
  unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;
  cycles_out[gid] = stop - start;
  // Force the chain results to be consumed so the compiler can't drop them.
  out[gid] = a + b + c + d;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

int main(int argc, char **argv) {
  int blocks = 16;
  int threads = 128;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--blocks") == 0) blocks = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--threads") == 0) threads = std::atoi(argv[i + 1]);
  }
  if (blocks <= 0) blocks = 16;
  if (threads <= 0) threads = 128;
  int total = blocks * threads;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  float *d_out = nullptr;
  unsigned long long *d_cycles = nullptr;
  AMORA_CHECK(cudaMalloc(&d_out, sizeof(float) * total));
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long) * total));

  // Warm-up.
  amora_baseline_fp32_independent_chains<<<blocks, threads>>>(d_out, d_cycles, 1.0f);
  AMORA_CHECK(cudaDeviceSynchronize());

  cudaEvent_t t0, t1;
  AMORA_CHECK(cudaEventCreate(&t0));
  AMORA_CHECK(cudaEventCreate(&t1));
  AMORA_CHECK(cudaEventRecord(t0));
  amora_baseline_fp32_independent_chains<<<blocks, threads>>>(d_out, d_cycles, 7.0f);
  AMORA_CHECK(cudaEventRecord(t1));
  AMORA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.0f;
  AMORA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * total);
  AMORA_CHECK(cudaMemcpy(samples, d_cycles, sizeof(unsigned long long) * total,
                          cudaMemcpyDeviceToHost));

  std::qsort(samples, total, sizeof(unsigned long long), cmp_ull);
  unsigned long long median = samples[total / 2];
  unsigned long long minc = samples[0];
  unsigned long long maxc = samples[total - 1];
  long fmas_per_thread = (long)AMORA_CHAIN_LEN * AMORA_INDEP_CHAINS;
  double cycles_per_fma_thread = (double)median / (double)fmas_per_thread;
  double total_fmas = (double)total * (double)fmas_per_thread;
  double approx_fma_per_cycle_per_sm = total_fmas / ((double)median * (double)prop.multiProcessorCount);

  std::printf(
      "{\"device_name\":\"%s\",\"multi_processor_count\":%d,\"blocks\":%d,\"threads\":%d,"
      "\"chain_length\":%d,\"independent_chains\":%d,"
      "\"cycles_min\":%llu,\"cycles_median\":%llu,\"cycles_max\":%llu,"
      "\"cycles_per_fma_per_thread\":%.4f,"
      "\"approx_fma_per_cycle_per_sm\":%.4f,"
      "\"elapsed_ms\":%.4f}\n",
      prop.name, prop.multiProcessorCount, blocks, threads,
      AMORA_CHAIN_LEN, AMORA_INDEP_CHAINS,
      (unsigned long long)minc, (unsigned long long)median, (unsigned long long)maxc,
      cycles_per_fma_thread, approx_fma_per_cycle_per_sm, ms);

  std::free(samples);
  cudaFree(d_out);
  cudaFree(d_cycles);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return 0;
}
