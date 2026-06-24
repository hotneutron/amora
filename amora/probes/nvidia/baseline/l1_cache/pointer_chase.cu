// L1-path dependent pointer-chase latency probe.
//
// One thread walks a randomized pointer-chase ring that fits inside the
// candidate L1 data cache, so steady-state loads hit in L1. Cycles-per-load is
// reported as the median across many launches. A larger DRAM-resident working
// set is also timed as a control so the analyzer can confirm the small ring is
// an L1-hit regime (small << large).

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

#ifndef AMORA_STEPS
#define AMORA_STEPS 4096
#endif

// Dependent pointer chase: each load address depends on the previous load.
extern "C" __global__ void
amora_l1_pointer_chase(const int *idx, int start, int steps,
                       unsigned long long *cycles_out, int *sink) {
  if (threadIdx.x != 0 || blockIdx.x != 0) return;
  int p = start;
  unsigned long long t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < steps; ++i) {
    p = idx[p];
  }
  unsigned long long t1 = clock64();
  cycles_out[0] = t1 - t0;
  sink[0] = p;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

// Build a single cycle that visits every slot once (a random permutation),
// so the hardware prefetcher cannot trivially predict the next address.
static void build_ring(int *host, int n) {
  for (int i = 0; i < n; ++i) host[i] = i;
  // Fisher-Yates over indices 1..n-1, then thread them into one cycle.
  for (int i = n - 1; i > 0; --i) {
    int j = (int)(((unsigned long long)rand() * (i + 1)) / ((unsigned long long)RAND_MAX + 1));
    int t = host[i]; host[i] = host[j]; host[j] = t;
  }
  // host[] now is a permutation order; turn it into next-pointers.
  int *next = (int *)std::malloc(sizeof(int) * n);
  for (int i = 0; i < n; ++i) {
    next[host[i]] = host[(i + 1) % n];
  }
  std::memcpy(host, next, sizeof(int) * n);
  std::free(next);
}

static double median_chase(const int *d_idx, int slots, int steps, int repeats,
                           unsigned long long *d_cycles, int *d_sink) {
  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);
  for (int r = 0; r < repeats; ++r) {
    amora_l1_pointer_chase<<<1, 32>>>(d_idx, r % slots, steps, d_cycles, d_sink);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }
  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  unsigned long long median = samples[repeats / 2];
  std::free(samples);
  return (double)median / (double)steps;
}

int main(int argc, char **argv) {
  int small_kb = 16;      // working set that should fit in L1
  int large_kb = 8192;    // working set that should miss to DRAM
  int repeats = 64;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--small-kb") == 0) small_kb = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--large-kb") == 0) large_kb = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 64;
  srand(1234);

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  int small_slots = (small_kb * 1024) / (int)sizeof(int);
  int large_slots = (large_kb * 1024) / (int)sizeof(int);

  unsigned long long *d_cycles = nullptr;
  int *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(int)));

  // Small (L1-hit candidate) ring.
  int *h_small = (int *)std::malloc(sizeof(int) * small_slots);
  build_ring(h_small, small_slots);
  int *d_small = nullptr;
  AMORA_CHECK(cudaMalloc(&d_small, sizeof(int) * small_slots));
  AMORA_CHECK(cudaMemcpy(d_small, h_small, sizeof(int) * small_slots, cudaMemcpyHostToDevice));

  // Large (DRAM) ring control.
  int *h_large = (int *)std::malloc(sizeof(int) * large_slots);
  build_ring(h_large, large_slots);
  int *d_large = nullptr;
  AMORA_CHECK(cudaMalloc(&d_large, sizeof(int) * large_slots));
  AMORA_CHECK(cudaMemcpy(d_large, h_large, sizeof(int) * large_slots, cudaMemcpyHostToDevice));

  // Warm-up.
  amora_l1_pointer_chase<<<1, 32>>>(d_small, 0, AMORA_STEPS, d_cycles, d_sink);
  AMORA_CHECK(cudaDeviceSynchronize());

  double small_cpl = median_chase(d_small, small_slots, AMORA_STEPS, repeats, d_cycles, d_sink);
  double large_cpl = median_chase(d_large, large_slots, AMORA_STEPS, repeats, d_cycles, d_sink);

  std::printf(
      "{\"device_name\":\"%s\",\"steps\":%d,\"repeats\":%d,"
      "\"small_kb\":%d,\"large_kb\":%d,"
      "\"l1_hit_cycles_per_load\":%.4f,\"dram_cycles_per_load\":%.4f}\n",
      prop.name, AMORA_STEPS, repeats, small_kb, large_kb, small_cpl, large_cpl);

  std::free(h_small);
  std::free(h_large);
  cudaFree(d_small);
  cudaFree(d_large);
  cudaFree(d_cycles);
  cudaFree(d_sink);
  return 0;
}
