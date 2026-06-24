// L2-path dependent pointer-chase latency probe.
//
// One thread walks a randomized pointer-chase ring sized to exceed the L1 data
// cache but fit inside the L2, so steady-state loads hit in L2. Cycles-per-load
// is reported as the median across many launches. A much larger DRAM-resident
// ring is also timed as a control so the analyzer can confirm the L2-fit ring is
// an L2-hit regime (l2 << dram).

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
amora_l2_pointer_chase(const int *idx, int start, int steps,
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
    amora_l2_pointer_chase<<<1, 32>>>(d_idx, r % slots, steps, d_cycles, d_sink);
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
  int l2_kb = 4096;       // working set that should miss L1 but fit in L2
  int dram_kb = 131072;   // working set that should miss to DRAM
  int repeats = 64;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--l2-kb") == 0) l2_kb = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--dram-kb") == 0) dram_kb = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 64;
  srand(1234);

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  int l2_slots = (l2_kb * 1024) / (int)sizeof(int);
  int dram_slots = (dram_kb * 1024) / (int)sizeof(int);

  unsigned long long *d_cycles = nullptr;
  int *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(int)));

  // L2-fit (L2-hit candidate) ring.
  int *h_l2 = (int *)std::malloc(sizeof(int) * l2_slots);
  build_ring(h_l2, l2_slots);
  int *d_l2 = nullptr;
  AMORA_CHECK(cudaMalloc(&d_l2, sizeof(int) * l2_slots));
  AMORA_CHECK(cudaMemcpy(d_l2, h_l2, sizeof(int) * l2_slots, cudaMemcpyHostToDevice));

  // Large (DRAM) ring control.
  int *h_dram = (int *)std::malloc(sizeof(int) * dram_slots);
  build_ring(h_dram, dram_slots);
  int *d_dram = nullptr;
  AMORA_CHECK(cudaMalloc(&d_dram, sizeof(int) * dram_slots));
  AMORA_CHECK(cudaMemcpy(d_dram, h_dram, sizeof(int) * dram_slots, cudaMemcpyHostToDevice));

  // Warm-up.
  amora_l2_pointer_chase<<<1, 32>>>(d_l2, 0, AMORA_STEPS, d_cycles, d_sink);
  AMORA_CHECK(cudaDeviceSynchronize());

  double l2_cpl = median_chase(d_l2, l2_slots, AMORA_STEPS, repeats, d_cycles, d_sink);
  double dram_cpl = median_chase(d_dram, dram_slots, AMORA_STEPS, repeats, d_cycles, d_sink);

  std::printf(
      "{\"device_name\":\"%s\",\"steps\":%d,\"repeats\":%d,"
      "\"l2_kb\":%d,\"dram_kb\":%d,"
      "\"l2_hit_cycles_per_load\":%.4f,\"dram_cycles_per_load\":%.4f}\n",
      prop.name, AMORA_STEPS, repeats, l2_kb, dram_kb, l2_cpl, dram_cpl);

  std::free(h_l2);
  std::free(h_dram);
  cudaFree(d_l2);
  cudaFree(d_dram);
  cudaFree(d_cycles);
  cudaFree(d_sink);
  return 0;
}
