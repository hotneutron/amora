// L1 conflict-set associativity probe.
//
// Builds pointer-chase rings whose elements all map to the same cache set by
// spacing them a fixed power-of-two stride apart, then grows the number of
// distinct lines in the ring. While the line count stays within the set
// associativity the ring is cache-resident (low latency); once it exceeds the
// associativity, every access conflict-misses (high latency). The knee in the
// latency-vs-ways curve estimates effective associativity.

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

extern "C" __global__ void
amora_l1_conflict(const int *idx, int start, int steps,
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

int main(int argc, char **argv) {
  int repeats = 32;
  int max_ways = 24;
  // Stride in bytes between conflicting lines (same set under a power-of-two
  // index). 4 KiB keeps lines in the same set for typical L1 geometries.
  int stride_bytes = 4096;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--max-ways") == 0) max_ways = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--stride-bytes") == 0) stride_bytes = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 32;
  if (max_ways <= 1) max_ways = 24;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  int stride_ints = stride_bytes / (int)sizeof(int);
  int total_slots = stride_ints * (max_ways + 1);

  int *h = (int *)std::malloc(sizeof(int) * total_slots);
  int *d = nullptr;
  AMORA_CHECK(cudaMalloc(&d, sizeof(int) * total_slots));

  unsigned long long *d_cycles = nullptr;
  int *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(int)));

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);

  std::printf("{\"device_name\":\"%s\",\"steps\":%d,\"stride_bytes\":%d,\"sweep\":[",
              prop.name, AMORA_STEPS, stride_bytes);

  for (int ways = 1; ways <= max_ways; ++ways) {
    // Build a ring over `ways` lines spaced `stride_ints` apart.
    for (int w = 0; w < ways; ++w) {
      int cur = w * stride_ints;
      int nxt = ((w + 1) % ways) * stride_ints;
      h[cur] = nxt;
    }
    AMORA_CHECK(cudaMemcpy(d, h, sizeof(int) * total_slots, cudaMemcpyHostToDevice));

    amora_l1_conflict<<<1, 32>>>(d, 0, AMORA_STEPS, d_cycles, d_sink);
    AMORA_CHECK(cudaDeviceSynchronize());

    for (int r = 0; r < repeats; ++r) {
      amora_l1_conflict<<<1, 32>>>(d, 0, AMORA_STEPS, d_cycles, d_sink);
      AMORA_CHECK(cudaDeviceSynchronize());
      AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }
    std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
    double cpl = (double)samples[repeats / 2] / (double)AMORA_STEPS;
    std::printf("%s{\"ways\":%d,\"cycles_per_load\":%.4f}", (ways == 1 ? "" : ","), ways, cpl);
  }

  std::printf("]}\n");

  std::free(h);
  std::free(samples);
  cudaFree(d);
  cudaFree(d_cycles);
  cudaFree(d_sink);
  return 0;
}
