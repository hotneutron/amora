// L1 working-set sweep probe.
//
// Runs the dependent pointer-chase at a series of working-set sizes and reports
// cycles-per-load at each size. The latency-vs-size curve exposes capacity knees
// (L1 -> L2 -> DRAM transitions) for the analyzer to fit.

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
amora_l1_working_set(const int *idx, int start, int steps,
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

static void build_ring(int *host, int n) {
  for (int i = 0; i < n; ++i) host[i] = i;
  for (int i = n - 1; i > 0; --i) {
    int j = (int)(((unsigned long long)rand() * (i + 1)) / ((unsigned long long)RAND_MAX + 1));
    int t = host[i]; host[i] = host[j]; host[j] = t;
  }
  int *next = (int *)std::malloc(sizeof(int) * n);
  for (int i = 0; i < n; ++i) next[host[i]] = host[(i + 1) % n];
  std::memcpy(host, next, sizeof(int) * n);
  std::free(next);
}

int main(int argc, char **argv) {
  int repeats = 32;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 32;
  srand(4321);

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  // Geometric working-set sweep in KiB from 4 KiB to 16 MiB.
  const int kb_points[] = {4, 8, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 4096, 16384};
  const int n_points = (int)(sizeof(kb_points) / sizeof(kb_points[0]));

  unsigned long long *d_cycles = nullptr;
  int *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(int)));

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);

  std::printf("{\"device_name\":\"%s\",\"steps\":%d,\"repeats\":%d,\"sweep\":[",
              prop.name, AMORA_STEPS, repeats);

  for (int k = 0; k < n_points; ++k) {
    int slots = (kb_points[k] * 1024) / (int)sizeof(int);
    int *h = (int *)std::malloc(sizeof(int) * slots);
    build_ring(h, slots);
    int *d = nullptr;
    AMORA_CHECK(cudaMalloc(&d, sizeof(int) * slots));
    AMORA_CHECK(cudaMemcpy(d, h, sizeof(int) * slots, cudaMemcpyHostToDevice));

    // Warm-up for this size.
    amora_l1_working_set<<<1, 32>>>(d, 0, AMORA_STEPS, d_cycles, d_sink);
    AMORA_CHECK(cudaDeviceSynchronize());

    for (int r = 0; r < repeats; ++r) {
      amora_l1_working_set<<<1, 32>>>(d, r % slots, AMORA_STEPS, d_cycles, d_sink);
      AMORA_CHECK(cudaDeviceSynchronize());
      AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }
    std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
    double cpl = (double)samples[repeats / 2] / (double)AMORA_STEPS;
    std::printf("%s{\"working_set_kb\":%d,\"cycles_per_load\":%.4f}",
                (k == 0 ? "" : ","), kb_points[k], cpl);

    std::free(h);
    cudaFree(d);
  }

  std::printf("]}\n");

  std::free(samples);
  cudaFree(d_cycles);
  cudaFree(d_sink);
  return 0;
}
