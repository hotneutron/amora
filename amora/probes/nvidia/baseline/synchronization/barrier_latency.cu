// CTA barrier latency probe.
//
// One CTA executes a long run of __syncthreads() barriers with a tiny amount of
// work between them, bracketed by clock64(). Cycles-per-barrier is the median
// across launches, swept over a few block sizes so the analyzer can report the
// scaling class. This is a timing-first measurement (no counters required).

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

#ifndef AMORA_BARRIERS
#define AMORA_BARRIERS 4096
#endif

extern "C" __global__ void
amora_baseline_barrier_latency(unsigned long long *cycles_out, int *sink) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  unsigned long long t0 = clock64();
#pragma unroll 8
  for (int i = 0; i < AMORA_BARRIERS; ++i) {
    if (threadIdx.x == 0) s += 1;
    __syncthreads();
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0) {
    cycles_out[0] = t1 - t0;
    sink[0] = s;
  }
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

int main(int argc, char **argv) {
  int repeats = 32;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 32;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  unsigned long long *d_cycles = nullptr;
  int *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(int)));

  const int block_sizes[] = {64, 128, 256, 512, 1024};
  const int n_blocks = (int)(sizeof(block_sizes) / sizeof(block_sizes[0]));

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);

  // Warm-up.
  amora_baseline_barrier_latency<<<1, 256>>>(d_cycles, d_sink);
  AMORA_CHECK(cudaDeviceSynchronize());

  std::printf("{\"device_name\":\"%s\",\"barriers\":%d,\"sweep\":[",
              prop.name, AMORA_BARRIERS);

  for (int b = 0; b < n_blocks; ++b) {
    int threads = block_sizes[b];
    for (int r = 0; r < repeats; ++r) {
      amora_baseline_barrier_latency<<<1, threads>>>(d_cycles, d_sink);
      AMORA_CHECK(cudaDeviceSynchronize());
      AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }
    std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
    double cpb = (double)samples[repeats / 2] / (double)AMORA_BARRIERS;
    std::printf("%s{\"threads_per_block\":%d,\"cycles_per_barrier\":%.4f}",
                (b == 0 ? "" : ","), threads, cpb);
  }

  std::printf("]}\n");

  std::free(samples);
  cudaFree(d_cycles);
  cudaFree(d_sink);
  return 0;
}
