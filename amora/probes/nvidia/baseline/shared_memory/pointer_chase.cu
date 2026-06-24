// Shared-memory pointer-chase latency probe.
//
// One thread chases a circular pointer chain in shared memory. The
// per-iteration cycle count is the shared-memory load-to-use latency.

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

#ifndef AMORA_CHASE_LEN
#define AMORA_CHASE_LEN 4096
#endif

extern "C" __global__ void amora_baseline_shared_pointer_chase(
    unsigned int *out, unsigned long long *cycles_out) {
  __shared__ unsigned int next[1024];
  unsigned int tid = threadIdx.x;
  if (tid < 1024) next[tid] = (tid + 1) & 1023;
  __syncthreads();
  if (tid != 0) return;

  unsigned int index = 0;
  unsigned long long start = clock64();
#pragma unroll 64
  for (int i = 0; i < AMORA_CHASE_LEN; ++i) {
    index = next[index];
  }
  unsigned long long stop = clock64();
  cycles_out[0] = stop - start;
  out[0] = index;
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

  unsigned int *d_out = nullptr;
  unsigned long long *d_cycles = nullptr;
  AMORA_CHECK(cudaMalloc(&d_out, sizeof(unsigned int)));
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

  // Warm-up.
  amora_baseline_shared_pointer_chase<<<1, 1024>>>(d_out, d_cycles);
  AMORA_CHECK(cudaDeviceSynchronize());

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);
  for (int r = 0; r < repeats; ++r) {
    amora_baseline_shared_pointer_chase<<<1, 1024>>>(d_out, d_cycles);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }

  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  unsigned long long median = samples[repeats / 2];
  double cycles_per_load = (double)median / (double)AMORA_CHASE_LEN;

  std::printf(
      "{\"device_name\":\"%s\",\"chase_len\":%d,\"repeats\":%d,"
      "\"cycles_min\":%llu,\"cycles_median\":%llu,\"cycles_max\":%llu,"
      "\"cycles_per_load\":%.4f}\n",
      prop.name, AMORA_CHASE_LEN, repeats,
      (unsigned long long)samples[0], (unsigned long long)median,
      (unsigned long long)samples[repeats - 1], cycles_per_load);

  std::free(samples);
  cudaFree(d_out);
  cudaFree(d_cycles);
  return 0;
}
