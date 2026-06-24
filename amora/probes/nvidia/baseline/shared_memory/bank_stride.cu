// Shared-memory bank-conflict stride sweep.
//
// One warp of 32 threads reads a 4-byte word at index (tid * stride) % N. For
// strides that are coprime to 32 the warp hits 32 distinct banks (no
// conflicts); strides that share factors with 32 produce conflict counts
// 2,4,8,16,32. The probe times each stride and emits a sweep curve.

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

#ifndef AMORA_INNER_LOOPS
#define AMORA_INNER_LOOPS 4096
#endif

// One warp does AMORA_INNER_LOOPS reads at the chosen stride.
extern "C" __global__ void amora_baseline_shared_bank_stride(
    unsigned int *sink, unsigned long long *cycles_out, int stride) {
  __shared__ unsigned int values[2048];
  unsigned int tid = threadIdx.x;
  if (tid < 2048) values[tid] = tid * 2654435761u;
  __syncthreads();
  if (tid >= 32) return;

  unsigned int idx = (tid * (unsigned int)stride) & 2047u;
  unsigned int acc = 0;
  unsigned long long start = clock64();
#pragma unroll 32
  for (int i = 0; i < AMORA_INNER_LOOPS; ++i) {
    acc ^= values[idx];
    idx = (idx + (unsigned int)stride) & 2047u;
  }
  unsigned long long stop = clock64();
  cycles_out[tid] = stop - start;
  if (tid == 0) sink[0] = acc;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

int main(int argc, char **argv) {
  // Strides chosen to span the 1,2,4,8,16,32-way bank-conflict regimes.
  int strides[] = {1, 2, 3, 4, 5, 7, 8, 11, 16, 17, 32, 33};
  int n_strides = sizeof(strides) / sizeof(strides[0]);
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--strides") == 0) {
      // Parse comma-separated list.
      n_strides = 0;
      char *tok = std::strtok((char *)argv[i + 1], ",");
      while (tok && n_strides < 32) {
        strides[n_strides++] = std::atoi(tok);
        tok = std::strtok(nullptr, ",");
      }
    }
  }

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  unsigned int *d_sink = nullptr;
  unsigned long long *d_cycles = nullptr;
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(unsigned int)));
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long) * 32));

  // Warm-up.
  amora_baseline_shared_bank_stride<<<1, 32>>>(d_sink, d_cycles, 1);
  AMORA_CHECK(cudaDeviceSynchronize());

  std::printf("{\"device_name\":\"%s\",\"inner_loops\":%d,\"sweep\":[",
              prop.name, AMORA_INNER_LOOPS);
  for (int s = 0; s < n_strides; ++s) {
    int stride = strides[s];
    int repeats = 32;
    unsigned long long medians[32];
    for (int r = 0; r < repeats; ++r) {
      amora_baseline_shared_bank_stride<<<1, 32>>>(d_sink, d_cycles, stride);
      AMORA_CHECK(cudaDeviceSynchronize());
      unsigned long long lane_cycles[32];
      AMORA_CHECK(cudaMemcpy(lane_cycles, d_cycles, sizeof(unsigned long long) * 32,
                              cudaMemcpyDeviceToHost));
      // Use lane 0's cycle count (it's representative for this single-warp probe).
      medians[r] = lane_cycles[0];
    }
    std::qsort(medians, repeats, sizeof(unsigned long long), cmp_ull);
    unsigned long long median = medians[repeats / 2];
    double cycles_per_access = (double)median / (double)AMORA_INNER_LOOPS;
    // gcd(stride, 32) gives expected bank-conflict factor.
    int a = stride > 0 ? stride : -stride;
    int b = 32;
    while (b) { int t = b; b = a % b; a = t; }
    int gcd_with_32 = a == 0 ? 1 : a;
    int conflict_factor = gcd_with_32; // 1=no conflict, 32=full conflict
    std::printf("%s{\"stride\":%d,\"conflict_factor\":%d,\"cycles_median\":%llu,"
                "\"cycles_per_access\":%.4f}",
                s ? "," : "", stride, conflict_factor,
                (unsigned long long)median, cycles_per_access);
  }
  std::printf("]}\n");

  cudaFree(d_sink);
  cudaFree(d_cycles);
  return 0;
}
