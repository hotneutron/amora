// Memory fence latency probe.
//
// One CTA executes a long loop of __threadfence() (device-scope memory fence)
// calls with minimal work, bracketed by clock64(). An empty-loop baseline with
// the same structure (minus the fence) measures the loop overhead so the wrapper
// can subtract it. Cycles-per-fence is the median across launches. This is a
// timing-first measurement (no counters required); SASS validation confirms the
// MEMBAR opcode is present in the timed kernel.

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

#ifndef AMORA_FENCES
#define AMORA_FENCES 4096
#endif

extern "C" __global__ void
amora_baseline_fence_latency(unsigned long long *cycles_out, int *sink) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  int acc = sink[0];
  unsigned long long t0 = clock64();
#pragma unroll 8
  for (int i = 0; i < AMORA_FENCES; ++i) {
    acc += i;
    __threadfence();
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0) {
    cycles_out[0] = t1 - t0;
    sink[0] = acc + s;
  }
}

extern "C" __global__ void
amora_baseline_fence_empty(unsigned long long *cycles_out, int *sink) {
  __shared__ int s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  int acc = sink[0];
  unsigned long long t0 = clock64();
#pragma unroll 8
  for (int i = 0; i < AMORA_FENCES; ++i) {
    acc += i;
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0) {
    cycles_out[0] = t1 - t0;
    sink[0] = acc + s;
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

  const int threads = 256;
  unsigned long long *d_cycles = nullptr;
  int *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(int)));
  AMORA_CHECK(cudaMemset(d_sink, 0, sizeof(int)));

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);

  // Warm-up.
  amora_baseline_fence_latency<<<1, threads>>>(d_cycles, d_sink);
  AMORA_CHECK(cudaDeviceSynchronize());

  // Fence loop.
  for (int r = 0; r < repeats; ++r) {
    amora_baseline_fence_latency<<<1, threads>>>(d_cycles, d_sink);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }
  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  double cycles_per_fence = (double)samples[repeats / 2] / (double)AMORA_FENCES;

  // Empty-loop baseline.
  for (int r = 0; r < repeats; ++r) {
    amora_baseline_fence_empty<<<1, threads>>>(d_cycles, d_sink);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }
  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  double cycles_per_empty = (double)samples[repeats / 2] / (double)AMORA_FENCES;

  double net_cycles_per_fence = cycles_per_fence - cycles_per_empty;

  std::printf(
      "{\"device_name\":\"%s\",\"fences\":%d,\"cycles_per_fence\":%.4f,"
      "\"cycles_per_empty\":%.4f,\"net_cycles_per_fence\":%.4f}\n",
      prop.name, AMORA_FENCES, cycles_per_fence, cycles_per_empty,
      net_cycles_per_fence);

  std::free(samples);
  cudaFree(d_cycles);
  cudaFree(d_sink);
  return 0;
}
