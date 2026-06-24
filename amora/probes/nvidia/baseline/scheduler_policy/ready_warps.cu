// Scheduler ready-warp issue-scaling probe.
//
// One CTA on one SM runs a varying number of warps, each executing an
// independent dependent FMA chain (no cross-warp dependency, no memory). With
// few ready warps the SM cannot hide arithmetic latency, so per-warp work
// finishes slowly; as ready warps increase, issue slots fill and aggregate
// throughput scales until the scheduler/pipe saturates. We report aggregate
// instructions-per-cycle vs warp count so the analyzer can find the saturation
// knee (an issue-capacity classification, not a named policy).

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
#define AMORA_CHAIN_LEN 2048
#endif

// Each warp does AMORA_CHAIN_LEN dependent FMAs. Total ops = active_warps*32*len.
extern "C" __global__ void
amora_sched_ready_warps(int active_warps, float seed,
                        unsigned long long *cycles_out, float *sink) {
  int warp = threadIdx.x >> 5;
  if (warp >= active_warps) return;
  float x = seed + (float)threadIdx.x;
  unsigned long long t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
    x = fmaf(x, 1.000001f, 0.000001f);
  }
  unsigned long long t1 = clock64();
  // Thread 0 of the block records elapsed cycles for the whole block.
  if (threadIdx.x == 0) cycles_out[0] = t1 - t0;
  sink[threadIdx.x % 32] = x;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

int main(int argc, char **argv) {
  int repeats = 32;
  int max_warps = 32;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--max-warps") == 0) max_warps = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 32;
  if (max_warps <= 0 || max_warps > 32) max_warps = 32;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  unsigned long long *d_cycles = nullptr;
  float *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(float) * 32));

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);

  // Warm-up.
  amora_sched_ready_warps<<<1, 32 * max_warps>>>(1, 1.0f, d_cycles, d_sink);
  AMORA_CHECK(cudaDeviceSynchronize());

  std::printf("{\"device_name\":\"%s\",\"chain_length\":%d,\"sweep\":[",
              prop.name, AMORA_CHAIN_LEN);

  for (int w = 1; w <= max_warps; ++w) {
    int threads = 32 * max_warps;  // launch full block; gate by active_warps
    for (int r = 0; r < repeats; ++r) {
      amora_sched_ready_warps<<<1, threads>>>(w, 1.0f + (float)r, d_cycles, d_sink);
      AMORA_CHECK(cudaDeviceSynchronize());
      AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }
    std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
    unsigned long long median = samples[repeats / 2];
    double total_ops = (double)w * 32.0 * (double)AMORA_CHAIN_LEN;
    double ipc = total_ops / (double)median;  // FMA-ops issued per cycle
    std::printf("%s{\"warps\":%d,\"cycles_median\":%llu,\"ops_per_cycle\":%.4f}",
                (w == 1 ? "" : ","), w, (unsigned long long)median, ipc);
  }

  std::printf("]}\n");

  std::free(samples);
  cudaFree(d_cycles);
  cudaFree(d_sink);
  return 0;
}
