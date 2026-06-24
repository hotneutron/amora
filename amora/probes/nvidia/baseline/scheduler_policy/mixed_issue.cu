// Scheduler mixed-issue / pipeline-overlap probe.
//
// Runs three independent-instruction microbenchmarks on one SM warp set:
//   1. FP32-only FMA stream,
//   2. INT-only MAD stream,
//   3. a mixed stream interleaving independent FP32 and INT ops.
// If the mixed stream's ops/cycle approaches the *sum* of the two single-pipe
// rates, the SM overlaps the FP32 and INT pipes (dual-issue-like). If it
// approaches the max of the two, it does not. We report all three rates so the
// analyzer can classify the overlap behaviorally.

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
#ifndef AMORA_LANES
#define AMORA_LANES 4   // independent chains per thread to expose ILP
#endif

extern "C" __global__ void
amora_mix_fp32(float seed, unsigned long long *cyc, float *sink) {
  float a = seed, b = seed + 1.f, c = seed + 2.f, d = seed + 3.f;
  unsigned long long t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
    a = fmaf(a, 1.000001f, 0.000001f);
    b = fmaf(b, 1.000001f, 0.000001f);
    c = fmaf(c, 1.000001f, 0.000001f);
    d = fmaf(d, 1.000001f, 0.000001f);
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
  sink[threadIdx.x % 32] = a + b + c + d;
}

extern "C" __global__ void
amora_mix_int(int seed, unsigned long long *cyc, int *sink) {
  int a = seed, b = seed + 1, c = seed + 2, d = seed + 3;
  unsigned long long t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
    a = a * 1664525 + 1013904223;
    b = b * 1664525 + 1013904223;
    c = c * 1664525 + 1013904223;
    d = d * 1664525 + 1013904223;
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
  sink[threadIdx.x % 32] = a + b + c + d;
}

extern "C" __global__ void
amora_mix_both(float fseed, int iseed, unsigned long long *cyc,
               float *fsink, int *isink) {
  float a = fseed, b = fseed + 1.f;
  int x = iseed, y = iseed + 1;
  unsigned long long t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
    a = fmaf(a, 1.000001f, 0.000001f);
    x = x * 1664525 + 1013904223;
    b = fmaf(b, 1.000001f, 0.000001f);
    y = y * 1664525 + 1013904223;
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
  fsink[threadIdx.x % 32] = a + b;
  isink[threadIdx.x % 32] = x + y;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

template <typename Launch>
static double median_ops(Launch launch, int repeats, double total_ops,
                         unsigned long long *d_cyc) {
  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);
  for (int r = 0; r < repeats; ++r) {
    launch(r);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cyc, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }
  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  unsigned long long median = samples[repeats / 2];
  std::free(samples);
  return total_ops / (double)median;
}

int main(int argc, char **argv) {
  int repeats = 32;
  int warps = 8;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--warps") == 0) warps = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 32;
  if (warps <= 0 || warps > 32) warps = 8;
  int threads = 32 * warps;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  unsigned long long *d_cyc = nullptr;
  float *d_fsink = nullptr;
  int *d_isink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cyc, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_fsink, sizeof(float) * 32));
  AMORA_CHECK(cudaMalloc(&d_isink, sizeof(int) * 32));

  double ops_per_thread = 4.0 * (double)AMORA_CHAIN_LEN;  // 4 ops/iter
  double total = (double)threads * ops_per_thread;

  // Warm-up.
  amora_mix_fp32<<<1, threads>>>(1.f, d_cyc, d_fsink);
  AMORA_CHECK(cudaDeviceSynchronize());

  double fp32 = median_ops([&](int r){ amora_mix_fp32<<<1, threads>>>(1.f + r, d_cyc, d_fsink); },
                           repeats, total, d_cyc);
  double i32 = median_ops([&](int r){ amora_mix_int<<<1, threads>>>(1 + r, d_cyc, d_isink); },
                          repeats, total, d_cyc);
  double both = median_ops([&](int r){ amora_mix_both<<<1, threads>>>(1.f + r, 1 + r, d_cyc, d_fsink, d_isink); },
                           repeats, total, d_cyc);

  std::printf(
      "{\"device_name\":\"%s\",\"warps\":%d,\"chain_length\":%d,"
      "\"fp32_ops_per_cycle\":%.4f,\"int_ops_per_cycle\":%.4f,"
      "\"mixed_ops_per_cycle\":%.4f}\n",
      prop.name, warps, AMORA_CHAIN_LEN, fp32, i32, both);

  cudaFree(d_cyc);
  cudaFree(d_fsink);
  cudaFree(d_isink);
  return 0;
}
