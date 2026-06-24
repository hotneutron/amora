// Differential register read-after-write / operand-delivery latency probe.
//
// Two dependent FMA chains of equal length:
//   1. "same" : a single accumulator (tight RAW dependency every op),
//   2. "rotating" : a small ring of accumulators advanced round-robin so each
//      op's source was written a few ops earlier (relaxed RAW distance).
// Both execute the same opcode and op count; the difference in cycles-per-op
// isolates the differential operand-delivery / scoreboard cost, separate from
// the absolute arithmetic latency measured by the baseline probe.

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
#define AMORA_CHAIN_LEN 4096
#endif
#ifndef AMORA_ROT
#define AMORA_ROT 8   // rotating ring depth (RAW distance)
#endif

extern "C" __global__ void
amora_reg_same(float seed, unsigned long long *cyc, float *sink) {
  if (threadIdx.x != 0) return;
  float x = seed;
  unsigned long long t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
    x = fmaf(x, 1.000001f, 0.000001f);  // RAW distance 1
  }
  unsigned long long t1 = clock64();
  cyc[0] = t1 - t0;
  sink[0] = x;
}

extern "C" __global__ void
amora_reg_rot(float seed, unsigned long long *cyc, float *sink) {
  if (threadIdx.x != 0) return;
  float r[AMORA_ROT];
#pragma unroll
  for (int k = 0; k < AMORA_ROT; ++k) r[k] = seed + (float)k;
  unsigned long long t0 = clock64();
#pragma unroll 16
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
    int k = i % AMORA_ROT;
    r[k] = fmaf(r[k], 1.000001f, 0.000001f);  // RAW distance AMORA_ROT
  }
  unsigned long long t1 = clock64();
  cyc[0] = t1 - t0;
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < AMORA_ROT; ++k) s += r[k];
  sink[0] = s;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

template <typename Launch>
static double median_cpo(Launch launch, int repeats, unsigned long long *d_cyc) {
  unsigned long long *s =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);
  for (int r = 0; r < repeats; ++r) {
    launch(r);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&s[r], d_cyc, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
  }
  std::qsort(s, repeats, sizeof(unsigned long long), cmp_ull);
  unsigned long long median = s[repeats / 2];
  std::free(s);
  return (double)median / (double)AMORA_CHAIN_LEN;
}

int main(int argc, char **argv) {
  int repeats = 64;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 64;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  unsigned long long *d_cyc = nullptr;
  float *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cyc, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(float)));

  amora_reg_same<<<1, 32>>>(1.f, d_cyc, d_sink);
  AMORA_CHECK(cudaDeviceSynchronize());

  double same = median_cpo([&](int r){ amora_reg_same<<<1, 32>>>(1.f + r, d_cyc, d_sink); },
                           repeats, d_cyc);
  double rot = median_cpo([&](int r){ amora_reg_rot<<<1, 32>>>(1.f + r, d_cyc, d_sink); },
                          repeats, d_cyc);

  std::printf(
      "{\"device_name\":\"%s\",\"chain_length\":%d,\"rot_depth\":%d,"
      "\"same_reg_cycles_per_op\":%.4f,\"rotating_reg_cycles_per_op\":%.4f,"
      "\"differential_cycles_per_op\":%.4f}\n",
      prop.name, AMORA_CHAIN_LEN, AMORA_ROT, same, rot, same - rot);

  cudaFree(d_cyc);
  cudaFree(d_sink);
  return 0;
}
