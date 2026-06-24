// Register-bank / operand-delivery sweep probe.
//
// Methodology note: the P1 plan specifies a `.sass`-controlled register-number
// sweep. Stable SASS register assignment is not portable through nvcc across
// architectures, so this CUDA approximation instead sweeps the number of
// independent live accumulators (operand-collector / register-bank pressure).
// Each variant runs a fixed number of independent FMA chains; throughput per
// op is reported per width. Periodic dips across widths indicate operand-
// delivery / register-bank conflicts. Results are reported as a candidate
// curve, not an exact bank count (downgraded vs the SASS-controlled ideal).

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

// Templated number of independent accumulators (operand width).
template <int W>
__global__ void amora_reg_width(float seed, unsigned long long *cyc, float *sink) {
  float acc[W];
#pragma unroll
  for (int k = 0; k < W; ++k) acc[k] = seed + (float)k;
  unsigned long long t0 = clock64();
#pragma unroll 8
  for (int i = 0; i < AMORA_CHAIN_LEN; ++i) {
#pragma unroll
    for (int k = 0; k < W; ++k) acc[k] = fmaf(acc[k], 1.000001f, 0.000001f);
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0) cyc[0] = t1 - t0;
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < W; ++k) s += acc[k];
  sink[threadIdx.x % 32] = s;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

template <typename Launch>
static double median_per_op(Launch launch, int repeats, double ops, unsigned long long *d_cyc) {
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
  return (double)median / ops;  // cycles per op
}

int main(int argc, char **argv) {
  int repeats = 32;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 32;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  unsigned long long *d_cyc = nullptr;
  float *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_cyc, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(float) * 32));

  // Warm-up.
  amora_reg_width<1><<<1, 32>>>(1.f, d_cyc, d_sink);
  AMORA_CHECK(cudaDeviceSynchronize());

#define SWEEP(W)                                                                \
  do {                                                                          \
    double ops = (double)(W) * (double)AMORA_CHAIN_LEN;                         \
    double cpo = median_per_op(                                                 \
        [&](int r) { amora_reg_width<W><<<1, 32>>>(1.f + r, d_cyc, d_sink); },  \
        repeats, ops, d_cyc);                                                   \
    std::printf("%s{\"width\":%d,\"cycles_per_op\":%.4f}", first ? "" : ",", W, \
                cpo);                                                           \
    first = false;                                                             \
  } while (0)

  bool first = true;
  std::printf("{\"device_name\":\"%s\",\"chain_length\":%d,\"sweep\":[",
              prop.name, AMORA_CHAIN_LEN);
  SWEEP(1);
  SWEEP(2);
  SWEEP(3);
  SWEEP(4);
  SWEEP(6);
  SWEEP(8);
  SWEEP(12);
  SWEEP(16);
  std::printf("]}\n");
#undef SWEEP

  cudaFree(d_cyc);
  cudaFree(d_sink);
  return 0;
}
