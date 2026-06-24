// Tensor-core MMA dependent-latency probe (FP16 16x16x16 via WMMA).
//
// One warp runs a dependent chain of wmma::mma_sync ops where each accumulator
// feeds the next, bracketed by clock64(). Cycles-per-MMA is the median across
// launches. The WMMA API lowers to HMMA on Volta+; SASS validation confirms the
// HMMA opcode is present. FP16 16x16x16 is supported on every tensor-core arch.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <mma.h>

using namespace nvcuda;

#define AMORA_CHECK(call)                                                      \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(_e));  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#ifndef AMORA_MMA_CHAIN
#define AMORA_MMA_CHAIN 512
#endif

extern "C" __global__ void
amora_tc_mma_latency(const __half *a, const __half *b, float *out,
                     unsigned long long *cycles_out) {
  if (blockIdx.x != 0 || threadIdx.x >= 32) return;
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::load_matrix_sync(fa, a, 16);
  wmma::load_matrix_sync(fb, b, 16);
  wmma::fill_fragment(acc, 0.0f);

  unsigned long long t0 = clock64();
  for (int i = 0; i < AMORA_MMA_CHAIN; ++i) {
    // Dependent: feed the accumulator back as matrix A each iteration by
    // keeping acc live across mma_sync calls (true RAW on the C fragment).
    wmma::mma_sync(acc, fa, fb, acc);
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0) cycles_out[0] = t1 - t0;
  wmma::store_matrix_sync(out, acc, 16, wmma::mem_row_major);
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

  const int n = 16 * 16;
  __half *d_a = nullptr, *d_b = nullptr;
  float *d_out = nullptr;
  unsigned long long *d_cycles = nullptr;
  AMORA_CHECK(cudaMalloc(&d_a, n * sizeof(__half)));
  AMORA_CHECK(cudaMalloc(&d_b, n * sizeof(__half)));
  AMORA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMemset(d_a, 0, n * sizeof(__half)));
  AMORA_CHECK(cudaMemset(d_b, 0, n * sizeof(__half)));

  amora_tc_mma_latency<<<1, 32>>>(d_a, d_b, d_out, d_cycles);
  AMORA_CHECK(cudaDeviceSynchronize());

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);
  for (int r = 0; r < repeats; ++r) {
    amora_tc_mma_latency<<<1, 32>>>(d_a, d_b, d_out, d_cycles);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }
  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  unsigned long long median = samples[repeats / 2];
  double cycles_per_mma = (double)median / (double)AMORA_MMA_CHAIN;

  std::printf(
      "{\"device_name\":\"%s\",\"mma_shape\":\"m16n16k16_fp16\","
      "\"chain\":%d,\"repeats\":%d,\"cycles_median\":%llu,"
      "\"cycles_per_mma\":%.4f}\n",
      prop.name, AMORA_MMA_CHAIN, repeats, (unsigned long long)median, cycles_per_mma);

  std::free(samples);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); cudaFree(d_cycles);
  return 0;
}
