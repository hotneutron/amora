// Tensor-core MMA throughput probe (FP16 16x16x16 via WMMA).
//
// Many warps each run independent MMA accumulators to saturate the tensor pipe.
// Reports MMA-ops per cycle (aggregate) so the analyzer/NCU can fit initiation
// interval. Independent chains (4 accumulators per warp) expose ILP.

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

#ifndef AMORA_MMA_ITERS
#define AMORA_MMA_ITERS 256
#endif
#define AMORA_MMA_LANES 4

extern "C" __global__ void
amora_tc_mma_throughput(const __half *a, const __half *b, float *out,
                        unsigned long long *cycles_out) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[AMORA_MMA_LANES];
  wmma::load_matrix_sync(fa, a, 16);
  wmma::load_matrix_sync(fb, b, 16);
#pragma unroll
  for (int k = 0; k < AMORA_MMA_LANES; ++k) wmma::fill_fragment(acc[k], 0.0f);

  unsigned long long t0 = clock64();
  for (int i = 0; i < AMORA_MMA_ITERS; ++i) {
#pragma unroll
    for (int k = 0; k < AMORA_MMA_LANES; ++k) wmma::mma_sync(acc[k], fa, fb, acc[k]);
  }
  unsigned long long t1 = clock64();
  if (threadIdx.x == 0 && blockIdx.x == 0) cycles_out[0] = t1 - t0;
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < AMORA_MMA_LANES; ++k) s += acc[k].x[0];
  if (s == -1.f) out[0] = s;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

int main(int argc, char **argv) {
  int repeats = 32;
  int warps = 4;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--warps") == 0) warps = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 32;
  if (warps <= 0 || warps > 32) warps = 4;
  int threads = 32 * warps;

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

  amora_tc_mma_throughput<<<1, threads>>>(d_a, d_b, d_out, d_cycles);
  AMORA_CHECK(cudaDeviceSynchronize());

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);
  for (int r = 0; r < repeats; ++r) {
    amora_tc_mma_throughput<<<1, threads>>>(d_a, d_b, d_out, d_cycles);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }
  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  unsigned long long median = samples[repeats / 2];
  // Total MMA ops by one warp = iters * lanes (per-warp throughput).
  double total_mma = (double)AMORA_MMA_ITERS * (double)AMORA_MMA_LANES;
  double mma_per_cycle = total_mma / (double)median;

  std::printf(
      "{\"device_name\":\"%s\",\"mma_shape\":\"m16n16k16_fp16\",\"warps\":%d,"
      "\"iters\":%d,\"lanes\":%d,\"cycles_median\":%llu,"
      "\"mma_per_cycle_per_warp\":%.4f}\n",
      prop.name, warps, AMORA_MMA_ITERS, AMORA_MMA_LANES,
      (unsigned long long)median, mma_per_cycle);

  std::free(samples);
  cudaFree(d_a); cudaFree(d_b); cudaFree(d_out); cudaFree(d_cycles);
  return 0;
}
