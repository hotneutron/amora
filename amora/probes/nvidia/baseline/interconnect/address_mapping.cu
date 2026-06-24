// Interconnect address-mapping probe.
//
// Streams a fixed amount of global memory while sweeping the base-address byte
// stride across large power-of-two strides. Throughput variation across strides
// reveals partition/slice periodicity (channel camping): a uniform mapping keeps
// GB/s flat, while a periodic mapping produces strong dips at strides aligned to
// the interleave. Timing-first with CUDA events.

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

typedef float4 vec_t;

// Each thread reads "count" elements separated by "vstride" vec_t elements,
// starting from a per-thread base. Large strides spread accesses across
// partitions; aligned strides can camp on a subset.
extern "C" __global__ void
amora_icn_address_mapping(const vec_t *in, float *partial, size_t n,
                          size_t vstride, int count) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t threads = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  size_t base = tid;
  for (int c = 0; c < count; ++c) {
    size_t idx = (base + (size_t)c * vstride * threads) % n;
    vec_t v = in[idx];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == -1.f) partial[0] = acc;  // prevent DCE
}

int main(int argc, char **argv) {
  int mb = 512;
  int iters = 5;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--mb") == 0) mb = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
  }
  if (mb <= 0) mb = 512;
  if (iters <= 0) iters = 5;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  size_t bytes = (size_t)mb * 1024 * 1024;
  size_t n = bytes / sizeof(vec_t);

  vec_t *d_in = nullptr;
  float *d_p = nullptr;
  AMORA_CHECK(cudaMalloc(&d_in, bytes));
  AMORA_CHECK(cudaMalloc(&d_p, sizeof(float)));
  AMORA_CHECK(cudaMemset(d_in, 1, bytes));

  const int block = 256;
  const int grid = prop.multiProcessorCount * 32;
  // Stride in KB of the per-step base displacement (power-of-two sweep).
  const int stride_kb[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  const int n_strides = (int)(sizeof(stride_kb) / sizeof(stride_kb[0]));
  const int count = 256;  // accesses per thread

  cudaEvent_t a, b;
  AMORA_CHECK(cudaEventCreate(&a));
  AMORA_CHECK(cudaEventCreate(&b));

  size_t total_threads = (size_t)grid * block;
  double bytes_moved = (double)total_threads * (double)count * sizeof(vec_t);

  // Warm-up.
  amora_icn_address_mapping<<<grid, block>>>(d_in, d_p, n, 1, count);
  AMORA_CHECK(cudaDeviceSynchronize());

  std::printf("{\"device_name\":\"%s\",\"sweep\":[", prop.name);

  for (int s = 0; s < n_strides; ++s) {
    int kb = stride_kb[s];
    size_t vstride = ((size_t)kb * 1024) / sizeof(vec_t);
    if (vstride == 0) vstride = 1;
    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
      AMORA_CHECK(cudaEventRecord(a));
      amora_icn_address_mapping<<<grid, block>>>(d_in, d_p, n, vstride, count);
      AMORA_CHECK(cudaEventRecord(b));
      AMORA_CHECK(cudaEventSynchronize(b));
      float ms = 0.f;
      AMORA_CHECK(cudaEventElapsedTime(&ms, a, b));
      if (ms < best) best = ms;
    }
    double gbps = bytes_moved / (best / 1000.0) / 1e9;
    std::printf("%s{\"stride_kb\":%d,\"gbps\":%.2f}", (s == 0 ? "" : ","), kb, gbps);
  }

  std::printf("]}\n");

  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaFree(d_in);
  cudaFree(d_p);
  return 0;
}
