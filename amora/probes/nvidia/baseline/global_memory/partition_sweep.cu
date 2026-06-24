// DRAM partition-camping sweep probe.
//
// Streams over a large DRAM buffer using a configurable base byte-offset so the
// grid's accesses start at different points relative to the memory-partition
// (channel) interleave. If achieved bandwidth varies strongly across offsets the
// device is sensitive to partition camping; a flat curve indicates balanced
// partition distribution. Bandwidth is timed with CUDA events per offset.

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

// Grid-stride read starting from `base` (in vec_t elements). The base offset
// shifts where each block's accesses land in the partition interleave.
extern "C" __global__ void
amora_gmem_partition(const vec_t *in, float *partial, size_t n, size_t base) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (size_t idx = i + base; idx < n; idx += stride) {
    vec_t v = in[idx];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == -1.f) partial[0] = acc;  // prevent DCE
}

static float time_ms(const vec_t *in, float *p, size_t n, size_t base,
                     int grid, int block) {
  cudaEvent_t a, b;
  AMORA_CHECK(cudaEventCreate(&a));
  AMORA_CHECK(cudaEventCreate(&b));
  AMORA_CHECK(cudaEventRecord(a));
  amora_gmem_partition<<<grid, block>>>(in, p, n, base);
  AMORA_CHECK(cudaEventRecord(b));
  AMORA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  AMORA_CHECK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms;
}

int main(int argc, char **argv) {
  int mb = 512;     // working set, far larger than cache
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

  int block = 256;
  int grid = prop.multiProcessorCount * 32;

  // Base offsets in KiB; chosen to straddle typical partition-stride sizes.
  const int offsets_kb[] = {0, 256, 512, 768, 1024, 1536};
  const int n_off = (int)(sizeof(offsets_kb) / sizeof(offsets_kb[0]));

  std::printf("{\"device_name\":\"%s\",\"working_set_mb\":%d,\"sweep\":[",
              prop.name, mb);

  for (int k = 0; k < n_off; ++k) {
    size_t base = (size_t)offsets_kb[k] * 1024 / sizeof(vec_t);
    if (base >= n) base = 0;
    size_t touched = (n > base) ? (n - base) : n;
    double moved = (double)touched * sizeof(vec_t);

    // Warm-up.
    amora_gmem_partition<<<grid, block>>>(d_in, d_p, n, base);
    AMORA_CHECK(cudaDeviceSynchronize());

    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
      float ms = time_ms(d_in, d_p, n, base, grid, block);
      if (ms < best) best = ms;
    }
    double gbps = moved / (best / 1000.0) / 1e9;

    std::printf("%s{\"offset_kb\":%d,\"gbps\":%.2f}", (k == 0 ? "" : ","),
                offsets_kb[k], gbps);
  }

  std::printf("]}\n");

  cudaFree(d_in);
  cudaFree(d_p);
  return 0;
}
