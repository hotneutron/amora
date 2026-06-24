// Async-copy (cp.async) transfer-size sweep probe.
//
// One CTA bulk-stages global memory into shared memory using the Ampere+
// async-copy pipeline (cuda_pipeline.h, lowers to LDGSTS on sm_80+), sweeping
// the shared-tile size (1KB..32KB). Per size we measure achieved GB/s with CUDA
// events over many iterations and report the curve. Timing-first.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#define AMORA_CHECK(call)                                                      \
  do {                                                                         \
    cudaError_t _e = (call);                                                   \
    if (_e != cudaSuccess) {                                                   \
      std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(_e));  \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

extern "C" __global__ void
amora_tma_transfer_sweep(const char *src, int tiles, int bytes_per_tile,
                         int *sink) {
  extern __shared__ char smem[];
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  const int chunk = 16;
  int chunks_per_tile = bytes_per_tile / chunk;

  int acc = 0;
  for (int t = 0; t < tiles; ++t) {
    const char *tile_src = src + (size_t)t * bytes_per_tile;
    for (int c = tid; c < chunks_per_tile; c += nthreads) {
      __pipeline_memcpy_async(smem + (size_t)c * chunk,
                              tile_src + (size_t)c * chunk, chunk);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();
    for (int c = tid; c < chunks_per_tile; c += nthreads) {
      acc += (int)smem[(size_t)c * chunk];
    }
    __syncthreads();
  }
  if (acc == 0x7fffffff) sink[0] = acc;  // prevent DCE
}

int main(int argc, char **argv) {
  int iters = 10;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
  }
  if (iters <= 0) iters = 10;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  const int tile_kb[] = {1, 4, 16, 32};
  const int n_sizes = (int)(sizeof(tile_kb) / sizeof(tile_kb[0]));

  // Move ~64MB of traffic per measurement regardless of tile size.
  const size_t target_bytes = (size_t)64 * 1024 * 1024;
  const int threads = 256;

  // Largest tile bounds the shared and per-tile global allocation.
  int max_kb = tile_kb[n_sizes - 1];
  size_t max_tile = (size_t)max_kb * 1024;

  char *d_src = nullptr;
  int *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_src, max_tile));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(int)));
  AMORA_CHECK(cudaMemset(d_src, 1, max_tile));

  cudaEvent_t a, b;
  AMORA_CHECK(cudaEventCreate(&a));
  AMORA_CHECK(cudaEventCreate(&b));

  std::printf("{\"device_name\":\"%s\",\"sweep\":[", prop.name);

  for (int s = 0; s < n_sizes; ++s) {
    int kb = tile_kb[s];
    int bytes_per_tile = kb * 1024;
    int tiles = (int)(target_bytes / (size_t)bytes_per_tile);
    if (tiles < 1) tiles = 1;
    size_t shmem = (size_t)bytes_per_tile;

    // Warm-up.
    amora_tma_transfer_sweep<<<1, threads, shmem>>>(d_src, tiles, bytes_per_tile, d_sink);
    AMORA_CHECK(cudaDeviceSynchronize());

    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
      AMORA_CHECK(cudaEventRecord(a));
      amora_tma_transfer_sweep<<<1, threads, shmem>>>(d_src, tiles, bytes_per_tile, d_sink);
      AMORA_CHECK(cudaEventRecord(b));
      AMORA_CHECK(cudaEventSynchronize(b));
      float ms = 0.f;
      AMORA_CHECK(cudaEventElapsedTime(&ms, a, b));
      if (ms < best) best = ms;
    }
    double bytes = (double)tiles * (double)bytes_per_tile;
    double gbps = bytes / (best / 1000.0) / 1e9;
    std::printf("%s{\"tile_kb\":%d,\"gbps\":%.2f}", (s == 0 ? "" : ","), kb, gbps);
  }

  std::printf("]}\n");

  cudaEventDestroy(a);
  cudaEventDestroy(b);
  cudaFree(d_src);
  cudaFree(d_sink);
  return 0;
}
