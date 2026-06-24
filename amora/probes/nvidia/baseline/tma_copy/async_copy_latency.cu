// Async-copy (cp.async) tile-latency probe.
//
// One CTA stages tiles from global to shared memory using the Ampere+ async-copy
// pipeline (__pipeline_memcpy_async + __pipeline_commit + __pipeline_wait_prior,
// from cuda_pipeline.h, which lowers to LDGSTS on sm_80+). We bracket the
// issue->wait->use sequence per tile with clock64() and report the median
// cycles-per-tile. Timing-first; no counters required.

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

#ifndef AMORA_TILE_BYTES
#define AMORA_TILE_BYTES 4096
#endif

extern "C" __global__ void
amora_tma_async_copy_latency(const char *src, int tiles, int bytes_per_tile,
                             unsigned long long *cycles_out, int *sink) {
  extern __shared__ char smem[];
  int tid = threadIdx.x;
  int nthreads = blockDim.x;
  // Each thread copies a 16B (float4-aligned) chunk per async transfer.
  const int chunk = 16;
  int chunks_per_tile = bytes_per_tile / chunk;

  __syncthreads();
  unsigned long long t0 = clock64();
  for (int t = 0; t < tiles; ++t) {
    const char *tile_src = src + (size_t)t * bytes_per_tile;
    // Issue: each thread stages its strided chunks into shared memory.
    for (int c = tid; c < chunks_per_tile; c += nthreads) {
      __pipeline_memcpy_async(smem + (size_t)c * chunk,
                              tile_src + (size_t)c * chunk, chunk);
    }
    __pipeline_commit();
    // Wait: drain the just-committed group.
    __pipeline_wait_prior(0);
    __syncthreads();
    // Use: touch the staged data so the copy is not eliminated.
    int acc = 0;
    for (int c = tid; c < chunks_per_tile; c += nthreads) {
      acc += (int)smem[(size_t)c * chunk];
    }
    if (acc == 0x7fffffff) sink[0] = acc;  // prevent DCE
    __syncthreads();
  }
  unsigned long long t1 = clock64();
  if (tid == 0) cycles_out[0] = t1 - t0;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

int main(int argc, char **argv) {
  int repeats = 32;
  int tiles = 64;
  int bytes_per_tile = AMORA_TILE_BYTES;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--tiles") == 0) tiles = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--bytes") == 0) bytes_per_tile = std::atoi(argv[i + 1]);
  }
  if (repeats <= 0) repeats = 32;
  if (tiles <= 0) tiles = 64;
  if (bytes_per_tile <= 0) bytes_per_tile = AMORA_TILE_BYTES;
  // Align tile to 16B chunk granularity.
  bytes_per_tile = (bytes_per_tile / 16) * 16;
  if (bytes_per_tile <= 0) bytes_per_tile = 16;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  // Global buffer holds all tiles and is larger than the shared tile.
  size_t total = (size_t)tiles * bytes_per_tile;
  char *d_src = nullptr;
  unsigned long long *d_cycles = nullptr;
  int *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_src, total));
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(int)));
  AMORA_CHECK(cudaMemset(d_src, 1, total));

  int threads = 128;
  size_t shmem = (size_t)bytes_per_tile;

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);

  // Warm-up.
  amora_tma_async_copy_latency<<<1, threads, shmem>>>(d_src, tiles, bytes_per_tile,
                                                      d_cycles, d_sink);
  AMORA_CHECK(cudaDeviceSynchronize());

  for (int r = 0; r < repeats; ++r) {
    amora_tma_async_copy_latency<<<1, threads, shmem>>>(d_src, tiles, bytes_per_tile,
                                                        d_cycles, d_sink);
    AMORA_CHECK(cudaDeviceSynchronize());
    AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                            cudaMemcpyDeviceToHost));
  }
  std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
  double cpt = (double)samples[repeats / 2] / (double)tiles;

  std::printf(
      "{\"device_name\":\"%s\",\"tiles\":%d,\"bytes_per_tile\":%d,"
      "\"cycles_per_tile\":%.4f}\n",
      prop.name, tiles, bytes_per_tile, cpt);

  std::free(samples);
  cudaFree(d_src);
  cudaFree(d_cycles);
  cudaFree(d_sink);
  return 0;
}
