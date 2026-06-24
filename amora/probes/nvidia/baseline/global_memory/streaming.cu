// DRAM/HBM streaming bandwidth probe.
//
// Runs grid-stride read, write, and copy kernels over a working set far larger
// than cache, timed with CUDA events. Reports achieved sustained GB/s per
// traffic class. Timing-first; NCU DRAM byte counters can corroborate later.

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

extern "C" __global__ void amora_stream_read(const vec_t *in, float *partial, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  for (; i < n; i += stride) {
    vec_t v = in[i];
    acc += v.x + v.y + v.z + v.w;
  }
  if (acc == -1.f) partial[0] = acc;  // prevent DCE
}

extern "C" __global__ void amora_stream_write(vec_t *out, size_t n, float seed) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  vec_t v = make_float4(seed, seed, seed, seed);
  for (; i < n; i += stride) out[i] = v;
}

extern "C" __global__ void amora_stream_copy(const vec_t *in, vec_t *out, size_t n) {
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x * blockDim.x;
  for (; i < n; i += stride) out[i] = in[i];
}

static float time_ms(void (*launch)(const vec_t *, vec_t *, float *, size_t, int, int),
                     const vec_t *in, vec_t *out, float *partial, size_t n,
                     int grid, int block) {
  cudaEvent_t a, b;
  AMORA_CHECK(cudaEventCreate(&a));
  AMORA_CHECK(cudaEventCreate(&b));
  AMORA_CHECK(cudaEventRecord(a));
  launch(in, out, partial, n, grid, block);
  AMORA_CHECK(cudaEventRecord(b));
  AMORA_CHECK(cudaEventSynchronize(b));
  float ms = 0.f;
  AMORA_CHECK(cudaEventElapsedTime(&ms, a, b));
  cudaEventDestroy(a);
  cudaEventDestroy(b);
  return ms;
}

static void launch_read(const vec_t *in, vec_t *out, float *p, size_t n, int g, int b) {
  (void)out; amora_stream_read<<<g, b>>>(in, p, n);
}
static void launch_write(const vec_t *in, vec_t *out, float *p, size_t n, int g, int b) {
  (void)in; (void)p; amora_stream_write<<<g, b>>>(out, n, 1.0f);
}
static void launch_copy(const vec_t *in, vec_t *out, float *p, size_t n, int g, int b) {
  (void)p; amora_stream_copy<<<g, b>>>(in, out, n);
}

static double best_gbps(const vec_t *in, vec_t *out, float *p, size_t n,
                        int grid, int block, double bytes, int iters,
                        void (*launch)(const vec_t *, vec_t *, float *, size_t, int, int)) {
  // Warm-up.
  launch(in, out, p, n, grid, block);
  AMORA_CHECK(cudaDeviceSynchronize());
  double best = 1e30;
  for (int it = 0; it < iters; ++it) {
    float ms = time_ms(launch, in, out, p, n, grid, block);
    if (ms < best) best = ms;
  }
  return bytes / (best / 1000.0) / 1e9;  // GB/s
}

int main(int argc, char **argv) {
  int mb = 512;     // working set per buffer
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

  vec_t *d_in = nullptr, *d_out = nullptr;
  float *d_p = nullptr;
  AMORA_CHECK(cudaMalloc(&d_in, bytes));
  AMORA_CHECK(cudaMalloc(&d_out, bytes));
  AMORA_CHECK(cudaMalloc(&d_p, sizeof(float)));
  AMORA_CHECK(cudaMemset(d_in, 1, bytes));

  int block = 256;
  int grid = prop.multiProcessorCount * 32;

  double read_gbps = best_gbps(d_in, d_out, d_p, n, grid, block, (double)bytes, iters, launch_read);
  double write_gbps = best_gbps(d_in, d_out, d_p, n, grid, block, (double)bytes, iters, launch_write);
  // copy moves 2x bytes (read+write).
  double copy_gbps = best_gbps(d_in, d_out, d_p, n, grid, block, 2.0 * (double)bytes, iters, launch_copy);

  std::printf(
      "{\"device_name\":\"%s\",\"working_set_mb\":%d,\"iters\":%d,"
      "\"read_gbps\":%.2f,\"write_gbps\":%.2f,\"copy_gbps\":%.2f}\n",
      prop.name, mb, iters, read_gbps, write_gbps, copy_gbps);

  cudaFree(d_in);
  cudaFree(d_out);
  cudaFree(d_p);
  return 0;
}
