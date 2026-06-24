// DRAM row-locality (row-buffer policy) sweep probe.
//
// Streams a large DRAM buffer with varied element strides to change how much
// DRAM row-buffer locality the access pattern exposes. A stride of 1 maximizes
// row hits; large strides scatter accesses across rows, exposing row-activation
// overhead. Bandwidth is timed with CUDA events per stride; the spread between
// best and worst bandwidth indicates the device's sensitivity to row locality.

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

// Each thread walks the buffer with a fixed element `stride`, wrapping modulo n,
// so consecutive accesses by a warp vary their DRAM-row locality with stride.
extern "C" __global__ void
amora_gmem_row_policy(const float *in, float *partial, size_t n, int stride,
                      int steps) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t threads = (size_t)gridDim.x * blockDim.x;
  float acc = 0.f;
  size_t idx = tid;
  for (int s = 0; s < steps; ++s) {
    acc += in[idx % n];
    idx += (size_t)stride * threads;
  }
  if (acc == -1.f) partial[0] = acc;  // prevent DCE
}

static float time_ms(const float *in, float *p, size_t n, int stride, int steps,
                     int grid, int block) {
  cudaEvent_t a, b;
  AMORA_CHECK(cudaEventCreate(&a));
  AMORA_CHECK(cudaEventCreate(&b));
  AMORA_CHECK(cudaEventRecord(a));
  amora_gmem_row_policy<<<grid, block>>>(in, p, n, stride, steps);
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
  size_t n = bytes / sizeof(float);

  float *d_in = nullptr;
  float *d_p = nullptr;
  AMORA_CHECK(cudaMalloc(&d_in, bytes));
  AMORA_CHECK(cudaMalloc(&d_p, sizeof(float)));
  AMORA_CHECK(cudaMemset(d_in, 1, bytes));

  int block = 256;
  int grid = prop.multiProcessorCount * 32;
  size_t threads = (size_t)grid * block;
  int steps = (int)(n / threads);
  if (steps < 1) steps = 1;
  double moved = (double)threads * (double)steps * sizeof(float);

  const int strides[] = {1, 8, 64, 512};
  const int n_str = (int)(sizeof(strides) / sizeof(strides[0]));

  std::printf("{\"device_name\":\"%s\",\"working_set_mb\":%d,\"sweep\":[",
              prop.name, mb);

  for (int k = 0; k < n_str; ++k) {
    int stride = strides[k];

    // Warm-up.
    amora_gmem_row_policy<<<grid, block>>>(d_in, d_p, n, stride, steps);
    AMORA_CHECK(cudaDeviceSynchronize());

    double best = 1e30;
    for (int it = 0; it < iters; ++it) {
      float ms = time_ms(d_in, d_p, n, stride, steps, grid, block);
      if (ms < best) best = ms;
    }
    double gbps = moved / (best / 1000.0) / 1e9;

    std::printf("%s{\"stride\":%d,\"gbps\":%.2f}", (k == 0 ? "" : ","),
                stride, gbps);
  }

  std::printf("]}\n");

  cudaFree(d_in);
  cudaFree(d_p);
  return 0;
}
