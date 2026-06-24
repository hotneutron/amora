// Memory-level-parallelism (outstanding requests) probe.
//
// Each thread issues N INDEPENDENT global loads from a cache-exceeding DRAM
// buffer before consuming them, exposing how many outstanding memory requests
// the load/store pipeline can keep in flight. We sweep the number of in-flight
// loads per thread and report achieved bytes/cycle for each setting. The
// saturation knee in the curve is the effective MLP. A single wave of blocks is
// launched so throughput is limited by outstanding-request capacity, not by an
// excess of resident warps hiding latency.

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

// Each thread issues `in_flight` independent float4 loads, accumulates them, and
// only writes the sink on an impossible condition (prevents DCE while keeping
// the loads independent). A grid-stride keeps the whole buffer touched.
extern "C" __global__ void
amora_mem_outstanding(const vec_t *in, float *sink, size_t n, int in_flight,
                      unsigned long long *cycles_out) {
  size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t threads = (size_t)gridDim.x * blockDim.x;
  size_t stride = threads * (size_t)in_flight;

  unsigned long long t0 = clock64();
  float acc = 0.f;
  for (size_t base = tid * (size_t)in_flight; base < n; base += stride) {
    float regs[32];
#pragma unroll
    for (int k = 0; k < 32; ++k) {
      if (k < in_flight) {
        size_t off = base + (size_t)k;
        if (off < n) {
          vec_t v = in[off];
          regs[k] = v.x + v.y + v.z + v.w;
        } else {
          regs[k] = 0.f;
        }
      } else {
        regs[k] = 0.f;
      }
    }
#pragma unroll
    for (int k = 0; k < 32; ++k) acc += regs[k];
  }
  unsigned long long t1 = clock64();
  if (acc == -1.f) sink[tid] = acc;  // impossible: prevents DCE
  if (tid == 0) cycles_out[0] = t1 - t0;
}

static int cmp_ull(const void *a, const void *b) {
  unsigned long long ua = *(const unsigned long long *)a;
  unsigned long long ub = *(const unsigned long long *)b;
  return (ua > ub) - (ua < ub);
}

int main(int argc, char **argv) {
  int mb = 256;     // working set, far larger than any cache
  int repeats = 16;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--mb") == 0) mb = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--repeats") == 0) repeats = std::atoi(argv[i + 1]);
  }
  if (mb <= 0) mb = 256;
  if (repeats <= 0) repeats = 16;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  size_t bytes = (size_t)mb * 1024 * 1024;
  size_t n = bytes / sizeof(vec_t);

  vec_t *d_in = nullptr;
  float *d_sink = nullptr;
  unsigned long long *d_cycles = nullptr;
  AMORA_CHECK(cudaMalloc(&d_in, bytes));
  AMORA_CHECK(cudaMemset(d_in, 1, bytes));

  int block = 256;
  // One wave: a few blocks per SM keeps the grid small so throughput is bound by
  // outstanding-request capacity rather than by latency hiding across many warps.
  int grid = prop.multiProcessorCount * 2;
  size_t threads = (size_t)grid * block;
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(float) * threads));
  AMORA_CHECK(cudaMalloc(&d_cycles, sizeof(unsigned long long)));

  const int in_flight_points[] = {1, 2, 4, 8, 16, 32};
  const int n_points = (int)(sizeof(in_flight_points) / sizeof(in_flight_points[0]));

  unsigned long long *samples =
      (unsigned long long *)std::malloc(sizeof(unsigned long long) * repeats);

  std::printf("{\"device_name\":\"%s\",\"buffer_mb\":%d,\"sweep\":[", prop.name, mb);

  for (int k = 0; k < n_points; ++k) {
    int in_flight = in_flight_points[k];

    // Warm-up for this setting.
    amora_mem_outstanding<<<grid, block>>>(d_in, d_sink, n, in_flight, d_cycles);
    AMORA_CHECK(cudaDeviceSynchronize());

    for (int r = 0; r < repeats; ++r) {
      amora_mem_outstanding<<<grid, block>>>(d_in, d_sink, n, in_flight, d_cycles);
      AMORA_CHECK(cudaDeviceSynchronize());
      AMORA_CHECK(cudaMemcpy(&samples[r], d_cycles, sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
    }
    std::qsort(samples, repeats, sizeof(unsigned long long), cmp_ull);
    unsigned long long median = samples[repeats / 2];
    // Thread 0 streams n/threads elements (grid-stride); the whole buffer of
    // `bytes` is read across the grid, so bytes/cycle is the per-thread bytes
    // divided by the time thread 0 spent, scaled by the thread count.
    double total_bytes = (double)bytes;
    double bytes_per_cycle = median > 0 ? total_bytes / (double)median : 0.0;

    std::printf("%s{\"in_flight\":%d,\"bytes_per_cycle\":%.4f}",
                (k == 0 ? "" : ","), in_flight, bytes_per_cycle);
  }

  std::printf("]}\n");

  std::free(samples);
  cudaFree(d_in);
  cudaFree(d_sink);
  cudaFree(d_cycles);
  return 0;
}
