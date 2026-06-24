// Memory-pipeline lane-address-pattern probe.
//
// One warp performs many global loads (LDG) under controlled per-lane address
// patterns. The kernel itself just exercises the access; the real signal is the
// NCU request/sector counters, which reveal how lane address patterns coalesce
// into memory sectors. Four named patterns are exercised:
//   0 contiguous: lane i -> addr i (fully coalesced)
//   1 stride2:    lane i -> addr 2*i
//   2 stride32:   lane i -> addr 32*i (one sector per lane)
//   3 broadcast:  all lanes -> addr 0 (single sector)
// The driver accepts --pattern <id 0..3> so NCU can profile each pattern in a
// separate launch; the default run exercises all four patterns.

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

// One warp issues `iters` rounds of LDG over `buf`, where each lane's address is
// determined by `pattern`. The buffer is large enough that strided patterns hit
// distinct sectors. The sink store is on an impossible condition (prevents DCE).
extern "C" __global__ void
amora_mem_lane_patterns(const float *buf, float *sink, size_t n, int pattern,
                        int iters) {
  int lane = threadIdx.x;
  float acc = 0.f;
  for (int it = 0; it < iters; ++it) {
    size_t addr;
    switch (pattern) {
      case 1:  addr = (size_t)lane * 2u; break;          // stride2
      case 2:  addr = (size_t)lane * 32u; break;         // stride32
      case 3:  addr = 0u; break;                         // broadcast
      default: addr = (size_t)lane; break;               // contiguous
    }
    addr = (addr + (size_t)it) % n;
    acc += buf[addr];
  }
  if (acc == -1.f) sink[lane] = acc;  // impossible: prevents DCE
}

int main(int argc, char **argv) {
  int pattern = -1;   // -1 == run all patterns
  int iters = 4096;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--pattern") == 0) pattern = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--iters") == 0) iters = std::atoi(argv[i + 1]);
  }
  if (iters <= 0) iters = 4096;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  // Large enough that stride32 over 32 lanes touches distinct cache lines.
  size_t n = (size_t)1 << 20;  // 1M floats == 4 MiB
  size_t bytes = n * sizeof(float);

  float *d_buf = nullptr;
  float *d_sink = nullptr;
  AMORA_CHECK(cudaMalloc(&d_buf, bytes));
  AMORA_CHECK(cudaMalloc(&d_sink, sizeof(float) * 32));
  AMORA_CHECK(cudaMemset(d_buf, 0, bytes));

  const char *names[] = {"contiguous", "stride2", "stride32", "broadcast"};
  const int n_patterns = 4;

  int first = (pattern >= 0 && pattern < n_patterns) ? pattern : 0;
  int last = (pattern >= 0 && pattern < n_patterns) ? pattern : n_patterns - 1;

  std::printf("{\"device_name\":\"%s\",\"patterns\":[", prop.name);
  bool printed = false;
  for (int p = first; p <= last; ++p) {
    amora_mem_lane_patterns<<<1, 32>>>(d_buf, d_sink, n, p, iters);
    AMORA_CHECK(cudaDeviceSynchronize());
    std::printf("%s{\"name\":\"%s\",\"iters\":%d}", (printed ? "," : ""),
                names[p], iters);
    printed = true;
  }
  std::printf("]}\n");

  cudaFree(d_buf);
  cudaFree(d_sink);
  return 0;
}
