// Persistent CTA residency probe.
//
// Each block writes (block_id, sm_id, start_clock, end_clock) into a global
// buffer. The host driver computes the maximum number of blocks observed
// concurrently per SM and prints a JSON summary.

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

struct BlockEntry {
  unsigned int block_id;
  unsigned int sm_id;
  unsigned long long start_clock;
  unsigned long long end_clock;
};

extern "C" __global__ void amora_baseline_persistent_cta(BlockEntry *entries,
                                                         unsigned long long busy_cycles) {
  if (threadIdx.x != 0) {
    return;
  }
  unsigned int sm_id;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm_id));
  unsigned long long start = clock64();
  // Spin for `busy_cycles` cycles so concurrent blocks overlap on each SM.
  while (clock64() - start < busy_cycles) {
    // intentionally empty
  }
  unsigned long long stop = clock64();
  BlockEntry e;
  e.block_id = blockIdx.x;
  e.sm_id = sm_id;
  e.start_clock = start;
  e.end_clock = stop;
  entries[blockIdx.x] = e;
}

int main(int argc, char **argv) {
  int blocks = 1024;
  int threads = 32;
  unsigned long long busy_cycles = 200000ULL;
  for (int i = 1; i + 1 < argc; i += 2) {
    if (std::strcmp(argv[i], "--blocks") == 0) blocks = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--threads") == 0) threads = std::atoi(argv[i + 1]);
    else if (std::strcmp(argv[i], "--busy-cycles") == 0)
      busy_cycles = std::strtoull(argv[i + 1], nullptr, 10);
  }
  if (blocks <= 0) blocks = 1024;
  if (threads <= 0) threads = 32;

  cudaDeviceProp prop{};
  AMORA_CHECK(cudaGetDeviceProperties(&prop, 0));

  BlockEntry *d_entries = nullptr;
  AMORA_CHECK(cudaMalloc(&d_entries, sizeof(BlockEntry) * blocks));
  AMORA_CHECK(cudaMemset(d_entries, 0, sizeof(BlockEntry) * blocks));

  cudaEvent_t t0, t1;
  AMORA_CHECK(cudaEventCreate(&t0));
  AMORA_CHECK(cudaEventCreate(&t1));
  AMORA_CHECK(cudaEventRecord(t0));
  amora_baseline_persistent_cta<<<blocks, threads>>>(d_entries, busy_cycles);
  AMORA_CHECK(cudaEventRecord(t1));
  AMORA_CHECK(cudaEventSynchronize(t1));
  float ms = 0.0f;
  AMORA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

  BlockEntry *h_entries = (BlockEntry *)std::malloc(sizeof(BlockEntry) * blocks);
  AMORA_CHECK(cudaMemcpy(h_entries, d_entries, sizeof(BlockEntry) * blocks,
                          cudaMemcpyDeviceToHost));

  // For each SM, count the maximum concurrent blocks by sweeping start/end events.
  // We bound sm_id at 256 (current arch limits well below this).
  const int kMaxSm = 256;
  int per_sm_count[kMaxSm];
  int per_sm_peak[kMaxSm];
  for (int i = 0; i < kMaxSm; ++i) { per_sm_count[i] = 0; per_sm_peak[i] = 0; }
  // Group entries by SM, then sort by start to compute peak concurrency via sweep-line.
  for (int sm = 0; sm < kMaxSm; ++sm) {
    int n = 0;
    static unsigned long long starts[8192];
    static unsigned long long ends[8192];
    for (int b = 0; b < blocks && n < 8192; ++b) {
      if ((int)h_entries[b].sm_id == sm) {
        starts[n] = h_entries[b].start_clock;
        ends[n] = h_entries[b].end_clock;
        ++n;
      }
    }
    if (n == 0) continue;
    // Insertion sort of starts/ends arrays (small n bounded by per-SM resident blocks).
    for (int i = 1; i < n; ++i) {
      unsigned long long s = starts[i];
      int j = i - 1;
      while (j >= 0 && starts[j] > s) { starts[j + 1] = starts[j]; --j; }
      starts[j + 1] = s;
    }
    for (int i = 1; i < n; ++i) {
      unsigned long long e = ends[i];
      int j = i - 1;
      while (j >= 0 && ends[j] > e) { ends[j + 1] = ends[j]; --j; }
      ends[j + 1] = e;
    }
    int active = 0, peak = 0, si = 0, ei = 0;
    while (si < n) {
      if (starts[si] <= ends[ei]) { ++active; if (active > peak) peak = active; ++si; }
      else { --active; ++ei; }
    }
    per_sm_count[sm] = n;
    per_sm_peak[sm] = peak;
  }

  int sm_count = 0;
  int peak_resident_blocks = 0;
  int total_resident = 0;
  for (int sm = 0; sm < kMaxSm; ++sm) {
    if (per_sm_count[sm] == 0) continue;
    ++sm_count;
    if (per_sm_peak[sm] > peak_resident_blocks) peak_resident_blocks = per_sm_peak[sm];
    total_resident += per_sm_peak[sm];
  }

  double mean_resident = sm_count ? (double)total_resident / (double)sm_count : 0.0;

  std::printf(
      "{\"device_name\":\"%s\",\"multi_processor_count\":%d,\"sm_count_observed\":%d,"
      "\"blocks_launched\":%d,\"threads_per_block\":%d,\"busy_cycles\":%llu,"
      "\"peak_resident_blocks_per_sm\":%d,\"mean_resident_blocks_per_sm\":%.4f,"
      "\"elapsed_ms\":%.4f}\n",
      prop.name, prop.multiProcessorCount, sm_count, blocks, threads,
      (unsigned long long)busy_cycles, peak_resident_blocks, mean_resident, ms);
  std::free(h_entries);
  cudaFree(d_entries);
  cudaEventDestroy(t0);
  cudaEventDestroy(t1);
  return 0;
}
