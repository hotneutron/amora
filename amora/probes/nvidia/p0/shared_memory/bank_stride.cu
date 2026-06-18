// P0 shared-memory bank-stride conflict probe.

#include <cuda_runtime.h>
#include <stdint.h>

struct SharedBankStrideResult {
  uint64_t total_cycles;
  uint32_t iterations;
  uint32_t stride_words;
  uint32_t sink;
};

extern "C" __global__ void amora_shared_bank_stride(
    SharedBankStrideResult *result,
    uint32_t iterations,
    uint32_t stride_words) {
  extern __shared__ uint32_t shared_values[];

  uint32_t lane = threadIdx.x & 31;
  uint32_t index = lane * stride_words;
  shared_values[index] = lane + 1;
  __syncthreads();

  uint32_t sink = 0;
  uint64_t start = clock64();
  for (uint32_t i = 0; i < iterations; ++i) {
    uint32_t value;
    asm volatile("ld.shared.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(&shared_values[index]));
    sink += value;
  }
  uint64_t stop = clock64();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    result->total_cycles = stop - start;
    result->iterations = iterations;
    result->stride_words = stride_words;
    result->sink = sink;
  }
}
