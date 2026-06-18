// P0 shared-memory dependent pointer-chase latency probe.

#include <cuda_runtime.h>
#include <stdint.h>

struct SharedPointerChaseResult {
  uint64_t total_cycles;
  uint32_t iterations;
  uint32_t final_index;
};

extern "C" __global__ void amora_shared_pointer_chase(
    SharedPointerChaseResult *result,
    uint32_t iterations,
    uint32_t stride_words,
    uint32_t element_count) {
  extern __shared__ uint32_t shared_indices[];

  for (uint32_t i = threadIdx.x; i < element_count; i += blockDim.x) {
    shared_indices[i] = (i + stride_words) % element_count;
  }
  __syncthreads();

  uint32_t index = threadIdx.x % element_count;
  uint64_t start = clock64();
  for (uint32_t i = 0; i < iterations; ++i) {
    asm volatile("ld.shared.u32 %0, [%1];"
                 : "=r"(index)
                 : "l"(&shared_indices[index]));
  }
  uint64_t stop = clock64();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    result->total_cycles = stop - start;
    result->iterations = iterations;
    result->final_index = index;
  }
}
