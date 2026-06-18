// P0 persistent CTA residency probe.
//
// This source is intentionally self-contained so the Python runner can compile
// it with nvcc on CUDA hosts. It records CTA entry information and keeps CTAs
// alive for a bounded spin window.

#include <cuda_runtime.h>
#include <stdint.h>

struct PersistentCtaRecord {
  uint32_t block_id;
  uint32_t sm_id;
  uint64_t entry_clock;
  uint32_t live_count;
};

__device__ __forceinline__ uint32_t amora_smid() {
  uint32_t smid;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
  return smid;
}

extern "C" __global__ void amora_persistent_cta(
    PersistentCtaRecord *records,
    unsigned int *live_counter,
    uint64_t spin_cycles) {
  extern __shared__ uint8_t dynamic_smem[];
  (void)dynamic_smem;

  if (threadIdx.x == 0) {
    uint32_t slot = atomicAdd(live_counter, 1);
    records[blockIdx.x].block_id = blockIdx.x;
    records[blockIdx.x].sm_id = amora_smid();
    records[blockIdx.x].entry_clock = clock64();
    records[blockIdx.x].live_count = slot + 1;
  }

  __syncthreads();

  uint64_t start = clock64();
  while ((clock64() - start) < spin_cycles) {
    asm volatile("" ::: "memory");
  }

  __syncthreads();

  if (threadIdx.x == 0) {
    atomicSub(live_counter, 1);
  }
}
