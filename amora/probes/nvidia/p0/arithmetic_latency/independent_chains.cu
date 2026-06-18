// P0 independent arithmetic throughput probe.

#include <cuda_runtime.h>
#include <stdint.h>

struct ArithmeticThroughputResult {
  uint64_t total_cycles;
  uint32_t iterations;
  uint32_t chains;
  uint32_t op_kind;
  float sink;
};

extern "C" __global__ void amora_independent_arithmetic_chains(
    ArithmeticThroughputResult *result,
    uint32_t iterations,
    uint32_t chains,
    uint32_t op_kind) {
  float x0 = 1.0f;
  float x1 = 2.0f;
  float x2 = 3.0f;
  float x3 = 4.0f;
  int i0 = 1;
  int i1 = 2;
  int i2 = 3;
  int i3 = 4;

  uint64_t start = clock64();
  for (uint32_t i = 0; i < iterations; ++i) {
    if (op_kind == 0) {
      asm volatile("fma.rn.f32 %0, %0, 1.0009765625, 0.5;" : "+f"(x0));
      if (chains > 1) asm volatile("fma.rn.f32 %0, %0, 1.0009765625, 0.5;" : "+f"(x1));
      if (chains > 2) asm volatile("fma.rn.f32 %0, %0, 1.0009765625, 0.5;" : "+f"(x2));
      if (chains > 3) asm volatile("fma.rn.f32 %0, %0, 1.0009765625, 0.5;" : "+f"(x3));
    } else {
      asm volatile("add.s32 %0, %0, 17;" : "+r"(i0));
      if (chains > 1) asm volatile("add.s32 %0, %0, 17;" : "+r"(i1));
      if (chains > 2) asm volatile("add.s32 %0, %0, 17;" : "+r"(i2));
      if (chains > 3) asm volatile("add.s32 %0, %0, 17;" : "+r"(i3));
    }
  }
  uint64_t stop = clock64();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    result->total_cycles = stop - start;
    result->iterations = iterations;
    result->chains = chains;
    result->op_kind = op_kind;
    result->sink = x0 + x1 + x2 + x3 + float(i0 + i1 + i2 + i3);
  }
}
