// P0 dependent arithmetic latency probe.

#include <cuda_runtime.h>
#include <stdint.h>

struct ArithmeticLatencyResult {
  uint64_t total_cycles;
  uint32_t iterations;
  uint32_t op_kind;
  float fp32_sink;
  int int_sink;
  double fp64_sink;
};

enum ArithmeticOpKind {
  AMORA_FP32_FMA = 0,
  AMORA_INT_ADD = 1,
  AMORA_SFU_RSQRT = 2,
  AMORA_FP64_FMA = 3
};

extern "C" __global__ void amora_dependent_arithmetic_chain(
    ArithmeticLatencyResult *result,
    uint32_t iterations,
    uint32_t op_kind) {
  float xf = 1.125f;
  int xi = 7;
  double xd = 1.125;

  uint64_t start = clock64();
  for (uint32_t i = 0; i < iterations; ++i) {
    if (op_kind == AMORA_FP32_FMA) {
      asm volatile("fma.rn.f32 %0, %0, 1.0009765625, 0.5;" : "+f"(xf));
    } else if (op_kind == AMORA_INT_ADD) {
      asm volatile("add.s32 %0, %0, 17;" : "+r"(xi));
    } else if (op_kind == AMORA_SFU_RSQRT) {
      asm volatile("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(xf));
      xf += 0.000001f;
    } else if (op_kind == AMORA_FP64_FMA) {
      asm volatile("fma.rn.f64 %0, %0, 1.0000000001, 0.5;" : "+d"(xd));
    }
  }
  uint64_t stop = clock64();

  if (threadIdx.x == 0 && blockIdx.x == 0) {
    result->total_cycles = stop - start;
    result->iterations = iterations;
    result->op_kind = op_kind;
    result->fp32_sink = xf;
    result->int_sink = xi;
    result->fp64_sink = xd;
  }
}
