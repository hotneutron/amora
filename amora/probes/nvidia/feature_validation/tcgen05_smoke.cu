// Blackwell PTX end-to-end audit. No matching native Blackwell device is available.
// Uniform MMA inputs deliberately avoid qualifying descriptor stride/swizzle mapping.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

__global__ void independent_tmem_roundtrip(uint32_t *out) {
  __shared__ unsigned base;
  unsigned tid = threadIdx.x;
  if (tid < 32) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&base));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                 :: "r"(addr) : "memory");
  }
  __syncthreads();
  unsigned taddr = base + ((tid / 32) * 32 << 16);
  unsigned value = 0x10000 + tid, result;
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};"
               :: "r"(taddr), "r"(value) : "memory");
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
               : "=r"(result) : "r"(taddr) : "memory");
  asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
  out[tid] = result;
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                 :: "r"(base) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 ::: "memory");
  }
}

__global__ void independent_tcgen05_uniform_mma(float *out) {
  __shared__ __align__(128) __half a[64*16];
  __shared__ __align__(128) __half b[16*8];
  __shared__ unsigned base;
  __shared__ __align__(8) uint64_t barrier;
  unsigned tid = threadIdx.x;
  for (unsigned i=tid; i<64*16; i+=128) a[i] = __float2half(1.0f);
  for (unsigned i=tid; i<16*8; i+=128) b[i] = __float2half(1.0f);
  if (tid < 32) {
    unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&base));
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;"
                 :: "r"(addr) : "memory");
  }
  unsigned bar = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
  if (tid == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar) : "memory");
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  if (tid == 0) {
    uint64_t ap = static_cast<unsigned>(__cvta_generic_to_shared(a));
    uint64_t bp = static_cast<unsigned>(__cvta_generic_to_shared(b));
    uint64_t adesc = (ap >> 4) | (1ull << 16) | (8ull << 32) | (1ull << 46);
    uint64_t bdesc = (bp >> 4) | (1ull << 16) | (8ull << 32) | (1ull << 46);
    unsigned idesc = (1u << 4) | (1u << 17) | (4u << 24);
    asm volatile("{ .reg .pred p; mov.pred p, 0; "
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
                 :: "r"(base), "l"(adesc), "l"(bdesc), "r"(idesc) : "memory");
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"
                 :: "r"(bar) : "memory");
  }
  asm volatile("{ .reg .pred p; TC_WAIT: "
               "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; "
               "@!p bra TC_WAIT; }" :: "r"(bar) : "memory");
  if (tid < 64) {
    unsigned taddr = base + ((tid / 32) * 32 << 16);
    unsigned v[8];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(v[0]),"=r"(v[1]),"=r"(v[2]),"=r"(v[3]),
                   "=r"(v[4]),"=r"(v[5]),"=r"(v[6]),"=r"(v[7])
                 : "r"(taddr) : "memory");
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    for (unsigned i=0; i<8; ++i) out[tid*8+i] = __uint_as_float(v[i]);
  }
  __syncthreads();
  if (tid < 32) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;"
                 :: "r"(base) : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;"
                 ::: "memory");
  }
  if (tid == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(bar) : "memory");
}

TEST(IndependentBlackwell, TmemFourWarpRoundTrip) {
  uint32_t *out = nullptr;
  ASSERT_EQ(cudaMalloc(&out, 128*sizeof(uint32_t)), cudaSuccess);
  independent_tmem_roundtrip<<<1,128>>>(out);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<uint32_t> result(128);
  ASSERT_EQ(cudaMemcpy(result.data(),out,128*sizeof(uint32_t),cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i=0;i<128;++i) EXPECT_EQ(result[i],0x10000+i);
  ASSERT_EQ(cudaFree(out),cudaSuccess);
}

TEST(IndependentBlackwell, UniformF16MmaCommitAndLoad) {
  float *out = nullptr;
  ASSERT_EQ(cudaMalloc(&out,512*sizeof(float)),cudaSuccess);
  independent_tcgen05_uniform_mma<<<1,128>>>(out);
  ASSERT_EQ(cudaGetLastError(),cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(),cudaSuccess);
  std::vector<float> result(512);
  ASSERT_EQ(cudaMemcpy(result.data(),out,512*sizeof(float),cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (float value:result) EXPECT_EQ(value,16.0f);
  ASSERT_EQ(cudaFree(out),cudaSuccess);
}

