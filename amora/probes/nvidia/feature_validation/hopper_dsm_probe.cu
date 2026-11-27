// Independent standard CUDA cluster/DSM checks; identical binary on H100 and PTX.
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
namespace cg = cooperative_groups;

__global__ void standard_dsm_store_atomic(uint32_t *out) {
  __shared__ uint32_t values[32];
  auto cluster = cg::this_cluster();
  unsigned rank = cluster.block_rank(), lane = threadIdx.x;
  values[lane] = rank * 1000 + lane;
  cluster.sync();
  if (rank == 0) {
    uint32_t *peer = cluster.map_shared_rank(values, 1);
    peer[lane] = 2000 + lane;
    atomicAdd(peer + lane, 7u);
  }
  cluster.sync();
  out[rank * 32 + lane] = values[lane];
  cluster.sync();
}

__global__ void standard_remote_arrive_local_wait(uint32_t *out) {
  __shared__ __align__(8) uint64_t barrier;
  auto cluster = cg::this_cluster();
  unsigned rank = cluster.block_rank();
  unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(&barrier));
  if (rank == 0 && threadIdx.x == 0)
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(addr) : "memory");
  cluster.sync();
  if (rank == 1 && threadIdx.x == 0) {
    unsigned remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(remote) : "r"(addr));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote) : "memory");
  }
  if (rank == 0) {
    asm volatile("{ .reg .pred p; WAIT_LABEL: "
                 "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0; "
                 "@!p bra WAIT_LABEL; }" :: "r"(addr) : "memory");
    out[threadIdx.x] = 1;
  }
  cluster.sync();
  if (rank == 0 && threadIdx.x == 0)
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" :: "r"(addr) : "memory");
}

static void launch_cluster(const void *kernel, uint32_t *output) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(2); cfg.blockDim = dim3(32);
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeClusterDimension;
  attr.val.clusterDim = {2, 1, 1};
  cfg.attrs = &attr; cfg.numAttrs = 1;
  void *args[] = {&output};
  ASSERT_EQ(cudaLaunchKernelExC(&cfg, kernel, args), cudaSuccess);
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

TEST(IndependentHopperDsm, ClusterRemoteStoreAndAtomicAdd) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, 64 * sizeof(uint32_t)), cudaSuccess);
  launch_cluster((const void *)standard_dsm_store_atomic, output);
  std::vector<uint32_t> result(64);
  ASSERT_EQ(cudaMemcpy(result.data(), output, 64 * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  for (unsigned i = 0; i < 32; ++i) {
    EXPECT_EQ(result[i], i);
    EXPECT_EQ(result[32+i], 2007+i);
  }
  ASSERT_EQ(cudaFree(output), cudaSuccess);
}

TEST(IndependentHopperDsm, RemoteMbarrierArriveAndOwnerWait) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output, 32 * sizeof(uint32_t)), cudaSuccess);
  ASSERT_EQ(cudaMemset(output, 0, 32 * sizeof(uint32_t)), cudaSuccess);
  launch_cluster((const void *)standard_remote_arrive_local_wait, output);
  std::vector<uint32_t> result(32);
  ASSERT_EQ(cudaMemcpy(result.data(), output, 32 * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost), cudaSuccess);
  for (auto value : result) EXPECT_EQ(value, 1u);
  ASSERT_EQ(cudaFree(output), cudaSuccess);
}

// Isolate the ordering mismatch without changing the original failing kernel.
template <int Mode>
__global__ void dsm_order_isolation(uint32_t *out) {
  __shared__ uint32_t values[32];
  auto cluster = cg::this_cluster();
  unsigned rank = cluster.block_rank(), lane = threadIdx.x;
  values[lane] = rank * 1000 + lane;
  cluster.sync();
  if (rank == 0) {
    uint32_t *peer = cluster.map_shared_rank(values, 1);
    if constexpr (Mode != 1) peer[lane] = 2000 + lane;
    if constexpr (Mode == 2) asm volatile("fence.sc.cluster;" ::: "memory");
    if constexpr (Mode != 0) atomicAdd(peer + lane, 7u);
  }
  cluster.sync();
  out[rank*32+lane] = values[lane];
  cluster.sync();
}

template <int Mode>
static void check_dsm_isolation(unsigned expected) {
  uint32_t *output = nullptr;
  ASSERT_EQ(cudaMalloc(&output,64*sizeof(uint32_t)),cudaSuccess);
  launch_cluster((const void *)dsm_order_isolation<Mode>,output);
  std::vector<uint32_t> result(64);
  ASSERT_EQ(cudaMemcpy(result.data(),output,64*sizeof(uint32_t),cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (unsigned i=0;i<32;++i) {
    EXPECT_EQ(result[i],i);
    EXPECT_EQ(result[32+i],expected+i);
  }
  ASSERT_EQ(cudaFree(output),cudaSuccess);
}

TEST(IndependentHopperDsm, RemoteStoreOnly) { check_dsm_isolation<0>(2000); }
TEST(IndependentHopperDsm, RemoteAtomicOnly) { check_dsm_isolation<1>(1007); }
TEST(IndependentHopperDsm, StoreFenceAtomic) { check_dsm_isolation<2>(2007); }

