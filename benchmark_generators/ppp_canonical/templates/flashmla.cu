// AMORA PPP bounded FlashMLA dense decode classification workload.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    exit(1);
  }
}

__global__ void amora_ppp_flashmla_dense_decode_kernel(
    const __half* __restrict__ q,
    const __half* __restrict__ kv,
    __half* __restrict__ out,
    int batch,
    int sequence,
    int heads,
    int sample_kv_blocks) {
  const int hidden = 576;
  const int query_tile = 64;
  const int kv_tile = 64;
  int query_blocks = (sequence + query_tile - 1) / query_tile;
  int batch_head = blockIdx.x / query_blocks;
  int query_block = blockIdx.x % query_blocks;
  if (batch_head >= batch * heads) return;

  int query_start = query_block * query_tile;
  int sampled_blocks = min(query_block + 1, sample_kv_blocks);
  int first_kv_block = query_block + 1 - sampled_blocks;
  float accum = 0.0f;
  for (int query_offset = 0; query_offset < 4 && query_start + query_offset < sequence;
       ++query_offset) {
    long query_base = ((long)batch_head * sequence + query_start + query_offset) * hidden;
    for (int sample = 0; sample < sampled_blocks; ++sample) {
      int kv_start = (first_kv_block + sample) * kv_tile;
      for (int kv_offset = 0; kv_offset < 4 && kv_start + kv_offset < sequence;
           ++kv_offset) {
        long kv_base = ((long)batch_head * sequence + kv_start + kv_offset) * hidden;
        for (int dim = threadIdx.x; dim < hidden; dim += blockDim.x) {
          accum += __expf(__half2float(q[query_base + dim]) *
                          __half2float(kv[kv_base + dim]) * 1.0e-4f);
        }
      }
    }
  }
  int output_row = query_start + (threadIdx.x % query_tile);
  int output_col = threadIdx.x / query_tile;
  if (output_row < sequence && output_col < hidden) {
    out[((long)batch_head * sequence + output_row) * hidden + output_col] =
        __float2half(accum);
  }
}

int main(int argc, char** argv) {
  int batch = argc > 1 ? atoi(argv[1]) : 1;
  int sequence = argc > 2 ? atoi(argv[2]) : 4096;
  int heads = argc > 3 ? atoi(argv[3]) : 8;
  int sample_kv_blocks = argc > 4 ? atoi(argv[4]) : 2;
  sample_kv_blocks = max(sample_kv_blocks, 1);
  const int hidden = 576;
  long elements = (long)batch * heads * sequence * hidden;

  __half* q = nullptr;
  __half* kv = nullptr;
  __half* out = nullptr;
  check(cudaMalloc(&q, (size_t)elements * sizeof(__half)), "cudaMalloc q");
  check(cudaMalloc(&kv, (size_t)elements * sizeof(__half)), "cudaMalloc kv");
  check(cudaMalloc(&out, (size_t)elements * sizeof(__half)), "cudaMalloc out");
  check(cudaMemset(q, 1, (size_t)elements * sizeof(__half)), "cudaMemset q");
  check(cudaMemset(kv, 2, (size_t)elements * sizeof(__half)), "cudaMemset kv");

  int blocks = batch * heads * ((sequence + 63) / 64);
  const int threads = 256;
#ifndef AMORA_GCOM_TRACE
  amora_ppp_flashmla_dense_decode_kernel<<<blocks, threads>>>(
      q, kv, out, batch, sequence, heads, sample_kv_blocks);
  check(cudaDeviceSynchronize(), "warmup flashmla");
#endif
  amora_ppp_flashmla_dense_decode_kernel<<<blocks, threads>>>(
      q, kv, out, batch, sequence, heads, sample_kv_blocks);
  check(cudaDeviceSynchronize(), "measured flashmla");

  cudaFree(q);
  cudaFree(kv);
  cudaFree(out);
  return 0;
}
