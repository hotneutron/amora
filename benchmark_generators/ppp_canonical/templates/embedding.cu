// AMORA PPP embedding lookup classification workload.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

static void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
    exit(1);
  }
}

__global__ void amora_ppp_embedding_lookup(
    const __half* __restrict__ table,
    const int* __restrict__ indices,
    __half* __restrict__ out,
    long rows,
    int hidden) {
  long row = blockIdx.x;
  if (row >= rows) return;
  const __half* source = table + (long)indices[row] * hidden;
  __half* destination = out + row * (long)hidden;
  for (int index = threadIdx.x; index < hidden; index += blockDim.x) {
    destination[index] = source[index];
  }
}

__global__ void amora_ppp_flush_l2(uint8_t* buffer, size_t bytes) {
  size_t index = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t stride = (size_t)blockDim.x * gridDim.x;
  for (; index < bytes; index += stride) {
    buffer[index] = (uint8_t)(buffer[index] + (uint8_t)index);
  }
}

static int randomish_id(long index, int unique) {
  unsigned value = (unsigned)index;
  value = value * 1664525u + 1013904223u;
  value ^= value >> 16;
  return (int)(value % (unsigned)unique);
}

int main(int argc, char** argv) {
  int batch = argc > 1 ? atoi(argv[1]) : 1;
  int sequence = argc > 2 ? atoi(argv[2]) : 8192;
  int hidden = argc > 3 ? atoi(argv[3]) : 128;
  int vocab = argc > 4 ? atoi(argv[4]) : 50000;
  int unique = argc > 5 ? atoi(argv[5]) : batch * sequence;
  int pattern = argc > 6 ? atoi(argv[6]) : 0;
  long rows = (long)batch * sequence;
  unique = min(max(unique, 1), min(vocab, (int)rows));

  std::vector<int> host_indices(rows);
  for (long index = 0; index < rows; ++index) {
    host_indices[index] = pattern == 1 ? (int)(index % unique) : randomish_id(index, unique);
  }
  __half* table = nullptr;
  __half* out = nullptr;
  int* indices = nullptr;
  uint8_t* flush = nullptr;
  const size_t flush_bytes = 128ull << 20;
  check(cudaMalloc(&table, (size_t)vocab * hidden * sizeof(__half)), "cudaMalloc table");
  check(cudaMalloc(&out, (size_t)rows * hidden * sizeof(__half)), "cudaMalloc out");
  check(cudaMalloc(&indices, (size_t)rows * sizeof(int)), "cudaMalloc indices");
  check(cudaMalloc(&flush, flush_bytes), "cudaMalloc flush");
  check(cudaMemset(table, 1, (size_t)vocab * hidden * sizeof(__half)), "cudaMemset table");
  check(cudaMemset(flush, 3, flush_bytes), "cudaMemset flush");
  check(cudaMemcpy(indices, host_indices.data(), (size_t)rows * sizeof(int),
                   cudaMemcpyHostToDevice), "cudaMemcpy indices");

  const int threads = 256;
#ifndef AMORA_GCOM_TRACE
  amora_ppp_embedding_lookup<<<rows, threads>>>(table, indices, out, rows, hidden);
  check(cudaDeviceSynchronize(), "warmup embedding");
  amora_ppp_flush_l2<<<4096, threads>>>(flush, flush_bytes);
  check(cudaDeviceSynchronize(), "flush l2");
#endif
  amora_ppp_embedding_lookup<<<rows, threads>>>(table, indices, out, rows, hidden);
  check(cudaDeviceSynchronize(), "measured embedding");

  cudaFree(table);
  cudaFree(out);
  cudaFree(indices);
  cudaFree(flush);
  return 0;
}
