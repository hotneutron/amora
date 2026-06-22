// Shared-memory pointer-chase latency probe template.

extern "C" __global__ void amora_baseline_shared_pointer_chase(unsigned int *out) {
  __shared__ unsigned int next[1024];
  const unsigned int tid = threadIdx.x;
  if (tid < 1024) {
    next[tid] = (tid + 1) & 1023;
  }
  __syncthreads();

  unsigned int index = 0;
  unsigned long long start = clock64();
#pragma unroll 64
  for (int i = 0; i < 1024; ++i) {
    index = next[index];
  }
  unsigned long long stop = clock64();
  if (tid == 0) {
    out[0] = index + static_cast<unsigned int>(stop - start);
  }
}
