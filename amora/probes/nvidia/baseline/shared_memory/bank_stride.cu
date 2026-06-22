// Shared-memory bank-stride probe template.

extern "C" __global__ void amora_baseline_shared_bank_stride(unsigned int *out, int stride) {
  __shared__ unsigned int values[2048];
  const unsigned int tid = threadIdx.x;
  if (tid < 2048) {
    values[tid] = tid;
  }
  __syncthreads();

  unsigned int lane = tid & 31;
  unsigned int index = (lane * static_cast<unsigned int>(stride)) & 2047;
  unsigned long long start = clock64();
  unsigned int value = values[index];
  unsigned long long stop = clock64();
  if (tid == 0) {
    out[0] = value + static_cast<unsigned int>(stop - start);
  }
}
