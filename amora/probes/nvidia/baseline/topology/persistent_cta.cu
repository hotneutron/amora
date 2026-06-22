// Persistent CTA residency probe template.
//
// This kernel is intentionally minimal in the baseline cutline. The Python runner
// currently reports this probe as unsupported until build orchestration and
// residency parsing are implemented.

extern "C" __global__ void amora_baseline_persistent_cta(unsigned long long *entries) {
  const unsigned int block_id = blockIdx.x;
  if (threadIdx.x == 0) {
    entries[block_id] = clock64();
  }
}
