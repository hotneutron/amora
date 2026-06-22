// Dependent arithmetic latency probe template.

extern "C" __global__ void amora_baseline_fp32_dependent_chain(float *out, float seed) {
  float x = seed;
  unsigned long long start = clock64();
#pragma unroll 128
  for (int i = 0; i < 1024; ++i) {
    x = fmaf(x, 1.000001f, 0.000001f);
  }
  unsigned long long stop = clock64();
  out[0] = x + static_cast<float>(stop - start);
}
