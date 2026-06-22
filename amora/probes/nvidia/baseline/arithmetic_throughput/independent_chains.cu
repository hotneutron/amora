// Independent arithmetic throughput probe template.

extern "C" __global__ void amora_baseline_fp32_independent_chains(float *out, float seed) {
  float a = seed;
  float b = seed + 1.0f;
  float c = seed + 2.0f;
  float d = seed + 3.0f;
  unsigned long long start = clock64();
#pragma unroll 128
  for (int i = 0; i < 1024; ++i) {
    a = fmaf(a, 1.000001f, 0.000001f);
    b = fmaf(b, 1.000002f, 0.000002f);
    c = fmaf(c, 1.000003f, 0.000003f);
    d = fmaf(d, 1.000004f, 0.000004f);
  }
  unsigned long long stop = clock64();
  out[0] = a + b + c + d + static_cast<float>(stop - start);
}
