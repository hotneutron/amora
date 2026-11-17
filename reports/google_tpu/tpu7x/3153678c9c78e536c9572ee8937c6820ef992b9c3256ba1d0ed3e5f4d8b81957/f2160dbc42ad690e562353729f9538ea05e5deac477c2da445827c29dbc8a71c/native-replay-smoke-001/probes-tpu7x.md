# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-replay-smoke-001
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 2
- `fit_status`: `direct`=2
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[vector.elementwise_throughput](#vectorelementwise_throughput) · [matrix.matmul_throughput](#matrixmatmul_throughput)

---

## vector.elementwise_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [128, 128]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `vector.elementwise_throughput` |

- launch.extras: `{"compile_ns": 311481883, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "cfb8cf80687e18bdb44acb686bed67109520e5488575a7c4f39adc3802459be8"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "28861aeb095bd3b001bc90ccbcfabcf6d4e34344d0159c42cf1588e711554c37"}}, "dynamic_argument_count": 1, "first_execution_ns": 942937, "input_digest": "cdc1a107d14e0f8289500fff522b2a93c219f55b476cfa6ff0fa8ed5a22ec12e", "kind": "vector", "logical_flops": 32768, "matrix": false, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 131072, "shape": [128, 128], "staging_ns": 5147923}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        128,
        128
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | vector.elementwise_throughput/0 |
| `probe_id` | vector.elementwise_throughput |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 0.0,
      "passed": true,
      "shape": [
        128,
        128
      ]
    }
  ],
  "passed": true,
  "rtol": 0.0002,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "compile_ns": 311481883,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "cfb8cf80687e18bdb44acb686bed67109520e5488575a7c4f39adc3802459be8"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "28861aeb095bd3b001bc90ccbcfabcf6d4e34344d0159c42cf1588e711554c37"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 942937,
  "input_digest": "cdc1a107d14e0f8289500fff522b2a93c219f55b476cfa6ff0fa8ed5a22ec12e",
  "kind": "vector",
  "logical_flops": 32768,
  "matrix": false,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 131072,
  "shape": [
    128,
    128
  ],
  "staging_ns": 5147923
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "measure": false,
  "probe_id": "vector.elementwise_throughput",
  "shape": [
    128,
    128
  ]
}
```

</details>

[↑ contents](#contents)

---

## matrix.matmul_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [128, 128, 128]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `matrix.matmul_throughput` |

- launch.extras: `{"compile_ns": 31858166, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "bb803c615e8f4cc2bcce738ee1445ee49d12b066af6071f5ce5f9bc10d8f17c1"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "e86ec563d754aa70c2e990b4e9bb14ddbcac89c683505bc049263f388d60047e"}}, "data_pattern": "random", "dynamic_argument_count": 2, "first_execution_ns": 630844, "input_digest": "697c61a7a350756e075148d73ab79e2ab61a74c92cef75cf6e72b73e7dc4d0e1", "kind": "matmul", "logical_flops": 4194304, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [128, 128, 128], "staging_ns": 30625322}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Measurement value

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 4.76837158203125e-07,
      "passed": true,
      "shape": [
        128,
        128
      ]
    }
  ],
  "passed": true,
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

### Raw values

| key | value |
| --- | --- |
| `case_id` | matrix.matmul_throughput/1 |
| `probe_id` | matrix.matmul_throughput |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 4.76837158203125e-07,
      "passed": true,
      "shape": [
        128,
        128
      ]
    }
  ],
  "passed": true,
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "compile_ns": 31858166,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "bb803c615e8f4cc2bcce738ee1445ee49d12b066af6071f5ce5f9bc10d8f17c1"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "e86ec563d754aa70c2e990b4e9bb14ddbcac89c683505bc049263f388d60047e"
    }
  },
  "data_pattern": "random",
  "dynamic_argument_count": 2,
  "first_execution_ns": 630844,
  "input_digest": "697c61a7a350756e075148d73ab79e2ab61a74c92cef75cf6e72b73e7dc4d0e1",
  "kind": "matmul",
  "logical_flops": 4194304,
  "matrix": true,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "shape": [
    128,
    128,
    128
  ],
  "staging_ns": 30625322
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "measure": false,
  "probe_id": "matrix.matmul_throughput",
  "shape": [
    128,
    128,
    128
  ]
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `284d3e7886f74496abd0716f3b5a078ad6cb407a047232537deda06586015063`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-final-verified/out/tpu/tpu-smoke-v1/8eb82045d34b30112403e6827bc74a8889e67c0348bc7d04310ba1fc03428d4e/native-replay-smoke-001/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
