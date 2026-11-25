# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-profile-replay-001
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 1
- `fit_status`: `direct`=1
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[matrix.matmul_throughput](#matrixmatmul_throughput)

---

## matrix.matmul_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [256, 256, 256]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `matrix.matmul_throughput` |

- launch.extras: `{"compile_ns": 46878646, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "88e69a098b5f2eb492cb82197d3a854aeb4f42ea47f2079198f2b1690834dcf8"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "b13550e55fd9179bc35444030528fb4b6e6794f0ea1cbf7fde859b94e2fd1b16"}}, "data_pattern": "random", "dynamic_argument_count": 2, "first_execution_ns": 709553, "input_digest": "29a114021862e99351d8a6183bf7211d58804b7309e004aa3de0710b8d54d51f", "kind": "matmul", "logical_flops": 33554432, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [256, 256, 256], "staging_ns": 326677872}`
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
      "max_abs_error": 9.5367431640625e-07,
      "passed": true,
      "shape": [
        256,
        256
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
| `case_id` | matrix.matmul_throughput/0 |
| `probe_id` | matrix.matmul_throughput |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 9.5367431640625e-07,
      "passed": true,
      "shape": [
        256,
        256
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
  "compile_ns": 46878646,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "88e69a098b5f2eb492cb82197d3a854aeb4f42ea47f2079198f2b1690834dcf8"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "b13550e55fd9179bc35444030528fb4b6e6794f0ea1cbf7fde859b94e2fd1b16"
    }
  },
  "data_pattern": "random",
  "dynamic_argument_count": 2,
  "first_execution_ns": 709553,
  "input_digest": "29a114021862e99351d8a6183bf7211d58804b7309e004aa3de0710b8d54d51f",
  "kind": "matmul",
  "logical_flops": 33554432,
  "matrix": true,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "shape": [
    256,
    256,
    256
  ],
  "staging_ns": 326677872
}
```

</details>

<details><summary><code>profiling</code> (JSON)</summary>

```json
{
  "artifacts": [
    "plugins/profile/2026_11_17_19_33_31/gke-tpu-f20b10b7-2pbv.xplane.pb",
    "plugins/profile/2026_11_17_19_33_31/perfetto_trace.json.gz",
    "plugins/profile/2026_11_17_19_33_31/gke-tpu-f20b10b7-2pbv.trace.json.gz"
  ],
  "reason": "trace captured; device interval semantics need explicit trace validation",
  "state": "observed"
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "measure": false,
  "probe_id": "matrix.matmul_throughput",
  "shape": [
    256,
    256,
    256
  ]
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `ebf591f437bf058c0ebcb96044f2f69e1d627d1eb9d69e273575564cb0fcb270`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-final-verified/out/tpu/tpu-profile-replay-v1/bef94bfaadccf1bcccb1a619afee3ea155c871e04e74e6049f9afb19f75c3fcc/native-profile-replay-001/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
