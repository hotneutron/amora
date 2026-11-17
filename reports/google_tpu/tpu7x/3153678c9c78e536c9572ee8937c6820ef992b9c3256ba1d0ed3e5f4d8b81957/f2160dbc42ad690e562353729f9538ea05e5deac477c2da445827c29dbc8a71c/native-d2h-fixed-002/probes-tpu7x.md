# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-d2h-fixed-002
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 1
- `fit_status`: `direct`=1
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[transfer.host_device](#transferhost_device)

---

## transfer.host_device

| field | value |
| --- | --- |
| launch | `transfer`  transfer {"shape": [1024, 1024]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 3.42726e+06 ns |
| simulator_param | `unsupported` = — |
| concept | `transfer.host_device` |

- launch.extras: `{"compiler_artifacts": {}, "device_copy_included": true, "direction": "device_to_host", "first_execution_ns": 4922944, "host_cache_policy": "fresh device buffer for every D2H invocation", "input_digest": "b7a8f7247102833f9aab3a204baa10bdb37eb9cff35fcca27cf0cc84b600d2a6", "kind": "transfer", "logical_flops": 2097152, "matrix": false, "mode": "transfer", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 4194304, "shape": [1024, 1024], "staging_ns": 17237257}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | transfer.host_device/0 |
| `effective_bytes_per_second` | 1.22381e+09 |
| `effective_flops_per_second` | 6.11904e+08 |
| `probe_id` | transfer.host_device |
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
        1024,
        1024
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
  "device_copy_included": true,
  "direction": "device_to_host",
  "first_execution_ns": 4922944,
  "host_cache_policy": "fresh device buffer for every D2H invocation",
  "input_digest": "b7a8f7247102833f9aab3a204baa10bdb37eb9cff35fcca27cf0cc84b600d2a6",
  "kind": "transfer",
  "logical_flops": 2097152,
  "matrix": false,
  "mode": "transfer",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 4194304,
  "shape": [
    1024,
    1024
  ],
  "staging_ns": 17237257
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "direction": "device_to_host",
  "probe_id": "transfer.host_device",
  "seed": 3,
  "shape": [
    1024,
    1024
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 1,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 3427255.0,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 18509.5,
      "max": 3508712.0,
      "median": 3425879.5,
      "min": 3377240.0,
      "relative_mad": 0.005402846188839975
    },
    {
      "count": 30,
      "mad": 21121.0,
      "max": 3508413.0,
      "median": 3430766.5,
      "min": 3387660.0,
      "relative_mad": 0.0061563501917137175
    },
    {
      "count": 30,
      "mad": 27841.0,
      "max": 4267501.0,
      "median": 3427255.0,
      "min": 3379351.0,
      "relative_mad": 0.008123410717906896
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 1375.5,
    "max": 3430766.5,
    "median": 3427255.0,
    "min": 3425879.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "single_invocation",
  "warmup_invocations": 20,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `14f9e14c3de814ae43b084dc8f27d7b78ec450da70982f3219f1cfc4402e6f56`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-final-verified/out/tpu/tpu-d2h-completion-v1/2d141f3888545fb25a7e00d6252a75d44df91928a14b06679fc2696aac75125e/native-d2h-fixed-002/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
