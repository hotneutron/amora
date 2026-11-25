# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-fixed-pallas-001
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 3
- `fit_status`: `direct`=3
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[global_memory.streaming](#global_memorystreaming) · [global_memory.streaming](#global_memorystreaming) · [local_memory.transfer_sweep](#local_memorytransfer_sweep)

---

## global_memory.streaming

| field | value |
| --- | --- |
| launch | `pallas`  pallas {"memory_spaces": "HBM input/output; VMEM tile refs", "shape": [128, 128], "tile": [128, 128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 163986 ns |
| simulator_param | `unsupported` = — |
| concept | `global_memory.streaming` |

- launch.extras: `{"compile_ns": 336983033, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "ad9f28a13953a9b5d7317a0246093ce73dd23aade84b61860b25d8b07dbe7f52"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "7506950b13e56493f5a6bb46da8f02418d5da9a496abf1651a336267177a0d9d"}}, "dma_semaphores": false, "dynamic_argument_count": 1, "first_execution_ns": 4300385, "input_digest": "cdc1a107d14e0f8289500fff522b2a93c219f55b476cfa6ff0fa8ed5a22ec12e", "interpretation": "controlled tiled stream; physical HBM traffic unmeasured", "kind": "pallas_stream", "logical_flops": 32768, "matrix": false, "memory_spaces": "HBM input/output; VMEM tile refs", "mode": "pallas", "probe_source_sha256": "f37c61f5e6bfe3c35795deab11621c2a07c691cccfb85dc9d53cf9d5743adae7", "requested_bytes": 131072, "shape": [128, 128], "staging_ns": 2181421, "tile": [128, 128]}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | global_memory.streaming/0 |
| `effective_bytes_per_second` | 7.99288e+08 |
| `effective_flops_per_second` | 1.99822e+08 |
| `probe_id` | global_memory.streaming |
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
  "compile_ns": 336983033,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "ad9f28a13953a9b5d7317a0246093ce73dd23aade84b61860b25d8b07dbe7f52"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "7506950b13e56493f5a6bb46da8f02418d5da9a496abf1651a336267177a0d9d"
    }
  },
  "dma_semaphores": false,
  "dynamic_argument_count": 1,
  "first_execution_ns": 4300385,
  "input_digest": "cdc1a107d14e0f8289500fff522b2a93c219f55b476cfa6ff0fa8ed5a22ec12e",
  "interpretation": "controlled tiled stream; physical HBM traffic unmeasured",
  "kind": "pallas_stream",
  "logical_flops": 32768,
  "matrix": false,
  "memory_spaces": "HBM input/output; VMEM tile refs",
  "mode": "pallas",
  "probe_source_sha256": "f37c61f5e6bfe3c35795deab11621c2a07c691cccfb85dc9d53cf9d5743adae7",
  "requested_bytes": 131072,
  "shape": [
    128,
    128
  ],
  "staging_ns": 2181421,
  "tile": [
    128,
    128
  ]
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "global_memory.streaming",
  "shape": [
    128,
    128
  ],
  "tile": [
    128,
    128
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
  "median_ns_per_invocation": 163986.0,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 5363.0,
      "max": 189995.0,
      "median": 164059.0,
      "min": 154234.0,
      "relative_mad": 0.03268945927989321
    },
    {
      "count": 30,
      "mad": 7791.5,
      "max": 191689.0,
      "median": 163986.0,
      "min": 148907.0,
      "relative_mad": 0.04751320234654178
    },
    {
      "count": 30,
      "mad": 3796.5,
      "max": 189183.0,
      "median": 158699.0,
      "min": 145883.0,
      "relative_mad": 0.023922646015412826
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 73.0,
    "max": 164059.0,
    "median": 163986.0,
    "min": 158699.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "single_invocation",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## global_memory.streaming

| field | value |
| --- | --- |
| launch | `pallas`  pallas {"memory_spaces": "HBM input/output; VMEM tile refs", "shape": [1024, 1024], "tile": [128, 128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 169246 ns |
| simulator_param | `unsupported` = — |
| concept | `global_memory.streaming` |

- launch.extras: `{"compile_ns": 55106013, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "50b4f98b04cc93a17cd3ad8a51f90ec1e4e61edae17701aea3f0b17a6389b610"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "20017d88c958b417d7b77dda49bfd0414a5ba81d39b13d515eab5b3ab6166883"}}, "dma_semaphores": false, "dynamic_argument_count": 1, "first_execution_ns": 795805, "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86", "interpretation": "controlled tiled stream; physical HBM traffic unmeasured", "kind": "pallas_stream", "logical_flops": 2097152, "matrix": false, "memory_spaces": "HBM input/output; VMEM tile refs", "mode": "pallas", "probe_source_sha256": "f37c61f5e6bfe3c35795deab11621c2a07c691cccfb85dc9d53cf9d5743adae7", "requested_bytes": 8388608, "shape": [1024, 1024], "staging_ns": 11556735, "tile": [128, 128]}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | global_memory.streaming/1 |
| `effective_bytes_per_second` | 4.95644e+10 |
| `effective_flops_per_second` | 1.23911e+10 |
| `probe_id` | global_memory.streaming |
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
  "compile_ns": 55106013,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "50b4f98b04cc93a17cd3ad8a51f90ec1e4e61edae17701aea3f0b17a6389b610"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "20017d88c958b417d7b77dda49bfd0414a5ba81d39b13d515eab5b3ab6166883"
    }
  },
  "dma_semaphores": false,
  "dynamic_argument_count": 1,
  "first_execution_ns": 795805,
  "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86",
  "interpretation": "controlled tiled stream; physical HBM traffic unmeasured",
  "kind": "pallas_stream",
  "logical_flops": 2097152,
  "matrix": false,
  "memory_spaces": "HBM input/output; VMEM tile refs",
  "mode": "pallas",
  "probe_source_sha256": "f37c61f5e6bfe3c35795deab11621c2a07c691cccfb85dc9d53cf9d5743adae7",
  "requested_bytes": 8388608,
  "shape": [
    1024,
    1024
  ],
  "staging_ns": 11556735,
  "tile": [
    128,
    128
  ]
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "global_memory.streaming",
  "shape": [
    1024,
    1024
  ],
  "tile": [
    128,
    128
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
  "median_ns_per_invocation": 169246.5,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 2755.0,
      "max": 192954.0,
      "median": 171840.5,
      "min": 163605.0,
      "relative_mad": 0.016032309030758175
    },
    {
      "count": 30,
      "mad": 8387.0,
      "max": 206670.0,
      "median": 169246.5,
      "min": 139091.0,
      "relative_mad": 0.0495549390977066
    },
    {
      "count": 30,
      "mad": 2759.5,
      "max": 161241.0,
      "median": 143049.5,
      "min": 138794.0,
      "relative_mad": 0.01929052530767322
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 2594.0,
    "max": 171840.5,
    "median": 169246.5,
    "min": 143049.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "single_invocation",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## local_memory.transfer_sweep

| field | value |
| --- | --- |
| launch | `pallas`  pallas {"memory_spaces": "HBM input/output; VMEM tile refs", "shape": [1024, 1024], "tile": [256, 128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 144031 ns |
| simulator_param | `unsupported` = — |
| concept | `local_memory.transfer_sweep` |

- launch.extras: `{"compile_ns": 56437364, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "069cdb320c4d0bf838b6d258a11e9619815f7ec11de8d18d9812444f4948c030"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "488e66a774559beb10b1be8fbb4bf878dae011d24ade50aaa35a9860bc40d633"}}, "dma_semaphores": true, "dynamic_argument_count": 1, "first_execution_ns": 1105892, "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86", "interpretation": "controlled tiled stream; physical HBM traffic unmeasured", "kind": "pallas_dma", "logical_flops": 2097152, "matrix": false, "memory_spaces": "HBM input/output; VMEM tile refs", "mode": "pallas", "probe_source_sha256": "f37c61f5e6bfe3c35795deab11621c2a07c691cccfb85dc9d53cf9d5743adae7", "requested_bytes": 8388608, "shape": [1024, 1024], "staging_ns": 10444145, "tile": [256, 128]}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | local_memory.transfer_sweep/2 |
| `effective_bytes_per_second` | 5.82417e+10 |
| `effective_flops_per_second` | 1.45604e+10 |
| `probe_id` | local_memory.transfer_sweep |
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
  "compile_ns": 56437364,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "069cdb320c4d0bf838b6d258a11e9619815f7ec11de8d18d9812444f4948c030"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "488e66a774559beb10b1be8fbb4bf878dae011d24ade50aaa35a9860bc40d633"
    }
  },
  "dma_semaphores": true,
  "dynamic_argument_count": 1,
  "first_execution_ns": 1105892,
  "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86",
  "interpretation": "controlled tiled stream; physical HBM traffic unmeasured",
  "kind": "pallas_dma",
  "logical_flops": 2097152,
  "matrix": false,
  "memory_spaces": "HBM input/output; VMEM tile refs",
  "mode": "pallas",
  "probe_source_sha256": "f37c61f5e6bfe3c35795deab11621c2a07c691cccfb85dc9d53cf9d5743adae7",
  "requested_bytes": 8388608,
  "shape": [
    1024,
    1024
  ],
  "staging_ns": 10444145,
  "tile": [
    256,
    128
  ]
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "local_memory.transfer_sweep",
  "shape": [
    1024,
    1024
  ],
  "tile": [
    256,
    128
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
  "median_ns_per_invocation": 144031.0,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 2871.0,
      "max": 154080.0,
      "median": 144267.5,
      "min": 139736.0,
      "relative_mad": 0.019900531997851214
    },
    {
      "count": 30,
      "mad": 1571.0,
      "max": 155106.0,
      "median": 142340.0,
      "min": 140025.0,
      "relative_mad": 0.011036953772657019
    },
    {
      "count": 30,
      "mad": 2873.0,
      "max": 151743.0,
      "median": 144031.0,
      "min": 138855.0,
      "relative_mad": 0.019947094722663872
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 236.5,
    "max": 144267.5,
    "median": 144031.0,
    "min": 142340.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "single_invocation",
  "warmup_iterations": 5
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `874916f3c9326480ef7bf4540c9b9388359895829645d022a0b5a1f9d3695847`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-final-verified/out/tpu/tpu-pallas-stream-v1/b040b4ec22a86b47eeb08823b6d0d7dbd26319f1504f907050e4d180f90736c7/native-fixed-pallas-001/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
