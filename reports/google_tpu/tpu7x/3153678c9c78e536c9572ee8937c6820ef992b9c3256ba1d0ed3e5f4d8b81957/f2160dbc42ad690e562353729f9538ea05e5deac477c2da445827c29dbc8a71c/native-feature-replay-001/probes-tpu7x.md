# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-feature-replay-001
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 8
- `fit_status`: `direct`=8
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[matrix.matmul_throughput](#matrixmatmul_throughput) · [transfer.host_device](#transferhost_device) · [transfer.host_device](#transferhost_device) · [global_memory.streaming](#global_memorystreaming) · [local_memory.transfer_sweep](#local_memorytransfer_sweep) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives) · [interconnect.collectives](#interconnectcollectives)

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

- launch.extras: `{"compile_ns": 42180452, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0000/optimized_hlo.txt", "sha256": "88e69a098b5f2eb492cb82197d3a854aeb4f42ea47f2079198f2b1690834dcf8"}, "stablehlo": {"path": "compiler/case-0000/stablehlo.txt", "sha256": "b13550e55fd9179bc35444030528fb4b6e6794f0ea1cbf7fde859b94e2fd1b16"}}, "data_pattern": "random", "dynamic_argument_count": 2, "first_execution_ns": 690960, "input_digest": "29a114021862e99351d8a6183bf7211d58804b7309e004aa3de0710b8d54d51f", "kind": "matmul", "logical_flops": 33554432, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [256, 256, 256], "staging_ns": 329887959}`
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
  "compile_ns": 42180452,
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
  "first_execution_ns": 690960,
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
  "staging_ns": 329887959
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

## transfer.host_device

| field | value |
| --- | --- |
| launch | `transfer`  transfer {"shape": [1024, 1024]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `transfer.host_device` |

- launch.extras: `{"compiler_artifacts": {}, "device_copy_included": false, "direction": "host_to_device", "first_execution_ns": 1494134, "host_cache_policy": "fresh device buffer for every D2H invocation", "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86", "kind": "transfer", "logical_flops": 2097152, "matrix": false, "mode": "transfer", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 4194304, "shape": [1024, 1024], "staging_ns": 11075029}`
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

### Raw values

| key | value |
| --- | --- |
| `case_id` | transfer.host_device/1 |
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
  "device_copy_included": false,
  "direction": "host_to_device",
  "first_execution_ns": 1494134,
  "host_cache_policy": "fresh device buffer for every D2H invocation",
  "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86",
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
  "staging_ns": 11075029
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "measure": false,
  "probe_id": "transfer.host_device",
  "shape": [
    1024,
    1024
  ]
}
```

</details>

[↑ contents](#contents)

---

## transfer.host_device

| field | value |
| --- | --- |
| launch | `transfer`  transfer {"shape": [1024, 1024]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `transfer.host_device` |

- launch.extras: `{"compiler_artifacts": {}, "device_copy_included": true, "direction": "device_to_host", "first_execution_ns": 3160034, "host_cache_policy": "fresh device buffer for every D2H invocation", "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86", "kind": "transfer", "logical_flops": 2097152, "matrix": false, "mode": "transfer", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 4194304, "shape": [1024, 1024], "staging_ns": 10138527}`
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

### Raw values

| key | value |
| --- | --- |
| `case_id` | transfer.host_device/2 |
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
  "first_execution_ns": 3160034,
  "host_cache_policy": "fresh device buffer for every D2H invocation",
  "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86",
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
  "staging_ns": 10138527
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "direction": "device_to_host",
  "measure": false,
  "probe_id": "transfer.host_device",
  "shape": [
    1024,
    1024
  ]
}
```

</details>

[↑ contents](#contents)

---

## global_memory.streaming

| field | value |
| --- | --- |
| launch | `pallas`  pallas {"memory_spaces": "HBM input/output; VMEM tile refs", "shape": [1024, 1024], "tile": [128, 128]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `global_memory.streaming` |

- launch.extras: `{"compile_ns": 54254900, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "2ccd6ff1d6e12fcd4bb42c9f34a62918ac9649f7393d37e220f1cda04fb0e68b"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "1365f3c1778feacc79938eca1d12284ffa4c5fe4149296f3e13113f8ff86b6f0"}}, "dma_semaphores": false, "dynamic_argument_count": 1, "first_execution_ns": 755529, "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86", "interpretation": "controlled tiled stream; physical HBM traffic unmeasured", "kind": "pallas_stream", "logical_flops": 2097152, "matrix": false, "memory_spaces": "HBM input/output; VMEM tile refs", "mode": "pallas", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 8388608, "shape": [1024, 1024], "staging_ns": 10131607, "tile": [128, 128]}`
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

### Raw values

| key | value |
| --- | --- |
| `case_id` | global_memory.streaming/3 |
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
  "compile_ns": 54254900,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "2ccd6ff1d6e12fcd4bb42c9f34a62918ac9649f7393d37e220f1cda04fb0e68b"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1365f3c1778feacc79938eca1d12284ffa4c5fe4149296f3e13113f8ff86b6f0"
    }
  },
  "dma_semaphores": false,
  "dynamic_argument_count": 1,
  "first_execution_ns": 755529,
  "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86",
  "interpretation": "controlled tiled stream; physical HBM traffic unmeasured",
  "kind": "pallas_stream",
  "logical_flops": 2097152,
  "matrix": false,
  "memory_spaces": "HBM input/output; VMEM tile refs",
  "mode": "pallas",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 8388608,
  "shape": [
    1024,
    1024
  ],
  "staging_ns": 10131607,
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
  "measure": false,
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

[↑ contents](#contents)

---

## local_memory.transfer_sweep

| field | value |
| --- | --- |
| launch | `pallas`  pallas {"memory_spaces": "HBM input/output; VMEM tile refs", "shape": [1024, 1024], "tile": [256, 128]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `local_memory.transfer_sweep` |

- launch.extras: `{"compile_ns": 55588509, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "0f79137b4f0c1b4f9ad3caffda835fe46fd6ce16f59a8d663a69b5e123e74a4a"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "f9a2a5a7ee25a53a867f51a0cf61ebea5595b5c2cba8a065eb8d2b755ec0d1c4"}}, "dma_semaphores": true, "dynamic_argument_count": 1, "first_execution_ns": 1046666, "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86", "interpretation": "controlled tiled stream; physical HBM traffic unmeasured", "kind": "pallas_dma", "logical_flops": 2097152, "matrix": false, "memory_spaces": "HBM input/output; VMEM tile refs", "mode": "pallas", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 8388608, "shape": [1024, 1024], "staging_ns": 10375983, "tile": [256, 128]}`
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

### Raw values

| key | value |
| --- | --- |
| `case_id` | local_memory.transfer_sweep/4 |
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
  "compile_ns": 55588509,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "0f79137b4f0c1b4f9ad3caffda835fe46fd6ce16f59a8d663a69b5e123e74a4a"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "f9a2a5a7ee25a53a867f51a0cf61ebea5595b5c2cba8a065eb8d2b755ec0d1c4"
    }
  },
  "dma_semaphores": true,
  "dynamic_argument_count": 1,
  "first_execution_ns": 1046666,
  "input_digest": "2522cd808b31128e3a0bb9832f105580545b9b77e092e8a695afa3d6e04b4c86",
  "interpretation": "controlled tiled stream; physical HBM traffic unmeasured",
  "kind": "pallas_dma",
  "logical_flops": 2097152,
  "matrix": false,
  "memory_spaces": "HBM input/output; VMEM tile refs",
  "mode": "pallas",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 8388608,
  "shape": [
    1024,
    1024
  ],
  "staging_ns": 10375983,
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
  "measure": false,
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

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [128]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 50377555, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "8b42bf6cc3ba328d2df034de3e76e926b81bcefecd47535a67b785b39eba97af"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "1cfe0acb37125b1eb475cd2a7df2cdbf1cab70e279770fbe60a4f4b3bcc6e376"}}, "dynamic_argument_count": 1, "first_execution_ns": 1667744, "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1", "kind": "collective", "matrix": false, "mode": "shard_map", "participant_coordinates": [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0]], "participant_device_ids": [0, 1, 2, 3, 4, 5, 6, 7], "participants": 8, "primitive": "psum", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 4096, "shape": [128], "staging_ns": 1233059}`
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
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        8,
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
| `case_id` | interconnect.collectives/5 |
| `probe_id` | interconnect.collectives |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 2e-05,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 2.384185791015625e-07,
      "passed": true,
      "shape": [
        8,
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
  "compile_ns": 50377555,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8b42bf6cc3ba328d2df034de3e76e926b81bcefecd47535a67b785b39eba97af"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "1cfe0acb37125b1eb475cd2a7df2cdbf1cab70e279770fbe60a4f4b3bcc6e376"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1667744,
  "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participant_coordinates": [
    [
      0,
      0,
      0
    ],
    [
      0,
      0,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      1,
      0
    ],
    [
      1,
      1,
      0
    ]
  ],
  "participant_device_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "participants": 8,
  "primitive": "psum",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 4096,
  "shape": [
    128
  ],
  "staging_ns": 1233059
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "measure": false,
  "primitive": "psum",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [128]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 363221134, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "95570fab46bd4598b681fb4b4281a9dadb1debfc4dab45b2b62b65fdae595fb0"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "7fa03db27220f4d32a9ad1f778983c079789b45f5b5b015d7ff6bb93ee8c4248"}}, "dynamic_argument_count": 1, "first_execution_ns": 3580544, "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1", "kind": "collective", "matrix": false, "mode": "shard_map", "participant_coordinates": [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0]], "participant_device_ids": [0, 1, 2, 3, 4, 5, 6, 7], "participants": 8, "primitive": "all_gather", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 4096, "shape": [128], "staging_ns": 812641}`
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
        8,
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
| `case_id` | interconnect.collectives/6 |
| `probe_id` | interconnect.collectives |
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
        8,
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
  "compile_ns": 363221134,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "95570fab46bd4598b681fb4b4281a9dadb1debfc4dab45b2b62b65fdae595fb0"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "7fa03db27220f4d32a9ad1f778983c079789b45f5b5b015d7ff6bb93ee8c4248"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 3580544,
  "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participant_coordinates": [
    [
      0,
      0,
      0
    ],
    [
      0,
      0,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      1,
      0
    ],
    [
      1,
      1,
      0
    ]
  ],
  "participant_device_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "participants": 8,
  "primitive": "all_gather",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 4096,
  "shape": [
    128
  ],
  "staging_ns": 812641
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "measure": false,
  "primitive": "all_gather",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

[↑ contents](#contents)

---

## interconnect.collectives

| field | value |
| --- | --- |
| launch | `shard_map`  shard_map {"participants": 8, "shape": [128]} |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `correctness` = _object_ (atol, outputs, passed, rtol, tolerance_policy) |
| simulator_param | `unsupported` = — |
| concept | `interconnect.collectives` |

- launch.extras: `{"compile_ns": 45637360, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0007/optimized_hlo.txt", "sha256": "6a2fd1d1b63f869098c59e4c1b41c5f839b6dffcfa159ade20e4e89a33307ded"}, "stablehlo": {"path": "compiler/case-0007/stablehlo.txt", "sha256": "46524df99c2ecb0375c9b47af93f54ace87574618ba763c71c22733ef08325e6"}}, "dynamic_argument_count": 1, "first_execution_ns": 1161943, "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1", "kind": "collective", "matrix": false, "mode": "shard_map", "participant_coordinates": [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 1, 0], [1, 1, 0]], "participant_device_ids": [0, 1, 2, 3, 4, 5, 6, 7], "participants": 8, "primitive": "ppermute", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 4096, "routes": [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0]], "shape": [128], "staging_ns": 818812}`
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
        8,
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
| `case_id` | interconnect.collectives/7 |
| `probe_id` | interconnect.collectives |
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
        8,
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
  "compile_ns": 45637360,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "6a2fd1d1b63f869098c59e4c1b41c5f839b6dffcfa159ade20e4e89a33307ded"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "46524df99c2ecb0375c9b47af93f54ace87574618ba763c71c22733ef08325e6"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1161943,
  "input_digest": "0bbc182aab537fd8d320f1f1224997dba3a2aa5b0a12a7107d56ef53d6f011d1",
  "kind": "collective",
  "matrix": false,
  "mode": "shard_map",
  "participant_coordinates": [
    [
      0,
      0,
      0
    ],
    [
      0,
      0,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      1,
      0,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      0,
      1,
      0
    ],
    [
      1,
      1,
      0
    ],
    [
      1,
      1,
      0
    ]
  ],
  "participant_device_ids": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7
  ],
  "participants": 8,
  "primitive": "ppermute",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 4096,
  "routes": [
    [
      0,
      1
    ],
    [
      1,
      2
    ],
    [
      2,
      3
    ],
    [
      3,
      4
    ],
    [
      4,
      5
    ],
    [
      5,
      6
    ],
    [
      6,
      7
    ],
    [
      7,
      0
    ]
  ],
  "shape": [
    128
  ],
  "staging_ns": 818812
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "devices": 8,
  "measure": false,
  "primitive": "ppermute",
  "probe_id": "interconnect.collectives",
  "shape": [
    128
  ]
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `ac7bf4f44071bb54f7cbf2e7871bed9e9e1b6da0118d6232a3929bb35cde58ec`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-final-verified/out/tpu/tpu-feature-replay-v1/9fc89db13489142a3a4db85a37f568594af6a70b04f8cc14f2312c906d082a75/native-feature-replay-001/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
