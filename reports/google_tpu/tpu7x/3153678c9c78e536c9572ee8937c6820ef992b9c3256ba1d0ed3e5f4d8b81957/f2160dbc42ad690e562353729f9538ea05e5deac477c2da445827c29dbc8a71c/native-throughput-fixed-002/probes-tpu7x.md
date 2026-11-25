# google_tpu / tpu7x / tpu7x — Probe Results

- Generated: native-throughput-fixed-002
- Device: TPU7x  ·  Backend: `tpu_jax`  ·  Probes: 12
- `fit_status`: `direct`=12
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[topology.device_attributes](#topologydevice_attributes) · [runtime.dispatch_latency](#runtimedispatch_latency) · [matrix.matmul_throughput](#matrixmatmul_throughput) · [matrix.matmul_throughput](#matrixmatmul_throughput) · [matrix.matmul_throughput](#matrixmatmul_throughput) · [vector.elementwise_throughput](#vectorelementwise_throughput) · [vector.elementwise_throughput](#vectorelementwise_throughput) · [transfer.host_device](#transferhost_device) · [transfer.host_device](#transferhost_device) · [matrix.matmul_throughput](#matrixmatmul_throughput) · [matrix.matmul_throughput](#matrixmatmul_throughput) · [matrix.dependent_chain](#matrixdependent_chain)

---

## topology.device_attributes

| field | value |
| --- | --- |
| launch | `metadata`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `topology.device_attributes` = _object_ (devices, generation, hardware_info, instance, instance_config_sha256, local_index, process_count, process_index, selected_device_id, target_config_digest, visibility) |
| simulator_param | `unsupported` = — |
| concept | `topology.device_attributes` |

- launch.extras: `{"compiler_artifacts": {}, "devices": [{"addressable": true, "coords": [0, 0, 0], "core_on_chip": 0, "id": 0, "kind": "TPU7x", "memory_stats": {"bytes_in_use": 65536, "bytes_limit": 101724450816, "bytes_reservable_limit": 101724450816, "bytes_reserved": 0, "largest_alloc_size": 64000, "largest_free_block_bytes": 101724385280, "num_allocs": 2, "peak_bytes_in_use": 65536, "peak_bytes_reserved": 0}, "name": "TPU7x", "process_index": 0}, {"addressable": true, "coords": [0, 0, 0], "core_on_chip": 1, "id": 1, "kind": "TPU7x", "memory_stats": {"bytes_in_use": 65536, "bytes_limit": 101724450816, "bytes_reservable_limit": 101724450816, "bytes_reserved": 0, "largest_alloc_size": 64000, "largest_free_block_bytes": 101724385280, "num_allocs": 2, "peak_bytes_in_use": 65536, "peak_bytes_reserved": 0}, "name": "TPU7x", "process_index": 0}, {"addressable": true, "coords": [1, 0, 0], "core_on_chip": 0, "id": 2, "kind": "TPU7x", "memory_stats": {"bytes_in_use": 65536, "bytes_limit": 101724450816, "bytes_reservable_limit": 101724450816, "bytes_reserved": 0, "largest_alloc_size": 64000, "largest_free_block_bytes": 101724385280, "num_allocs": 2, "peak_bytes_in_use": 65536, "peak_bytes_reserved": 0}, "name": "TPU7x", "process_index": 0}, {"addressable": true, "coords": [1, 0, 0], "core_on_chip": 1, "id": 3, "kind": "TPU7x", "memory_stats": {"bytes_in_use": 65536, "bytes_limit": 101724450816, "bytes_reservable_limit": 101724450816, "bytes_reserved": 0, "largest_alloc_size": 64000, "largest_free_block_bytes": 101724385280, "num_allocs": 2, "peak_bytes_in_use": 65536, "peak_bytes_reserved": 0}, "name": "TPU7x", "process_index": 0}, {"addressable": true, "coords": [0, 1, 0], "core_on_chip": 0, "id": 4, "kind": "TPU7x", "memory_stats": {"bytes_in_use": 65536, "bytes_limit": 101724450816, "bytes_reservable_limit": 101724450816, "bytes_reserved": 0, "largest_alloc_size": 64000, "largest_free_block_bytes": 101724385280, "num_allocs": 2, "peak_bytes_in_use": 65536, "peak_bytes_reserved": 0}, "name": "TPU7x", "process_index": 0}, {"addressable": true, "coords": [0, 1, 0], "core_on_chip": 1, "id": 5, "kind": "TPU7x", "memory_stats": {"bytes_in_use": 65536, "bytes_limit": 101724450816, "bytes_reservable_limit": 101724450816, "bytes_reserved": 0, "largest_alloc_size": 64000, "largest_free_block_bytes": 101724385280, "num_allocs": 2, "peak_bytes_in_use": 65536, "peak_bytes_reserved": 0}, "name": "TPU7x", "process_index": 0}, {"addressable": true, "coords": [1, 1, 0], "core_on_chip": 0, "id": 6, "kind": "TPU7x", "memory_stats": {"bytes_in_use": 65536, "bytes_limit": 101724450816, "bytes_reservable_limit": 101724450816, "bytes_reserved": 0, "largest_alloc_size": 64000, "largest_free_block_bytes": 101724385280, "num_allocs": 2, "peak_bytes_in_use": 65536, "peak_bytes_reserved": 0}, "name": "TPU7x", "process_index": 0}, {"addressable": true, "coords": [1, 1, 0], "core_on_chip": 1, "id": 7, "kind": "TPU7x", "memory_stats": {"bytes_in_use": 65536, "bytes_limit": 101724450816, "bytes_reservable_limit": 101724450816, "bytes_reserved": 0, "largest_alloc_size": 64000, "largest_free_block_bytes": 101724385280, "num_allocs": 2, "peak_bytes_in_use": 65536, "peak_bytes_reserved": 0}, "name": "TPU7x", "process_index": 0}], "generation": "tpu7x", "hardware_info": {"chip_version": "7x", "hbm_capacity_bytes": 103000000000, "mxu_column_size": 256, "num_cores": 1, "num_lanes": 128, "num_mxus": 2, "num_sublanes": 8, "smem_capacity_bytes": 1048576, "source": "JAX TPU registry resolved by runtime identity; architecture metadata", "vmem_capacity_bytes": 67108864}, "instance": "gke-tpu-f20b10b7-2pbv", "instance_config_sha256": "be356a51bebe34edb3a42c66ba54e3c26c16c00ca96aa1a5cd98b93bfc914c8d", "local_index": 0, "process_count": 1, "process_index": 0, "selected_device_id": 0, "target_config_digest": "3153678c9c78e536c9572ee8937c6820ef992b9c3256ba1d0ed3e5f4d8b81957", "visibility": {"TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1", "TPU_HOST_BOUNDS": "1,1,1", "TPU_LOG_DIR": "/home/maas/amora-tpu-20261116-1659/out/tpu/runtime-logs", "TPU_SKIP_MDS_QUERY": "1", "TPU_TOPOLOGY": "2x2x1", "TPU_WORKER_HOSTNAMES": "localhost", "TPU_WORKER_ID": "0"}}`
- mapping_contract: unsupported

### Measurement value

<details><summary>value (JSON)</summary>

```json
{
  "devices": [
    {
      "addressable": true,
      "coords": [
        0,
        0,
        0
      ],
      "core_on_chip": 0,
      "id": 0,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        0,
        0,
        0
      ],
      "core_on_chip": 1,
      "id": 1,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        1,
        0,
        0
      ],
      "core_on_chip": 0,
      "id": 2,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        1,
        0,
        0
      ],
      "core_on_chip": 1,
      "id": 3,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        0,
        1,
        0
      ],
      "core_on_chip": 0,
      "id": 4,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        0,
        1,
        0
      ],
      "core_on_chip": 1,
      "id": 5,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        1,
        1,
        0
      ],
      "core_on_chip": 0,
      "id": 6,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        1,
        1,
        0
      ],
      "core_on_chip": 1,
      "id": 7,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    }
  ],
  "generation": "tpu7x",
  "hardware_info": {
    "chip_version": "7x",
    "hbm_capacity_bytes": 103000000000,
    "mxu_column_size": 256,
    "num_cores": 1,
    "num_lanes": 128,
    "num_mxus": 2,
    "num_sublanes": 8,
    "smem_capacity_bytes": 1048576,
    "source": "JAX TPU registry resolved by runtime identity; architecture metadata",
    "vmem_capacity_bytes": 67108864
  },
  "instance": "gke-tpu-f20b10b7-2pbv",
  "instance_config_sha256": "be356a51bebe34edb3a42c66ba54e3c26c16c00ca96aa1a5cd98b93bfc914c8d",
  "local_index": 0,
  "process_count": 1,
  "process_index": 0,
  "selected_device_id": 0,
  "target_config_digest": "3153678c9c78e536c9572ee8937c6820ef992b9c3256ba1d0ed3e5f4d8b81957",
  "visibility": {
    "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_LOG_DIR": "/home/maas/amora-tpu-20261116-1659/out/tpu/runtime-logs",
    "TPU_SKIP_MDS_QUERY": "1",
    "TPU_TOPOLOGY": "2x2x1",
    "TPU_WORKER_HOSTNAMES": "localhost",
    "TPU_WORKER_ID": "0"
  }
}
```

</details>

### Raw values

| key | value |
| --- | --- |
| `case_id` | topology.device_attributes/0 |
| `probe_id` | topology.device_attributes |
| `status` | passed |

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "devices": [
    {
      "addressable": true,
      "coords": [
        0,
        0,
        0
      ],
      "core_on_chip": 0,
      "id": 0,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        0,
        0,
        0
      ],
      "core_on_chip": 1,
      "id": 1,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        1,
        0,
        0
      ],
      "core_on_chip": 0,
      "id": 2,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        1,
        0,
        0
      ],
      "core_on_chip": 1,
      "id": 3,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        0,
        1,
        0
      ],
      "core_on_chip": 0,
      "id": 4,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        0,
        1,
        0
      ],
      "core_on_chip": 1,
      "id": 5,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        1,
        1,
        0
      ],
      "core_on_chip": 0,
      "id": 6,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    },
    {
      "addressable": true,
      "coords": [
        1,
        1,
        0
      ],
      "core_on_chip": 1,
      "id": 7,
      "kind": "TPU7x",
      "memory_stats": {
        "bytes_in_use": 65536,
        "bytes_limit": 101724450816,
        "bytes_reservable_limit": 101724450816,
        "bytes_reserved": 0,
        "largest_alloc_size": 64000,
        "largest_free_block_bytes": 101724385280,
        "num_allocs": 2,
        "peak_bytes_in_use": 65536,
        "peak_bytes_reserved": 0
      },
      "name": "TPU7x",
      "process_index": 0
    }
  ],
  "generation": "tpu7x",
  "hardware_info": {
    "chip_version": "7x",
    "hbm_capacity_bytes": 103000000000,
    "mxu_column_size": 256,
    "num_cores": 1,
    "num_lanes": 128,
    "num_mxus": 2,
    "num_sublanes": 8,
    "smem_capacity_bytes": 1048576,
    "source": "JAX TPU registry resolved by runtime identity; architecture metadata",
    "vmem_capacity_bytes": 67108864
  },
  "instance": "gke-tpu-f20b10b7-2pbv",
  "instance_config_sha256": "be356a51bebe34edb3a42c66ba54e3c26c16c00ca96aa1a5cd98b93bfc914c8d",
  "local_index": 0,
  "process_count": 1,
  "process_index": 0,
  "selected_device_id": 0,
  "target_config_digest": "3153678c9c78e536c9572ee8937c6820ef992b9c3256ba1d0ed3e5f4d8b81957",
  "visibility": {
    "TPU_CHIPS_PER_HOST_BOUNDS": "2,2,1",
    "TPU_HOST_BOUNDS": "1,1,1",
    "TPU_LOG_DIR": "/home/maas/amora-tpu-20261116-1659/out/tpu/runtime-logs",
    "TPU_SKIP_MDS_QUERY": "1",
    "TPU_TOPOLOGY": "2x2x1",
    "TPU_WORKER_HOSTNAMES": "localhost",
    "TPU_WORKER_ID": "0"
  }
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "topology.device_attributes"
}
```

</details>

[↑ contents](#contents)

---

## runtime.dispatch_latency

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [1]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 53115.6 ns |
| simulator_param | `unsupported` = — |
| concept | `runtime.dispatch_latency` |

- launch.extras: `{"compile_ns": 315415080, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0001/optimized_hlo.txt", "sha256": "8780a9e8d18a217d50cc6a21f05b78e68d35a49f371bc1d414dbaa67f29f3391"}, "stablehlo": {"path": "compiler/case-0001/stablehlo.txt", "sha256": "3ebf7ad527b00ecb5d0c7be73e95a221f43815713c64f2014bb99bc39ed952ff"}}, "dynamic_argument_count": 1, "first_execution_ns": 1135150, "input_digest": "53ef229298e1626f91e61c9c7b9e12a00bdd994d57f4fe201388524d9a63aa4f", "kind": "dispatch", "logical_flops": 2, "matrix": false, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 8, "shape": [1], "staging_ns": 5015466}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | runtime.dispatch_latency/1 |
| `effective_bytes_per_second` | 150615 |
| `effective_flops_per_second` | 37653.7 |
| `probe_id` | runtime.dispatch_latency |
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
        1
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
  "compile_ns": 315415080,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "8780a9e8d18a217d50cc6a21f05b78e68d35a49f371bc1d414dbaa67f29f3391"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3ebf7ad527b00ecb5d0c7be73e95a221f43815713c64f2014bb99bc39ed952ff"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 1135150,
  "input_digest": "53ef229298e1626f91e61c9c7b9e12a00bdd994d57f4fe201388524d9a63aa4f",
  "kind": "dispatch",
  "logical_flops": 2,
  "matrix": false,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 8,
  "shape": [
    1
  ],
  "staging_ns": 5015466
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "runtime.dispatch_latency"
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 53115.625,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 20460.0,
      "max": 1774577.0,
      "median": 1699700.0,
      "min": 1641839.0,
      "relative_mad": 0.012037418367947285
    },
    {
      "count": 30,
      "mad": 27844.0,
      "max": 1770351.0,
      "median": 1699464.0,
      "min": 1603112.0,
      "relative_mad": 0.01638398930486318
    },
    {
      "count": 30,
      "mad": 32731.0,
      "max": 1808822.0,
      "median": 1718459.5,
      "min": 1627957.0,
      "relative_mad": 0.01904671014940998
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 236.0,
    "max": 1718459.5,
    "median": 1699700.0,
    "min": 1699464.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## matrix.matmul_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [128, 128, 128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 54021.4 ns |
| simulator_param | `unsupported` = — |
| concept | `matrix.matmul_throughput` |

- launch.extras: `{"compile_ns": 33043190, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0002/optimized_hlo.txt", "sha256": "bb803c615e8f4cc2bcce738ee1445ee49d12b066af6071f5ce5f9bc10d8f17c1"}, "stablehlo": {"path": "compiler/case-0002/stablehlo.txt", "sha256": "e86ec563d754aa70c2e990b4e9bb14ddbcac89c683505bc049263f388d60047e"}}, "data_pattern": "random", "dynamic_argument_count": 2, "first_execution_ns": 645124, "input_digest": "2413574619ea1078bcbc9c83d54a7469c0f76f714605186fef8a46a35c6fc8ac", "kind": "matmul", "logical_flops": 4194304, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [128, 128, 128], "staging_ns": 33156957}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | matrix.matmul_throughput/2 |
| `effective_flops_per_second` | 7.76415e+10 |
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
  "compile_ns": 33043190,
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
  "first_execution_ns": 645124,
  "input_digest": "2413574619ea1078bcbc9c83d54a7469c0f76f714605186fef8a46a35c6fc8ac",
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
  "staging_ns": 33156957
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "matrix.matmul_throughput",
  "seed": 1,
  "shape": [
    128,
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
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 54021.40625,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 26029.5,
      "max": 2435631.0,
      "median": 1753419.5,
      "min": 1665208.0,
      "relative_mad": 0.01484499288390485
    },
    {
      "count": 30,
      "mad": 19008.5,
      "max": 1801691.0,
      "median": 1720671.5,
      "min": 1669808.0,
      "relative_mad": 0.01104714060760581
    },
    {
      "count": 30,
      "mad": 23501.5,
      "max": 1792378.0,
      "median": 1728685.0,
      "min": 1679640.0,
      "relative_mad": 0.013595015864660132
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 8013.5,
    "max": 1753419.5,
    "median": 1728685.0,
    "min": 1720671.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## matrix.matmul_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [256, 256, 256]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 54128.9 ns |
| simulator_param | `unsupported` = — |
| concept | `matrix.matmul_throughput` |

- launch.extras: `{"compile_ns": 39330050, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0003/optimized_hlo.txt", "sha256": "88e69a098b5f2eb492cb82197d3a854aeb4f42ea47f2079198f2b1690834dcf8"}, "stablehlo": {"path": "compiler/case-0003/stablehlo.txt", "sha256": "b13550e55fd9179bc35444030528fb4b6e6794f0ea1cbf7fde859b94e2fd1b16"}}, "data_pattern": "random", "dynamic_argument_count": 2, "first_execution_ns": 633999, "input_digest": "0e2500abc4830aa905b0896e12a0305f226a8c428b2a4de3c5dbf051aab8817a", "kind": "matmul", "logical_flops": 33554432, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [256, 256, 256], "staging_ns": 35083326}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | matrix.matmul_throughput/3 |
| `effective_flops_per_second` | 6.19899e+11 |
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
  "compile_ns": 39330050,
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
  "first_execution_ns": 633999,
  "input_digest": "0e2500abc4830aa905b0896e12a0305f226a8c428b2a4de3c5dbf051aab8817a",
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
  "staging_ns": 35083326
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "matrix.matmul_throughput",
  "seed": 1,
  "shape": [
    256,
    256,
    256
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 54128.859375,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 19403.5,
      "max": 1869326.0,
      "median": 1742519.0,
      "min": 1681282.0,
      "relative_mad": 0.011135316171588373
    },
    {
      "count": 30,
      "mad": 31170.0,
      "max": 2184875.0,
      "median": 1728986.5,
      "min": 1664495.0,
      "relative_mad": 0.018027902473501094
    },
    {
      "count": 30,
      "mad": 27785.0,
      "max": 1788069.0,
      "median": 1732123.5,
      "min": 1626754.0,
      "relative_mad": 0.01604100400462207
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 3137.0,
    "max": 1742519.0,
    "median": 1732123.5,
    "min": 1728986.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## matrix.matmul_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [511, 513, 257]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 54637.2 ns |
| simulator_param | `unsupported` = — |
| concept | `matrix.matmul_throughput` |

- launch.extras: `{"compile_ns": 86739975, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0004/optimized_hlo.txt", "sha256": "2e0423705d0b5ebcce46418043f4e530a19f905b71cd4c33f4de238da44d437f"}, "stablehlo": {"path": "compiler/case-0004/stablehlo.txt", "sha256": "f933c7b45bb06ec272a7c0bf7ff8b46cedd4af5ae9571f26f7776f2e15233180"}}, "data_pattern": "random", "dynamic_argument_count": 2, "first_execution_ns": 646936, "input_digest": "5ecd7fa4494eecfa488c0435c9dcbdddd95444e61c9a87635da1355040409c08", "kind": "matmul", "logical_flops": 134741502, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [511, 513, 257], "staging_ns": 83003030}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | matrix.matmul_throughput/4 |
| `effective_flops_per_second` | 2.46611e+12 |
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
        511,
        513
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
  "compile_ns": 86739975,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "2e0423705d0b5ebcce46418043f4e530a19f905b71cd4c33f4de238da44d437f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "f933c7b45bb06ec272a7c0bf7ff8b46cedd4af5ae9571f26f7776f2e15233180"
    }
  },
  "data_pattern": "random",
  "dynamic_argument_count": 2,
  "first_execution_ns": 646936,
  "input_digest": "5ecd7fa4494eecfa488c0435c9dcbdddd95444e61c9a87635da1355040409c08",
  "kind": "matmul",
  "logical_flops": 134741502,
  "matrix": true,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "shape": [
    511,
    513,
    257
  ],
  "staging_ns": 83003030
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "matrix.matmul_throughput",
  "seed": 1,
  "shape": [
    511,
    513,
    257
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 54637.21875,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 21002.0,
      "max": 1816791.0,
      "median": 1748391.0,
      "min": 1658116.0,
      "relative_mad": 0.012012187205264726
    },
    {
      "count": 30,
      "mad": 22230.5,
      "max": 1826867.0,
      "median": 1751257.0,
      "min": 1686122.0,
      "relative_mad": 0.012694024920385757
    },
    {
      "count": 30,
      "mad": 27956.0,
      "max": 1809059.0,
      "median": 1745669.5,
      "min": 1669861.0,
      "relative_mad": 0.016014486132684337
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 2721.5,
    "max": 1751257.0,
    "median": 1748391.0,
    "min": 1745669.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## vector.elementwise_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [128, 128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 53762.6 ns |
| simulator_param | `unsupported` = — |
| concept | `vector.elementwise_throughput` |

- launch.extras: `{"compile_ns": 30110745, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0005/optimized_hlo.txt", "sha256": "cfb8cf80687e18bdb44acb686bed67109520e5488575a7c4f39adc3802459be8"}, "stablehlo": {"path": "compiler/case-0005/stablehlo.txt", "sha256": "28861aeb095bd3b001bc90ccbcfabcf6d4e34344d0159c42cf1588e711554c37"}}, "dynamic_argument_count": 1, "first_execution_ns": 710239, "input_digest": "106d4049a8a3205e6c2463b6e6f6e9fa46097ea981986162e0fb87099530c329", "kind": "vector", "logical_flops": 32768, "matrix": false, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 131072, "shape": [128, 128], "staging_ns": 765448}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | vector.elementwise_throughput/5 |
| `effective_bytes_per_second` | 2.43798e+09 |
| `effective_flops_per_second` | 6.09495e+08 |
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
  "compile_ns": 30110745,
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
  "first_execution_ns": 710239,
  "input_digest": "106d4049a8a3205e6c2463b6e6f6e9fa46097ea981986162e0fb87099530c329",
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
  "staging_ns": 765448
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "vector.elementwise_throughput",
  "seed": 2,
  "shape": [
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
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 53762.578125,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 15845.0,
      "max": 1769213.0,
      "median": 1720402.5,
      "min": 1668140.0,
      "relative_mad": 0.009210054042585965
    },
    {
      "count": 30,
      "mad": 29065.0,
      "max": 1773892.0,
      "median": 1732140.5,
      "min": 1644846.0,
      "relative_mad": 0.016779816648822655
    },
    {
      "count": 30,
      "mad": 13429.0,
      "max": 1756500.0,
      "median": 1709834.5,
      "min": 1637227.0,
      "relative_mad": 0.007853976510592107
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 10568.0,
    "max": 1732140.5,
    "median": 1720402.5,
    "min": 1709834.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## vector.elementwise_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [1024, 1024]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 53410.1 ns |
| simulator_param | `unsupported` = — |
| concept | `vector.elementwise_throughput` |

- launch.extras: `{"compile_ns": 152094588, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0006/optimized_hlo.txt", "sha256": "ecbb1de483665c291a789847f17068b2483bd0ab606e772029e293d15dbf31e5"}, "stablehlo": {"path": "compiler/case-0006/stablehlo.txt", "sha256": "3c9a3b5f8145d09d8212220f87675b9366609c66ded117bf9b5c9c6a1248b75a"}}, "dynamic_argument_count": 1, "first_execution_ns": 866278, "input_digest": "673d16fca96ea0b8ca391ef984f1e5d254f986299862bac945a0a9dfc64b531e", "kind": "vector", "logical_flops": 2097152, "matrix": false, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 8388608, "shape": [1024, 1024], "staging_ns": 11113478}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | vector.elementwise_throughput/6 |
| `effective_bytes_per_second` | 1.5706e+11 |
| `effective_flops_per_second` | 3.92651e+10 |
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
  "compile_ns": 152094588,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "ecbb1de483665c291a789847f17068b2483bd0ab606e772029e293d15dbf31e5"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "3c9a3b5f8145d09d8212220f87675b9366609c66ded117bf9b5c9c6a1248b75a"
    }
  },
  "dynamic_argument_count": 1,
  "first_execution_ns": 866278,
  "input_digest": "673d16fca96ea0b8ca391ef984f1e5d254f986299862bac945a0a9dfc64b531e",
  "kind": "vector",
  "logical_flops": 2097152,
  "matrix": false,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 8388608,
  "shape": [
    1024,
    1024
  ],
  "staging_ns": 11113478
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "vector.elementwise_throughput",
  "seed": 2,
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
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 53410.0625,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 17809.5,
      "max": 2498166.0,
      "median": 1715980.0,
      "min": 1657793.0,
      "relative_mad": 0.010378617466404037
    },
    {
      "count": 30,
      "mad": 15586.5,
      "max": 1787154.0,
      "median": 1708133.5,
      "min": 1640911.0,
      "relative_mad": 0.009124872265545989
    },
    {
      "count": 30,
      "mad": 21623.5,
      "max": 1784709.0,
      "median": 1709122.0,
      "min": 1662506.0,
      "relative_mad": 0.012651817716933023
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 988.5,
    "max": 1715980.0,
    "median": 1709122.0,
    "min": 1708133.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## transfer.host_device

| field | value |
| --- | --- |
| launch | `transfer`  transfer {"shape": [128, 128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 67514.6 ns |
| simulator_param | `unsupported` = — |
| concept | `transfer.host_device` |

- launch.extras: `{"compiler_artifacts": {}, "device_copy_included": false, "direction": "host_to_device", "first_execution_ns": 266899, "host_cache_policy": "fresh device buffer for every D2H invocation", "input_digest": "c11094a14c100bdc43048a4d30b8bacc528cb6c3bac4755216d3ca554a0dad6a", "kind": "transfer", "logical_flops": 32768, "matrix": false, "mode": "transfer", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 65536, "shape": [128, 128], "staging_ns": 806131}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | transfer.host_device/7 |
| `effective_bytes_per_second` | 9.70694e+08 |
| `effective_flops_per_second` | 4.85347e+08 |
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
  "device_copy_included": false,
  "direction": "host_to_device",
  "first_execution_ns": 266899,
  "host_cache_policy": "fresh device buffer for every D2H invocation",
  "input_digest": "c11094a14c100bdc43048a4d30b8bacc528cb6c3bac4755216d3ca554a0dad6a",
  "kind": "transfer",
  "logical_flops": 32768,
  "matrix": false,
  "mode": "transfer",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "requested_bytes": 65536,
  "shape": [
    128,
    128
  ],
  "staging_ns": 806131
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "transfer.host_device",
  "seed": 3,
  "shape": [
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
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 67514.59375,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 22948.0,
      "max": 2246313.0,
      "median": 2154097.5,
      "min": 2100915.0,
      "relative_mad": 0.010653185382741496
    },
    {
      "count": 30,
      "mad": 27409.5,
      "max": 2219931.0,
      "median": 2160467.0,
      "min": 2093297.0,
      "relative_mad": 0.012686840391452403
    },
    {
      "count": 30,
      "mad": 17864.0,
      "max": 2211725.0,
      "median": 2169478.5,
      "min": 2125694.0,
      "relative_mad": 0.008234236937586613
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 6369.5,
    "max": 2169478.5,
    "median": 2160467.0,
    "min": 2154097.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## transfer.host_device

| field | value |
| --- | --- |
| launch | `transfer`  transfer {"shape": [1024, 1024]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 891505 ns |
| simulator_param | `unsupported` = — |
| concept | `transfer.host_device` |

- launch.extras: `{"compiler_artifacts": {}, "device_copy_included": false, "direction": "host_to_device", "first_execution_ns": 1729941, "host_cache_policy": "fresh device buffer for every D2H invocation", "input_digest": "b7a8f7247102833f9aab3a204baa10bdb37eb9cff35fcca27cf0cc84b600d2a6", "kind": "transfer", "logical_flops": 2097152, "matrix": false, "mode": "transfer", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "requested_bytes": 4194304, "shape": [1024, 1024], "staging_ns": 10801134}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | transfer.host_device/8 |
| `effective_bytes_per_second` | 4.70474e+09 |
| `effective_flops_per_second` | 2.35237e+09 |
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
  "first_execution_ns": 1729941,
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
  "staging_ns": 10801134
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
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
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 891505.25,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 136150.5,
      "max": 29052130.0,
      "median": 28528168.0,
      "min": 28127221.0,
      "relative_mad": 0.004772493628052106
    },
    {
      "count": 30,
      "mad": 118195.0,
      "max": 30314691.0,
      "median": 28491386.0,
      "min": 28179323.0,
      "relative_mad": 0.0041484468323162655
    },
    {
      "count": 30,
      "mad": 218494.0,
      "max": 29308954.0,
      "median": 28543439.5,
      "min": 28055141.0,
      "relative_mad": 0.00765478876503303
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 15271.5,
    "max": 28543439.5,
    "median": 28528168.0,
    "min": 28491386.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## matrix.matmul_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [1024, 1024, 1024]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 53655.8 ns |
| simulator_param | `unsupported` = — |
| concept | `matrix.matmul_throughput` |

- launch.extras: `{"compile_ns": 364918221, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0009/optimized_hlo.txt", "sha256": "60f96b43a437d0a749a239fb1224ed0c3dc6e7076a4a2672c711c2b7232e1c9e"}, "stablehlo": {"path": "compiler/case-0009/stablehlo.txt", "sha256": "aa478b37735f811cbc8b76f019f4622035a5bd57d07e6f0840dcb33916a98abb"}}, "data_pattern": "random", "dynamic_argument_count": 2, "first_execution_ns": 806892, "input_digest": "e0bbde64d54662c0614a34bd59280a8330db16841d87a23503591d7ec90af639", "kind": "matmul", "logical_flops": 2147483648, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [1024, 1024, 1024], "staging_ns": 179622512}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | matrix.matmul_throughput/9 |
| `effective_flops_per_second` | 4.00233e+13 |
| `probe_id` | matrix.matmul_throughput |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 1.9073486328125e-06,
      "passed": true,
      "shape": [
        1024,
        1024
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
  "compile_ns": 364918221,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "60f96b43a437d0a749a239fb1224ed0c3dc6e7076a4a2672c711c2b7232e1c9e"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "aa478b37735f811cbc8b76f019f4622035a5bd57d07e6f0840dcb33916a98abb"
    }
  },
  "data_pattern": "random",
  "dynamic_argument_count": 2,
  "first_execution_ns": 806892,
  "input_digest": "e0bbde64d54662c0614a34bd59280a8330db16841d87a23503591d7ec90af639",
  "kind": "matmul",
  "logical_flops": 2147483648,
  "matrix": true,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "shape": [
    1024,
    1024,
    1024
  ],
  "staging_ns": 179622512
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "matrix.matmul_throughput",
  "seed": 1,
  "shape": [
    1024,
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
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 53655.8125,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 19286.0,
      "max": 1819567.0,
      "median": 1726037.5,
      "min": 1663793.0,
      "relative_mad": 0.01117356951978158
    },
    {
      "count": 30,
      "mad": 24966.5,
      "max": 1846592.0,
      "median": 1716986.0,
      "min": 1646511.0,
      "relative_mad": 0.014540887345616097
    },
    {
      "count": 30,
      "mad": 23874.0,
      "max": 1810593.0,
      "median": 1702923.5,
      "min": 1646057.0,
      "relative_mad": 0.014019420132495675
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 9051.5,
    "max": 1726037.5,
    "median": 1716986.0,
    "min": 1702923.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## matrix.matmul_throughput

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [2048, 2048, 2048]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 53769.5 ns |
| simulator_param | `unsupported` = — |
| concept | `matrix.matmul_throughput` |

- launch.extras: `{"compile_ns": 1296429876, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0010/optimized_hlo.txt", "sha256": "bbc1e4f1186862e83fa0af51ad2ab84f81707f834b752c60583ce10ea47d2b07"}, "stablehlo": {"path": "compiler/case-0010/stablehlo.txt", "sha256": "72fd74a55150cc10a58f0d60f5010f7a7691b3443893373830dc402069b27212"}}, "data_pattern": "random", "dynamic_argument_count": 2, "first_execution_ns": 1894710, "input_digest": "fe62b59d7223503fdd042f45c3faa2ef2c11c4068b108f29bc416ce25032f0d0", "kind": "matmul", "logical_flops": 17179869184, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [2048, 2048, 2048], "staging_ns": 325693088}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | matrix.matmul_throughput/10 |
| `effective_flops_per_second` | 3.1951e+14 |
| `probe_id` | matrix.matmul_throughput |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
  "outputs": [
    {
      "dtype": "float32",
      "max_abs_error": 3.814697265625e-06,
      "passed": true,
      "shape": [
        2048,
        2048
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
  "compile_ns": 1296429876,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "bbc1e4f1186862e83fa0af51ad2ab84f81707f834b752c60583ce10ea47d2b07"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "72fd74a55150cc10a58f0d60f5010f7a7691b3443893373830dc402069b27212"
    }
  },
  "data_pattern": "random",
  "dynamic_argument_count": 2,
  "first_execution_ns": 1894710,
  "input_digest": "fe62b59d7223503fdd042f45c3faa2ef2c11c4068b108f29bc416ce25032f0d0",
  "kind": "matmul",
  "logical_flops": 17179869184,
  "matrix": true,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "shape": [
    2048,
    2048,
    2048
  ],
  "staging_ns": 325693088
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "probe_id": "matrix.matmul_throughput",
  "seed": 1,
  "shape": [
    2048,
    2048,
    2048
  ]
}
```

</details>

<details><summary><code>timing</code> (JSON)</summary>

```json
{
  "allocation_policy": "outputs allocated by executable; all outputs retained until drain",
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 53769.46875,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 18049.5,
      "max": 1795465.0,
      "median": 1723431.0,
      "min": 1647801.0,
      "relative_mad": 0.010473004141157958
    },
    {
      "count": 30,
      "mad": 15232.5,
      "max": 1797924.0,
      "median": 1720623.0,
      "min": 1667829.0,
      "relative_mad": 0.008852898049136854
    },
    {
      "count": 30,
      "mad": 27779.5,
      "max": 1781994.0,
      "median": 1716299.0,
      "min": 1663077.0,
      "relative_mad": 0.016185699577987284
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 2808.0,
    "max": 1723431.0,
    "median": 1720623.0,
    "min": 1716299.0
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## matrix.dependent_chain

| field | value |
| --- | --- |
| launch | `jax_jit`  jax_jit {"shape": [128, 128, 128]} |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `dispatch_to_completion_ns` = 55283 ns |
| simulator_param | `unsupported` = — |
| concept | `matrix.dependent_chain` |

- launch.extras: `{"compile_ns": 42787304, "compiler_artifacts": {"optimized_hlo": {"path": "compiler/case-0011/optimized_hlo.txt", "sha256": "53ba9e6bbaab872f9e230951c663db638992510d3b5bd667159128e291d07d9f"}, "stablehlo": {"path": "compiler/case-0011/stablehlo.txt", "sha256": "6cb1a98e7a94aa8122e5b5de70abffc19652f4f9d8033952864ed472aa611281"}}, "data_pattern": "dyadic_hadamard", "dynamic_argument_count": 2, "first_execution_ns": 758538, "input_digest": "694932a40cd61e535ade8e7444c6fc75be554c00619ca191001be732c979364f", "kind": "matrix_chain", "logical_flops": 16777216, "matrix": true, "mode": "jax_jit", "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d", "shape": [128, 128, 128], "staging_ns": 3337811}`
- mapping_contract: unsupported

**assumptions:**

- includes host dispatch and completion overhead
- IR/correctness do not prove ISA instruction counts or physical memory traffic

### Raw values

| key | value |
| --- | --- |
| `case_id` | matrix.dependent_chain/11 |
| `effective_flops_per_second` | 3.03478e+11 |
| `probe_id` | matrix.dependent_chain |
| `status` | passed |

<details><summary><code>correctness</code> (JSON)</summary>

```json
{
  "atol": 0.0005,
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
  "rtol": 0.005,
  "tolerance_policy": "portable-core-r1"
}
```

</details>

<details><summary><code>metadata</code> (JSON)</summary>

```json
{
  "compile_ns": 42787304,
  "compiler_artifacts": {
    "optimized_hlo": {
      "path": "optimized_hlo.txt",
      "sha256": "53ba9e6bbaab872f9e230951c663db638992510d3b5bd667159128e291d07d9f"
    },
    "stablehlo": {
      "path": "stablehlo.txt",
      "sha256": "6cb1a98e7a94aa8122e5b5de70abffc19652f4f9d8033952864ed472aa611281"
    }
  },
  "data_pattern": "dyadic_hadamard",
  "dynamic_argument_count": 2,
  "first_execution_ns": 758538,
  "input_digest": "694932a40cd61e535ade8e7444c6fc75be554c00619ca191001be732c979364f",
  "kind": "matrix_chain",
  "logical_flops": 16777216,
  "matrix": true,
  "mode": "jax_jit",
  "probe_source_sha256": "db6abd7366489b31320872bfdd7ed324a8926e7442aaf06ddcd6987db393727d",
  "shape": [
    128,
    128,
    128
  ],
  "staging_ns": 3337811
}
```

</details>

<details><summary><code>spec</code> (JSON)</summary>

```json
{
  "data_pattern": "dyadic_hadamard",
  "iterations": 4,
  "probe_id": "matrix.dependent_chain",
  "seed": 4,
  "shape": [
    128,
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
  "batch_size": 32,
  "input_donation": false,
  "measured_samples_per_repeat": 30,
  "median_ns_per_invocation": 55283.046875,
  "output_reuse": false,
  "repeat_summaries": [
    {
      "count": 30,
      "mad": 24235.0,
      "max": 1847788.0,
      "median": 1769057.5,
      "min": 1681204.0,
      "relative_mad": 0.013699385124564917
    },
    {
      "count": 30,
      "mad": 14245.0,
      "max": 1820334.0,
      "median": 1767761.5,
      "min": 1731202.0,
      "relative_mad": 0.008058213735280467
    },
    {
      "count": 30,
      "mad": 30314.0,
      "max": 1841785.0,
      "median": 1776615.5,
      "min": 1728474.0,
      "relative_mad": 0.01706278032584991
    }
  ],
  "repeats": 3,
  "stable": true,
  "summary": {
    "count": 3,
    "mad": 1296.0,
    "max": 1776615.5,
    "median": 1769057.5,
    "min": 1767761.5
  },
  "time_domain": "synchronized_host",
  "timed_unit": "batched_invocations",
  "warmup_invocations": 640,
  "warmup_iterations": 20,
  "warmup_policy": "same batch, output retention and drain as measured units"
}
```

</details>

[↑ contents](#contents)

---

## Source evidence

- Run digest: `f2068f6b91444bfec96673debca055eeec9374b747de7a6f9cdf604a5ab1c8bb`
- [Verified source manifest](../../../../../../out/tpu_backend_execution/evidence/native-final-verified/out/tpu/tpu-single-device-throughput-v1/644fb027ce3b6230b052e49dbc17c6144bab80b68a4563b51324178a8369b4ef/native-throughput-fixed-002/manifest.json)
- Compiler and sample paths in the JSON report are relative to the source evidence root.
