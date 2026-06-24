# nvidia / hopper / h100-80g — Probe Results

- Generated: 2026-06-24T08:46Z
- Device: NVIDIA H100 80GB HBM3  ·  Backend: `nvidia_cuda`  ·  Probes: 18
- `fit_status`: `behavioral_only`=2, `conditionally_identified`=2, `direct`=5, `underconstrained`=5, `uniquely_identified`=4
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[topology.device_attributes](#topologydevice_attributes) · [topology.occupancy](#topologyoccupancy) · [topology.persistent_cta](#topologypersistent_cta) · [arithmetic_latency.dependent_chain](#arithmetic_latencydependent_chain) · [arithmetic_throughput.independent_chains](#arithmetic_throughputindependent_chains) · [shared_memory.pointer_chase](#shared_memorypointer_chase) · [shared_memory.bank_stride](#shared_memorybank_stride) · [shared_memory.analyze](#shared_memoryanalyze) · [l1_cache.pointer_chase](#l1_cachepointer_chase) · [l1_cache.working_set](#l1_cacheworking_set) · [l1_cache.conflict_sets](#l1_cacheconflict_sets) · [l1_cache.analyze](#l1_cacheanalyze) · [scheduler_policy.ready_warps](#scheduler_policyready_warps) · [scheduler_policy.mixed_issue](#scheduler_policymixed_issue) · [scheduler_policy.analyze](#scheduler_policyanalyze) · [register_file.register_bank_sweep](#register_fileregister_bank_sweep) · [register_file.register_latency](#register_fileregister_latency) · [register_file.analyze](#register_fileanalyze)

---

## topology.device_attributes

| field | value |
| --- | --- |
| launch | `metadata`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `cuda_device_identity` = _object_ (device_index, device_name, driver_version, uuid) |
| simulator_param | `device_identity` = _object_ (device_index, device_name, driver_version, uuid) |
| concept | `runtime_visible_device_identity` |

- interpretation: device identity is available; resource limits need CUDA API helper in the next cutline
- mapping_contract: identity metadata is recorded for traceability and is not a simulator structural parameter

**assumptions:**

- nvidia-smi identity metadata is treated as direct runtime metadata

### Measurement value

```json
{
  "device_index": 0,
  "device_name": "NVIDIA H100 80GB HBM3",
  "driver_version": "595.71.05",
  "uuid": "GPU-c80fe196-e3ba-82ff-7233-d9804f701864"
}
```

### Raw values

| key | value |
| --- | --- |
| `device_index` | 0 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `driver_version` | 595.71.05 |
| `uuid` | GPU-c80fe196-e3ba-82ff-7233-d9804f701864 |

[↑ contents](#contents)

---

## topology.occupancy

| field | value |
| --- | --- |
| launch | `planning`  — |
| evidence_tier | `direct_metadata` |
| fit_status | `direct` |
| measurement | `occupancy_sweep_plan` = _object_ (block_sizes, dynamic_shared_memory_bytes, point_count, registers_per_thread, sweep_points) |
| simulator_param | `launch_shape_sweep` = _object_ (block_sizes, dynamic_shared_memory_bytes, point_count, registers_per_thread, sweep_points) |
| concept | `cuda_launch_shape_grid` |

- interpretation: block/register/shared-memory cross-product for downstream occupancy fits
- mapping_contract: planning artifact; not a structural simulator parameter

**assumptions:**

- occupancy sweep is a planning artifact; resident-block fitting requires CUDA Occupancy API

### Measurement value

<details><summary>value (JSON)</summary>

```json
{
  "block_sizes": [
    32,
    64,
    128,
    256,
    512,
    1024
  ],
  "dynamic_shared_memory_bytes": [
    0,
    1024,
    8192,
    32768
  ],
  "point_count": 120,
  "registers_per_thread": [
    16,
    32,
    64,
    96,
    128
  ],
  "sweep_points": [
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 16,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 16,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 16,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 16,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 32,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 32,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 32,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 32,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 64,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 64,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 64,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 64,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 96,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 96,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 96,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 96,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 128,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 128,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 128,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 128,
      "threads_per_block": 32,
      "warps_per_block": 1
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 16,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 16,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 16,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 16,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 32,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 32,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 32,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 32,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 64,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 64,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 64,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 64,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 96,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 96,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 96,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 96,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 128,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 128,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 128,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 128,
      "threads_per_block": 64,
      "warps_per_block": 2
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 16,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 16,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 16,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 16,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 32,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 32,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 32,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 32,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 64,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 64,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 64,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 64,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 96,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 96,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 96,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 96,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 128,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 128,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 128,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 128,
      "threads_per_block": 128,
      "warps_per_block": 4
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 16,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 16,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 16,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 16,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 32,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 32,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 32,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 32,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 64,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 64,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 64,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 64,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 96,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 96,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 96,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 96,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 128,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 128,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 128,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 128,
      "threads_per_block": 256,
      "warps_per_block": 8
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 16,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 16,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 16,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 16,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 32,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 32,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 32,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 32,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 64,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 64,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 64,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 64,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 96,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 96,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 96,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 96,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 128,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 128,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 128,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 128,
      "threads_per_block": 512,
      "warps_per_block": 16
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 16,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 16,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 16,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 16,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 32,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 32,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 32,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 32,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 64,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 64,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 64,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 64,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 96,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 96,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 96,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 96,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 0,
      "registers_per_thread": 128,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 1024,
      "registers_per_thread": 128,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 8192,
      "registers_per_thread": 128,
      "threads_per_block": 1024,
      "warps_per_block": 32
    },
    {
      "dynamic_shared_memory_bytes": 32768,
      "registers_per_thread": 128,
      "threads_per_block": 1024,
      "warps_per_block": 32
    }
  ]
}
```

</details>

### Raw values

| key | value |
| --- | --- |
| `point_count` | 120 |

#### `sweep_points` (120 rows)

| dynamic_shared_memory_bytes | registers_per_thread | threads_per_block | warps_per_block |
| --- | --- | --- | --- |
| 0 | 16 | 32 | 1 |
| 1024 | 16 | 32 | 1 |
| 8192 | 16 | 32 | 1 |
| 32768 | 16 | 32 | 1 |
| 0 | 32 | 32 | 1 |
| 1024 | 32 | 32 | 1 |
| 8192 | 32 | 32 | 1 |
| 32768 | 32 | 32 | 1 |
| 0 | 64 | 32 | 1 |
| 1024 | 64 | 32 | 1 |
| 8192 | 64 | 32 | 1 |
| 32768 | 64 | 32 | 1 |
| 0 | 96 | 32 | 1 |
| 1024 | 96 | 32 | 1 |
| 8192 | 96 | 32 | 1 |
| 32768 | 96 | 32 | 1 |
| 0 | 128 | 32 | 1 |
| 1024 | 128 | 32 | 1 |
| 8192 | 128 | 32 | 1 |
| 32768 | 128 | 32 | 1 |
| 0 | 16 | 64 | 2 |
| 1024 | 16 | 64 | 2 |
| 8192 | 16 | 64 | 2 |
| 32768 | 16 | 64 | 2 |
| 0 | 32 | 64 | 2 |
| 1024 | 32 | 64 | 2 |
| 8192 | 32 | 64 | 2 |
| 32768 | 32 | 64 | 2 |
| 0 | 64 | 64 | 2 |
| 1024 | 64 | 64 | 2 |
| 8192 | 64 | 64 | 2 |
| 32768 | 64 | 64 | 2 |
| 0 | 96 | 64 | 2 |
| 1024 | 96 | 64 | 2 |
| 8192 | 96 | 64 | 2 |
| 32768 | 96 | 64 | 2 |
| 0 | 128 | 64 | 2 |
| 1024 | 128 | 64 | 2 |
| 8192 | 128 | 64 | 2 |
| 32768 | 128 | 64 | 2 |
| 0 | 16 | 128 | 4 |
| 1024 | 16 | 128 | 4 |
| 8192 | 16 | 128 | 4 |
| 32768 | 16 | 128 | 4 |
| 0 | 32 | 128 | 4 |
| 1024 | 32 | 128 | 4 |
| 8192 | 32 | 128 | 4 |
| 32768 | 32 | 128 | 4 |
| 0 | 64 | 128 | 4 |
| 1024 | 64 | 128 | 4 |
| 8192 | 64 | 128 | 4 |
| 32768 | 64 | 128 | 4 |
| 0 | 96 | 128 | 4 |
| 1024 | 96 | 128 | 4 |
| 8192 | 96 | 128 | 4 |
| 32768 | 96 | 128 | 4 |
| 0 | 128 | 128 | 4 |
| 1024 | 128 | 128 | 4 |
| 8192 | 128 | 128 | 4 |
| 32768 | 128 | 128 | 4 |
| 0 | 16 | 256 | 8 |
| 1024 | 16 | 256 | 8 |
| 8192 | 16 | 256 | 8 |
| 32768 | 16 | 256 | 8 |
| 0 | 32 | 256 | 8 |
| 1024 | 32 | 256 | 8 |
| 8192 | 32 | 256 | 8 |
| 32768 | 32 | 256 | 8 |
| 0 | 64 | 256 | 8 |
| 1024 | 64 | 256 | 8 |
| 8192 | 64 | 256 | 8 |
| 32768 | 64 | 256 | 8 |
| 0 | 96 | 256 | 8 |
| 1024 | 96 | 256 | 8 |
| 8192 | 96 | 256 | 8 |
| 32768 | 96 | 256 | 8 |
| 0 | 128 | 256 | 8 |
| 1024 | 128 | 256 | 8 |
| 8192 | 128 | 256 | 8 |
| 32768 | 128 | 256 | 8 |
| 0 | 16 | 512 | 16 |
| 1024 | 16 | 512 | 16 |
| 8192 | 16 | 512 | 16 |
| 32768 | 16 | 512 | 16 |
| 0 | 32 | 512 | 16 |
| 1024 | 32 | 512 | 16 |
| 8192 | 32 | 512 | 16 |
| 32768 | 32 | 512 | 16 |
| 0 | 64 | 512 | 16 |
| 1024 | 64 | 512 | 16 |
| 8192 | 64 | 512 | 16 |
| 32768 | 64 | 512 | 16 |
| 0 | 96 | 512 | 16 |
| 1024 | 96 | 512 | 16 |
| 8192 | 96 | 512 | 16 |
| 32768 | 96 | 512 | 16 |
| 0 | 128 | 512 | 16 |
| 1024 | 128 | 512 | 16 |
| 8192 | 128 | 512 | 16 |
| 32768 | 128 | 512 | 16 |
| 0 | 16 | 1024 | 32 |
| 1024 | 16 | 1024 | 32 |
| 8192 | 16 | 1024 | 32 |
| 32768 | 16 | 1024 | 32 |
| 0 | 32 | 1024 | 32 |
| 1024 | 32 | 1024 | 32 |
| 8192 | 32 | 1024 | 32 |
| 32768 | 32 | 1024 | 32 |
| 0 | 64 | 1024 | 32 |
| 1024 | 64 | 1024 | 32 |
| 8192 | 64 | 1024 | 32 |
| 32768 | 64 | 1024 | 32 |
| 0 | 96 | 1024 | 32 |
| 1024 | 96 | 1024 | 32 |
| 8192 | 96 | 1024 | 32 |
| 32768 | 96 | 1024 | 32 |
| 0 | 128 | 1024 | 32 |
| 1024 | 128 | 1024 | 32 |
| 8192 | 128 | 1024 | 32 |
| 32768 | 128 | 1024 | 32 |

<details><summary><code>block_sizes</code> (JSON)</summary>

```json
[
  32,
  64,
  128,
  256,
  512,
  1024
]
```

</details>

<details><summary><code>dynamic_shared_memory_bytes</code> (JSON)</summary>

```json
[
  0,
  1024,
  8192,
  32768
]
```

</details>

<details><summary><code>registers_per_thread</code> (JSON)</summary>

```json
[
  16,
  32,
  64,
  96,
  128
]
```

</details>

[↑ contents](#contents)

---

## topology.persistent_cta

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1024, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `uniquely_identified` |
| measurement | `resident_blocks_per_sm` = 8 blocks |
| simulator_param | `max_resident_ctas_per_sm` = 8 ctas |
| concept | `cuda_resident_blocks_per_sm` |

- binary_hash: `63a05dac0b90b9e0249635f5d7de57915abc85c61ba4fd9d78d7c732f4f732a1`
- launch.extras: `{"busy_cycles": 200000}`
- interpretation: peak resident CTAs per SM under the configured launch shape
- mapping_contract: observed peak block residency under busy-spin → simulator max_resident_ctas_per_sm

**assumptions:**

- concurrency derived from sweep-line over per-SM (start,end) cycle pairs
- kernel uses %smid plus a busy-spin to keep blocks resident long enough to overlap

### Metrics

| key | value | unit |
| --- | --- | --- |
| `elapsed_ms` | 0.1902 | ms |
| `mean_resident_blocks_per_sm` | 7.7576 | — |
| `multi_processor_count` | 132 | — |
| `peak_resident_blocks_per_sm` | 8 | — |
| `sm_count_observed` | 132 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 63a05dac0b90b9e0249635f5d7de57915abc85c61ba4fd9d78d7c732f4f732a1 |
| `blocks_launched` | 1024 |
| `busy_cycles` | 200000 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `elapsed_ms` | 0.1902 |
| `mean_resident_blocks_per_sm` | 7.7576 |
| `multi_processor_count` | 132 |
| `peak_resident_blocks_per_sm` | 8 |
| `sm_count_observed` | 132 |
| `threads_per_block` | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "6bd1d01c6a6144a242e8c629fdcd8d907dfc334f9f97b050f1d895f83fbba8a5",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 3,
    "EXIT": 2,
    "IADD3": 1,
    "IMAD": 2,
    "ISETP": 3,
    "LDC": 2,
    "NOP": 9,
    "S2R": 3,
    "STG": 3,
    "ULDC": 2
  },
  "satisfied": [],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/topology/persistent_cta.cu`
- bytes: `5507`  ·  sha256: `ce2f1944aa3ad53742efcae13166e51ee6c0b88f9517d01fce3a0699dcb8a770`

### SASS validation

- validated: `True`
- disassembly_hash: `6bd1d01c6a6144a242e8c629fdcd8d907dfc334f9f97b050f1d895f83fbba8a5`
- opcode_histogram: `{"BRA": 2, "CS2R": 3, "EXIT": 2, "IADD3": 1, "IMAD": 2, "ISETP": 3, "LDC": 2, "NOP": 9, "S2R": 3, "STG": 3, "ULDC": 2}`

[↑ contents](#contents)

---

## arithmetic_latency.dependent_chain

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `fp32_fma_dependent_latency` = 4.377 cycles_per_op |
| simulator_param | `fp32_fma_pipeline_depth` = 4.377 cycles_per_op |
| concept | `fp32_fma_dependent_pipeline_latency` |

- binary_hash: `4fbcd8154347573ed0323ef8220ca7fc7aa7fde79d57e6f47f4b6b6b9bbeb732`
- interpretation: cycles between issue and writeback for a dependent FMA
- mapping_contract: dependent FMA cycles-per-op → simulator FP32 FMA latency depth

**assumptions:**

- FP32 FMA dependent chain timed via clock64 inside a single warp
- median across N launches is reported to suppress one-shot kernel-launch jitter

### Metrics

| key | value | unit |
| --- | --- | --- |
| `chain_length` | 4096 | — |
| `cycles_median` | 17928 | — |
| `cycles_per_fma` | 4.377 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 4fbcd8154347573ed0323ef8220ca7fc7aa7fde79d57e6f47f4b6b6b9bbeb732 |
| `chain_length` | 4096 |
| `cycles_max` | 17928 |
| `cycles_median` | 17928 |
| `cycles_min` | 17928 |
| `cycles_per_fma` | 4.377 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `repeats` | 64 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "dependency_confirmed": true,
  "disassembly_hash": "8e35104a69bfdb10f61454278b232b323a6336063c51641923cd5a6bf910e0a1",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "FFMA": 32,
    "HFMA2": 1,
    "IADD3": 3,
    "ISETP": 1,
    "LDC": 3,
    "LOP3": 1,
    "MOV": 2,
    "NOP": 9,
    "S2R": 1,
    "S2UR": 1,
    "STG": 2,
    "ULDC": 2
  },
  "satisfied": [
    "FFMA>=8 (32)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/arithmetic_latency/dependent_chain.cu`
- bytes: `3276`  ·  sha256: `75b8231ef6b256c8eabe9f5586ad06ba00260c15ee1752415fe139a6a82ab880`

### SASS validation

- validated: `True`
- disassembly_hash: `8e35104a69bfdb10f61454278b232b323a6336063c51641923cd5a6bf910e0a1`
- satisfied: FFMA>=8 (32)
- dependency_confirmed: `True`
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 2, "FFMA": 32, "HFMA2": 1, "IADD3": 3, "ISETP": 1, "LDC": 3, "LOP3": 1, "MOV": 2, "NOP": 9, "S2R": 1, "S2UR": 1, "STG": 2, "ULDC": 2}`

[↑ contents](#contents)

---

## arithmetic_throughput.independent_chains

| field | value |
| --- | --- |
| launch | `kernel`  grid=[16, 1, 1] block=[128, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `uniquely_identified` |
| measurement | `fp32_fma_throughput` = 1.1471 cycles_per_op |
| simulator_param | `fp32_fma_throughput` = 1.1471 cycles_per_op |
| concept | `fp32_fma_independent_pipeline_throughput` |

- binary_hash: `e30d16ab2f4848347d3522bb4049e1b971f4be2e2e4c201e68e70af66f25b5aa`
- interpretation: effective FMA cycles-per-op once ILP saturates the FP32 pipe
- mapping_contract: independent FMA cycles-per-op → simulator FP32 FMA throughput

**assumptions:**

- 4 independent FMA chains per thread to expose ILP
- throughput is per-thread cycles-per-op; per-SM is approximate (assumes resident across all SMs)

### Metrics

| key | value | unit |
| --- | --- | --- |
| `approx_fma_per_cycle_per_sm` | 13.5256 | fma/cycle/sm |
| `cycles_median` | 18794 | — |
| `cycles_per_fma_per_thread` | 1.1471 | cycles |

### Raw values

| key | value |
| --- | --- |
| `approx_fma_per_cycle_per_sm` | 13.5256 |
| `binary_sha256` | e30d16ab2f4848347d3522bb4049e1b971f4be2e2e4c201e68e70af66f25b5aa |
| `blocks` | 16 |
| `chain_length` | 4096 |
| `cycles_max` | 18800 |
| `cycles_median` | 18794 |
| `cycles_min` | 18784 |
| `cycles_per_fma_per_thread` | 1.1471 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `elapsed_ms` | 0.0179 |
| `independent_chains` | 4 |
| `multi_processor_count` | 132 |
| `threads` | 128 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "47d695f861cf42bc1b6e14ad9763b708d219f5555b1caa947f2ace6f8f3a7e1d",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 1,
    "FADD": 6,
    "FFMA": 128,
    "HFMA2": 1,
    "IADD3": 3,
    "IMAD": 3,
    "ISETP": 1,
    "LDC": 4,
    "MOV": 5,
    "NOP": 13,
    "S2R": 2,
    "STG": 2,
    "ULDC": 3
  },
  "satisfied": [
    "FFMA>=8 (128)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/arithmetic_throughput/independent_chains.cu`
- bytes: `4638`  ·  sha256: `3707dce55ee9b9715b5343b1543301bb9f4de54b691b8af1b50ec509b209428b`

### SASS validation

- validated: `True`
- disassembly_hash: `47d695f861cf42bc1b6e14ad9763b708d219f5555b1caa947f2ace6f8f3a7e1d`
- satisfied: FFMA>=8 (128)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 1, "FADD": 6, "FFMA": 128, "HFMA2": 1, "IADD3": 3, "IMAD": 3, "ISETP": 1, "LDC": 4, "MOV": 5, "NOP": 13, "S2R": 2, "STG": 2, "ULDC": 3}`

[↑ contents](#contents)

---

## shared_memory.pointer_chase

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[1024, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `shared_memory_load_latency` = 29.0146 cycles |
| simulator_param | `shared_memory_load_latency_cycles` = 29.0146 cycles |
| concept | `shared_memory_load_to_use_latency` |

- binary_hash: `18d423cc6c50bfb2c83d570fe14887d712fa4e8437883055c251c34c79d07662`
- interpretation: LDS dependent-load latency in cycles
- mapping_contract: dependent shared-memory chase cycles-per-load → simulator shared-mem latency

**assumptions:**

- single-thread pointer chase over a 1024-entry shared-memory ring
- median cycles-per-load is reported across N kernel launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `chase_len` | 4096 | — |
| `cycles_median` | 118844 | — |
| `cycles_per_load` | 29.0146 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 18d423cc6c50bfb2c83d570fe14887d712fa4e8437883055c251c34c79d07662 |
| `chase_len` | 4096 |
| `cycles_max` | 118851 |
| `cycles_median` | 118844 |
| `cycles_min` | 118843 |
| `cycles_per_load` | 29.0146 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `repeats` | 64 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "e54306e2b42ece6936af213a72d4c6330874a98ef377e898b7ee71e215bc7d39",
  "opcode_histogram": {
    "BAR": 1,
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 1,
    "IMAD": 4,
    "ISETP": 3,
    "LDC": 3,
    "LDS": 64,
    "LEA": 65,
    "LOP3": 1,
    "MOV": 1,
    "NOP": 8,
    "S2R": 2,
    "S2UR": 1,
    "STG": 2,
    "STS": 1,
    "ULDC": 1,
    "ULEA": 1,
    "UMOV": 1,
    "VIADD": 2
  },
  "satisfied": [
    "LDS>=1 (64)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/shared_memory/pointer_chase.cu`
- bytes: `3320`  ·  sha256: `8708d8a5a03239f88fa1520247d007ff6be5319b5d83e1ab1aa8dd310c4ceedd`

### SASS validation

- validated: `True`
- disassembly_hash: `e54306e2b42ece6936af213a72d4c6330874a98ef377e898b7ee71e215bc7d39`
- satisfied: LDS>=1 (64)
- opcode_histogram: `{"BAR": 1, "BRA": 2, "CS2R": 2, "EXIT": 2, "IADD3": 1, "IMAD": 4, "ISETP": 3, "LDC": 3, "LDS": 64, "LEA": 65, "LOP3": 1, "MOV": 1, "NOP": 8, "S2R": 2, "S2UR": 1, "STG": 2, "STS": 1, "ULDC": 1, "ULEA": 1, "UMOV": 1, "VIADD": 2}`

[↑ contents](#contents)

---

## shared_memory.bank_stride

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `uniquely_identified` |
| measurement | `shared_memory_bank_count` = 32 banks |
| simulator_param | `shared_memory_banks` = 32 banks |
| concept | `shared_memory_bank_count` |

- binary_hash: `0506f91b97f22a88b65d8356ffd1228b3da9b20693f5848125c312d42653da96`
- interpretation: shared-memory bank count inferred from cycles-per-access vs stride curve
- mapping_contract: bank-stride sweep peak conflict factor → simulator shared-memory bank count

**assumptions:**

- single warp probes shared memory with stride sweep covering conflict-factors 1..32
- conflict factor reported as gcd(stride, 32) which holds for shipping NVIDIA archs

### Metrics

| key | value | unit |
| --- | --- | --- |
| `full_conflict_cycles_per_access` | 65.4795 | cycles |
| `inferred_bank_count` | 32 | — |
| `no_conflict_cycles_per_access` | 6.8857 | cycles |
| `sweep_points` | 12 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 0506f91b97f22a88b65d8356ffd1228b3da9b20693f5848125c312d42653da96 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `inner_loops` | 4096 |

#### `sweep` (12 rows)

| conflict_factor | cycles_median | cycles_per_access | stride |
| --- | --- | --- | --- |
| 1 | 28204 | 6.8857 | 1 |
| 2 | 28588 | 6.9795 | 2 |
| 1 | 28204 | 6.8857 | 3 |
| 4 | 40748 | 9.9482 | 4 |
| 1 | 28204 | 6.8857 | 5 |
| 1 | 28204 | 6.8857 | 7 |
| 8 | 72492 | 17.6982 | 8 |
| 1 | 28204 | 6.8857 | 11 |
| 16 | 137132 | 33.4795 | 16 |
| 1 | 28204 | 6.8857 | 17 |
| 32 | 268204 | 65.4795 | 32 |
| 1 | 28204 | 6.8857 | 33 |

<details><summary><code>ncu</code> (JSON)</summary>

```json
{
  "launches_profiled": 64,
  "logical": "shared_conflicts",
  "metric": "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
  "role": "validation",
  "value": 4096.0
}
```

</details>

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "0d72d07d030e7393d173f30263b1ca22bda4a26d640c3c9e5e37674e195303d5",
  "opcode_histogram": {
    "BAR": 1,
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 3,
    "IADD3": 2,
    "IMAD": 32,
    "ISETP": 4,
    "LDC": 3,
    "LDS": 32,
    "LEA": 1,
    "LOP3": 49,
    "MOV": 1,
    "NOP": 11,
    "S2R": 2,
    "S2UR": 1,
    "SHF": 7,
    "STG": 2,
    "STS": 1,
    "ULDC": 2,
    "ULEA": 1,
    "UMOV": 1,
    "VIADD": 32
  },
  "satisfied": [
    "LDS>=1 (32)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/shared_memory/bank_stride.cu`
- bytes: `4535`  ·  sha256: `deee44a45bdafc0270320e62f77d735ec08f3c481b8db8985cbb6ec1bf0f7de9`

### SASS validation

- validated: `True`
- disassembly_hash: `0d72d07d030e7393d173f30263b1ca22bda4a26d640c3c9e5e37674e195303d5`
- satisfied: LDS>=1 (32)
- opcode_histogram: `{"BAR": 1, "BRA": 2, "CS2R": 2, "EXIT": 3, "IADD3": 2, "IMAD": 32, "ISETP": 4, "LDC": 3, "LDS": 32, "LEA": 1, "LOP3": 49, "MOV": 1, "NOP": 11, "S2R": 2, "S2UR": 1, "SHF": 7, "STG": 2, "STS": 1, "ULDC": 2, "ULEA": 1, "UMOV": 1, "VIADD": 32}`

[↑ contents](#contents)

---

## shared_memory.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `uniquely_identified` |
| measurement | `shared_memory_summary` = _object_ (bank_count, bank_serialization_factor, shared_load_latency_cycles) |
| simulator_param | `shared_memory_summary` = _object_ (bank_count, bank_serialization_factor, shared_load_latency_cycles) |
| concept | `shared_memory_summary` |

- interpretation: merged shared-memory characterization derived from pointer-chase and bank-stride probes
- mapping_contract: cross-probe summary suitable for simulator shared-memory model parameters

**assumptions:**

- consumes the cycles-per-load median from pointer_chase and the stride sweep from bank_stride
- bank_serialization_factor is full_conflict / no_conflict cycles-per-access

### Measurement value

```json
{
  "bank_count": 32,
  "bank_serialization_factor": 9.50949068359063,
  "shared_load_latency_cycles": 29.0146
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `bank_count` | 32 | — |
| `bank_serialization_factor` | 9.50949 | — |
| `shared_load_latency_cycles` | 29.0146 | — |

### Raw values

<details><summary><code>bank_stride</code> (JSON)</summary>

```json
{
  "binary_sha256": "0506f91b97f22a88b65d8356ffd1228b3da9b20693f5848125c312d42653da96",
  "full_conflict_cycles_per_access": 65.4795,
  "inferred_bank_count": 32,
  "no_conflict_cycles_per_access": 6.8857
}
```

</details>

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "bank_count": 32,
  "bank_serialization_factor": 9.50949068359063,
  "shared_load_latency_cycles": 29.0146
}
```

</details>

<details><summary><code>pointer_chase</code> (JSON)</summary>

```json
{
  "binary_sha256": "18d423cc6c50bfb2c83d570fe14887d712fa4e8437883055c251c34c79d07662",
  "cycles_per_load": 29.0146
}
```

</details>

[↑ contents](#contents)

---

## l1_cache.pointer_chase

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `l1_hit_load_latency` = 70.6108 cycles |
| simulator_param | `l1_latency` = 70.6108 cycles |
| concept | `l1_path_hit_latency` |

- binary_hash: `aea1c593856320979dc411cf66981a009cb13437e410ceff83a5256d388dc94b`
- interpretation: dependent-load latency for an L1-resident working set in cycles
- mapping_contract: dependent L1-hit chase cycles-per-load → simulator L1 hit latency

**assumptions:**

- single-thread dependent pointer chase over a randomized ring sized to fit L1
- a DRAM-resident ring is timed as a control; L1-hit regime requires small << large
- median cycles-per-load reported across N launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `dram_cycles_per_load` | 317.958 | cycles |
| `hit_to_dram_ratio` | 4.50296 | — |
| `l1_hit_cycles_per_load` | 70.6108 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | aea1c593856320979dc411cf66981a009cb13437e410ceff83a5256d388dc94b |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `dram_cycles_per_load` | 317.958 |
| `l1_hit_cycles_per_load` | 70.6108 |
| `large_kb` | 8192 |
| `repeats` | 64 |
| `small_kb` | 16 |
| `steps` | 4096 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "6275e1d880988c28a7b893d366d589cbe5bb2efb2dab55c07fed4b71ba42ff0c",
  "opcode_histogram": {
    "BRA": 6,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 6,
    "IMAD": 17,
    "ISETP": 5,
    "LDC": 6,
    "LDG": 17,
    "LOP3": 2,
    "MOV": 1,
    "NOP": 10,
    "S2R": 1,
    "S2UR": 1,
    "STG": 2,
    "ULDC": 2
  },
  "satisfied": [
    "LDG>=1 (17)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/l1_cache/pointer_chase.cu`
- bytes: `5643`  ·  sha256: `198f6a1e50bc281f93330623a68670586bd6f83f49dab47b704c831b35c3edff`

### SASS validation

- validated: `True`
- disassembly_hash: `6275e1d880988c28a7b893d366d589cbe5bb2efb2dab55c07fed4b71ba42ff0c`
- satisfied: LDG>=1 (17)
- opcode_histogram: `{"BRA": 6, "CS2R": 2, "EXIT": 2, "IADD3": 6, "IMAD": 17, "ISETP": 5, "LDC": 6, "LDG": 17, "LOP3": 2, "MOV": 1, "NOP": 10, "S2R": 1, "S2UR": 1, "STG": 2, "ULDC": 2}`

[↑ contents](#contents)

---

## l1_cache.working_set

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `underconstrained` |
| measurement | `l1_effective_capacity` = _object_ |
| simulator_param | `l1d_cache_capacity` = _object_ |
| concept | `l1_effective_capacity_knee` |

- binary_hash: `7369adb033f38fdbe48c0ee7a5fc5f0047e9666f09a77d78446f2a72866f747e`
- interpretation: effective L1 capacity bounded by the first latency knee in the working-set sweep
- mapping_contract: working-set latency knee → simulator L1 capacity range

**assumptions:**

- dependent pointer-chase latency swept across geometric working-set sizes
- first >40% latency jump marks the effective L1 capacity knee
- capacity is reported as a bounded range, not an exact scalar

### Measurement value

```json
{}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `sweep_points` | 13 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 7369adb033f38fdbe48c0ee7a5fc5f0047e9666f09a77d78446f2a72866f747e |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `repeats` | 32 |
| `steps` | 4096 |

#### `sweep` (13 rows)

| cycles_per_load | working_set_kb |
| --- | --- |
| 47.3569 | 4 |
| 55.4131 | 8 |
| 70.6562 | 16 |
| 86.3101 | 24 |
| 101.26 | 32 |
| 128.85 | 48 |
| 151.03 | 64 |
| 202.842 | 128 |
| 238.56 | 256 |
| 264.61 | 512 |
| 275.678 | 1024 |
| 311.713 | 4096 |
| 352.958 | 16384 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "27c2fe00efb6e73da81a1bc1f140267fc6cb401a11dbe978e45437e8bbe54641",
  "opcode_histogram": {
    "BRA": 6,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 6,
    "IMAD": 17,
    "ISETP": 5,
    "LDC": 6,
    "LDG": 17,
    "LOP3": 2,
    "MOV": 1,
    "NOP": 10,
    "S2R": 1,
    "S2UR": 1,
    "STG": 2,
    "ULDC": 2
  },
  "satisfied": [
    "LDG>=1 (17)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/l1_cache/working_set.cu`
- bytes: `4223`  ·  sha256: `0bdfc85ec29e0f1847fd8bca024ae2637a57427be3c0796383ed180cc75987b6`

### SASS validation

- validated: `True`
- disassembly_hash: `27c2fe00efb6e73da81a1bc1f140267fc6cb401a11dbe978e45437e8bbe54641`
- satisfied: LDG>=1 (17)
- opcode_histogram: `{"BRA": 6, "CS2R": 2, "EXIT": 2, "IADD3": 6, "IMAD": 17, "ISETP": 5, "LDC": 6, "LDG": 17, "LOP3": 2, "MOV": 1, "NOP": 10, "S2R": 1, "S2UR": 1, "STG": 2, "ULDC": 2}`

[↑ contents](#contents)

---

## l1_cache.conflict_sets

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `underconstrained` |
| measurement | `l1_effective_associativity` = — |
| simulator_param | `l1d_cache_assoc` = — |
| concept | `l1_effective_associativity` |

- binary_hash: `75786b9d9ba51b01fccc2e935eea020724418cb77c755dab4d86fa208568f54f`
- interpretation: effective L1 associativity bounded by the conflict-set latency knee
- mapping_contract: conflict-set latency knee → simulator L1 associativity (bounded)

**assumptions:**

- ring of same-set lines grown one way at a time at a fixed power-of-two stride
- latency knee marks where the conflict set exceeds the effective associativity
- associativity is bounded: indexing/replacement/hashing can mimic the same curve

### Metrics

| key | value | unit |
| --- | --- | --- |
| `stride_bytes` | 4096 | — |
| `sweep_points` | 24 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 75786b9d9ba51b01fccc2e935eea020724418cb77c755dab4d86fa208568f54f |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `steps` | 4096 |
| `stride_bytes` | 4096 |

#### `sweep` (24 rows)

| cycles_per_load | ways |
| --- | --- |
| 39.7166 | 1 |
| 39.7747 | 2 |
| 39.8391 | 3 |
| 39.9033 | 4 |
| 39.9692 | 5 |
| 40.0269 | 6 |
| 40.0842 | 7 |
| 40.1462 | 8 |
| 40.2012 | 9 |
| 40.2585 | 10 |
| 40.3223 | 11 |
| 40.3875 | 12 |
| 40.4526 | 13 |
| 40.5093 | 14 |
| 40.5654 | 15 |
| 40.6262 | 16 |
| 40.6836 | 17 |
| 40.7424 | 18 |
| 40.8042 | 19 |
| 40.8716 | 20 |
| 40.9353 | 21 |
| 40.9922 | 22 |
| 41.0488 | 23 |
| 41.1138 | 24 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "1545a684c1493ce4a53dffe7a33207a3efc8ab08a17cba953138aefd1733ec6f",
  "opcode_histogram": {
    "BRA": 6,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 6,
    "IMAD": 17,
    "ISETP": 5,
    "LDC": 6,
    "LDG": 17,
    "LOP3": 2,
    "MOV": 1,
    "NOP": 10,
    "S2R": 1,
    "S2UR": 1,
    "STG": 2,
    "ULDC": 2
  },
  "satisfied": [
    "LDG>=1 (17)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/l1_cache/conflict_sets.cu`
- bytes: `4359`  ·  sha256: `a1c9a63a7abe678c35db2a1c1f96e95658eb687a8d6b157b54981d06b72ee984`

### SASS validation

- validated: `True`
- disassembly_hash: `1545a684c1493ce4a53dffe7a33207a3efc8ab08a17cba953138aefd1733ec6f`
- satisfied: LDG>=1 (17)
- opcode_histogram: `{"BRA": 6, "CS2R": 2, "EXIT": 2, "IADD3": 6, "IMAD": 17, "ISETP": 5, "LDC": 6, "LDG": 17, "LOP3": 2, "MOV": 1, "NOP": 10, "S2R": 1, "S2UR": 1, "STG": 2, "ULDC": 2}`

[↑ contents](#contents)

---

## l1_cache.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `underconstrained` |
| measurement | `l1_cache_summary` = _object_ (l1_effective_capacity_kb, l1_hit_latency_cycles) |
| simulator_param | `l1d_cache_summary` = _object_ (l1_effective_capacity_kb, l1_hit_latency_cycles) |
| concept | `l1_cache_summary` |

- interpretation: merged L1 path characterization from latency, capacity, and conflict probes
- mapping_contract: cross-probe L1 summary for simulator L1 cache model parameters

**assumptions:**

- merges L1 hit latency, capacity knee, and associativity knee
- merged fit status is the weakest of the contributing probe fits

### Measurement value

```json
{
  "l1_effective_capacity_kb": {},
  "l1_hit_latency_cycles": 70.6057
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `l1_effective_capacity_kb` | `{}` | — |
| `l1_hit_latency_cycles` | 70.6057 | — |

### Raw values

<details><summary><code>conflict_sets</code> (JSON)</summary>

```json
{
  "binary_sha256": "75786b9d9ba51b01fccc2e935eea020724418cb77c755dab4d86fa208568f54f"
}
```

</details>

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "l1_effective_capacity_kb": {},
  "l1_hit_latency_cycles": 70.6057
}
```

</details>

<details><summary><code>pointer_chase</code> (JSON)</summary>

```json
{
  "binary_sha256": "aea1c593856320979dc411cf66981a009cb13437e410ceff83a5256d388dc94b",
  "l1_hit_cycles_per_load": 70.6057
}
```

</details>

<details><summary><code>working_set</code> (JSON)</summary>

```json
{
  "binary_sha256": "7369adb033f38fdbe48c0ee7a5fc5f0047e9666f09a77d78446f2a72866f747e",
  "effective_capacity": {}
}
```

</details>

[↑ contents](#contents)

---

## scheduler_policy.ready_warps

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[1024, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `conditionally_identified` |
| measurement | `issue_saturation_warps` = 16 warps |
| simulator_param | `gpgpu_num_sched_per_core` = 16 warps |
| concept | `scheduler_issue_scaling` |

- binary_hash: `5a65abee5f03e65c0424e0c8f0ef7ab60a5c3110c9fb5b566e6351698d7f595a`
- interpretation: ready-warp count at which issue throughput saturates on one SM
- mapping_contract: issue-scaling saturation knee → simulator scheduler issue capacity (conditional)

**assumptions:**

- one CTA on one SM runs N independent dependent-FMA warps (no memory)
- saturation warp count = smallest warp count reaching 95% of peak ops/cycle
- scheduler policy name is behavioral; only issue scaling is reported

### Metrics

| key | value | unit |
| --- | --- | --- |
| `peak_ops_per_cycle` | 106.411 | ops/cycle |
| `saturation_warps` | 16 | — |
| `sweep_points` | 32 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 5a65abee5f03e65c0424e0c8f0ef7ab60a5c3110c9fb5b566e6351698d7f595a |
| `chain_length` | 2048 |
| `device_name` | NVIDIA H100 80GB HBM3 |

#### `sweep` (32 rows)

| cycles_median | ops_per_cycle | warps |
| --- | --- | --- |
| 9096 | 7.2049 | 1 |
| 9096 | 14.4099 | 2 |
| 9096 | 21.6148 | 3 |
| 9096 | 28.8197 | 4 |
| 9227 | 35.5132 | 5 |
| 9227 | 42.6158 | 6 |
| 9228 | 49.713 | 7 |
| 9228 | 56.8149 | 8 |
| 9622 | 61.2995 | 9 |
| 9618 | 68.1389 | 10 |
| 9622 | 74.9216 | 11 |
| 9622 | 81.7327 | 12 |
| 10356 | 82.2681 | 13 |
| 10338 | 88.7506 | 14 |
| 10354 | 94.943 | 15 |
| 10314 | 101.665 | 16 |
| 12580 | 88.5622 | 17 |
| 12582 | 93.7568 | 18 |
| 12588 | 98.9183 | 19 |
| 12575 | 104.232 | 20 |
| 14887 | 92.4468 | 21 |
| 14887 | 96.8491 | 22 |
| 14886 | 101.258 | 23 |
| 14892 | 105.618 | 24 |
| 17288 | 94.7709 | 25 |
| 17294 | 98.5276 | 26 |
| 17285 | 102.37 | 27 |
| 17306 | 106.033 | 28 |
| 19711 | 96.4205 | 29 |
| 19714 | 99.7301 | 30 |
| 19709 | 103.081 | 31 |
| 19708 | 106.411 | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "9ae202af275fa1b6abfa288fd733b7a3a6e68137611ca61c211187b9283c7cb2",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "FADD": 1,
    "FFMA": 16,
    "HFMA2": 1,
    "I2FP": 1,
    "IADD3": 5,
    "ISETP": 3,
    "LDC": 2,
    "LOP3": 1,
    "MOV": 1,
    "NOP": 10,
    "S2R": 1,
    "SHF": 2,
    "STG": 2,
    "ULDC": 4
  },
  "satisfied": [
    "FFMA>=8 (16)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/scheduler_policy/ready_warps.cu`
- bytes: `4167`  ·  sha256: `66e8d665ec60350e3c5fa942833f7065db231c1fd1ca64eb36650e36571860de`

### SASS validation

- validated: `True`
- disassembly_hash: `9ae202af275fa1b6abfa288fd733b7a3a6e68137611ca61c211187b9283c7cb2`
- satisfied: FFMA>=8 (16)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 2, "FADD": 1, "FFMA": 16, "HFMA2": 1, "I2FP": 1, "IADD3": 5, "ISETP": 3, "LDC": 2, "LOP3": 1, "MOV": 1, "NOP": 10, "S2R": 1, "SHF": 2, "STG": 2, "ULDC": 4}`

[↑ contents](#contents)

---

## scheduler_policy.mixed_issue

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[256, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `behavioral_only` |
| measurement | `mixed_issue_overlap` = single_issue_like |
| simulator_param | `gpgpu_dual_issue_diff_exec_units` = single_issue_like |
| concept | `mixed_pipeline_issue_overlap` |

- binary_hash: `4826d4e62529a9a8c1404c12e237a322565b36f4b206c901dd1d0c6096e2038e`
- interpretation: FP32/INT pipe overlap classified from mixed vs single-pipe throughput
- mapping_contract: mixed/single-pipe overlap ratio → simulator dual-issue behavioral class

**assumptions:**

- independent FP32 (FMA) and INT (MAD) streams run alone and interleaved
- overlap_ratio = mixed / max(fp32, int); higher means more pipe overlap
- mixed-issue capability is a behavioral class, not a named policy

### Metrics

| key | value | unit |
| --- | --- | --- |
| `fp32_ops_per_cycle` | 115.425 | ops/cycle |
| `int_ops_per_cycle` | 483.215 | ops/cycle |
| `mixed_ops_per_cycle` | 178.178 | ops/cycle |
| `overlap_ratio` | 0.368734 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 4826d4e62529a9a8c1404c12e237a322565b36f4b206c901dd1d0c6096e2038e |
| `chain_length` | 2048 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `fp32_ops_per_cycle` | 115.425 |
| `int_ops_per_cycle` | 483.215 |
| `mixed_ops_per_cycle` | 178.178 |
| `warps` | 8 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "9ce43af19d846c32f2c7c50872d53ad5cb588a9826a9915dfee8eaf63a0ad046",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 1,
    "FADD": 2,
    "FFMA": 32,
    "HFMA2": 1,
    "IADD3": 9,
    "IMAD": 2,
    "ISETP": 2,
    "LDC": 3,
    "LOP3": 1,
    "MOV": 4,
    "NOP": 10,
    "S2R": 1,
    "SHF": 1,
    "STG": 3,
    "ULDC": 4
  },
  "satisfied": [
    "FFMA>=4 (32)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/scheduler_policy/mixed_issue.cu`
- bytes: `5669`  ·  sha256: `7ed20977f0719cdb66f2939d1f30f8d2004ff321fa5dab14bdb86a39940fc8e7`

### SASS validation

- validated: `True`
- disassembly_hash: `9ce43af19d846c32f2c7c50872d53ad5cb588a9826a9915dfee8eaf63a0ad046`
- satisfied: FFMA>=4 (32)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 1, "FADD": 2, "FFMA": 32, "HFMA2": 1, "IADD3": 9, "IMAD": 2, "ISETP": 2, "LDC": 3, "LOP3": 1, "MOV": 4, "NOP": 10, "S2R": 1, "SHF": 1, "STG": 3, "ULDC": 4}`

[↑ contents](#contents)

---

## scheduler_policy.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `behavioral_only` |
| measurement | `scheduler_summary` = _object_ (issue_saturation_warps, mixed_issue_class, peak_ops_per_cycle) |
| simulator_param | `gpgpu_scheduler_summary` = _object_ (issue_saturation_warps, mixed_issue_class, peak_ops_per_cycle) |
| concept | `scheduler_summary` |

- interpretation: scheduler issue scaling and pipe-overlap behavior on one SM
- mapping_contract: cross-probe scheduler summary for simulator scheduler model (behavioral)

**assumptions:**

- combines ready-warp issue saturation with mixed-issue overlap class
- scheduler policy reported as a behavioral class with a conditional issue-capacity value

### Measurement value

```json
{
  "issue_saturation_warps": 16,
  "mixed_issue_class": "single_issue_like",
  "peak_ops_per_cycle": 106.4058
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `issue_saturation_warps` | 16 | — |
| `mixed_issue_class` | single_issue_like | — |
| `peak_ops_per_cycle` | 106.406 | — |

### Raw values

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "issue_saturation_warps": 16,
  "mixed_issue_class": "single_issue_like",
  "peak_ops_per_cycle": 106.4058
}
```

</details>

<details><summary><code>mixed_issue</code> (JSON)</summary>

```json
{
  "binary_sha256": "4826d4e62529a9a8c1404c12e237a322565b36f4b206c901dd1d0c6096e2038e",
  "overlap_class": "single_issue_like",
  "overlap_ratio": 0.36873402237142205
}
```

</details>

<details><summary><code>ready_warps</code> (JSON)</summary>

```json
{
  "binary_sha256": "5a65abee5f03e65c0424e0c8f0ef7ab60a5c3110c9fb5b566e6351698d7f595a",
  "saturation_warps": 16
}
```

</details>

[↑ contents](#contents)

---

## register_file.register_bank_sweep

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `underconstrained` |
| measurement | `operand_delivery_plateau` = 16 accumulators |
| simulator_param | `gpgpu_num_reg_banks` = 16 accumulators |
| concept | `register_bank_operand_delivery` |

- binary_hash: `d87426f5daf1450c015a3c0c78d8bfbd62c3517f32bbb7790b9bd6e08e11c9c8`
- interpretation: operand-delivery throughput plateau across register-pressure widths
- mapping_contract: operand-width plateau → simulator register-bank pressure (candidate, multi-fit)

**assumptions:**

- operand-width sweep of independent FMA accumulators (register pressure proxy)
- CUDA approximation of the SASS-controlled register sweep; bank count is not uniquely identified
- plateau width marks where added ILP stops improving cycles-per-op

### Metrics

| key | value | unit |
| --- | --- | --- |
| `ilp_plateau_width` | 16 | accumulators |
| `sweep_points` | 8 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | d87426f5daf1450c015a3c0c78d8bfbd62c3517f32bbb7790b9bd6e08e11c9c8 |
| `chain_length` | 2048 |
| `device_name` | NVIDIA H100 80GB HBM3 |

#### `sweep` (8 rows)

| cycles_per_op | width |
| --- | --- |
| 4.8789 | 1 |
| 2.6265 | 2 |
| 1.9596 | 3 |
| 1.501 | 4 |
| 1.334 | 6 |
| 1.2505 | 8 |
| 1.167 | 12 |
| 1.1252 | 16 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "32b00f24f3dc9631660e2a3852a2d60551786eee1d0f2d93facad6cc44a49f1d",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 1,
    "FADD": 32,
    "FFMA": 128,
    "HFMA2": 1,
    "IADD3": 5,
    "ISETP": 2,
    "LDC": 3,
    "LOP3": 1,
    "MOV": 1,
    "NOP": 8,
    "S2R": 1,
    "SHF": 1,
    "STG": 2,
    "ULDC": 2
  },
  "satisfied": [
    "FFMA>=8 (128)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/register_file/register_bank_sweep.cu`
- bytes: `4516`  ·  sha256: `7aa75fca4c6681820fe6c22be0b38aee5698824f1639b55dbcc13ccfbed950a5`

### SASS validation

- validated: `True`
- disassembly_hash: `32b00f24f3dc9631660e2a3852a2d60551786eee1d0f2d93facad6cc44a49f1d`
- satisfied: FFMA>=8 (128)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 1, "FADD": 32, "FFMA": 128, "HFMA2": 1, "IADD3": 5, "ISETP": 2, "LDC": 3, "LOP3": 1, "MOV": 1, "NOP": 8, "S2R": 1, "SHF": 1, "STG": 2, "ULDC": 2}`

[↑ contents](#contents)

---

## register_file.register_latency

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `conditionally_identified` |
| measurement | `operand_delivery_differential_latency` = 2.3606 cycles |
| simulator_param | `max_latency_regular_register_file_latency` = 2.3606 cycles |
| concept | `register_operand_delivery_latency` |

- binary_hash: `67595a7ac5ed5219f1337b3cd1c638c9753514e314b3c4b87bab7c8313166a8b`
- interpretation: extra per-op cost of tight RAW dependence attributable to operand delivery
- mapping_contract: RAW-distance differential cycles → simulator operand-delivery latency (conditional)

**assumptions:**

- same-register (RAW distance 1) vs rotating-register (relaxed RAW) chains of equal length
- differential cycles-per-op isolates operand-delivery cost from absolute arithmetic latency
- operand-collector parameters stay conditional: scoreboard/bank effects are entangled

### Metrics

| key | value | unit |
| --- | --- | --- |
| `differential_cycles_per_op` | 2.3606 | cycles |
| `rotating_reg_cycles_per_op` | 2.0789 | cycles |
| `same_reg_cycles_per_op` | 4.4395 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 67595a7ac5ed5219f1337b3cd1c638c9753514e314b3c4b87bab7c8313166a8b |
| `chain_length` | 4096 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `differential_cycles_per_op` | 2.3606 |
| `rot_depth` | 8 |
| `rotating_reg_cycles_per_op` | 2.0789 |
| `same_reg_cycles_per_op` | 4.4395 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "68845fd951f326d632215e2cfc47e11225bc03dd62097a16316e823c0fb4da51",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "FFMA": 16,
    "HFMA2": 1,
    "IADD3": 3,
    "ISETP": 2,
    "LDC": 3,
    "MOV": 2,
    "NOP": 10,
    "S2R": 1,
    "STG": 2,
    "ULDC": 2
  },
  "satisfied": [
    "FFMA>=8 (16)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/register_file/register_latency.cu`
- bytes: `4327`  ·  sha256: `e19a118fab98c5f6a260b19f5d10357e42bade8a5c98e02080c3560932c19975`

### SASS validation

- validated: `True`
- disassembly_hash: `68845fd951f326d632215e2cfc47e11225bc03dd62097a16316e823c0fb4da51`
- satisfied: FFMA>=8 (16)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 2, "FFMA": 16, "HFMA2": 1, "IADD3": 3, "ISETP": 2, "LDC": 3, "MOV": 2, "NOP": 10, "S2R": 1, "STG": 2, "ULDC": 2}`

[↑ contents](#contents)

---

## register_file.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `underconstrained` |
| measurement | `register_file_summary` = _object_ (operand_delivery_differential_cycles, operand_delivery_plateau_accumulators) |
| simulator_param | `register_file_summary` = _object_ (operand_delivery_differential_cycles, operand_delivery_plateau_accumulators) |
| concept | `register_file_summary` |

- interpretation: merged register-bank pressure and operand-delivery differential latency
- mapping_contract: cross-probe register-file summary for simulator operand-delivery model (candidate)

**assumptions:**

- keeps register-bank evidence (plateau) separate from operand-delivery latency (differential)
- bank count remains a candidate; differential latency remains conditional

### Measurement value

```json
{
  "operand_delivery_differential_cycles": 2.3606,
  "operand_delivery_plateau_accumulators": 16
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `operand_delivery_differential_cycles` | 2.3606 | — |
| `operand_delivery_plateau_accumulators` | 16 | — |

### Raw values

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "operand_delivery_differential_cycles": 2.3606,
  "operand_delivery_plateau_accumulators": 16
}
```

</details>

<details><summary><code>register_bank_sweep</code> (JSON)</summary>

```json
{
  "binary_sha256": "d87426f5daf1450c015a3c0c78d8bfbd62c3517f32bbb7790b9bd6e08e11c9c8",
  "ilp_plateau_width": 16
}
```

</details>

<details><summary><code>register_latency</code> (JSON)</summary>

```json
{
  "binary_sha256": "67595a7ac5ed5219f1337b3cd1c638c9753514e314b3c4b87bab7c8313166a8b",
  "differential_cycles_per_op": 2.3606
}
```

</details>

[↑ contents](#contents)

---
