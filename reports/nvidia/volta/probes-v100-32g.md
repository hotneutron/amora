# nvidia / volta / v100-32g — Probe Results

- Generated: 2026-06-24T06:32Z
- Device: Tesla V100-SXM2-32GB  ·  Backend: `nvidia_cuda`  ·  Probes: 8
- `fit_status`: `direct`=4, `uniquely_identified`=4
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[topology.device_attributes](#topologydevice_attributes) · [topology.occupancy](#topologyoccupancy) · [topology.persistent_cta](#topologypersistent_cta) · [arithmetic_latency.dependent_chain](#arithmetic_latencydependent_chain) · [arithmetic_throughput.independent_chains](#arithmetic_throughputindependent_chains) · [shared_memory.pointer_chase](#shared_memorypointer_chase) · [shared_memory.bank_stride](#shared_memorybank_stride) · [shared_memory.analyze](#shared_memoryanalyze)

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
  "device_name": "Tesla V100-SXM2-32GB",
  "driver_version": "535.261.03",
  "uuid": "GPU-c4ef444c-e2c7-3bbe-8441-45d607d9d3a7"
}
```

### Raw values

| key | value |
| --- | --- |
| `device_index` | 0 |
| `device_name` | Tesla V100-SXM2-32GB |
| `driver_version` | 535.261.03 |
| `uuid` | GPU-c4ef444c-e2c7-3bbe-8441-45d607d9d3a7 |

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
| measurement | `resident_blocks_per_sm` = 13 blocks |
| simulator_param | `max_resident_ctas_per_sm` = 13 ctas |
| concept | `cuda_resident_blocks_per_sm` |

- binary_hash: `f54ec41a6d76daa5874e82e1c1165c02971cb78bf6364e9230c0f1862e0b3a33`
- launch.extras: `{"busy_cycles": 200000}`
- interpretation: peak resident CTAs per SM under the configured launch shape
- mapping_contract: observed peak block residency under busy-spin → simulator max_resident_ctas_per_sm

**assumptions:**

- concurrency derived from sweep-line over per-SM (start,end) cycle pairs
- kernel uses %smid plus a busy-spin to keep blocks resident long enough to overlap

### Metrics

| key | value | unit |
| --- | --- | --- |
| `elapsed_ms` | 1.2001 | ms |
| `mean_resident_blocks_per_sm` | 12.8 | — |
| `multi_processor_count` | 80 | — |
| `peak_resident_blocks_per_sm` | 13 | — |
| `sm_count_observed` | 80 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | f54ec41a6d76daa5874e82e1c1165c02971cb78bf6364e9230c0f1862e0b3a33 |
| `blocks_launched` | 1024 |
| `busy_cycles` | 200000 |
| `device_name` | Tesla V100-SXM2-32GB |
| `elapsed_ms` | 1.2001 |
| `mean_resident_blocks_per_sm` | 12.8 |
| `multi_processor_count` | 80 |
| `peak_resident_blocks_per_sm` | 13 |
| `sm_count_observed` | 80 |
| `threads_per_block` | 32 |

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/topology/persistent_cta.cu`
- bytes: `5507`  ·  sha256: `ce2f1944aa3ad53742efcae13166e51ee6c0b88f9517d01fce3a0699dcb8a770`

[↑ contents](#contents)

---

## arithmetic_latency.dependent_chain

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `fp32_fma_dependent_latency` = 4.376 cycles_per_op |
| simulator_param | `fp32_fma_pipeline_depth` = 4.376 cycles_per_op |
| concept | `fp32_fma_dependent_pipeline_latency` |

- binary_hash: `756eac08d4a520f2882d2b5cf2734b160456dcb3975d04cdd2e63c6d671904b2`
- interpretation: cycles between issue and writeback for a dependent FMA
- mapping_contract: dependent FMA cycles-per-op → simulator FP32 FMA latency depth

**assumptions:**

- FP32 FMA dependent chain timed via clock64 inside a single warp
- median across N launches is reported to suppress one-shot kernel-launch jitter

### Metrics

| key | value | unit |
| --- | --- | --- |
| `chain_length` | 4096 | — |
| `cycles_median` | 17924 | — |
| `cycles_per_fma` | 4.376 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 756eac08d4a520f2882d2b5cf2734b160456dcb3975d04cdd2e63c6d671904b2 |
| `chain_length` | 4096 |
| `cycles_max` | 17924 |
| `cycles_median` | 17924 |
| `cycles_min` | 17924 |
| `cycles_per_fma` | 4.376 |
| `device_name` | Tesla V100-SXM2-32GB |
| `repeats` | 64 |

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/arithmetic_latency/dependent_chain.cu`
- bytes: `3276`  ·  sha256: `75b8231ef6b256c8eabe9f5586ad06ba00260c15ee1752415fe139a6a82ab880`

[↑ contents](#contents)

---

## arithmetic_throughput.independent_chains

| field | value |
| --- | --- |
| launch | `kernel`  grid=[16, 1, 1] block=[128, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `uniquely_identified` |
| measurement | `fp32_fma_throughput` = 2.1097 cycles_per_op |
| simulator_param | `fp32_fma_throughput` = 2.1097 cycles_per_op |
| concept | `fp32_fma_independent_pipeline_throughput` |

- binary_hash: `e7a164cbe82da75dce9e6494dea27367609a624ac23bf24b0a49c1d29dacbf0c`
- interpretation: effective FMA cycles-per-op once ILP saturates the FP32 pipe
- mapping_contract: independent FMA cycles-per-op → simulator FP32 FMA throughput

**assumptions:**

- 4 independent FMA chains per thread to expose ILP
- throughput is per-thread cycles-per-op; per-SM is approximate (assumes resident across all SMs)

### Metrics

| key | value | unit |
| --- | --- | --- |
| `approx_fma_per_cycle_per_sm` | 12.1342 | fma/cycle/sm |
| `cycles_median` | 34566 | — |
| `cycles_per_fma_per_thread` | 2.1097 | cycles |

### Raw values

| key | value |
| --- | --- |
| `approx_fma_per_cycle_per_sm` | 12.1342 |
| `binary_sha256` | e7a164cbe82da75dce9e6494dea27367609a624ac23bf24b0a49c1d29dacbf0c |
| `blocks` | 16 |
| `chain_length` | 4096 |
| `cycles_max` | 34566 |
| `cycles_median` | 34566 |
| `cycles_min` | 34566 |
| `cycles_per_fma_per_thread` | 2.1097 |
| `device_name` | Tesla V100-SXM2-32GB |
| `elapsed_ms` | 0.0348 |
| `independent_chains` | 4 |
| `multi_processor_count` | 80 |
| `threads` | 128 |

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/arithmetic_throughput/independent_chains.cu`
- bytes: `4638`  ·  sha256: `3707dce55ee9b9715b5343b1543301bb9f4de54b691b8af1b50ec509b209428b`

[↑ contents](#contents)

---

## shared_memory.pointer_chase

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[1024, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `direct` |
| measurement | `shared_memory_load_latency` = 26.9988 cycles |
| simulator_param | `shared_memory_load_latency_cycles` = 26.9988 cycles |
| concept | `shared_memory_load_to_use_latency` |

- binary_hash: `2863c0360952265661278e7379759fba28c7f4f83e42d8e4dd6b4164f00eec38`
- interpretation: LDS dependent-load latency in cycles
- mapping_contract: dependent shared-memory chase cycles-per-load → simulator shared-mem latency

**assumptions:**

- single-thread pointer chase over a 1024-entry shared-memory ring
- median cycles-per-load is reported across N kernel launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `chase_len` | 4096 | — |
| `cycles_median` | 110587 | — |
| `cycles_per_load` | 26.9988 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 2863c0360952265661278e7379759fba28c7f4f83e42d8e4dd6b4164f00eec38 |
| `chase_len` | 4096 |
| `cycles_max` | 110587 |
| `cycles_median` | 110587 |
| `cycles_min` | 110587 |
| `cycles_per_load` | 26.9988 |
| `device_name` | Tesla V100-SXM2-32GB |
| `repeats` | 64 |

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/shared_memory/pointer_chase.cu`
- bytes: `3320`  ·  sha256: `8708d8a5a03239f88fa1520247d007ff6be5319b5d83e1ab1aa8dd310c4ceedd`

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

- binary_hash: `1798a1a33811d27465eba1dc23835b2d6b8ad3e29acd13c52d07a34041e1a007`
- interpretation: shared-memory bank count inferred from cycles-per-access vs stride curve
- mapping_contract: bank-stride sweep peak conflict factor → simulator shared-memory bank count

**assumptions:**

- single warp probes shared memory with stride sweep covering conflict-factors 1..32
- conflict factor reported as gcd(stride, 32) which holds for shipping NVIDIA archs

### Metrics

| key | value | unit |
| --- | --- | --- |
| `full_conflict_cycles_per_access` | 65.345 | cycles |
| `inferred_bank_count` | 32 | — |
| `no_conflict_cycles_per_access` | 6.9387 | cycles |
| `sweep_points` | 12 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 1798a1a33811d27465eba1dc23835b2d6b8ad3e29acd13c52d07a34041e1a007 |
| `device_name` | Tesla V100-SXM2-32GB |
| `inner_loops` | 4096 |

#### `sweep` (12 rows)

| conflict_factor | cycles_median | cycles_per_access | stride |
| --- | --- | --- | --- |
| 1 | 28421 | 6.9387 | 1 |
| 2 | 28677 | 7.0012 | 2 |
| 1 | 28421 | 6.9387 | 3 |
| 4 | 38917 | 9.5012 | 4 |
| 1 | 28421 | 6.9387 | 5 |
| 1 | 28421 | 6.9387 | 7 |
| 8 | 71045 | 17.345 | 8 |
| 1 | 28421 | 6.9387 | 11 |
| 16 | 136581 | 33.345 | 16 |
| 1 | 28421 | 6.9387 | 17 |
| 32 | 267653 | 65.345 | 32 |
| 1 | 28421 | 6.9387 | 33 |

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/shared_memory/bank_stride.cu`
- bytes: `4535`  ·  sha256: `deee44a45bdafc0270320e62f77d735ec08f3c481b8db8985cbb6ec1bf0f7de9`

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
  "bank_serialization_factor": 9.417470131292605,
  "shared_load_latency_cycles": 26.9988
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `bank_count` | 32 | — |
| `bank_serialization_factor` | 9.41747 | — |
| `shared_load_latency_cycles` | 26.9988 | — |

### Raw values

<details><summary><code>bank_stride</code> (JSON)</summary>

```json
{
  "binary_sha256": "1798a1a33811d27465eba1dc23835b2d6b8ad3e29acd13c52d07a34041e1a007",
  "full_conflict_cycles_per_access": 65.345,
  "inferred_bank_count": 32,
  "no_conflict_cycles_per_access": 6.9387
}
```

</details>

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "bank_count": 32,
  "bank_serialization_factor": 9.417470131292605,
  "shared_load_latency_cycles": 26.9988
}
```

</details>

<details><summary><code>pointer_chase</code> (JSON)</summary>

```json
{
  "binary_sha256": "2863c0360952265661278e7379759fba28c7f4f83e42d8e4dd6b4164f00eec38",
  "cycles_per_load": 26.9988
}
```

</details>

[↑ contents](#contents)

---
