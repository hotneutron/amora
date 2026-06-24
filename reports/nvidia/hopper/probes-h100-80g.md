# nvidia / hopper / h100-80g — Probe Results

- Generated: 2026-06-24T09:48Z
- Device: NVIDIA H100 80GB HBM3  ·  Backend: `nvidia_cuda`  ·  Probes: 36
- `fit_status`: `behavioral_only`=6, `bounded`=8, `conditionally_identified`=4, `direct`=6, `underconstrained`=5, `uniquely_identified`=7
- Back to [family index](README.md)

<a id="contents"></a>
## Contents

[topology.device_attributes](#topologydevice_attributes) · [topology.occupancy](#topologyoccupancy) · [topology.persistent_cta](#topologypersistent_cta) · [arithmetic_latency.dependent_chain](#arithmetic_latencydependent_chain) · [arithmetic_throughput.independent_chains](#arithmetic_throughputindependent_chains) · [shared_memory.pointer_chase](#shared_memorypointer_chase) · [shared_memory.bank_stride](#shared_memorybank_stride) · [shared_memory.analyze](#shared_memoryanalyze) · [l1_cache.pointer_chase](#l1_cachepointer_chase) · [l1_cache.working_set](#l1_cacheworking_set) · [l1_cache.conflict_sets](#l1_cacheconflict_sets) · [l1_cache.analyze](#l1_cacheanalyze) · [scheduler_policy.ready_warps](#scheduler_policyready_warps) · [scheduler_policy.mixed_issue](#scheduler_policymixed_issue) · [scheduler_policy.analyze](#scheduler_policyanalyze) · [register_file.register_bank_sweep](#register_fileregister_bank_sweep) · [register_file.register_latency](#register_fileregister_latency) · [register_file.analyze](#register_fileanalyze) · [synchronization.barrier_latency](#synchronizationbarrier_latency) · [global_memory.streaming](#global_memorystreaming) · [l2_cache.pointer_chase](#l2_cachepointer_chase) · [memory_pipeline.outstanding_requests](#memory_pipelineoutstanding_requests) · [memory_pipeline.lane_patterns](#memory_pipelinelane_patterns) · [memory_pipeline.analyze](#memory_pipelineanalyze) · [global_memory.partition_sweep](#global_memorypartition_sweep) · [global_memory.row_policy_sweep](#global_memoryrow_policy_sweep) · [global_memory.analyze](#global_memoryanalyze) · [tensor_core.mma_latency](#tensor_coremma_latency) · [tensor_core.mma_throughput](#tensor_coremma_throughput) · [synchronization.fence_latency](#synchronizationfence_latency) · [tma_copy.async_copy_latency](#tma_copyasync_copy_latency) · [tma_copy.tma_transfer_sweep](#tma_copytma_transfer_sweep) · [tma_copy.analyze](#tma_copyanalyze) · [interconnect.injection_rate](#interconnectinjection_rate) · [interconnect.address_mapping](#interconnectaddress_mapping) · [interconnect.analyze](#interconnectanalyze)

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
| `elapsed_ms` | 0.1911 | ms |
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
| `elapsed_ms` | 0.1911 |
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
| `cycles_max` | 18796 |
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
| `cycles_max` | 118972 |
| `cycles_median` | 118844 |
| `cycles_min` | 118844 |
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
| measurement | `l1_hit_load_latency` = 70.6099 cycles |
| simulator_param | `l1_latency` = 70.6099 cycles |
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
| `dram_cycles_per_load` | 318.034 | cycles |
| `hit_to_dram_ratio` | 4.5041 | — |
| `l1_hit_cycles_per_load` | 70.6099 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | aea1c593856320979dc411cf66981a009cb13437e410ceff83a5256d388dc94b |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `dram_cycles_per_load` | 318.034 |
| `l1_hit_cycles_per_load` | 70.6099 |
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
| 47.3457 | 4 |
| 55.3882 | 8 |
| 70.6104 | 16 |
| 86.2385 | 24 |
| 101.23 | 32 |
| 128.846 | 48 |
| 151.023 | 64 |
| 202.856 | 128 |
| 238.55 | 256 |
| 264.627 | 512 |
| 275.799 | 1024 |
| 311.615 | 4096 |
| 352.978 | 16384 |

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
| 39.8386 | 3 |
| 39.9023 | 4 |
| 39.968 | 5 |
| 40.0256 | 6 |
| 40.0828 | 7 |
| 40.1455 | 8 |
| 40.1997 | 9 |
| 40.2581 | 10 |
| 40.3208 | 11 |
| 40.3865 | 12 |
| 40.4512 | 13 |
| 40.5073 | 14 |
| 40.5647 | 15 |
| 40.626 | 16 |
| 40.6826 | 17 |
| 40.7383 | 18 |
| 40.8005 | 19 |
| 40.866 | 20 |
| 40.9331 | 21 |
| 40.9895 | 22 |
| 41.0474 | 23 |
| 41.1099 | 24 |

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
  "l1_hit_latency_cycles": 70.6062
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `l1_effective_capacity_kb` | `{}` | — |
| `l1_hit_latency_cycles` | 70.6062 | — |

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
  "l1_hit_latency_cycles": 70.6062
}
```

</details>

<details><summary><code>pointer_chase</code> (JSON)</summary>

```json
{
  "binary_sha256": "aea1c593856320979dc411cf66981a009cb13437e410ceff83a5256d388dc94b",
  "l1_hit_cycles_per_load": 70.6062
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
| 9228 | 35.5093 | 5 |
| 9227 | 42.6158 | 6 |
| 9227 | 49.7184 | 7 |
| 9228 | 56.8149 | 8 |
| 9488 | 62.1653 | 9 |
| 9865 | 66.4328 | 10 |
| 9483 | 76.0198 | 11 |
| 9865 | 79.7194 | 12 |
| 10354 | 82.2839 | 13 |
| 10370 | 88.4768 | 14 |
| 10361 | 94.8789 | 15 |
| 10354 | 101.273 | 16 |
| 12583 | 88.541 | 17 |
| 12569 | 93.8538 | 18 |
| 12575 | 99.0206 | 19 |
| 12560 | 104.357 | 20 |
| 14887 | 92.4468 | 21 |
| 14892 | 96.8165 | 22 |
| 14898 | 101.177 | 23 |
| 14892 | 105.618 | 24 |
| 17288 | 94.7709 | 25 |
| 17296 | 98.5162 | 26 |
| 17306 | 102.246 | 27 |
| 17306 | 106.033 | 28 |
| 19730 | 96.3276 | 29 |
| 19709 | 99.7554 | 30 |
| 19718 | 103.034 | 31 |
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
| `fp32_ops_per_cycle` | 115.387 | ops/cycle |
| `int_ops_per_cycle` | 483.215 | ops/cycle |
| `mixed_ops_per_cycle` | 178.178 | ops/cycle |
| `overlap_ratio` | 0.368734 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 4826d4e62529a9a8c1404c12e237a322565b36f4b206c901dd1d0c6096e2038e |
| `chain_length` | 2048 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `fp32_ops_per_cycle` | 115.387 |
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

## synchronization.barrier_latency

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[64, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `uniquely_identified` |
| measurement | `cta_barrier_latency` = 45.2683 cycles |
| simulator_param | `barrier_latency` = 45.2683 cycles |
| concept | `cta_barrier_latency` |

- binary_hash: `b9d3f9842c0817d9f222f4cb3d49c2e6f961833e6873e705dc9cce8b3eac66e0`
- interpretation: cycles per __syncthreads() barrier for the measured CTA shape
- mapping_contract: cycles-per-barrier for a named CTA shape → simulator barrier latency (conditional)

**assumptions:**

- one CTA runs a long __syncthreads() loop with minimal inter-barrier work
- cycles-per-barrier reported for the smallest block; scaling curve retained
- barrier cost is occupancy-coupled; reported per the launch class measured

### Metrics

| key | value | unit |
| --- | --- | --- |
| `cycles_per_barrier` | 45.2683 | cycles |
| `sweep_points` | 5 | — |

### Raw values

| key | value |
| --- | --- |
| `barriers` | 4096 |
| `binary_sha256` | b9d3f9842c0817d9f222f4cb3d49c2e6f961833e6873e705dc9cce8b3eac66e0 |
| `device_name` | NVIDIA H100 80GB HBM3 |

#### `sweep` (5 rows)

| cycles_per_barrier | threads_per_block |
| --- | --- |
| 45.2683 | 64 |
| 49.2761 | 128 |
| 57.2917 | 256 |
| 74.2996 | 512 |
| 106.439 | 1024 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "932b58f27c0bd25f9c2103250de18e88688a22dc47ad357bce4c4c2e6fc904c1",
  "opcode_histogram": {
    "BAR": 9,
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 1,
    "IMAD": 2,
    "ISETP": 2,
    "LDC": 3,
    "LDS": 9,
    "LEA": 9,
    "MOV": 8,
    "NOP": 13,
    "S2R": 10,
    "S2UR": 1,
    "STG": 2,
    "STS": 9,
    "ULDC": 1,
    "ULEA": 1,
    "UMOV": 1,
    "VIADD": 9
  },
  "satisfied": [
    "BAR>=1 (9)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/synchronization/barrier_latency.cu`
- bytes: `3443`  ·  sha256: `247a7f3cbd7851717429c4368ad51992191dc098cabc27ffa3b5d208e3112af9`

### SASS validation

- validated: `True`
- disassembly_hash: `932b58f27c0bd25f9c2103250de18e88688a22dc47ad357bce4c4c2e6fc904c1`
- satisfied: BAR>=1 (9)
- opcode_histogram: `{"BAR": 9, "BRA": 2, "CS2R": 2, "EXIT": 2, "IADD3": 1, "IMAD": 2, "ISETP": 2, "LDC": 3, "LDS": 9, "LEA": 9, "MOV": 8, "NOP": 13, "S2R": 10, "S2UR": 1, "STG": 2, "STS": 9, "ULDC": 1, "ULEA": 1, "UMOV": 1, "VIADD": 9}`

[↑ contents](#contents)

---

## global_memory.streaming

| field | value |
| --- | --- |
| launch | `kernel`  — |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `dram_sustained_bandwidth` = _object_ (copy_gbps, peak_gbps, read_gbps, write_gbps) |
| simulator_param | `dram_bandwidth` = _object_ (copy_gbps, peak_gbps, read_gbps, write_gbps) |
| concept | `dram_streaming_bandwidth` |

- binary_hash: `cdebd98312f31519f81f9dde721a59205bf713a99250e0bfdbcf43e28d19460b`
- interpretation: sustained DRAM/HBM bandwidth per traffic class from streaming kernels
- mapping_contract: achieved sustained bandwidth per traffic class → simulator DRAM bandwidth (bounded)

**assumptions:**

- grid-stride read/write/copy over a working set far larger than cache
- best-of-N CUDA-event timing; bandwidth is bounded by clock variation
- copy moves 2x bytes (read+write); reported as achieved sustained GB/s

### Measurement value

```json
{
  "copy_gbps": 2896.62,
  "peak_gbps": 3138.86,
  "read_gbps": 3072.75,
  "write_gbps": 3138.86
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `copy_gbps` | 2896.62 | GB/s |
| `peak_gbps` | 3138.86 | GB/s |
| `read_gbps` | 3072.75 | GB/s |
| `write_gbps` | 3138.86 | GB/s |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | cdebd98312f31519f81f9dde721a59205bf713a99250e0bfdbcf43e28d19460b |
| `copy_gbps` | 2896.62 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `iters` | 5 |
| `read_gbps` | 3072.75 |
| `working_set_mb` | 512 |
| `write_gbps` | 3138.86 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "c30763fca850ef9713f0567d26a00cfeb9ccda321a6041b92e7f4b012bedf49b",
  "opcode_histogram": {
    "BRA": 2,
    "EXIT": 2,
    "IADD3": 5,
    "IMAD": 7,
    "ISETP": 4,
    "LDC": 2,
    "LDG": 1,
    "NOP": 14,
    "S2R": 1,
    "S2UR": 1,
    "SHF": 1,
    "STG": 1,
    "ULDC": 5,
    "UMOV": 1,
    "VIADD": 1
  },
  "satisfied": [
    "LDG>=1 (1)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/global_memory/streaming.cu`
- bytes: `4889`  ·  sha256: `fb36b1238f0bfc414d6c1cfafe8b60b3351beff0a18e535a42228578cd083ea4`

### SASS validation

- validated: `True`
- disassembly_hash: `c30763fca850ef9713f0567d26a00cfeb9ccda321a6041b92e7f4b012bedf49b`
- satisfied: LDG>=1 (1)
- opcode_histogram: `{"BRA": 2, "EXIT": 2, "IADD3": 5, "IMAD": 7, "ISETP": 4, "LDC": 2, "LDG": 1, "NOP": 14, "S2R": 1, "S2UR": 1, "SHF": 1, "STG": 1, "ULDC": 5, "UMOV": 1, "VIADD": 1}`

[↑ contents](#contents)

---

## l2_cache.pointer_chase

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `l2_hit_load_latency` = 329.514 cycles |
| simulator_param | `l2_latency` = 329.514 cycles |
| concept | `l2_hit_latency` |

- binary_hash: `8d6ad27fcbe7009059299109b14e3bf004478562d72c76abac457afc77d95657`
- interpretation: dependent-load latency for an L2-resident working set in cycles
- mapping_contract: dependent L2-resident chase cycles-per-load -> simulator L2 hit latency (bounded)

**assumptions:**

- single-thread dependent pointer chase over a randomized ring sized to exceed L1 but fit L2
- a DRAM-resident ring is timed as a control; L2-hit regime requires l2 << dram
- median cycles-per-load reported across N launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `dram_cycles_per_load` | 648.649 | cycles |
| `hit_to_dram_ratio` | 1.9685 | — |
| `l2_hit_cycles_per_load` | 329.514 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 8d6ad27fcbe7009059299109b14e3bf004478562d72c76abac457afc77d95657 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `dram_cycles_per_load` | 648.649 |
| `dram_kb` | 131072 |
| `l2_hit_cycles_per_load` | 329.514 |
| `l2_kb` | 4096 |
| `repeats` | 64 |
| `steps` | 4096 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "57854a182316821c4e71649538cd6cf750f2c073b7c32a1c05f7c54b21a15e6f",
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

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/l2_cache/pointer_chase.cu`
- bytes: `5571`  ·  sha256: `75cd7ad66553dcc3219494c7902e6c0d93b86f77f03f490480bb5a09755f6ea9`

### SASS validation

- validated: `True`
- disassembly_hash: `57854a182316821c4e71649538cd6cf750f2c073b7c32a1c05f7c54b21a15e6f`
- satisfied: LDG>=1 (17)
- opcode_histogram: `{"BRA": 6, "CS2R": 2, "EXIT": 2, "IADD3": 6, "IMAD": 17, "ISETP": 5, "LDC": 6, "LDG": 17, "LOP3": 2, "MOV": 1, "NOP": 10, "S2R": 1, "S2UR": 1, "STG": 2, "ULDC": 2}`

[↑ contents](#contents)

---

## memory_pipeline.outstanding_requests

| field | value |
| --- | --- |
| launch | `kernel`  grid=None block=[256, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `effective_outstanding_requests` = 4 loads |
| simulator_param | `ldst_queue_capacity` = 4 loads |
| concept | `memory_level_parallelism` |

- binary_hash: `bee80266eac9776cceb7e972e46446936f31d5ea3932f2c630bf47df7ff9dd4b`
- interpretation: in-flight independent loads at which memory throughput saturates
- mapping_contract: outstanding-load saturation knee -> simulator load/store queue capacity (bounded)

**assumptions:**

- each thread issues a swept number of independent global loads before consuming them
- a single wave of blocks is launched so throughput is bound by outstanding-request capacity
- saturation knee = smallest in-flight count reaching >=95% of peak bytes/cycle

### Metrics

| key | value | unit |
| --- | --- | --- |
| `effective_outstanding_requests` | 4 | loads |
| `sweep_points` | 6 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | bee80266eac9776cceb7e972e46446936f31d5ea3932f2c630bf47df7ff9dd4b |
| `buffer_mb` | 256 |
| `device_name` | NVIDIA H100 80GB HBM3 |

#### `sweep` (6 rows)

| bytes_per_cycle | in_flight |
| --- | --- |
| 621.246 | 1 |
| 1173.3 | 2 |
| 1620.45 | 4 |
| 1427.32 | 8 |
| 1227.18 | 16 |
| 1189.98 | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "2c0bedbeb0ec3bffed89587ef659094e73c87dee3353ab7c951ae5b4a00e766a",
  "opcode_histogram": {
    "BRA": 3,
    "BSSY": 2,
    "BSYNC": 2,
    "CS2R": 4,
    "EXIT": 2,
    "FADD": 128,
    "FSETP": 1,
    "HFMA2": 1,
    "IADD3": 50,
    "IMAD": 46,
    "ISETP": 102,
    "LDC": 5,
    "LDG": 32,
    "LEA": 4,
    "MOV": 13,
    "NOP": 9,
    "PLOP3": 1,
    "S2R": 1,
    "S2UR": 1,
    "STG": 2,
    "ULDC": 6,
    "USHF": 1
  },
  "satisfied": [
    "LDG>=1 (32)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/memory_pipeline/outstanding_requests.cu`
- bytes: `5450`  ·  sha256: `259f29323a8c75b0145391d145b15705f40c65db79252bb0b9f7607030db5e96`

### SASS validation

- validated: `True`
- disassembly_hash: `2c0bedbeb0ec3bffed89587ef659094e73c87dee3353ab7c951ae5b4a00e766a`
- satisfied: LDG>=1 (32)
- opcode_histogram: `{"BRA": 3, "BSSY": 2, "BSYNC": 2, "CS2R": 4, "EXIT": 2, "FADD": 128, "FSETP": 1, "HFMA2": 1, "IADD3": 50, "IMAD": 46, "ISETP": 102, "LDC": 5, "LDG": 32, "LEA": 4, "MOV": 13, "NOP": 9, "PLOP3": 1, "S2R": 1, "S2UR": 1, "STG": 2, "ULDC": 6, "USHF": 1}`

[↑ contents](#contents)

---

## memory_pipeline.lane_patterns

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `direct_counter` |
| fit_status | `direct` |
| measurement | `coalescing_sectors_per_request` = 32 sectors/request |
| simulator_param | `memory_coalescing_rule` = 32 sectors/request |
| concept | `memory_coalescing` |

- binary_hash: `a361eac8d2978edc47a74f0e885a9a58c324d57e0082095089efc3b06b4f571c`
- interpretation: global-load sectors per request from controlled lane address patterns
- mapping_contract: NCU sectors/request under lane patterns -> simulator memory coalescing rule

**assumptions:**

- one warp issues many global loads under named lane address patterns
- NCU sectors/request is the primary coalescing signal; timing only confirms LDG activity
- max counter value across profiled lane patterns characterizes the worst-case coalescing

### Metrics

| key | value | unit |
| --- | --- | --- |
| `global_load_requests` | 4096 | — |
| `global_load_sectors` | 131072 | — |
| `sectors_per_request` | 32 | sectors/request |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | a361eac8d2978edc47a74f0e885a9a58c324d57e0082095089efc3b06b4f571c |
| `device_name` | NVIDIA H100 80GB HBM3 |

#### `patterns` (4 rows)

| iters | name |
| --- | --- |
| 4096 | contiguous |
| 4096 | stride2 |
| 4096 | stride32 |
| 4096 | broadcast |

<details><summary><code>ncu</code> (JSON)</summary>

```json
{
  "global_load_requests": {
    "launches_profiled": 4,
    "logical": "global_load_requests",
    "metric": "l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum",
    "role": "primary",
    "value": 4096.0
  },
  "global_load_sectors": {
    "launches_profiled": 4,
    "logical": "global_load_sectors",
    "metric": "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum",
    "role": "primary",
    "value": 131072.0
  },
  "sectors_per_request": 32.0
}
```

</details>

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "12052dc5cb5f1b5d96e9aec452a4b66a115cb4624fe2688209a7a2b4e39b256f",
  "opcode_histogram": {
    "BRA": 30,
    "BSSY": 7,
    "BSYNC": 7,
    "CALL": 8,
    "EXIT": 3,
    "F2I": 9,
    "FADD": 8,
    "FSETP": 1,
    "HFMA2": 6,
    "I2F": 9,
    "IADD3": 32,
    "IMAD": 120,
    "ISETP": 49,
    "LDC": 7,
    "LDG": 8,
    "LEA": 19,
    "LOP3": 16,
    "MOV": 21,
    "MUFU": 9,
    "NOP": 12,
    "RET": 1,
    "S2R": 1,
    "SEL": 6,
    "SHF": 2,
    "STG": 1,
    "UIADD3": 4,
    "ULDC": 17,
    "ULOP3": 1,
    "UMOV": 4,
    "VIADD": 22
  },
  "satisfied": [
    "LDG>=1 (8)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/memory_pipeline/lane_patterns.cu`
- bytes: `3801`  ·  sha256: `e1a7b386e271d8f7fb446bfcf766ec8118e1193d5397dee1daef39f17e3ed6b6`

### SASS validation

- validated: `True`
- disassembly_hash: `12052dc5cb5f1b5d96e9aec452a4b66a115cb4624fe2688209a7a2b4e39b256f`
- satisfied: LDG>=1 (8)
- opcode_histogram: `{"BRA": 30, "BSSY": 7, "BSYNC": 7, "CALL": 8, "EXIT": 3, "F2I": 9, "FADD": 8, "FSETP": 1, "HFMA2": 6, "I2F": 9, "IADD3": 32, "IMAD": 120, "ISETP": 49, "LDC": 7, "LDG": 8, "LEA": 19, "LOP3": 16, "MOV": 21, "MUFU": 9, "NOP": 12, "RET": 1, "S2R": 1, "SEL": 6, "SHF": 2, "STG": 1, "UIADD3": 4, "ULDC": 17, "ULOP3": 1, "UMOV": 4, "VIADD": 22}`

[↑ contents](#contents)

---

## memory_pipeline.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `bounded` |
| measurement | `memory_pipeline_summary` = _object_ (coalescing_sectors_per_request, effective_outstanding_requests) |
| simulator_param | `memory_pipeline_summary` = _object_ (coalescing_sectors_per_request, effective_outstanding_requests) |
| concept | `memory_pipeline_summary` |

- interpretation: merged memory-pipeline characterization from coalescing and outstanding-request probes
- mapping_contract: cross-probe memory-pipeline summary for simulator coalescing + load/store-queue parameters

**assumptions:**

- merges lane-pattern coalescing (sectors/request) with the outstanding-request knee
- merged fit status is the weakest of the contributing probe fits

### Measurement value

```json
{
  "coalescing_sectors_per_request": 32.0,
  "effective_outstanding_requests": 4
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `coalescing_sectors_per_request` | 32 | — |
| `effective_outstanding_requests` | 4 | — |

### Raw values

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "coalescing_sectors_per_request": 32.0,
  "effective_outstanding_requests": 4
}
```

</details>

<details><summary><code>lane_patterns</code> (JSON)</summary>

```json
{
  "binary_sha256": "a361eac8d2978edc47a74f0e885a9a58c324d57e0082095089efc3b06b4f571c",
  "sectors_per_request": 32.0
}
```

</details>

<details><summary><code>outstanding_requests</code> (JSON)</summary>

```json
{
  "binary_sha256": "bee80266eac9776cceb7e972e46446936f31d5ea3932f2c630bf47df7ff9dd4b",
  "effective_outstanding_requests": 4
}
```

</details>

[↑ contents](#contents)

---

## global_memory.partition_sweep

| field | value |
| --- | --- |
| launch | `kernel`  — |
| evidence_tier | `timing_direct` |
| fit_status | `behavioral_only` |
| measurement | `partition_camping_class` = balanced |
| simulator_param | `memory_partition_class` = balanced |
| concept | `memory_partition_behavior` |

- binary_hash: `a9da48d787f1ca31a6343f61f8308fd18bf70fde63962b240b9ead6e94a81950`
- interpretation: DRAM partition-camping sensitivity from base-offset bandwidth sweep
- mapping_contract: base-offset bandwidth variation -> simulator memory-partition camping class

**assumptions:**

- grid-stride read from several base offsets relative to the partition interleave
- best-of-N CUDA-event timing per offset; bandwidth varies with clock/partition balance
- max/min bandwidth ratio < 1.15 classifies as balanced else camping_sensitive

### Metrics

| key | value | unit |
| --- | --- | --- |
| `bandwidth_ratio` | 1.00633 | — |
| `max_gbps` | 3092.77 | GB/s |
| `min_gbps` | 3073.32 | GB/s |
| `partition_camping_class` | balanced | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | a9da48d787f1ca31a6343f61f8308fd18bf70fde63962b240b9ead6e94a81950 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `working_set_mb` | 512 |

#### `sweep` (6 rows)

| gbps | offset_kb |
| --- | --- |
| 3078.39 | 0 |
| 3092.77 | 256 |
| 3078.21 | 512 |
| 3083.5 | 768 |
| 3088.81 | 1024 |
| 3073.32 | 1536 |

<details><summary><code>ncu</code> (JSON)</summary>

```json
{
  "launches_profiled": 16,
  "logical": "dram_bytes_read",
  "metric": "dram__bytes_read.sum",
  "role": "corroboration",
  "value": 536903424.0
}
```

</details>

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "2f746a377da2909761fa342fa3d8912032b518c7d3abb0eb825132b23cfa52d5",
  "opcode_histogram": {
    "BRA": 2,
    "BSSY": 1,
    "BSYNC": 1,
    "EXIT": 3,
    "FADD": 4,
    "FSETP": 1,
    "HFMA2": 1,
    "IADD3": 4,
    "IMAD": 5,
    "ISETP": 4,
    "LDC": 3,
    "LDG": 1,
    "LEA": 2,
    "NOP": 15,
    "S2R": 1,
    "S2UR": 1,
    "STG": 1,
    "ULDC": 5,
    "UMOV": 1
  },
  "satisfied": [
    "LDG>=1 (1)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/global_memory/partition_sweep.cu`
- bytes: `4041`  ·  sha256: `d386c53533f50ae549669e8ad6a650e828d196ce75a03339b2356f039bb55147`

### SASS validation

- validated: `True`
- disassembly_hash: `2f746a377da2909761fa342fa3d8912032b518c7d3abb0eb825132b23cfa52d5`
- satisfied: LDG>=1 (1)
- opcode_histogram: `{"BRA": 2, "BSSY": 1, "BSYNC": 1, "EXIT": 3, "FADD": 4, "FSETP": 1, "HFMA2": 1, "IADD3": 4, "IMAD": 5, "ISETP": 4, "LDC": 3, "LDG": 1, "LEA": 2, "NOP": 15, "S2R": 1, "S2UR": 1, "STG": 1, "ULDC": 5, "UMOV": 1}`

[↑ contents](#contents)

---

## global_memory.row_policy_sweep

| field | value |
| --- | --- |
| launch | `kernel`  — |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `row_locality_sensitivity` = 1.7605 ratio |
| simulator_param | `dram_row_policy_class` = 1.7605 ratio |
| concept | `dram_row_locality` |

- binary_hash: `eff230df9f9dd83883c5a742f90c4c94c4b2d51b3ab45063bcdc60b00d4c7752`
- interpretation: DRAM row-locality sensitivity from a stride bandwidth sweep
- mapping_contract: stride bandwidth spread -> simulator DRAM row-buffer policy class (bounded)

**assumptions:**

- grid-stride read with several element strides to vary DRAM row-buffer locality
- best-of-N CUDA-event timing per stride; bandwidth bounded by clock variation
- row_locality_sensitivity = best_gbps / worst_gbps across strides (bounded)

### Metrics

| key | value | unit |
| --- | --- | --- |
| `best_gbps` | 1951.43 | GB/s |
| `row_locality_sensitivity` | 1.7605 | — |
| `worst_gbps` | 1108.45 | GB/s |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | eff230df9f9dd83883c5a742f90c4c94c4b2d51b3ab45063bcdc60b00d4c7752 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `working_set_mb` | 512 |

#### `sweep` (4 rows)

| gbps | stride |
| --- | --- |
| 1925.65 | 1 |
| 1951.43 | 8 |
| 1389.33 | 64 |
| 1108.45 | 512 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "c8f2bf60e102a1a9dc0d829041ec98c142a9cabeb4fdd3da8cb6ff4aa6b2b2ff",
  "opcode_histogram": {
    "BRA": 15,
    "BSSY": 5,
    "BSYNC": 5,
    "CALL": 5,
    "EXIT": 3,
    "F2I": 6,
    "FADD": 5,
    "FSETP": 1,
    "HFMA2": 6,
    "I2F": 6,
    "IADD3": 36,
    "IMAD": 99,
    "ISETP": 31,
    "LDC": 4,
    "LDG": 5,
    "LEA": 12,
    "LOP3": 11,
    "MOV": 15,
    "MUFU": 6,
    "NOP": 14,
    "RET": 1,
    "S2R": 2,
    "SEL": 6,
    "SHF": 3,
    "STG": 1,
    "UIMAD": 1,
    "ULDC": 10,
    "VIADD": 14
  },
  "satisfied": [
    "LDG>=1 (5)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/global_memory/row_policy_sweep.cu`
- bytes: `4021`  ·  sha256: `6d12a1a626488fcec2d13de1676d0160c0c0fc5e21969abfbdb9d678476bd012`

### SASS validation

- validated: `True`
- disassembly_hash: `c8f2bf60e102a1a9dc0d829041ec98c142a9cabeb4fdd3da8cb6ff4aa6b2b2ff`
- satisfied: LDG>=1 (5)
- opcode_histogram: `{"BRA": 15, "BSSY": 5, "BSYNC": 5, "CALL": 5, "EXIT": 3, "F2I": 6, "FADD": 5, "FSETP": 1, "HFMA2": 6, "I2F": 6, "IADD3": 36, "IMAD": 99, "ISETP": 31, "LDC": 4, "LDG": 5, "LEA": 12, "LOP3": 11, "MOV": 15, "MUFU": 6, "NOP": 14, "RET": 1, "S2R": 2, "SEL": 6, "SHF": 3, "STG": 1, "UIMAD": 1, "ULDC": 10, "VIADD": 14}`

[↑ contents](#contents)

---

## global_memory.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `behavioral_only` |
| measurement | `global_memory_summary` = _object_ (partition_class, peak_gbps, row_locality_sensitivity) |
| simulator_param | `global_memory_summary` = _object_ (partition_class, peak_gbps, row_locality_sensitivity) |
| concept | `global_memory_summary` |

- interpretation: merged global-memory characterization from streaming, partition, and row-locality probes
- mapping_contract: cross-probe global-memory summary for simulator DRAM bandwidth + partition + row-policy parameters

**assumptions:**

- merges streaming peak bandwidth, partition-camping class, and row-locality sensitivity
- merged fit status is the weakest of the contributing probe fits

### Measurement value

```json
{
  "partition_class": "balanced",
  "peak_gbps": 3142.98,
  "row_locality_sensitivity": 1.7528273142754727
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `partition_class` | balanced | — |
| `peak_gbps` | 3142.98 | — |
| `row_locality_sensitivity` | 1.75283 | — |

### Raw values

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "partition_class": "balanced",
  "peak_gbps": 3142.98,
  "row_locality_sensitivity": 1.7528273142754727
}
```

</details>

<details><summary><code>partition_sweep</code> (JSON)</summary>

```json
{
  "binary_sha256": "a9da48d787f1ca31a6343f61f8308fd18bf70fde63962b240b9ead6e94a81950",
  "partition_class": "balanced"
}
```

</details>

<details><summary><code>row_policy_sweep</code> (JSON)</summary>

```json
{
  "binary_sha256": "eff230df9f9dd83883c5a742f90c4c94c4b2d51b3ab45063bcdc60b00d4c7752",
  "row_locality_sensitivity": 1.7528273142754727
}
```

</details>

<details><summary><code>streaming</code> (JSON)</summary>

```json
{
  "binary_sha256": "cdebd98312f31519f81f9dde721a59205bf713a99250e0bfdbcf43e28d19460b",
  "peak_gbps": 3142.98
}
```

</details>

[↑ contents](#contents)

---

## tensor_core.mma_latency

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `uniquely_identified` |
| measurement | `tensor_mma_latency` = 24.4648 cycles_per_op |
| simulator_param | `tensor_core_mma_latency` = 24.4648 cycles_per_op |
| concept | `tensor_core_mma_latency` |

- binary_hash: `07c5b2d89de384194bfc0b9b31e724a5db1df99917c52a7bf15f0fd393c1870f`
- interpretation: dependent FP16 16x16x16 MMA latency in cycles
- mapping_contract: dependent MMA cycles-per-op -> simulator tensor-core pipeline latency

**assumptions:**

- dependent wmma::mma_sync chain (FP16 m16n16k16) timed via clock64 in one warp
- median across launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `cycles_median` | 12526 | — |
| `cycles_per_mma` | 24.4648 | cycles_per_op |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 07c5b2d89de384194bfc0b9b31e724a5db1df99917c52a7bf15f0fd393c1870f |
| `chain` | 512 |
| `cycles_median` | 12526 |
| `cycles_per_mma` | 24.4648 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `mma_shape` | m16n16k16_fp16 |
| `repeats` | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "1891aada7634856f0a86fb35bcc917f5d6be995f799465179ce082db1458e69c",
  "opcode_histogram": {
    "BRA": 1,
    "CS2R": 2,
    "EXIT": 2,
    "HMMA": 1024,
    "IADD3": 1,
    "IMAD": 3,
    "ISETP": 3,
    "LDC": 2,
    "LDG": 8,
    "LEA": 6,
    "LOP3": 1,
    "NOP": 526,
    "S2R": 2,
    "S2UR": 1,
    "SHF": 1,
    "STG": 5,
    "ULDC": 4
  },
  "satisfied": [
    "HMMA>=1 (1024)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/tensor_core/mma_latency.cu`
- bytes: `4145`  ·  sha256: `4116de2f34110423c466efbb44e5e518a29864cf4fabbd1576ac8d031ce45c1b`

### SASS validation

- validated: `True`
- disassembly_hash: `1891aada7634856f0a86fb35bcc917f5d6be995f799465179ce082db1458e69c`
- satisfied: HMMA>=1 (1024)
- opcode_histogram: `{"BRA": 1, "CS2R": 2, "EXIT": 2, "HMMA": 1024, "IADD3": 1, "IMAD": 3, "ISETP": 3, "LDC": 2, "LDG": 8, "LEA": 6, "LOP3": 1, "NOP": 526, "S2R": 2, "S2UR": 1, "SHF": 1, "STG": 5, "ULDC": 4}`

[↑ contents](#contents)

---

## tensor_core.mma_throughput

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[128, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `uniquely_identified` |
| measurement | `tensor_mma_throughput` = 0.1598 mma/cycle |
| simulator_param | `tensor_core_initiation_interval` = 6.2578 cycles_per_op |
| concept | `tensor_core_mma_throughput` |

- binary_hash: `57fb5eb9fac953ae055b712cd2649ef7b251768cfbaf527d602c6a97d32ba015`
- interpretation: independent FP16 16x16x16 MMA throughput in MMA-ops per cycle per warp
- mapping_contract: independent MMA throughput -> simulator tensor-core initiation interval

**assumptions:**

- independent wmma::mma_sync accumulators (FP16 m16n16k16) expose ILP to saturate the tensor pipe
- median across launches; throughput reported per warp

### Metrics

| key | value | unit |
| --- | --- | --- |
| `cycles_median` | 6407 | — |
| `mma_per_cycle_per_warp` | 0.1598 | mma/cycle |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 57fb5eb9fac953ae055b712cd2649ef7b251768cfbaf527d602c6a97d32ba015 |
| `cycles_median` | 6407 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `iters` | 256 |
| `lanes` | 4 |
| `mma_per_cycle_per_warp` | 0.1598 |
| `mma_shape` | m16n16k16_fp16 |
| `warps` | 4 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "9eee915e7387048ed494f35b39a311c5baaaf63e497fadf2bd2bd552e2a2c547",
  "opcode_histogram": {
    "BRA": 1,
    "CS2R": 2,
    "EXIT": 2,
    "FADD": 4,
    "FSETP": 1,
    "HMMA": 256,
    "IADD3": 1,
    "IMAD": 4,
    "LDC": 3,
    "LDG": 6,
    "LEA": 4,
    "LOP3": 2,
    "NOP": 265,
    "S2R": 2,
    "S2UR": 1,
    "SHF": 1,
    "STG": 2,
    "ULDC": 3
  },
  "satisfied": [
    "HMMA>=1 (256)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/tensor_core/mma_throughput.cu`
- bytes: `4451`  ·  sha256: `da95487187f13dc1d97ec1d6b965387f783d7f4f0111d9f2000df9d6e4d8e83a`

### SASS validation

- validated: `True`
- disassembly_hash: `9eee915e7387048ed494f35b39a311c5baaaf63e497fadf2bd2bd552e2a2c547`
- satisfied: HMMA>=1 (256)
- opcode_histogram: `{"BRA": 1, "CS2R": 2, "EXIT": 2, "FADD": 4, "FSETP": 1, "HMMA": 256, "IADD3": 1, "IMAD": 4, "LDC": 3, "LDG": 6, "LEA": 4, "LOP3": 2, "NOP": 265, "S2R": 2, "S2UR": 1, "SHF": 1, "STG": 2, "ULDC": 3}`

[↑ contents](#contents)

---

## synchronization.fence_latency

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[256, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `conditionally_identified` |
| measurement | `memory_fence_latency` = 927.239 cycles |
| simulator_param | `fence_latency` = 927.239 cycles |
| concept | `memory_fence_latency` |

- binary_hash: `c085b9ee5779e1845e75772c38109e6fdd2945486e4072609f5fcd4fb8889dae`
- interpretation: net cycles per __threadfence() after subtracting empty-loop overhead
- mapping_contract: net per-fence cycles -> simulator memory fence latency (conditional)

**assumptions:**

- device-scope __threadfence() loop timed via clock64 in one CTA
- empty-loop baseline subtracted to remove loop/branch overhead from the per-fence cost
- net per-fence cost reflects fence scope as measured; median across launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `cycles_per_empty` | 0.0151 | — |
| `cycles_per_fence` | 927.254 | — |
| `net_cycles_per_fence` | 927.239 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | c085b9ee5779e1845e75772c38109e6fdd2945486e4072609f5fcd4fb8889dae |
| `cycles_per_empty` | 0.0151 |
| `cycles_per_fence` | 927.254 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `fences` | 4096 |
| `net_cycles_per_fence` | 927.239 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "36e81b9a3e65f8134290648d34c07de7aa66853d8bbe80bf636063a55bb13249",
  "opcode_histogram": {
    "BAR": 1,
    "BRA": 2,
    "CCTL": 8,
    "CGAERRBAR": 8,
    "CS2R": 2,
    "ERRBAR": 8,
    "EXIT": 2,
    "IADD3": 1,
    "IMAD": 2,
    "ISETP": 2,
    "LDC": 4,
    "LDG": 1,
    "MEMBAR": 8,
    "NOP": 9,
    "S2R": 1,
    "STG": 2,
    "ULDC": 1,
    "VIADD": 2
  },
  "satisfied": [
    "MEMBAR>=1 (8)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/synchronization/fence_latency.cu`
- bytes: `4401`  ·  sha256: `94d5e03aba0ff1d6281cd58fed7ffe508fc6c464b0a6d107f9f6766355528552`

### SASS validation

- validated: `True`
- disassembly_hash: `36e81b9a3e65f8134290648d34c07de7aa66853d8bbe80bf636063a55bb13249`
- satisfied: MEMBAR>=1 (8)
- opcode_histogram: `{"BAR": 1, "BRA": 2, "CCTL": 8, "CGAERRBAR": 8, "CS2R": 2, "ERRBAR": 8, "EXIT": 2, "IADD3": 1, "IMAD": 2, "ISETP": 2, "LDC": 4, "LDG": 1, "MEMBAR": 8, "NOP": 9, "S2R": 1, "STG": 2, "ULDC": 1, "VIADD": 2}`

[↑ contents](#contents)

---

## tma_copy.async_copy_latency

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[128, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `conditionally_identified` |
| measurement | `async_copy_tile_latency` = 723.672 cycles |
| simulator_param | `async_copy_completion_latency` = 723.672 cycles |
| concept | `async_copy_latency` |

- binary_hash: `7ddaad00b56af1c7ea60f9df42d3e8ca2d9a04eb45fb0a05a3633fa4517bd65b`
- interpretation: cycles per cp.async-staged tile (issue->wait->use)
- mapping_contract: async-copy issue->wait->use cycles -> simulator async-copy completion latency (conditional)

**assumptions:**

- one CTA stages tiles global->shared via cp.async (__pipeline_memcpy_async)
- cycles-per-tile brackets issue->commit->wait->use with clock64()
- completion latency is conditional on cp.async being emitted (LDGSTS)

### Metrics

| key | value | unit |
| --- | --- | --- |
| `bytes_per_tile` | 4096 | — |
| `cycles_per_tile` | 723.672 | cycles |
| `tiles` | 64 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 7ddaad00b56af1c7ea60f9df42d3e8ca2d9a04eb45fb0a05a3633fa4517bd65b |
| `bytes_per_tile` | 4096 |
| `cycles_per_tile` | 723.672 |
| `device_name` | NVIDIA H100 80GB HBM3 |
| `tiles` | 64 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "605531674cd1f9df974bcab0a3cc7b7edef1b8e1b6133a651a52b73f4f4477bb",
  "opcode_histogram": {
    "BAR": 3,
    "BRA": 15,
    "BREAK": 1,
    "BSSY": 6,
    "BSYNC": 6,
    "CS2R": 2,
    "DEPBAR": 1,
    "EXIT": 2,
    "F2I": 1,
    "HFMA2": 1,
    "I2F": 1,
    "IADD3": 32,
    "IMAD": 56,
    "ISETP": 19,
    "LDC": 7,
    "LDGDEPBAR": 1,
    "LDGSTS": 1,
    "LDS": 31,
    "LEA": 6,
    "LOP3": 3,
    "MUFU": 1,
    "NOP": 14,
    "PLOP3": 3,
    "S2R": 1,
    "S2UR": 1,
    "SHF": 1,
    "STG": 2,
    "ULDC": 3,
    "ULEA": 2,
    "UMOV": 1,
    "USHF": 3,
    "VIADD": 9,
    "WARPSYNC": 1
  },
  "satisfied": [
    "LDGSTS>=1 (1)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/tma_copy/async_copy_latency.cu`
- bytes: `5026`  ·  sha256: `7c2c7184a938a323596a5c6327c81baeeb2f23d82c3c44cb3a6d7e88a9e78a0d`

### SASS validation

- validated: `True`
- disassembly_hash: `605531674cd1f9df974bcab0a3cc7b7edef1b8e1b6133a651a52b73f4f4477bb`
- satisfied: LDGSTS>=1 (1)
- opcode_histogram: `{"BAR": 3, "BRA": 15, "BREAK": 1, "BSSY": 6, "BSYNC": 6, "CS2R": 2, "DEPBAR": 1, "EXIT": 2, "F2I": 1, "HFMA2": 1, "I2F": 1, "IADD3": 32, "IMAD": 56, "ISETP": 19, "LDC": 7, "LDGDEPBAR": 1, "LDGSTS": 1, "LDS": 31, "LEA": 6, "LOP3": 3, "MUFU": 1, "NOP": 14, "PLOP3": 3, "S2R": 1, "S2UR": 1, "SHF": 1, "STG": 2, "ULDC": 3, "ULEA": 2, "UMOV": 1, "USHF": 3, "VIADD": 9, "WARPSYNC": 1}`

[↑ contents](#contents)

---

## tma_copy.tma_transfer_sweep

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[256, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `async_copy_throughput` = 37.46 GB/s |
| simulator_param | `tma_transfer_throughput` = 37.46 GB/s |
| concept | `async_copy_throughput` |

- binary_hash: `a63a25f79ed2fa621135f704b3dfde2a8713213c703655ea86c15b7166b85a31`
- interpretation: peak achieved cp.async global->shared bandwidth across tile sizes
- mapping_contract: peak async-copy bandwidth across tile sizes -> simulator async-copy transfer throughput (bounded)

**assumptions:**

- one CTA bulk-stages global->shared via cp.async over a tile-size sweep
- best-of-N CUDA-event timing per size; throughput is bounded by clock variation
- peak GB/s across the sweep is reported as achieved async-copy bandwidth

### Metrics

| key | value | unit |
| --- | --- | --- |
| `peak_gbps` | 37.46 | GB/s |
| `sweep_points` | 4 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | a63a25f79ed2fa621135f704b3dfde2a8713213c703655ea86c15b7166b85a31 |
| `device_name` | NVIDIA H100 80GB HBM3 |

#### `sweep` (4 rows)

| gbps | tile_kb |
| --- | --- |
| 2 | 1 |
| 7.35 | 4 |
| 22.57 | 16 |
| 37.46 | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "eb509538763f52da23d1c1ae20936490396b7d2f008c423d6bc414c1e99733ab",
  "opcode_histogram": {
    "BAR": 2,
    "BRA": 10,
    "BSSY": 3,
    "BSYNC": 3,
    "DEPBAR": 1,
    "EXIT": 2,
    "F2I": 1,
    "I2F": 1,
    "IADD3": 9,
    "IMAD": 29,
    "ISETP": 13,
    "LDC": 7,
    "LDGDEPBAR": 1,
    "LDGSTS": 1,
    "LDS": 7,
    "LEA": 6,
    "LOP3": 3,
    "MOV": 1,
    "MUFU": 1,
    "NOP": 8,
    "S2R": 1,
    "S2UR": 1,
    "SHF": 3,
    "STG": 1,
    "UIADD3": 1,
    "ULDC": 1,
    "ULEA": 1,
    "UMOV": 2,
    "USHF": 1,
    "VIADD": 4
  },
  "satisfied": [
    "LDGSTS>=1 (1)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/tma_copy/tma_transfer_sweep.cu`
- bytes: `4077`  ·  sha256: `d751a4fa721ffdc7f16e6e508af89a8567476f94b65068d814e2d150fe958ddf`

### SASS validation

- validated: `True`
- disassembly_hash: `eb509538763f52da23d1c1ae20936490396b7d2f008c423d6bc414c1e99733ab`
- satisfied: LDGSTS>=1 (1)
- opcode_histogram: `{"BAR": 2, "BRA": 10, "BSSY": 3, "BSYNC": 3, "DEPBAR": 1, "EXIT": 2, "F2I": 1, "I2F": 1, "IADD3": 9, "IMAD": 29, "ISETP": 13, "LDC": 7, "LDGDEPBAR": 1, "LDGSTS": 1, "LDS": 7, "LEA": 6, "LOP3": 3, "MOV": 1, "MUFU": 1, "NOP": 8, "S2R": 1, "S2UR": 1, "SHF": 3, "STG": 1, "UIADD3": 1, "ULDC": 1, "ULEA": 1, "UMOV": 2, "USHF": 1, "VIADD": 4}`

[↑ contents](#contents)

---

## tma_copy.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `bounded` |
| measurement | `async_copy_summary` = _object_ (async_copy_peak_gbps, async_copy_tile_latency) |
| simulator_param | `async_copy_summary` = _object_ (async_copy_peak_gbps, async_copy_tile_latency) |
| concept | `async_copy_summary` |

- interpretation: merged async-copy characterization from latency and throughput probes
- mapping_contract: cross-probe async-copy summary for simulator async-copy latency + throughput parameters

**assumptions:**

- merges async-copy tile latency and peak async-copy throughput
- merged fit status is the weakest of the contributing probe fits

### Measurement value

```json
{
  "async_copy_peak_gbps": 37.5,
  "async_copy_tile_latency": 724.0938
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `async_copy_peak_gbps` | 37.5 | — |
| `async_copy_tile_latency` | 724.094 | — |

### Raw values

<details><summary><code>async_copy_latency</code> (JSON)</summary>

```json
{
  "async_copy_tile_latency": 724.0938,
  "binary_sha256": "7ddaad00b56af1c7ea60f9df42d3e8ca2d9a04eb45fb0a05a3633fa4517bd65b"
}
```

</details>

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "async_copy_peak_gbps": 37.5,
  "async_copy_tile_latency": 724.0938
}
```

</details>

<details><summary><code>tma_transfer_sweep</code> (JSON)</summary>

```json
{
  "async_copy_peak_gbps": 37.5,
  "binary_sha256": "a63a25f79ed2fa621135f704b3dfde2a8713213c703655ea86c15b7166b85a31"
}
```

</details>

[↑ contents](#contents)

---

## interconnect.injection_rate

| field | value |
| --- | --- |
| launch | `kernel`  — |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `injection_saturation_gbps` = 3064.33 GB/s |
| simulator_param | `interconnect_injection_bandwidth` = 3064.33 GB/s |
| concept | `interconnect_injection` |

- binary_hash: `ce58ed8dc33e87bcc371366ffd09984a193f2f4a8a74fce1cb909a8d48f5ff38`
- interpretation: peak aggregate injection bandwidth vs offered load (blocks per SM)
- mapping_contract: peak aggregate injection bandwidth vs offered load -> simulator interconnect injection bandwidth (bounded)

**assumptions:**

- multi-SM grid-stride stream over a working set far larger than cache
- offered load swept via blocks-per-SM = {1,2,4,8}; best-of-N CUDA-event timing
- peak aggregate GB/s across offered loads is the injection-saturation bandwidth

### Metrics

| key | value | unit |
| --- | --- | --- |
| `saturation_gbps` | 3064.33 | GB/s |
| `sweep_points` | 4 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | ce58ed8dc33e87bcc371366ffd09984a193f2f4a8a74fce1cb909a8d48f5ff38 |
| `device_name` | NVIDIA H100 80GB HBM3 |

#### `sweep` (4 rows)

| blocks_per_sm | gbps |
| --- | --- |
| 1 | 1259.27 |
| 2 | 2194.53 |
| 4 | 2947.51 |
| 8 | 3064.33 |

<details><summary><code>ncu</code> (JSON)</summary>

```json
{
  "launches_profiled": 21,
  "logical": "dram_bytes_read",
  "metric": "dram__bytes_read.sum",
  "role": "corroboration",
  "value": 536929024.0
}
```

</details>

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "60c68e02232a07f46ad92b8faf00c6d656c40cfb201700852d2859080d537635",
  "opcode_histogram": {
    "BRA": 2,
    "BSSY": 1,
    "BSYNC": 1,
    "EXIT": 3,
    "FADD": 4,
    "FSETP": 1,
    "HFMA2": 1,
    "IADD3": 2,
    "IMAD": 5,
    "ISETP": 4,
    "LDC": 3,
    "LDG": 1,
    "LEA": 2,
    "MOV": 1,
    "NOP": 8,
    "S2R": 1,
    "S2UR": 1,
    "STG": 1,
    "ULDC": 4,
    "UMOV": 1,
    "VIADD": 1
  },
  "satisfied": [
    "LDG>=1 (1)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/interconnect/injection_rate.cu`
- bytes: `3326`  ·  sha256: `9e952d520fa3eb98451a091df1dc0ed9ff0a06c487db5661967b618619d7ee0a`

### SASS validation

- validated: `True`
- disassembly_hash: `60c68e02232a07f46ad92b8faf00c6d656c40cfb201700852d2859080d537635`
- satisfied: LDG>=1 (1)
- opcode_histogram: `{"BRA": 2, "BSSY": 1, "BSYNC": 1, "EXIT": 3, "FADD": 4, "FSETP": 1, "HFMA2": 1, "IADD3": 2, "IMAD": 5, "ISETP": 4, "LDC": 3, "LDG": 1, "LEA": 2, "MOV": 1, "NOP": 8, "S2R": 1, "S2UR": 1, "STG": 1, "ULDC": 4, "UMOV": 1, "VIADD": 1}`

[↑ contents](#contents)

---

## interconnect.address_mapping

| field | value |
| --- | --- |
| launch | `kernel`  — |
| evidence_tier | `timing_direct` |
| fit_status | `behavioral_only` |
| measurement | `address_mapping_class` = uniform |
| simulator_param | `address_mapping_class` = uniform |
| concept | `address_partition_mapping` |

- binary_hash: `59113f2d43c024972777406284751e13fa48a7c2674fbe0ba65b61bf17ec09af`
- interpretation: partition/slice periodicity from base-stride bandwidth variation
- mapping_contract: base-stride bandwidth variation -> simulator address-partition mapping class (candidate/behavioral)

**assumptions:**

- grid-stride reads with a per-step base displacement swept across power-of-two strides
- best-of-N CUDA-event timing per stride; bandwidth varies with partition interleave
- max/min bandwidth ratio < 1.2 classifies as uniform else periodic_camping

### Metrics

| key | value | unit |
| --- | --- | --- |
| `address_mapping_class` | uniform | — |
| `bandwidth_ratio` | 1.08196 | — |
| `max_gbps` | 4517.95 | GB/s |
| `min_gbps` | 4175.7 | GB/s |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 59113f2d43c024972777406284751e13fa48a7c2674fbe0ba65b61bf17ec09af |
| `device_name` | NVIDIA H100 80GB HBM3 |

#### `sweep` (10 rows)

| gbps | stride_kb |
| --- | --- |
| 4216.79 | 1 |
| 4517.95 | 2 |
| 4340.7 | 4 |
| 4261.06 | 8 |
| 4217.56 | 16 |
| 4194.94 | 32 |
| 4185.43 | 64 |
| 4186.44 | 128 |
| 4175.7 | 256 |
| 4180.87 | 512 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "134a2b333904fc6a693341322aca7cb612c591505f11e876d9174552baaf7099",
  "opcode_histogram": {
    "BRA": 15,
    "BSSY": 5,
    "BSYNC": 5,
    "CALL": 5,
    "EXIT": 3,
    "F2I": 6,
    "FADD": 20,
    "FSETP": 1,
    "HFMA2": 5,
    "I2F": 6,
    "IADD3": 41,
    "IMAD": 96,
    "ISETP": 31,
    "LDC": 4,
    "LDG": 5,
    "LEA": 12,
    "LOP3": 11,
    "MOV": 19,
    "MUFU": 6,
    "NOP": 9,
    "RET": 1,
    "S2R": 2,
    "SEL": 6,
    "SHF": 1,
    "STG": 1,
    "UIADD3": 3,
    "ULDC": 10,
    "UMOV": 4,
    "USHF": 1,
    "VIADD": 10
  },
  "satisfied": [
    "LDG>=1 (5)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/home/cliu/wk/amora/amora/probes/nvidia/baseline/interconnect/address_mapping.cu`
- bytes: `4060`  ·  sha256: `851e9735460bc75e719ae687b8aa0072ddf57fc19a80625f747739ea9b5025d9`

### SASS validation

- validated: `True`
- disassembly_hash: `134a2b333904fc6a693341322aca7cb612c591505f11e876d9174552baaf7099`
- satisfied: LDG>=1 (5)
- opcode_histogram: `{"BRA": 15, "BSSY": 5, "BSYNC": 5, "CALL": 5, "EXIT": 3, "F2I": 6, "FADD": 20, "FSETP": 1, "HFMA2": 5, "I2F": 6, "IADD3": 41, "IMAD": 96, "ISETP": 31, "LDC": 4, "LDG": 5, "LEA": 12, "LOP3": 11, "MOV": 19, "MUFU": 6, "NOP": 9, "RET": 1, "S2R": 2, "SEL": 6, "SHF": 1, "STG": 1, "UIADD3": 3, "ULDC": 10, "UMOV": 4, "USHF": 1, "VIADD": 10}`

[↑ contents](#contents)

---

## interconnect.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `behavioral_only` |
| measurement | `interconnect_summary` = _object_ (address_mapping_class, injection_saturation_gbps) |
| simulator_param | `interconnect_summary` = _object_ (address_mapping_class, injection_saturation_gbps) |
| concept | `interconnect_summary` |

- interpretation: merged interconnect characterization from injection-rate and address-mapping probes
- mapping_contract: cross-probe interconnect summary for simulator injection bandwidth + address-mapping parameters

**assumptions:**

- merges injection-saturation bandwidth and address-mapping behavioral class
- merged fit status is the weakest of the contributing probe fits

### Measurement value

```json
{
  "address_mapping_class": "uniform",
  "injection_saturation_gbps": 3077.26
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `address_mapping_class` | uniform | — |
| `injection_saturation_gbps` | 3077.26 | — |

### Raw values

<details><summary><code>address_mapping</code> (JSON)</summary>

```json
{
  "address_mapping_class": "uniform",
  "binary_sha256": "59113f2d43c024972777406284751e13fa48a7c2674fbe0ba65b61bf17ec09af"
}
```

</details>

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "address_mapping_class": "uniform",
  "injection_saturation_gbps": 3077.26
}
```

</details>

<details><summary><code>injection_rate</code> (JSON)</summary>

```json
{
  "binary_sha256": "ce58ed8dc33e87bcc371366ffd09984a193f2f4a8a74fce1cb909a8d48f5ff38",
  "injection_saturation_gbps": 3077.26
}
```

</details>

[↑ contents](#contents)

---
