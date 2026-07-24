# nvidia / volta / v100-32g — Probe Results

- Generated: 2026-07-22T05:23Z
- Device: Tesla V100-SXM2-32GB  ·  Backend: `nvidia_cuda`  ·  Probes: 36
- `fit_status`: `behavioral_only`=8, `bounded`=7, `conditionally_identified`=3, `direct`=5, `underconstrained`=3, `uniquely_identified`=7, `unsupported`=3
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
| measurement | `cuda_device_identity` = _object_ (device_index, device_name, driver_version, published_facts, uuid) |
| simulator_param | `device_identity` = _object_ (device_index, device_name, driver_version, published_facts, uuid) |
| concept | `runtime_visible_device_identity` |

- interpretation: device identity is available; resource limits need CUDA API helper in the next cutline
- mapping_contract: identity metadata is recorded for traceability and is not a simulator structural parameter

**assumptions:**

- nvidia-smi identity metadata is treated as direct runtime metadata
- published_facts are curated trust-and-verify anchors, not runtime measurements

### Measurement value

```json
{
  "device_index": 0,
  "device_name": "Tesla V100-SXM2-32GB",
  "driver_version": "535.261.03",
  "published_facts": {
    "compute_capability": "7.0",
    "family": "volta",
    "features": [
      "tensor_core"
    ],
    "l2_cache_mb": 6.0,
    "memory_bandwidth_gbps": 900.0,
    "model": "v100",
    "shared_memory_per_sm_kb": 96,
    "sm_count": 80
  },
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

<details><summary><code>published_facts</code> (JSON)</summary>

```json
{
  "compute_capability": "7.0",
  "family": "volta",
  "features": [
    "tensor_core"
  ],
  "l2_cache_mb": 6.0,
  "memory_bandwidth_gbps": 900.0,
  "model": "v100",
  "shared_memory_per_sm_kb": 96,
  "sm_count": 80
}
```

</details>

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

- binary_hash: `33d7b0dde9faab632bd5522c056c5326f889c33355aa13e4e5192595f2812995`
- launch.extras: `{"busy_cycles": 200000}`
- interpretation: peak resident CTAs per SM under the configured launch shape
- mapping_contract: observed peak block residency under busy-spin → simulator max_resident_ctas_per_sm

**assumptions:**

- concurrency derived from sweep-line over per-SM (start,end) cycle pairs
- kernel uses %smid plus a busy-spin to keep blocks resident long enough to overlap

### Metrics

| key | value | unit |
| --- | --- | --- |
| `elapsed_ms` | 0.2386 | ms |
| `mean_resident_blocks_per_sm` | 12.8 | — |
| `multi_processor_count` | 80 | — |
| `peak_resident_blocks_per_sm` | 13 | — |
| `sm_count_observed` | 80 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 33d7b0dde9faab632bd5522c056c5326f889c33355aa13e4e5192595f2812995 |
| `blocks_launched` | 1024 |
| `busy_cycles` | 200000 |
| `device_name` | Tesla V100-SXM2-32GB |
| `elapsed_ms` | 0.2386 |
| `mean_resident_blocks_per_sm` | 12.8 |
| `multi_processor_count` | 80 |
| `peak_resident_blocks_per_sm` | 13 |
| `sm_count_observed` | 80 |
| `threads_per_block` | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "7348d8aa596d8caab388f9e352b1b550be91e9a30ff40033b2b535a7126241d4",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 3,
    "EXIT": 2,
    "IADD3": 1,
    "IMAD": 4,
    "ISETP": 3,
    "NOP": 1,
    "S2R": 3,
    "STG": 3
  },
  "satisfied": [],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/topology/persistent_cta.cu`
- bytes: `5507`  ·  sha256: `ce2f1944aa3ad53742efcae13166e51ee6c0b88f9517d01fce3a0699dcb8a770`

### SASS validation

- validated: `True`
- disassembly_hash: `7348d8aa596d8caab388f9e352b1b550be91e9a30ff40033b2b535a7126241d4`
- opcode_histogram: `{"BRA": 2, "CS2R": 3, "EXIT": 2, "IADD3": 1, "IMAD": 4, "ISETP": 3, "NOP": 1, "S2R": 3, "STG": 3}`

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

- binary_hash: `ba40225903664924d0140311f6e53b049a80b62049b47ba8f0e75ce951fd4049`
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
| `binary_sha256` | ba40225903664924d0140311f6e53b049a80b62049b47ba8f0e75ce951fd4049 |
| `chain_length` | 4096 |
| `cycles_max` | 17924 |
| `cycles_median` | 17924 |
| `cycles_min` | 17924 |
| `cycles_per_fma` | 4.376 |
| `device_name` | Tesla V100-SXM2-32GB |
| `repeats` | 64 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "dependency_confirmed": true,
  "disassembly_hash": "4d7646823125f171be417d7a7d1d2298c02c3cde97db1cc13cb5e850b7cb06ab",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "FFMA": 32,
    "IADD3": 3,
    "ISETP": 1,
    "LOP3": 1,
    "MOV": 8,
    "NOP": 7,
    "S2R": 2,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/arithmetic_latency/dependent_chain.cu`
- bytes: `3276`  ·  sha256: `75b8231ef6b256c8eabe9f5586ad06ba00260c15ee1752415fe139a6a82ab880`

### SASS validation

- validated: `True`
- disassembly_hash: `4d7646823125f171be417d7a7d1d2298c02c3cde97db1cc13cb5e850b7cb06ab`
- satisfied: FFMA>=8 (32)
- dependency_confirmed: `True`
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 2, "FFMA": 32, "IADD3": 3, "ISETP": 1, "LOP3": 1, "MOV": 8, "NOP": 7, "S2R": 2, "STG": 2}`

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

- binary_hash: `adf1a220ee169d21e3556bcab412e3230520c0ad00df923be0ad91ec6162bfa8`
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
| `binary_sha256` | adf1a220ee169d21e3556bcab412e3230520c0ad00df923be0ad91ec6162bfa8 |
| `blocks` | 16 |
| `chain_length` | 4096 |
| `cycles_max` | 34566 |
| `cycles_median` | 34566 |
| `cycles_min` | 34566 |
| `cycles_per_fma_per_thread` | 2.1097 |
| `device_name` | Tesla V100-SXM2-32GB |
| `elapsed_ms` | 0.0328 |
| `independent_chains` | 4 |
| `multi_processor_count` | 80 |
| `threads` | 128 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "993e05daee5aea972f8b65f9fb985ea57332729cb82a42272288c6c4ea6bcb9a",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 1,
    "FADD": 6,
    "FFMA": 128,
    "IADD3": 3,
    "IMAD": 3,
    "ISETP": 1,
    "MOV": 10,
    "NOP": 6,
    "S2R": 2,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/arithmetic_throughput/independent_chains.cu`
- bytes: `4638`  ·  sha256: `3707dce55ee9b9715b5343b1543301bb9f4de54b691b8af1b50ec509b209428b`

### SASS validation

- validated: `True`
- disassembly_hash: `993e05daee5aea972f8b65f9fb985ea57332729cb82a42272288c6c4ea6bcb9a`
- satisfied: FFMA>=8 (128)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 1, "FADD": 6, "FFMA": 128, "IADD3": 3, "IMAD": 3, "ISETP": 1, "MOV": 10, "NOP": 6, "S2R": 2, "STG": 2}`

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

- binary_hash: `972238eab6021b9b4cdfef8a8f5778206a52da359133badf4452df283072b950`
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
| `binary_sha256` | 972238eab6021b9b4cdfef8a8f5778206a52da359133badf4452df283072b950 |
| `chase_len` | 4096 |
| `cycles_max` | 110587 |
| `cycles_median` | 110587 |
| `cycles_min` | 110587 |
| `cycles_per_load` | 26.9988 |
| `device_name` | Tesla V100-SXM2-32GB |
| `repeats` | 64 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "35a2c66e1c2eaf61081045c66e8135cef8a6c512962b819d1d3f8a637d8bb04e",
  "opcode_histogram": {
    "BAR": 1,
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 4,
    "IMAD": 39,
    "ISETP": 3,
    "LDS": 64,
    "LOP3": 1,
    "MOV": 3,
    "NOP": 4,
    "S2R": 1,
    "SHF": 30,
    "STG": 2,
    "STS": 1
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/shared_memory/pointer_chase.cu`
- bytes: `3320`  ·  sha256: `8708d8a5a03239f88fa1520247d007ff6be5319b5d83e1ab1aa8dd310c4ceedd`

### SASS validation

- validated: `True`
- disassembly_hash: `35a2c66e1c2eaf61081045c66e8135cef8a6c512962b819d1d3f8a637d8bb04e`
- satisfied: LDS>=1 (64)
- opcode_histogram: `{"BAR": 1, "BRA": 2, "CS2R": 2, "EXIT": 2, "IADD3": 4, "IMAD": 39, "ISETP": 3, "LDS": 64, "LOP3": 1, "MOV": 3, "NOP": 4, "S2R": 1, "SHF": 30, "STG": 2, "STS": 1}`

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

- binary_hash: `5f54e1d2cbc40cbc2152a9ba738f7635124c6c3645f8193966e9ea0a74fa3bdd`
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
| `binary_sha256` | 5f54e1d2cbc40cbc2152a9ba738f7635124c6c3645f8193966e9ea0a74fa3bdd |
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

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "40a72beec5c52c0aa53f8a8f5f291aae74f80a2f586bef195d8a4d23684ab81c",
  "opcode_histogram": {
    "BAR": 1,
    "BRA": 2,
    "CS2R": 3,
    "EXIT": 3,
    "IADD3": 34,
    "IMAD": 41,
    "ISETP": 4,
    "LDS": 32,
    "LOP3": 49,
    "NOP": 2,
    "S2R": 1,
    "STG": 2,
    "STS": 1
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/shared_memory/bank_stride.cu`
- bytes: `4535`  ·  sha256: `deee44a45bdafc0270320e62f77d735ec08f3c481b8db8985cbb6ec1bf0f7de9`

### SASS validation

- validated: `True`
- disassembly_hash: `40a72beec5c52c0aa53f8a8f5f291aae74f80a2f586bef195d8a4d23684ab81c`
- satisfied: LDS>=1 (32)
- opcode_histogram: `{"BAR": 1, "BRA": 2, "CS2R": 3, "EXIT": 3, "IADD3": 34, "IMAD": 41, "ISETP": 4, "LDS": 32, "LOP3": 49, "NOP": 2, "S2R": 1, "STG": 2, "STS": 1}`

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
  "binary_sha256": "5f54e1d2cbc40cbc2152a9ba738f7635124c6c3645f8193966e9ea0a74fa3bdd",
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
  "binary_sha256": "972238eab6021b9b4cdfef8a8f5778206a52da359133badf4452df283072b950",
  "cycles_per_load": 26.9988
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
| measurement | `l1_hit_load_latency` = 59.0574 cycles |
| simulator_param | `l1_latency` = 59.0574 cycles |
| concept | `l1_path_hit_latency` |

- binary_hash: `3c8c4f628b884733d8b1a2ae288ee539829c85efcade61dcedc2cef6420426e9`
- interpretation: dependent-load latency for an L1-resident working set in cycles
- mapping_contract: dependent L1-hit chase cycles-per-load → simulator L1 hit latency

**assumptions:**

- single-thread dependent pointer chase over a randomized ring sized to fit L1
- a DRAM-resident ring is timed as a control; L1-hit regime requires small << large
- median cycles-per-load reported across N launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `dram_cycles_per_load` | 303.937 | cycles |
| `hit_to_dram_ratio` | 5.14647 | — |
| `l1_hit_cycles_per_load` | 59.0574 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 3c8c4f628b884733d8b1a2ae288ee539829c85efcade61dcedc2cef6420426e9 |
| `device_name` | Tesla V100-SXM2-32GB |
| `dram_cycles_per_load` | 303.937 |
| `l1_hit_cycles_per_load` | 59.0574 |
| `large_kb` | 8192 |
| `repeats` | 64 |
| `small_kb` | 16 |
| `steps` | 4096 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "49336fd59095bc0561decad7988919e5d2d5f06b230458909722679ed0e2b576",
  "opcode_histogram": {
    "BRA": 6,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 6,
    "IMAD": 19,
    "ISETP": 5,
    "LDG": 17,
    "LOP3": 2,
    "MOV": 7,
    "NOP": 1,
    "S2R": 2,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/l1_cache/pointer_chase.cu`
- bytes: `5643`  ·  sha256: `198f6a1e50bc281f93330623a68670586bd6f83f49dab47b704c831b35c3edff`

### SASS validation

- validated: `True`
- disassembly_hash: `49336fd59095bc0561decad7988919e5d2d5f06b230458909722679ed0e2b576`
- satisfied: LDG>=1 (17)
- opcode_histogram: `{"BRA": 6, "CS2R": 2, "EXIT": 2, "IADD3": 6, "IMAD": 19, "ISETP": 5, "LDG": 17, "LOP3": 2, "MOV": 7, "NOP": 1, "S2R": 2, "STG": 2}`

[↑ contents](#contents)

---

## l1_cache.working_set

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `l1_effective_capacity` = _object_ (effective_l1_kb_high, effective_l1_kb_low) |
| simulator_param | `l1d_cache_capacity` = _object_ (effective_l1_kb_high, effective_l1_kb_low) |
| concept | `l1_effective_capacity_knee` |

- binary_hash: `0449c2a2f6337138b7110fc52abfdaf5367f1dba2584bd0104df736ef03eeed8`
- interpretation: effective L1 capacity bounded by the first latency knee in the working-set sweep
- mapping_contract: working-set latency knee → simulator L1 capacity range

**assumptions:**

- dependent pointer-chase latency swept across geometric working-set sizes
- first >40% latency jump marks the effective L1 capacity knee
- capacity is reported as a bounded range, not an exact scalar

### Measurement value

```json
{
  "effective_l1_kb_high": 16384,
  "effective_l1_kb_low": 4096
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `effective_l1_kb_high` | 16384 | KiB |
| `effective_l1_kb_low` | 4096 | KiB |
| `sweep_points` | 13 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 0449c2a2f6337138b7110fc52abfdaf5367f1dba2584bd0104df736ef03eeed8 |
| `device_name` | Tesla V100-SXM2-32GB |
| `repeats` | 32 |
| `steps` | 4096 |

#### `sweep` (13 rows)

| cycles_per_load | working_set_kb |
| --- | --- |
| 41.8481 | 4 |
| 47.7341 | 8 |
| 59.2166 | 16 |
| 70.9373 | 24 |
| 82.1182 | 32 |
| 102.923 | 48 |
| 119.724 | 64 |
| 159.441 | 128 |
| 197.726 | 256 |
| 211.471 | 512 |
| 216.942 | 1024 |
| 220.732 | 4096 |
| 400.467 | 16384 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "dca2da69bdc9c0b340e532d4ab36ecd91e01e8652daa854cadbc8ed2adfd3244",
  "opcode_histogram": {
    "BRA": 6,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 6,
    "IMAD": 19,
    "ISETP": 5,
    "LDG": 17,
    "LOP3": 2,
    "MOV": 7,
    "NOP": 1,
    "S2R": 2,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/l1_cache/working_set.cu`
- bytes: `4223`  ·  sha256: `0bdfc85ec29e0f1847fd8bca024ae2637a57427be3c0796383ed180cc75987b6`

### SASS validation

- validated: `True`
- disassembly_hash: `dca2da69bdc9c0b340e532d4ab36ecd91e01e8652daa854cadbc8ed2adfd3244`
- satisfied: LDG>=1 (17)
- opcode_histogram: `{"BRA": 6, "CS2R": 2, "EXIT": 2, "IADD3": 6, "IMAD": 19, "ISETP": 5, "LDG": 17, "LOP3": 2, "MOV": 7, "NOP": 1, "S2R": 2, "STG": 2}`

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

- binary_hash: `42351f9945abba935835e367150b668d6e866f67241604602938172a18048bd5`
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
| `binary_sha256` | 42351f9945abba935835e367150b668d6e866f67241604602938172a18048bd5 |
| `device_name` | Tesla V100-SXM2-32GB |
| `steps` | 4096 |
| `stride_bytes` | 4096 |

#### `sweep` (24 rows)

| cycles_per_load | ways |
| --- | --- |
| 36.0591 | 1 |
| 36.1055 | 2 |
| 36.1533 | 3 |
| 36.2017 | 4 |
| 36.2498 | 5 |
| 36.2976 | 6 |
| 36.3462 | 7 |
| 36.3931 | 8 |
| 36.437 | 9 |
| 36.4802 | 10 |
| 36.5217 | 11 |
| 36.5637 | 12 |
| 36.6072 | 13 |
| 36.6506 | 14 |
| 36.6934 | 15 |
| 36.7349 | 16 |
| 36.7827 | 17 |
| 36.8306 | 18 |
| 36.8777 | 19 |
| 36.9263 | 20 |
| 36.9729 | 21 |
| 37.02 | 22 |
| 37.0688 | 23 |
| 37.1165 | 24 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "7e023fa8bc3886f6031dcff0c978ad6085bed21624dcf7ae539168b8bd8474df",
  "opcode_histogram": {
    "BRA": 6,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 6,
    "IMAD": 19,
    "ISETP": 5,
    "LDG": 17,
    "LOP3": 2,
    "MOV": 7,
    "NOP": 1,
    "S2R": 2,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/l1_cache/conflict_sets.cu`
- bytes: `4359`  ·  sha256: `a1c9a63a7abe678c35db2a1c1f96e95658eb687a8d6b157b54981d06b72ee984`

### SASS validation

- validated: `True`
- disassembly_hash: `7e023fa8bc3886f6031dcff0c978ad6085bed21624dcf7ae539168b8bd8474df`
- satisfied: LDG>=1 (17)
- opcode_histogram: `{"BRA": 6, "CS2R": 2, "EXIT": 2, "IADD3": 6, "IMAD": 19, "ISETP": 5, "LDG": 17, "LOP3": 2, "MOV": 7, "NOP": 1, "S2R": 2, "STG": 2}`

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
  "l1_effective_capacity_kb": {
    "effective_l1_kb_high": 16384,
    "effective_l1_kb_low": 4096
  },
  "l1_hit_latency_cycles": 59.217
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `l1_effective_capacity_kb` | `{"effective_l1_kb_high": 16384, "effective_l1_kb_low": 4096}` | — |
| `l1_hit_latency_cycles` | 59.217 | — |

### Raw values

<details><summary><code>conflict_sets</code> (JSON)</summary>

```json
{
  "binary_sha256": "42351f9945abba935835e367150b668d6e866f67241604602938172a18048bd5"
}
```

</details>

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "l1_effective_capacity_kb": {
    "effective_l1_kb_high": 16384,
    "effective_l1_kb_low": 4096
  },
  "l1_hit_latency_cycles": 59.217
}
```

</details>

<details><summary><code>pointer_chase</code> (JSON)</summary>

```json
{
  "binary_sha256": "3c8c4f628b884733d8b1a2ae288ee539829c85efcade61dcedc2cef6420426e9",
  "l1_hit_cycles_per_load": 59.217
}
```

</details>

<details><summary><code>working_set</code> (JSON)</summary>

```json
{
  "binary_sha256": "0449c2a2f6337138b7110fc52abfdaf5367f1dba2584bd0104df736ef03eeed8",
  "effective_capacity": {
    "effective_l1_kb_high": 16384,
    "effective_l1_kb_low": 4096
  }
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
| measurement | `issue_saturation_warps` = 12 warps |
| simulator_param | `gpgpu_num_sched_per_core` = 12 warps |
| concept | `scheduler_issue_scaling` |

- binary_hash: `aefe7e63a47e81e91d2ed7b0eb1dce28e3bb2f273c4a4cb4ce289e468ac3a74b`
- interpretation: ready-warp count at which issue throughput saturates on one SM
- mapping_contract: issue-scaling saturation knee → simulator scheduler issue capacity (conditional)

**assumptions:**

- one CTA on one SM runs N independent dependent-FMA warps (no memory)
- saturation warp count = smallest warp count reaching 95% of peak ops/cycle
- scheduler policy name is behavioral; only issue scaling is reported

### Metrics

| key | value | unit |
| --- | --- | --- |
| `peak_ops_per_cycle` | 62.1194 | ops/cycle |
| `saturation_warps` | 12 | — |
| `sweep_points` | 32 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | aefe7e63a47e81e91d2ed7b0eb1dce28e3bb2f273c4a4cb4ce289e468ac3a74b |
| `chain_length` | 2048 |
| `device_name` | Tesla V100-SXM2-32GB |

#### `sweep` (32 rows)

| cycles_median | ops_per_cycle | warps |
| --- | --- | --- |
| 9093 | 7.2073 | 1 |
| 9093 | 14.4146 | 2 |
| 9093 | 21.6219 | 3 |
| 9093 | 28.8292 | 4 |
| 9224 | 35.5247 | 5 |
| 9224 | 42.6297 | 6 |
| 9224 | 49.7346 | 7 |
| 9224 | 56.8395 | 8 |
| 12868 | 45.8365 | 9 |
| 12869 | 50.9255 | 10 |
| 12869 | 56.018 | 11 |
| 12869 | 61.1106 | 12 |
| 16909 | 50.3855 | 13 |
| 16908 | 54.2645 | 14 |
| 16909 | 58.1371 | 15 |
| 16917 | 61.9836 | 16 |
| 21131 | 52.7241 | 17 |
| 21118 | 55.8598 | 18 |
| 21117 | 58.966 | 19 |
| 21118 | 62.0665 | 20 |
| 25373 | 54.241 | 21 |
| 25374 | 56.8216 | 22 |
| 25370 | 59.4138 | 23 |
| 25369 | 61.9994 | 24 |
| 29541 | 55.4619 | 25 |
| 29542 | 57.6784 | 26 |
| 29543 | 59.8948 | 27 |
| 29540 | 62.1194 | 28 |
| 33809 | 56.2141 | 29 |
| 33808 | 58.1543 | 30 |
| 33809 | 60.091 | 31 |
| 33810 | 62.0276 | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "4cb932f69b0064111ad8c8648da66ac613fbfcbd59ff901c1d2ab136d49e1ee9",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "FADD": 1,
    "FFMA": 16,
    "I2F": 1,
    "IADD3": 5,
    "ISETP": 3,
    "LOP3": 1,
    "MOV": 5,
    "NOP": 3,
    "S2R": 1,
    "SHF": 2,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/scheduler_policy/ready_warps.cu`
- bytes: `4167`  ·  sha256: `66e8d665ec60350e3c5fa942833f7065db231c1fd1ca64eb36650e36571860de`

### SASS validation

- validated: `True`
- disassembly_hash: `4cb932f69b0064111ad8c8648da66ac613fbfcbd59ff901c1d2ab136d49e1ee9`
- satisfied: FFMA>=8 (16)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 2, "FADD": 1, "FFMA": 16, "I2F": 1, "IADD3": 5, "ISETP": 3, "LOP3": 1, "MOV": 5, "NOP": 3, "S2R": 1, "SHF": 2, "STG": 2}`

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

- binary_hash: `af1bee341d6dcaf256c1d822cf590a4dd0b471b447df355abf36871303c39862`
- interpretation: FP32/INT pipe overlap classified from mixed vs single-pipe throughput
- mapping_contract: mixed/single-pipe overlap ratio → simulator dual-issue behavioral class

**assumptions:**

- independent FP32 (FMA) and INT (MAD) streams run alone and interleaved
- overlap_ratio = mixed / max(fp32, int); higher means more pipe overlap
- mixed-issue capability is a behavioral class, not a named policy

### Metrics

| key | value | unit |
| --- | --- | --- |
| `fp32_ops_per_cycle` | 63.4271 | ops/cycle |
| `int_ops_per_cycle` | 495.429 | ops/cycle |
| `mixed_ops_per_cycle` | 118.55 | ops/cycle |
| `overlap_ratio` | 0.239288 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | af1bee341d6dcaf256c1d822cf590a4dd0b471b447df355abf36871303c39862 |
| `chain_length` | 2048 |
| `device_name` | Tesla V100-SXM2-32GB |
| `fp32_ops_per_cycle` | 63.4271 |
| `int_ops_per_cycle` | 495.429 |
| `mixed_ops_per_cycle` | 118.55 |
| `warps` | 8 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "90d506f174b37173aef1616c1e5c7125b376da105f8783477bdc4e6ac75a5cf4",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 1,
    "FADD": 2,
    "FFMA": 32,
    "IADD3": 9,
    "IMAD": 2,
    "ISETP": 2,
    "LOP3": 1,
    "MOV": 10,
    "NOP": 2,
    "S2R": 1,
    "SHF": 1,
    "STG": 3
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/scheduler_policy/mixed_issue.cu`
- bytes: `5669`  ·  sha256: `7ed20977f0719cdb66f2939d1f30f8d2004ff321fa5dab14bdb86a39940fc8e7`

### SASS validation

- validated: `True`
- disassembly_hash: `90d506f174b37173aef1616c1e5c7125b376da105f8783477bdc4e6ac75a5cf4`
- satisfied: FFMA>=4 (32)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 1, "FADD": 2, "FFMA": 32, "IADD3": 9, "IMAD": 2, "ISETP": 2, "LOP3": 1, "MOV": 10, "NOP": 2, "S2R": 1, "SHF": 1, "STG": 3}`

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
  "issue_saturation_warps": 12,
  "mixed_issue_class": "single_issue_like",
  "peak_ops_per_cycle": 62.1194
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `issue_saturation_warps` | 12 | — |
| `mixed_issue_class` | single_issue_like | — |
| `peak_ops_per_cycle` | 62.1194 | — |

### Raw values

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "issue_saturation_warps": 12,
  "mixed_issue_class": "single_issue_like",
  "peak_ops_per_cycle": 62.1194
}
```

</details>

<details><summary><code>mixed_issue</code> (JSON)</summary>

```json
{
  "binary_sha256": "af1bee341d6dcaf256c1d822cf590a4dd0b471b447df355abf36871303c39862",
  "overlap_class": "single_issue_like",
  "overlap_ratio": 0.2392876721840376
}
```

</details>

<details><summary><code>ready_warps</code> (JSON)</summary>

```json
{
  "binary_sha256": "aefe7e63a47e81e91d2ed7b0eb1dce28e3bb2f273c4a4cb4ce289e468ac3a74b",
  "saturation_warps": 12
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
| fit_status | `bounded` |
| measurement | `operand_delivery_plateau` = 12 accumulators |
| simulator_param | `gpgpu_num_reg_banks` = 12 accumulators |
| concept | `register_bank_operand_delivery` |

- binary_hash: `58fb10a3e58ba747b62467daa21c21c0f8f892003910746e9cb981a1f08deb0a`
- interpretation: operand-delivery throughput plateau across register-pressure widths
- mapping_contract: operand-width plateau → simulator register-bank pressure (candidate, multi-fit)

**assumptions:**

- operand-width sweep of independent FMA accumulators (register pressure proxy)
- SASS confirms distinct FFMA register operands so the sweep is register-controlled
- plateau width marks where added ILP stops improving cycles-per-op

### Metrics

| key | value | unit |
| --- | --- | --- |
| `ilp_plateau_width` | 12 | accumulators |
| `sass_distinct_ffma_registers` | 17 | — |
| `sweep_points` | 8 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 58fb10a3e58ba747b62467daa21c21c0f8f892003910746e9cb981a1f08deb0a |
| `chain_length` | 2048 |
| `device_name` | Tesla V100-SXM2-32GB |

#### `sweep` (8 rows)

| cycles_per_op | width |
| --- | --- |
| 5.002 | 1 |
| 2.5647 | 2 |
| 2.584 | 3 |
| 2.438 | 4 |
| 2.292 | 6 |
| 2.219 | 8 |
| 2.146 | 12 |
| 2.1173 | 16 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "b239acfc443b0956356962d71af806f30be946c1d2157f36a1b8146b016bfb0f",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 1,
    "FADD": 32,
    "FFMA": 128,
    "IADD3": 5,
    "ISETP": 2,
    "LOP3": 1,
    "MOV": 6,
    "NOP": 7,
    "S2R": 1,
    "SHF": 1,
    "STG": 2
  },
  "register_count": 17,
  "satisfied": [
    "FFMA>=8 (128)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/register_file/register_bank_sweep.cu`
- bytes: `4516`  ·  sha256: `7aa75fca4c6681820fe6c22be0b38aee5698824f1639b55dbcc13ccfbed950a5`

### SASS validation

- validated: `True`
- disassembly_hash: `b239acfc443b0956356962d71af806f30be946c1d2157f36a1b8146b016bfb0f`
- satisfied: FFMA>=8 (128)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 1, "FADD": 32, "FFMA": 128, "IADD3": 5, "ISETP": 2, "LOP3": 1, "MOV": 6, "NOP": 7, "S2R": 1, "SHF": 1, "STG": 2}`

[↑ contents](#contents)

---

## register_file.register_latency

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `conditionally_identified` |
| measurement | `operand_delivery_differential_latency` = 1.4382 cycles |
| simulator_param | `max_latency_regular_register_file_latency` = 1.4382 cycles |
| concept | `register_operand_delivery_latency` |

- binary_hash: `b543894477aaf06bf347c2d994a275931e3388076f435bd4946957e90822c26a`
- interpretation: extra per-op cost of tight RAW dependence attributable to operand delivery
- mapping_contract: RAW-distance differential cycles → simulator operand-delivery latency (conditional)

**assumptions:**

- same-register (RAW distance 1) vs rotating-register (relaxed RAW) chains of equal length
- differential cycles-per-op isolates operand-delivery cost from absolute arithmetic latency
- operand-collector parameters stay conditional: scoreboard/bank effects are entangled

### Metrics

| key | value | unit |
| --- | --- | --- |
| `differential_cycles_per_op` | 1.4382 | cycles |
| `rotating_reg_cycles_per_op` | 3.0015 | cycles |
| `same_reg_cycles_per_op` | 4.4397 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | b543894477aaf06bf347c2d994a275931e3388076f435bd4946957e90822c26a |
| `chain_length` | 4096 |
| `device_name` | Tesla V100-SXM2-32GB |
| `differential_cycles_per_op` | 1.4382 |
| `rot_depth` | 8 |
| `rotating_reg_cycles_per_op` | 3.0015 |
| `same_reg_cycles_per_op` | 4.4397 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "18cec8c58c94f89ebf92ecd619a6e054e1175c3d8df11c49e209f7fa8bef6d66",
  "opcode_histogram": {
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "FFMA": 16,
    "IADD3": 3,
    "ISETP": 2,
    "MOV": 8,
    "S2R": 1,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/register_file/register_latency.cu`
- bytes: `4327`  ·  sha256: `e19a118fab98c5f6a260b19f5d10357e42bade8a5c98e02080c3560932c19975`

### SASS validation

- validated: `True`
- disassembly_hash: `18cec8c58c94f89ebf92ecd619a6e054e1175c3d8df11c49e209f7fa8bef6d66`
- satisfied: FFMA>=8 (16)
- opcode_histogram: `{"BRA": 2, "CS2R": 2, "EXIT": 2, "FFMA": 16, "IADD3": 3, "ISETP": 2, "MOV": 8, "S2R": 1, "STG": 2}`

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
  "operand_delivery_differential_cycles": 1.4382,
  "operand_delivery_plateau_accumulators": 12
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `operand_delivery_differential_cycles` | 1.4382 | — |
| `operand_delivery_plateau_accumulators` | 12 | — |

### Raw values

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "operand_delivery_differential_cycles": 1.4382,
  "operand_delivery_plateau_accumulators": 12
}
```

</details>

<details><summary><code>register_bank_sweep</code> (JSON)</summary>

```json
{
  "binary_sha256": "58fb10a3e58ba747b62467daa21c21c0f8f892003910746e9cb981a1f08deb0a",
  "ilp_plateau_width": 12
}
```

</details>

<details><summary><code>register_latency</code> (JSON)</summary>

```json
{
  "binary_sha256": "b543894477aaf06bf347c2d994a275931e3388076f435bd4946957e90822c26a",
  "differential_cycles_per_op": 1.4382
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
| measurement | `cta_barrier_latency` = 57.001 cycles |
| simulator_param | `barrier_latency` = 57.001 cycles |
| concept | `cta_barrier_latency` |

- binary_hash: `c7ad5f9f66bfef70d8a465baa5581dacba04862c5b1553ea51e71dddaff10b17`
- interpretation: cycles per __syncthreads() barrier for the measured CTA shape
- mapping_contract: cycles-per-barrier for a named CTA shape → simulator barrier latency (conditional)

**assumptions:**

- one CTA runs a long __syncthreads() loop with minimal inter-barrier work
- cycles-per-barrier reported for the smallest block; scaling curve retained
- barrier cost is occupancy-coupled; reported per the launch class measured

### Metrics

| key | value | unit |
| --- | --- | --- |
| `cycles_per_barrier` | 57.001 | cycles |
| `sweep_points` | 5 | — |

### Raw values

| key | value |
| --- | --- |
| `barriers` | 4096 |
| `binary_sha256` | c7ad5f9f66bfef70d8a465baa5581dacba04862c5b1553ea51e71dddaff10b17 |
| `device_name` | Tesla V100-SXM2-32GB |

#### `sweep` (5 rows)

| cycles_per_barrier | threads_per_block |
| --- | --- |
| 57.001 | 64 |
| 61.001 | 128 |
| 69.0007 | 256 |
| 85.8767 | 512 |
| 119.457 | 1024 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "291b5b081b58ec81e77f2db58e7e39e38a8922ae95e273e9f25b0f0f4624c3ce",
  "opcode_histogram": {
    "BAR": 9,
    "BRA": 2,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 10,
    "IMAD": 7,
    "ISETP": 2,
    "LDS": 9,
    "NOP": 16,
    "S2R": 1,
    "STG": 2,
    "STS": 9
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/synchronization/barrier_latency.cu`
- bytes: `3443`  ·  sha256: `247a7f3cbd7851717429c4368ad51992191dc098cabc27ffa3b5d208e3112af9`

### SASS validation

- validated: `True`
- disassembly_hash: `291b5b081b58ec81e77f2db58e7e39e38a8922ae95e273e9f25b0f0f4624c3ce`
- satisfied: BAR>=1 (9)
- opcode_histogram: `{"BAR": 9, "BRA": 2, "CS2R": 2, "EXIT": 2, "IADD3": 10, "IMAD": 7, "ISETP": 2, "LDS": 9, "NOP": 16, "S2R": 1, "STG": 2, "STS": 9}`

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

- binary_hash: `1fb090ab6cbf6b64bcf9c8f90fe4bb1da791f3336947787b9a8780939b314686`
- interpretation: sustained DRAM/HBM bandwidth per traffic class from streaming kernels
- mapping_contract: achieved sustained bandwidth per traffic class → simulator DRAM bandwidth (bounded)

**assumptions:**

- grid-stride read/write/copy over a working set far larger than cache
- best-of-N CUDA-event timing; bandwidth is bounded by clock variation
- copy moves 2x bytes (read+write); reported as achieved sustained GB/s

### Measurement value

```json
{
  "copy_gbps": 749.52,
  "peak_gbps": 859.49,
  "read_gbps": 859.49,
  "write_gbps": 764.27
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `copy_gbps` | 749.52 | GB/s |
| `peak_gbps` | 859.49 | GB/s |
| `read_gbps` | 859.49 | GB/s |
| `write_gbps` | 764.27 | GB/s |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 1fb090ab6cbf6b64bcf9c8f90fe4bb1da791f3336947787b9a8780939b314686 |
| `copy_gbps` | 749.52 |
| `device_name` | Tesla V100-SXM2-32GB |
| `iters` | 5 |
| `read_gbps` | 859.49 |
| `working_set_mb` | 512 |
| `write_gbps` | 764.27 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "f77428923132ec7a7c6e7ea3dc0db720641b38516ded24da7ba6358976597574",
  "opcode_histogram": {
    "BRA": 2,
    "EXIT": 2,
    "IADD3": 5,
    "IMAD": 9,
    "ISETP": 4,
    "LDG": 1,
    "MOV": 1,
    "NOP": 2,
    "S2R": 2,
    "SHF": 2,
    "STG": 1
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/global_memory/streaming.cu`
- bytes: `4889`  ·  sha256: `fb36b1238f0bfc414d6c1cfafe8b60b3351beff0a18e535a42228578cd083ea4`

### SASS validation

- validated: `True`
- disassembly_hash: `f77428923132ec7a7c6e7ea3dc0db720641b38516ded24da7ba6358976597574`
- satisfied: LDG>=1 (1)
- opcode_histogram: `{"BRA": 2, "EXIT": 2, "IADD3": 5, "IMAD": 9, "ISETP": 4, "LDG": 1, "MOV": 1, "NOP": 2, "S2R": 2, "SHF": 2, "STG": 1}`

[↑ contents](#contents)

---

## l2_cache.pointer_chase

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `l2_hit_load_latency` = 246.257 cycles |
| simulator_param | `l2_latency` = 246.257 cycles |
| concept | `l2_hit_latency` |

- binary_hash: `d18663be7312772d5b49d5a5430ce240b7750a4b62393474684315fae266896c`
- interpretation: dependent-load latency for an L2-resident working set in cycles
- mapping_contract: dependent L2-resident chase cycles-per-load -> simulator L2 hit latency (bounded)

**assumptions:**

- single-thread dependent pointer chase over a randomized ring sized to exceed L1 but fit L2
- a DRAM-resident ring is timed as a control; L2-hit regime requires l2 << dram
- median cycles-per-load reported across N launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `dram_cycles_per_load` | 445.168 | cycles |
| `hit_to_dram_ratio` | 1.80774 | — |
| `l2_hit_cycles_per_load` | 246.257 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | d18663be7312772d5b49d5a5430ce240b7750a4b62393474684315fae266896c |
| `device_name` | Tesla V100-SXM2-32GB |
| `dram_cycles_per_load` | 445.168 |
| `dram_kb` | 131072 |
| `l2_hit_cycles_per_load` | 246.257 |
| `l2_kb` | 4096 |
| `repeats` | 64 |
| `steps` | 4096 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "7d4b2b1720dfa93b91d27cfbaee0848fb976e7a8278392a9c42f49ce7a04086e",
  "opcode_histogram": {
    "BRA": 6,
    "CS2R": 2,
    "EXIT": 2,
    "IADD3": 6,
    "IMAD": 19,
    "ISETP": 5,
    "LDG": 17,
    "LOP3": 2,
    "MOV": 7,
    "NOP": 1,
    "S2R": 2,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/l2_cache/pointer_chase.cu`
- bytes: `5571`  ·  sha256: `75cd7ad66553dcc3219494c7902e6c0d93b86f77f03f490480bb5a09755f6ea9`

### SASS validation

- validated: `True`
- disassembly_hash: `7d4b2b1720dfa93b91d27cfbaee0848fb976e7a8278392a9c42f49ce7a04086e`
- satisfied: LDG>=1 (17)
- opcode_histogram: `{"BRA": 6, "CS2R": 2, "EXIT": 2, "IADD3": 6, "IMAD": 19, "ISETP": 5, "LDG": 17, "LOP3": 2, "MOV": 7, "NOP": 1, "S2R": 2, "STG": 2}`

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

- binary_hash: `f6b93893d33108843bc5ce60a0dfcbe9ad75369cb06c61a51991f13be607edc2`
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
| `binary_sha256` | f6b93893d33108843bc5ce60a0dfcbe9ad75369cb06c61a51991f13be607edc2 |
| `buffer_mb` | 256 |
| `device_name` | Tesla V100-SXM2-32GB |

#### `sweep` (6 rows)

| bytes_per_cycle | in_flight |
| --- | --- |
| 331.555 | 1 |
| 518.37 | 2 |
| 579.158 | 4 |
| 480.292 | 8 |
| 475.583 | 16 |
| 424.759 | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "800691862f1b8be100a038526a668dba00abe81299662e3504658e14ed5c30ef",
  "opcode_histogram": {
    "BMOV": 2,
    "BRA": 3,
    "BSSY": 2,
    "BSYNC": 2,
    "CS2R": 3,
    "EXIT": 2,
    "FADD": 128,
    "FSETP": 1,
    "IADD3": 61,
    "IMAD": 48,
    "ISETP": 102,
    "LDG": 32,
    "LEA": 4,
    "MOV": 9,
    "NOP": 2,
    "PLOP3": 1,
    "S2R": 2,
    "SHF": 1,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/memory_pipeline/outstanding_requests.cu`
- bytes: `5450`  ·  sha256: `259f29323a8c75b0145391d145b15705f40c65db79252bb0b9f7607030db5e96`

### SASS validation

- validated: `True`
- disassembly_hash: `800691862f1b8be100a038526a668dba00abe81299662e3504658e14ed5c30ef`
- satisfied: LDG>=1 (32)
- opcode_histogram: `{"BMOV": 2, "BRA": 3, "BSSY": 2, "BSYNC": 2, "CS2R": 3, "EXIT": 2, "FADD": 128, "FSETP": 1, "IADD3": 61, "IMAD": 48, "ISETP": 102, "LDG": 32, "LEA": 4, "MOV": 9, "NOP": 2, "PLOP3": 1, "S2R": 2, "SHF": 1, "STG": 2}`

[↑ contents](#contents)

---

## memory_pipeline.lane_patterns

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[32, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `behavioral_only` |
| measurement | `coalescing_sectors_per_request` = — |
| simulator_param | `memory_coalescing_rule` = — |
| concept | `memory_coalescing` |

- binary_hash: `a8d789965a2d79f68bfd6720fba5e33fe441edfd449ba7d1f600a9c5b4a8ad98`
- interpretation: global-load sectors per request from controlled lane address patterns
- mapping_contract: NCU sectors/request under lane patterns -> simulator memory coalescing rule

**assumptions:**

- one warp issues many global loads under named lane address patterns
- NCU sectors/request is the primary coalescing signal; timing only confirms LDG activity
- max counter value across profiled lane patterns characterizes the worst-case coalescing

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | a8d789965a2d79f68bfd6720fba5e33fe441edfd449ba7d1f600a9c5b4a8ad98 |
| `device_name` | Tesla V100-SXM2-32GB |

#### `patterns` (4 rows)

| iters | name |
| --- | --- |
| 4096 | contiguous |
| 4096 | stride2 |
| 4096 | stride32 |
| 4096 | broadcast |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "4347d218cb428bccd008b31ca1ffbc227fff3f168ab5eef1597eb3dcb90bf17c",
  "opcode_histogram": {
    "BMOV": 7,
    "BRA": 30,
    "BSSY": 7,
    "BSYNC": 7,
    "CALL": 8,
    "CS2R": 1,
    "EXIT": 3,
    "F2I": 9,
    "FADD": 8,
    "FSETP": 1,
    "I2F": 9,
    "IADD3": 53,
    "IMAD": 155,
    "ISETP": 50,
    "LDG": 8,
    "LEA": 19,
    "LOP3": 17,
    "MOV": 15,
    "MUFU": 9,
    "NOP": 4,
    "RET": 1,
    "S2R": 1,
    "SEL": 7,
    "SHF": 3,
    "STG": 1
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/memory_pipeline/lane_patterns.cu`
- bytes: `3801`  ·  sha256: `e1a7b386e271d8f7fb446bfcf766ec8118e1193d5397dee1daef39f17e3ed6b6`

### SASS validation

- validated: `True`
- disassembly_hash: `4347d218cb428bccd008b31ca1ffbc227fff3f168ab5eef1597eb3dcb90bf17c`
- satisfied: LDG>=1 (8)
- opcode_histogram: `{"BMOV": 7, "BRA": 30, "BSSY": 7, "BSYNC": 7, "CALL": 8, "CS2R": 1, "EXIT": 3, "F2I": 9, "FADD": 8, "FSETP": 1, "I2F": 9, "IADD3": 53, "IMAD": 155, "ISETP": 50, "LDG": 8, "LEA": 19, "LOP3": 17, "MOV": 15, "MUFU": 9, "NOP": 4, "RET": 1, "S2R": 1, "SEL": 7, "SHF": 3, "STG": 1}`

[↑ contents](#contents)

---

## memory_pipeline.analyze

| field | value |
| --- | --- |
| launch | `analysis`  — |
| evidence_tier | `coupled_inference` |
| fit_status | `behavioral_only` |
| measurement | `memory_pipeline_summary` = _object_ (effective_outstanding_requests) |
| simulator_param | `memory_pipeline_summary` = _object_ (effective_outstanding_requests) |
| concept | `memory_pipeline_summary` |

- interpretation: merged memory-pipeline characterization from coalescing and outstanding-request probes
- mapping_contract: cross-probe memory-pipeline summary for simulator coalescing + load/store-queue parameters

**assumptions:**

- merges lane-pattern coalescing (sectors/request) with the outstanding-request knee
- merged fit status is the weakest of the contributing probe fits

### Measurement value

```json
{
  "effective_outstanding_requests": 4
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `effective_outstanding_requests` | 4 | — |

### Raw values

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "effective_outstanding_requests": 4
}
```

</details>

<details><summary><code>lane_patterns</code> (JSON)</summary>

```json
{
  "binary_sha256": "a8d789965a2d79f68bfd6720fba5e33fe441edfd449ba7d1f600a9c5b4a8ad98"
}
```

</details>

<details><summary><code>outstanding_requests</code> (JSON)</summary>

```json
{
  "binary_sha256": "f6b93893d33108843bc5ce60a0dfcbe9ad75369cb06c61a51991f13be607edc2",
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

- binary_hash: `311a6b94d1174dcb80014c2959490d64cc7addc168d0db3e9d97e5378457a78f`
- interpretation: DRAM partition-camping sensitivity from base-offset bandwidth sweep
- mapping_contract: base-offset bandwidth variation -> simulator memory-partition camping class

**assumptions:**

- grid-stride read from several base offsets relative to the partition interleave
- best-of-N CUDA-event timing per offset; bandwidth varies with clock/partition balance
- max/min bandwidth ratio < 1.15 classifies as balanced else camping_sensitive

### Metrics

| key | value | unit |
| --- | --- | --- |
| `bandwidth_ratio` | 1.01016 | — |
| `max_gbps` | 871.08 | GB/s |
| `min_gbps` | 862.32 | GB/s |
| `partition_camping_class` | balanced | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 311a6b94d1174dcb80014c2959490d64cc7addc168d0db3e9d97e5378457a78f |
| `device_name` | Tesla V100-SXM2-32GB |
| `working_set_mb` | 512 |

#### `sweep` (6 rows)

| gbps | offset_kb |
| --- | --- |
| 862.32 | 0 |
| 870.49 | 256 |
| 870.06 | 512 |
| 871.08 | 768 |
| 869.21 | 1024 |
| 865.48 | 1536 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "e33351bd18985eacb006006259ba391ba54db90b836844813958b68e489de3b3",
  "opcode_histogram": {
    "BMOV": 1,
    "BRA": 2,
    "BSSY": 1,
    "BSYNC": 1,
    "EXIT": 3,
    "FADD": 4,
    "FSETP": 1,
    "IADD3": 4,
    "IMAD": 8,
    "ISETP": 4,
    "LDG": 1,
    "LEA": 2,
    "MOV": 3,
    "NOP": 1,
    "S2R": 2,
    "STG": 1
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/global_memory/partition_sweep.cu`
- bytes: `4041`  ·  sha256: `d386c53533f50ae549669e8ad6a650e828d196ce75a03339b2356f039bb55147`

### SASS validation

- validated: `True`
- disassembly_hash: `e33351bd18985eacb006006259ba391ba54db90b836844813958b68e489de3b3`
- satisfied: LDG>=1 (1)
- opcode_histogram: `{"BMOV": 1, "BRA": 2, "BSSY": 1, "BSYNC": 1, "EXIT": 3, "FADD": 4, "FSETP": 1, "IADD3": 4, "IMAD": 8, "ISETP": 4, "LDG": 1, "LEA": 2, "MOV": 3, "NOP": 1, "S2R": 2, "STG": 1}`

[↑ contents](#contents)

---

## global_memory.row_policy_sweep

| field | value |
| --- | --- |
| launch | `kernel`  — |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `row_locality_sensitivity` = 1.5328 ratio |
| simulator_param | `dram_row_policy_class` = 1.5328 ratio |
| concept | `dram_row_locality` |

- binary_hash: `3dcaee29a1bef74819d5c73b207f879108935f515ee3d1ef4825cf777674c472`
- interpretation: DRAM row-locality sensitivity from a stride bandwidth sweep
- mapping_contract: stride bandwidth spread -> simulator DRAM row-buffer policy class (bounded)

**assumptions:**

- grid-stride read with several element strides to vary DRAM row-buffer locality
- best-of-N CUDA-event timing per stride; bandwidth bounded by clock variation
- row_locality_sensitivity = best_gbps / worst_gbps across strides (bounded)

### Metrics

| key | value | unit |
| --- | --- | --- |
| `best_gbps` | 713.44 | GB/s |
| `row_locality_sensitivity` | 1.5328 | — |
| `worst_gbps` | 465.45 | GB/s |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 3dcaee29a1bef74819d5c73b207f879108935f515ee3d1ef4825cf777674c472 |
| `device_name` | Tesla V100-SXM2-32GB |
| `working_set_mb` | 512 |

#### `sweep` (4 rows)

| gbps | stride |
| --- | --- |
| 556.17 | 1 |
| 515.03 | 8 |
| 713.44 | 64 |
| 465.45 | 512 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "1790ae2aa7957eebf3b425c10153e62652f22ea556adc585b6be14be360d6c27",
  "opcode_histogram": {
    "BMOV": 5,
    "BRA": 15,
    "BSSY": 5,
    "BSYNC": 5,
    "CALL": 5,
    "EXIT": 3,
    "F2I": 6,
    "FADD": 5,
    "FSETP": 1,
    "I2F": 6,
    "IADD3": 48,
    "IMAD": 129,
    "ISETP": 31,
    "LDG": 5,
    "LEA": 12,
    "LOP3": 11,
    "MOV": 25,
    "MUFU": 6,
    "NOP": 5,
    "RET": 1,
    "S2R": 2,
    "SEL": 6,
    "SHF": 2,
    "STG": 1
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/global_memory/row_policy_sweep.cu`
- bytes: `4021`  ·  sha256: `6d12a1a626488fcec2d13de1676d0160c0c0fc5e21969abfbdb9d678476bd012`

### SASS validation

- validated: `True`
- disassembly_hash: `1790ae2aa7957eebf3b425c10153e62652f22ea556adc585b6be14be360d6c27`
- satisfied: LDG>=1 (5)
- opcode_histogram: `{"BMOV": 5, "BRA": 15, "BSSY": 5, "BSYNC": 5, "CALL": 5, "EXIT": 3, "F2I": 6, "FADD": 5, "FSETP": 1, "I2F": 6, "IADD3": 48, "IMAD": 129, "ISETP": 31, "LDG": 5, "LEA": 12, "LOP3": 11, "MOV": 25, "MUFU": 6, "NOP": 5, "RET": 1, "S2R": 2, "SEL": 6, "SHF": 2, "STG": 1}`

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
  "peak_gbps": 858.08,
  "row_locality_sensitivity": 1.484794352495534
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `partition_class` | balanced | — |
| `peak_gbps` | 858.08 | — |
| `row_locality_sensitivity` | 1.48479 | — |

### Raw values

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "partition_class": "balanced",
  "peak_gbps": 858.08,
  "row_locality_sensitivity": 1.484794352495534
}
```

</details>

<details><summary><code>partition_sweep</code> (JSON)</summary>

```json
{
  "binary_sha256": "311a6b94d1174dcb80014c2959490d64cc7addc168d0db3e9d97e5378457a78f",
  "partition_class": "balanced"
}
```

</details>

<details><summary><code>row_policy_sweep</code> (JSON)</summary>

```json
{
  "binary_sha256": "3dcaee29a1bef74819d5c73b207f879108935f515ee3d1ef4825cf777674c472",
  "row_locality_sensitivity": 1.484794352495534
}
```

</details>

<details><summary><code>streaming</code> (JSON)</summary>

```json
{
  "binary_sha256": "1fb090ab6cbf6b64bcf9c8f90fe4bb1da791f3336947787b9a8780939b314686",
  "peak_gbps": 858.08
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
| measurement | `tensor_mma_latency` = 64.3555 cycles_per_op |
| simulator_param | `tensor_core_mma_latency` = 64.3555 cycles_per_op |
| concept | `tensor_core_mma_latency` |

- binary_hash: `2dd40dd5f316a8c4172e66acb3f7f726bad2e85d8e7db2a6c4f4d133d22f1fbf`
- interpretation: dependent FP16 16x16x16 MMA latency in cycles
- mapping_contract: dependent MMA cycles-per-op -> simulator tensor-core pipeline latency

**assumptions:**

- dependent wmma::mma_sync chain (FP16 m16n16k16) timed via clock64 in one warp
- median across launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `cycles_median` | 32950 | — |
| `cycles_per_mma` | 64.3555 | cycles_per_op |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 2dd40dd5f316a8c4172e66acb3f7f726bad2e85d8e7db2a6c4f4d133d22f1fbf |
| `chain` | 512 |
| `cycles_median` | 32950 |
| `cycles_per_mma` | 64.3555 |
| `device_name` | Tesla V100-SXM2-32GB |
| `mma_shape` | m16n16k16_fp16 |
| `repeats` | 32 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "e8c32db0fba1974fb759690745f92f520015ec24d09082f5d5eb0fc44c9b1de6",
  "opcode_histogram": {
    "BRA": 1,
    "CS2R": 2,
    "EXIT": 2,
    "HMMA": 8192,
    "IADD3": 1,
    "IMAD": 16,
    "ISETP": 3,
    "LDG": 4,
    "LEA": 2,
    "LOP3": 8,
    "MOV": 1,
    "NOP": 3,
    "S2R": 3,
    "SHF": 4,
    "STG": 5
  },
  "satisfied": [
    "HMMA>=1 (8192)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/tensor_core/mma_latency.cu`
- bytes: `4145`  ·  sha256: `4116de2f34110423c466efbb44e5e518a29864cf4fabbd1576ac8d031ce45c1b`

### SASS validation

- validated: `True`
- disassembly_hash: `e8c32db0fba1974fb759690745f92f520015ec24d09082f5d5eb0fc44c9b1de6`
- satisfied: HMMA>=1 (8192)
- opcode_histogram: `{"BRA": 1, "CS2R": 2, "EXIT": 2, "HMMA": 8192, "IADD3": 1, "IMAD": 16, "ISETP": 3, "LDG": 4, "LEA": 2, "LOP3": 8, "MOV": 1, "NOP": 3, "S2R": 3, "SHF": 4, "STG": 5}`

[↑ contents](#contents)

---

## tensor_core.mma_throughput

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[128, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `uniquely_identified` |
| measurement | `tensor_mma_throughput` = 0.009 mma/cycle |
| simulator_param | `tensor_core_initiation_interval` = 111.111 cycles_per_op |
| concept | `tensor_core_mma_throughput` |

- binary_hash: `c6f11ce5ff8e549aebfcf9b678b42f76c33a183adc824a60e6f884624f80d799`
- interpretation: independent FP16 16x16x16 MMA throughput in MMA-ops per cycle per warp
- mapping_contract: independent MMA throughput -> simulator tensor-core initiation interval

**assumptions:**

- independent wmma::mma_sync accumulators (FP16 m16n16k16) expose ILP to saturate the tensor pipe
- median across launches; throughput reported per warp

### Metrics

| key | value | unit |
| --- | --- | --- |
| `cycles_median` | 114242 | — |
| `mma_per_cycle_per_warp` | 0.009 | mma/cycle |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | c6f11ce5ff8e549aebfcf9b678b42f76c33a183adc824a60e6f884624f80d799 |
| `cycles_median` | 114242 |
| `device_name` | Tesla V100-SXM2-32GB |
| `iters` | 256 |
| `lanes` | 4 |
| `mma_per_cycle_per_warp` | 0.009 |
| `mma_shape` | m16n16k16_fp16 |
| `warps` | 4 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "38d6cc967a766f4f6010e24defb0cafba44c60ae60fde8f92e535ccce0e34427",
  "opcode_histogram": {
    "BRA": 1,
    "CS2R": 2,
    "EXIT": 2,
    "FADD": 4,
    "FSETP": 1,
    "HMMA": 16384,
    "IADD3": 1,
    "IMAD": 10,
    "LDG": 4,
    "LEA": 1,
    "LOP3": 5,
    "MOV": 4,
    "NOP": 3,
    "S2R": 3,
    "SHF": 4,
    "STG": 2
  },
  "satisfied": [
    "HMMA>=1 (16384)"
  ],
  "validated": true,
  "violations": []
}
```

</details>

### Registered source

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/tensor_core/mma_throughput.cu`
- bytes: `4451`  ·  sha256: `da95487187f13dc1d97ec1d6b965387f783d7f4f0111d9f2000df9d6e4d8e83a`

### SASS validation

- validated: `True`
- disassembly_hash: `38d6cc967a766f4f6010e24defb0cafba44c60ae60fde8f92e535ccce0e34427`
- satisfied: HMMA>=1 (16384)
- opcode_histogram: `{"BRA": 1, "CS2R": 2, "EXIT": 2, "FADD": 4, "FSETP": 1, "HMMA": 16384, "IADD3": 1, "IMAD": 10, "LDG": 4, "LEA": 1, "LOP3": 5, "MOV": 4, "NOP": 3, "S2R": 3, "SHF": 4, "STG": 2}`

[↑ contents](#contents)

---

## synchronization.fence_latency

| field | value |
| --- | --- |
| launch | `kernel`  grid=[1, 1, 1] block=[256, 1, 1] |
| evidence_tier | `timing_direct` |
| fit_status | `conditionally_identified` |
| measurement | `memory_fence_latency` = 163.074 cycles |
| simulator_param | `fence_latency` = 163.074 cycles |
| concept | `memory_fence_latency` |

- binary_hash: `ba238c4f4934ae1e29a1fb5d8b2a4617556d3f2b6f0630038ef642c11655ab87`
- interpretation: net cycles per __threadfence() after subtracting empty-loop overhead
- mapping_contract: net per-fence cycles -> simulator memory fence latency (conditional)

**assumptions:**

- device-scope __threadfence() loop timed via clock64 in one CTA
- empty-loop baseline subtracted to remove loop/branch overhead from the per-fence cost
- net per-fence cost reflects fence scope as measured; median across launches

### Metrics

| key | value | unit |
| --- | --- | --- |
| `cycles_per_empty` | 0.0095 | — |
| `cycles_per_fence` | 163.083 | — |
| `net_cycles_per_fence` | 163.074 | cycles |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | ba238c4f4934ae1e29a1fb5d8b2a4617556d3f2b6f0630038ef642c11655ab87 |
| `cycles_per_empty` | 0.0095 |
| `cycles_per_fence` | 163.083 |
| `device_name` | Tesla V100-SXM2-32GB |
| `fences` | 4096 |
| `net_cycles_per_fence` | 163.074 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "0689671ba9f742249acc82d7bee9ed558a194f1730bde03f6d4b560a620cb874",
  "opcode_histogram": {
    "BAR": 1,
    "BRA": 2,
    "CCTL": 8,
    "CS2R": 2,
    "ERRBAR": 8,
    "EXIT": 2,
    "IADD3": 4,
    "IMAD": 5,
    "ISETP": 2,
    "LDG": 1,
    "MEMBAR": 8,
    "MOV": 1,
    "NOP": 8,
    "S2R": 1,
    "STG": 2
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/synchronization/fence_latency.cu`
- bytes: `4401`  ·  sha256: `94d5e03aba0ff1d6281cd58fed7ffe508fc6c464b0a6d107f9f6766355528552`

### SASS validation

- validated: `True`
- disassembly_hash: `0689671ba9f742249acc82d7bee9ed558a194f1730bde03f6d4b560a620cb874`
- satisfied: MEMBAR>=1 (8)
- opcode_histogram: `{"BAR": 1, "BRA": 2, "CCTL": 8, "CS2R": 2, "ERRBAR": 8, "EXIT": 2, "IADD3": 4, "IMAD": 5, "ISETP": 2, "LDG": 1, "MEMBAR": 8, "MOV": 1, "NOP": 8, "S2R": 1, "STG": 2}`

[↑ contents](#contents)

---

## tma_copy.async_copy_latency

| field | value |
| --- | --- |
| launch | `metadata`  — |
| evidence_tier | `unsupported` |
| fit_status | `unsupported` |
| measurement | `tma_copy.async_copy_latency` = — |
| simulator_param | `tma_copy.async_copy_latency` = — |
| concept | `tma_copy.async_copy_latency` |

- mapping_contract: unsupported

**assumptions:**

- async_copy is not available on volta/v100 (compute capability 7.0)

### Raw values

| key | value |
| --- | --- |
| `required_feature` | async_copy |

<details><summary><code>arch_facts</code> (JSON)</summary>

```json
{
  "compute_capability": "7.0",
  "family": "volta",
  "features": [
    "tensor_core"
  ],
  "l2_cache_mb": 6.0,
  "memory_bandwidth_gbps": 900.0,
  "model": "v100",
  "shared_memory_per_sm_kb": 96,
  "sm_count": 80
}
```

</details>

[↑ contents](#contents)

---

## tma_copy.tma_transfer_sweep

| field | value |
| --- | --- |
| launch | `metadata`  — |
| evidence_tier | `unsupported` |
| fit_status | `unsupported` |
| measurement | `tma_copy.tma_transfer_sweep` = — |
| simulator_param | `tma_copy.tma_transfer_sweep` = — |
| concept | `tma_copy.tma_transfer_sweep` |

- mapping_contract: unsupported

**assumptions:**

- async_copy is not available on volta/v100 (compute capability 7.0)

### Raw values

| key | value |
| --- | --- |
| `required_feature` | async_copy |

<details><summary><code>arch_facts</code> (JSON)</summary>

```json
{
  "compute_capability": "7.0",
  "family": "volta",
  "features": [
    "tensor_core"
  ],
  "l2_cache_mb": 6.0,
  "memory_bandwidth_gbps": 900.0,
  "model": "v100",
  "shared_memory_per_sm_kb": 96,
  "sm_count": 80
}
```

</details>

[↑ contents](#contents)

---

## tma_copy.analyze

| field | value |
| --- | --- |
| launch | `metadata`  — |
| evidence_tier | `unsupported` |
| fit_status | `unsupported` |
| measurement | `tma_copy.analyze` = — |
| simulator_param | `tma_copy.analyze` = — |
| concept | `tma_copy.analyze` |

- mapping_contract: unsupported

**assumptions:**

- async-copy analyzer cannot run: missing inputs from async_copy_latency, tma_transfer_sweep

### Raw values

| key | value |
| --- | --- |
| `async_copy_latency` | unsupported |
| `tma_transfer_sweep` | unsupported |

[↑ contents](#contents)

---

## interconnect.injection_rate

| field | value |
| --- | --- |
| launch | `kernel`  — |
| evidence_tier | `timing_direct` |
| fit_status | `bounded` |
| measurement | `injection_saturation_gbps` = 869.47 GB/s |
| simulator_param | `interconnect_injection_bandwidth` = 869.47 GB/s |
| concept | `interconnect_injection` |

- binary_hash: `dec8f0d22364d43d63baf8cc1046075a55288869b35db8aabf3689762ce4a387`
- interpretation: peak aggregate injection bandwidth vs offered load (blocks per SM)
- mapping_contract: peak aggregate injection bandwidth vs offered load -> simulator interconnect injection bandwidth (bounded)

**assumptions:**

- multi-SM grid-stride stream over a working set far larger than cache
- offered load swept via blocks-per-SM = {1,2,4,8}; best-of-N CUDA-event timing
- peak aggregate GB/s across offered loads is the injection-saturation bandwidth

### Metrics

| key | value | unit |
| --- | --- | --- |
| `saturation_gbps` | 869.47 | GB/s |
| `sweep_points` | 4 | — |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | dec8f0d22364d43d63baf8cc1046075a55288869b35db8aabf3689762ce4a387 |
| `device_name` | Tesla V100-SXM2-32GB |

#### `sweep` (4 rows)

| blocks_per_sm | gbps |
| --- | --- |
| 1 | 597.14 |
| 2 | 783.69 |
| 4 | 862.32 |
| 8 | 869.47 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "df29f3c5ffd87b634e42d4a01f2fff059af329211119fc3cd64ae3230b008852",
  "opcode_histogram": {
    "BMOV": 1,
    "BRA": 2,
    "BSSY": 1,
    "BSYNC": 1,
    "EXIT": 3,
    "FADD": 4,
    "FSETP": 1,
    "IADD3": 2,
    "IMAD": 9,
    "ISETP": 4,
    "LDG": 1,
    "LEA": 2,
    "MOV": 4,
    "NOP": 1,
    "S2R": 2,
    "STG": 1
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/interconnect/injection_rate.cu`
- bytes: `3326`  ·  sha256: `9e952d520fa3eb98451a091df1dc0ed9ff0a06c487db5661967b618619d7ee0a`

### SASS validation

- validated: `True`
- disassembly_hash: `df29f3c5ffd87b634e42d4a01f2fff059af329211119fc3cd64ae3230b008852`
- satisfied: LDG>=1 (1)
- opcode_histogram: `{"BMOV": 1, "BRA": 2, "BSSY": 1, "BSYNC": 1, "EXIT": 3, "FADD": 4, "FSETP": 1, "IADD3": 2, "IMAD": 9, "ISETP": 4, "LDG": 1, "LEA": 2, "MOV": 4, "NOP": 1, "S2R": 2, "STG": 1}`

[↑ contents](#contents)

---

## interconnect.address_mapping

| field | value |
| --- | --- |
| launch | `kernel`  — |
| evidence_tier | `timing_direct` |
| fit_status | `behavioral_only` |
| measurement | `address_mapping_class` = periodic_camping |
| simulator_param | `address_mapping_class` = periodic_camping |
| concept | `address_partition_mapping` |

- binary_hash: `5d7ae78cf27dccfb7189000eb9e6ac22b4d287654529fe41ea8a623d69692ad8`
- interpretation: partition/slice periodicity from base-stride bandwidth variation
- mapping_contract: base-stride bandwidth variation -> simulator address-partition mapping class (candidate/behavioral)

**assumptions:**

- grid-stride reads with a per-step base displacement swept across power-of-two strides
- best-of-N CUDA-event timing per stride; bandwidth varies with partition interleave
- max/min bandwidth ratio < 1.2 classifies as uniform else periodic_camping

### Metrics

| key | value | unit |
| --- | --- | --- |
| `address_mapping_class` | periodic_camping | — |
| `bandwidth_ratio` | 1.4008 | — |
| `max_gbps` | 2097.15 | GB/s |
| `min_gbps` | 1497.11 | GB/s |

### Raw values

| key | value |
| --- | --- |
| `binary_sha256` | 5d7ae78cf27dccfb7189000eb9e6ac22b4d287654529fe41ea8a623d69692ad8 |
| `device_name` | Tesla V100-SXM2-32GB |

#### `sweep` (10 rows)

| gbps | stride_kb |
| --- | --- |
| 2097.15 | 1 |
| 1779.66 | 2 |
| 1558.53 | 4 |
| 1497.11 | 8 |
| 1553.91 | 16 |
| 1542.02 | 32 |
| 1533.01 | 64 |
| 1528.54 | 128 |
| 1528.54 | 256 |
| 1528.54 | 512 |

<details><summary><code>sass</code> (JSON)</summary>

```json
{
  "disassembly_hash": "8f333a83b76f96972fccb9b863423c13b4196b699c58b6eb9f32a128aa14c7ef",
  "opcode_histogram": {
    "BMOV": 5,
    "BRA": 15,
    "BSSY": 5,
    "BSYNC": 5,
    "CALL": 5,
    "EXIT": 3,
    "F2I": 6,
    "FADD": 20,
    "FSETP": 1,
    "I2F": 6,
    "IADD3": 52,
    "IMAD": 117,
    "ISETP": 31,
    "LDG": 5,
    "LEA": 12,
    "LOP3": 11,
    "MOV": 25,
    "MUFU": 6,
    "NOP": 6,
    "RET": 1,
    "S2R": 2,
    "SEL": 6,
    "SHF": 2,
    "STG": 1
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

- path: `/data00/home/chun.liu/wk/amora/amora/probes/nvidia/baseline/interconnect/address_mapping.cu`
- bytes: `4060`  ·  sha256: `851e9735460bc75e719ae687b8aa0072ddf57fc19a80625f747739ea9b5025d9`

### SASS validation

- validated: `True`
- disassembly_hash: `8f333a83b76f96972fccb9b863423c13b4196b699c58b6eb9f32a128aa14c7ef`
- satisfied: LDG>=1 (5)
- opcode_histogram: `{"BMOV": 5, "BRA": 15, "BSSY": 5, "BSYNC": 5, "CALL": 5, "EXIT": 3, "F2I": 6, "FADD": 20, "FSETP": 1, "I2F": 6, "IADD3": 52, "IMAD": 117, "ISETP": 31, "LDG": 5, "LEA": 12, "LOP3": 11, "MOV": 25, "MUFU": 6, "NOP": 6, "RET": 1, "S2R": 2, "SEL": 6, "SHF": 2, "STG": 1}`

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
  "address_mapping_class": "periodic_camping",
  "injection_saturation_gbps": 875.27
}
```

### Metrics

| key | value | unit |
| --- | --- | --- |
| `address_mapping_class` | periodic_camping | — |
| `injection_saturation_gbps` | 875.27 | — |

### Raw values

<details><summary><code>address_mapping</code> (JSON)</summary>

```json
{
  "address_mapping_class": "periodic_camping",
  "binary_sha256": "5d7ae78cf27dccfb7189000eb9e6ac22b4d287654529fe41ea8a623d69692ad8"
}
```

</details>

<details><summary><code>derived</code> (JSON)</summary>

```json
{
  "address_mapping_class": "periodic_camping",
  "injection_saturation_gbps": 875.27
}
```

</details>

<details><summary><code>injection_rate</code> (JSON)</summary>

```json
{
  "binary_sha256": "dec8f0d22364d43d63baf8cc1046075a55288869b35db8aabf3689762ce4a387",
  "injection_saturation_gbps": 875.27
}
```

</details>

[↑ contents](#contents)

---
