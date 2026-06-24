# AMORA Report — nvidia / hopper

- SKUs: 1
- Metadata: [manifest.json](manifest.json) · environment: [environment.md](environment.md)

## SKUs

| sku | device | probes | fit_status | generated | report |
| --- | --- | ---: | --- | --- | --- |
| `h100-80g` | NVIDIA H100 80GB HBM3 | 8 | `direct`=4, `uniquely_identified`=4 | 2026-06-24T02:08Z | [probes](probes-h100-80g.md) |

## `h100-80g` outcomes

| probe_id | mode | evidence_tier | fit_status | measurement |
| --- | --- | --- | --- | --- |
| [topology.device_attributes](probes-h100-80g.md#topologydevice_attributes) | `metadata` | `direct_metadata` | `direct` | _object_ (device_index, device_name, driver_version, uuid) |
| [topology.occupancy](probes-h100-80g.md#topologyoccupancy) | `planning` | `direct_metadata` | `direct` | _object_ (block_sizes, dynamic_shared_memory_bytes, point_count, registers_per_thread, sweep_points) |
| [topology.persistent_cta](probes-h100-80g.md#topologypersistent_cta) | `kernel` | `timing_direct` | `uniquely_identified` | 8 blocks |
| [arithmetic_latency.dependent_chain](probes-h100-80g.md#arithmetic_latencydependent_chain) | `kernel` | `timing_direct` | `direct` | 4.377 cycles_per_op |
| [arithmetic_throughput.independent_chains](probes-h100-80g.md#arithmetic_throughputindependent_chains) | `kernel` | `timing_direct` | `uniquely_identified` | 1.1471 cycles_per_op |
| [shared_memory.pointer_chase](probes-h100-80g.md#shared_memorypointer_chase) | `kernel` | `timing_direct` | `direct` | 29.0129 cycles |
| [shared_memory.bank_stride](probes-h100-80g.md#shared_memorybank_stride) | `kernel` | `timing_direct` | `uniquely_identified` | 32 banks |
| [shared_memory.analyze](probes-h100-80g.md#shared_memoryanalyze) | `analysis` | `coupled_inference` | `uniquely_identified` | _object_ (bank_count, bank_serialization_factor, shared_load_latency_cycles) |
