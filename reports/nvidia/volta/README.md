# AMORA Report — nvidia / volta

- SKUs: 1
- Metadata: [manifest.json](manifest.json) · environment: [environment.md](environment.md)

## SKUs

| sku | device | probes | fit_status | generated | report |
| --- | --- | ---: | --- | --- | --- |
| `v100-32g` | Tesla V100-SXM2-32GB | 8 | `direct`=4, `uniquely_identified`=4 | 2026-06-24T06:32Z | [probes](probes-v100-32g.md) |

## `v100-32g` outcomes

| probe_id | mode | evidence_tier | fit_status | measurement |
| --- | --- | --- | --- | --- |
| [topology.device_attributes](probes-v100-32g.md#topologydevice_attributes) | `metadata` | `direct_metadata` | `direct` | _object_ (device_index, device_name, driver_version, uuid) |
| [topology.occupancy](probes-v100-32g.md#topologyoccupancy) | `planning` | `direct_metadata` | `direct` | _object_ (block_sizes, dynamic_shared_memory_bytes, point_count, registers_per_thread, sweep_points) |
| [topology.persistent_cta](probes-v100-32g.md#topologypersistent_cta) | `kernel` | `timing_direct` | `uniquely_identified` | 13 blocks |
| [arithmetic_latency.dependent_chain](probes-v100-32g.md#arithmetic_latencydependent_chain) | `kernel` | `timing_direct` | `direct` | 4.376 cycles_per_op |
| [arithmetic_throughput.independent_chains](probes-v100-32g.md#arithmetic_throughputindependent_chains) | `kernel` | `timing_direct` | `uniquely_identified` | 2.1097 cycles_per_op |
| [shared_memory.pointer_chase](probes-v100-32g.md#shared_memorypointer_chase) | `kernel` | `timing_direct` | `direct` | 26.9988 cycles |
| [shared_memory.bank_stride](probes-v100-32g.md#shared_memorybank_stride) | `kernel` | `timing_direct` | `uniquely_identified` | 32 banks |
| [shared_memory.analyze](probes-v100-32g.md#shared_memoryanalyze) | `analysis` | `coupled_inference` | `uniquely_identified` | _object_ (bank_count, bank_serialization_factor, shared_load_latency_cycles) |
