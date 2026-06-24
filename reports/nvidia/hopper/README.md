# AMORA Report — nvidia / hopper

- SKUs: 1
- Metadata: [manifest.json](manifest.json) · environment: [environment.md](environment.md)

## SKUs

| sku | device | probes | fit_status | generated | report |
| --- | --- | ---: | --- | --- | --- |
| `h100-80g` | NVIDIA H100 80GB HBM3 | 36 | `behavioral_only`=6, `bounded`=8, `conditionally_identified`=4, `direct`=6, `underconstrained`=5, `uniquely_identified`=7 | 2026-06-24T09:48Z | [probes](probes-h100-80g.md) |

## `h100-80g` outcomes

| probe_id | mode | evidence_tier | fit_status | measurement |
| --- | --- | --- | --- | --- |
| [topology.device_attributes](probes-h100-80g.md#topologydevice_attributes) | `metadata` | `direct_metadata` | `direct` | _object_ (device_index, device_name, driver_version, uuid) |
| [topology.occupancy](probes-h100-80g.md#topologyoccupancy) | `planning` | `direct_metadata` | `direct` | _object_ (block_sizes, dynamic_shared_memory_bytes, point_count, registers_per_thread, sweep_points) |
| [topology.persistent_cta](probes-h100-80g.md#topologypersistent_cta) | `kernel` | `timing_direct` | `uniquely_identified` | 8 blocks |
| [arithmetic_latency.dependent_chain](probes-h100-80g.md#arithmetic_latencydependent_chain) | `kernel` | `timing_direct` | `direct` | 4.377 cycles_per_op |
| [arithmetic_throughput.independent_chains](probes-h100-80g.md#arithmetic_throughputindependent_chains) | `kernel` | `timing_direct` | `uniquely_identified` | 1.1471 cycles_per_op |
| [shared_memory.pointer_chase](probes-h100-80g.md#shared_memorypointer_chase) | `kernel` | `timing_direct` | `direct` | 29.0146 cycles |
| [shared_memory.bank_stride](probes-h100-80g.md#shared_memorybank_stride) | `kernel` | `timing_direct` | `uniquely_identified` | 32 banks |
| [shared_memory.analyze](probes-h100-80g.md#shared_memoryanalyze) | `analysis` | `coupled_inference` | `uniquely_identified` | _object_ (bank_count, bank_serialization_factor, shared_load_latency_cycles) |
| [l1_cache.pointer_chase](probes-h100-80g.md#l1_cachepointer_chase) | `kernel` | `timing_direct` | `direct` | 70.6099 cycles |
| [l1_cache.working_set](probes-h100-80g.md#l1_cacheworking_set) | `kernel` | `timing_direct` | `underconstrained` | _object_ |
| [l1_cache.conflict_sets](probes-h100-80g.md#l1_cacheconflict_sets) | `kernel` | `timing_direct` | `underconstrained` | — |
| [l1_cache.analyze](probes-h100-80g.md#l1_cacheanalyze) | `analysis` | `coupled_inference` | `underconstrained` | _object_ (l1_effective_capacity_kb, l1_hit_latency_cycles) |
| [scheduler_policy.ready_warps](probes-h100-80g.md#scheduler_policyready_warps) | `kernel` | `timing_direct` | `conditionally_identified` | 16 warps |
| [scheduler_policy.mixed_issue](probes-h100-80g.md#scheduler_policymixed_issue) | `kernel` | `timing_direct` | `behavioral_only` | single_issue_like |
| [scheduler_policy.analyze](probes-h100-80g.md#scheduler_policyanalyze) | `analysis` | `coupled_inference` | `behavioral_only` | _object_ (issue_saturation_warps, mixed_issue_class, peak_ops_per_cycle) |
| [register_file.register_bank_sweep](probes-h100-80g.md#register_fileregister_bank_sweep) | `kernel` | `timing_direct` | `underconstrained` | 16 accumulators |
| [register_file.register_latency](probes-h100-80g.md#register_fileregister_latency) | `kernel` | `timing_direct` | `conditionally_identified` | 2.3606 cycles |
| [register_file.analyze](probes-h100-80g.md#register_fileanalyze) | `analysis` | `coupled_inference` | `underconstrained` | _object_ (operand_delivery_differential_cycles, operand_delivery_plateau_accumulators) |
| [synchronization.barrier_latency](probes-h100-80g.md#synchronizationbarrier_latency) | `kernel` | `timing_direct` | `uniquely_identified` | 45.2683 cycles |
| [global_memory.streaming](probes-h100-80g.md#global_memorystreaming) | `kernel` | `timing_direct` | `bounded` | _object_ (copy_gbps, peak_gbps, read_gbps, write_gbps) |
| [l2_cache.pointer_chase](probes-h100-80g.md#l2_cachepointer_chase) | `kernel` | `timing_direct` | `bounded` | 329.514 cycles |
| [memory_pipeline.outstanding_requests](probes-h100-80g.md#memory_pipelineoutstanding_requests) | `kernel` | `timing_direct` | `bounded` | 4 loads |
| [memory_pipeline.lane_patterns](probes-h100-80g.md#memory_pipelinelane_patterns) | `kernel` | `direct_counter` | `direct` | 32 sectors/request |
| [memory_pipeline.analyze](probes-h100-80g.md#memory_pipelineanalyze) | `analysis` | `coupled_inference` | `bounded` | _object_ (coalescing_sectors_per_request, effective_outstanding_requests) |
| [global_memory.partition_sweep](probes-h100-80g.md#global_memorypartition_sweep) | `kernel` | `timing_direct` | `behavioral_only` | balanced |
| [global_memory.row_policy_sweep](probes-h100-80g.md#global_memoryrow_policy_sweep) | `kernel` | `timing_direct` | `bounded` | 1.7605 ratio |
| [global_memory.analyze](probes-h100-80g.md#global_memoryanalyze) | `analysis` | `coupled_inference` | `behavioral_only` | _object_ (partition_class, peak_gbps, row_locality_sensitivity) |
| [tensor_core.mma_latency](probes-h100-80g.md#tensor_coremma_latency) | `kernel` | `timing_direct` | `uniquely_identified` | 24.4648 cycles_per_op |
| [tensor_core.mma_throughput](probes-h100-80g.md#tensor_coremma_throughput) | `kernel` | `timing_direct` | `uniquely_identified` | 0.1598 mma/cycle |
| [synchronization.fence_latency](probes-h100-80g.md#synchronizationfence_latency) | `kernel` | `timing_direct` | `conditionally_identified` | 927.239 cycles |
| [tma_copy.async_copy_latency](probes-h100-80g.md#tma_copyasync_copy_latency) | `kernel` | `timing_direct` | `conditionally_identified` | 723.672 cycles |
| [tma_copy.tma_transfer_sweep](probes-h100-80g.md#tma_copytma_transfer_sweep) | `kernel` | `timing_direct` | `bounded` | 37.46 GB/s |
| [tma_copy.analyze](probes-h100-80g.md#tma_copyanalyze) | `analysis` | `coupled_inference` | `bounded` | _object_ (async_copy_peak_gbps, async_copy_tile_latency) |
| [interconnect.injection_rate](probes-h100-80g.md#interconnectinjection_rate) | `kernel` | `timing_direct` | `bounded` | 3064.33 GB/s |
| [interconnect.address_mapping](probes-h100-80g.md#interconnectaddress_mapping) | `kernel` | `timing_direct` | `behavioral_only` | uniform |
| [interconnect.analyze](probes-h100-80g.md#interconnectanalyze) | `analysis` | `coupled_inference` | `behavioral_only` | _object_ (address_mapping_class, injection_saturation_gbps) |
