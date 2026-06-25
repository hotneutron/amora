# AMORA Report — nvidia / volta

- SKUs: 1
- Metadata: [manifest.json](manifest.json) · environment: [environment.md](environment.md)

## SKUs

| sku | device | probes | fit_status | generated | report |
| --- | --- | ---: | --- | --- | --- |
| `v100-32g` | Tesla V100-SXM2-32GB | 36 | `behavioral_only`=8, `bounded`=7, `conditionally_identified`=3, `direct`=5, `underconstrained`=3, `uniquely_identified`=7, `unsupported`=3 | 2026-06-25T18:24Z | [probes](probes-v100-32g.md) |

## `v100-32g` outcomes

| probe_id | mode | evidence_tier | fit_status | measurement |
| --- | --- | --- | --- | --- |
| [topology.device_attributes](probes-v100-32g.md#topologydevice_attributes) | `metadata` | `direct_metadata` | `direct` | _object_ (device_index, device_name, driver_version, published_facts, uuid) |
| [topology.occupancy](probes-v100-32g.md#topologyoccupancy) | `planning` | `direct_metadata` | `direct` | _object_ (block_sizes, dynamic_shared_memory_bytes, point_count, registers_per_thread, sweep_points) |
| [topology.persistent_cta](probes-v100-32g.md#topologypersistent_cta) | `kernel` | `timing_direct` | `uniquely_identified` | 13 blocks |
| [arithmetic_latency.dependent_chain](probes-v100-32g.md#arithmetic_latencydependent_chain) | `kernel` | `timing_direct` | `direct` | 4.376 cycles_per_op |
| [arithmetic_throughput.independent_chains](probes-v100-32g.md#arithmetic_throughputindependent_chains) | `kernel` | `timing_direct` | `uniquely_identified` | 2.1097 cycles_per_op |
| [shared_memory.pointer_chase](probes-v100-32g.md#shared_memorypointer_chase) | `kernel` | `timing_direct` | `direct` | 26.9988 cycles |
| [shared_memory.bank_stride](probes-v100-32g.md#shared_memorybank_stride) | `kernel` | `timing_direct` | `uniquely_identified` | 32 banks |
| [shared_memory.analyze](probes-v100-32g.md#shared_memoryanalyze) | `analysis` | `coupled_inference` | `uniquely_identified` | _object_ (bank_count, bank_serialization_factor, shared_load_latency_cycles) |
| [l1_cache.pointer_chase](probes-v100-32g.md#l1_cachepointer_chase) | `kernel` | `timing_direct` | `direct` | 59.0576 cycles |
| [l1_cache.working_set](probes-v100-32g.md#l1_cacheworking_set) | `kernel` | `timing_direct` | `bounded` | _object_ (effective_l1_kb_high, effective_l1_kb_low) |
| [l1_cache.conflict_sets](probes-v100-32g.md#l1_cacheconflict_sets) | `kernel` | `timing_direct` | `underconstrained` | — |
| [l1_cache.analyze](probes-v100-32g.md#l1_cacheanalyze) | `analysis` | `coupled_inference` | `underconstrained` | _object_ (l1_effective_capacity_kb, l1_hit_latency_cycles) |
| [scheduler_policy.ready_warps](probes-v100-32g.md#scheduler_policyready_warps) | `kernel` | `timing_direct` | `conditionally_identified` | 12 warps |
| [scheduler_policy.mixed_issue](probes-v100-32g.md#scheduler_policymixed_issue) | `kernel` | `timing_direct` | `behavioral_only` | single_issue_like |
| [scheduler_policy.analyze](probes-v100-32g.md#scheduler_policyanalyze) | `analysis` | `coupled_inference` | `behavioral_only` | _object_ (issue_saturation_warps, mixed_issue_class, peak_ops_per_cycle) |
| [register_file.register_bank_sweep](probes-v100-32g.md#register_fileregister_bank_sweep) | `kernel` | `timing_direct` | `bounded` | 12 accumulators |
| [register_file.register_latency](probes-v100-32g.md#register_fileregister_latency) | `kernel` | `timing_direct` | `conditionally_identified` | 1.4382 cycles |
| [register_file.analyze](probes-v100-32g.md#register_fileanalyze) | `analysis` | `coupled_inference` | `underconstrained` | _object_ (operand_delivery_differential_cycles, operand_delivery_plateau_accumulators) |
| [synchronization.barrier_latency](probes-v100-32g.md#synchronizationbarrier_latency) | `kernel` | `timing_direct` | `uniquely_identified` | 57.001 cycles |
| [global_memory.streaming](probes-v100-32g.md#global_memorystreaming) | `kernel` | `timing_direct` | `bounded` | _object_ (copy_gbps, peak_gbps, read_gbps, write_gbps) |
| [l2_cache.pointer_chase](probes-v100-32g.md#l2_cachepointer_chase) | `kernel` | `timing_direct` | `bounded` | 248.352 cycles |
| [memory_pipeline.outstanding_requests](probes-v100-32g.md#memory_pipelineoutstanding_requests) | `kernel` | `timing_direct` | `bounded` | 4 loads |
| [memory_pipeline.lane_patterns](probes-v100-32g.md#memory_pipelinelane_patterns) | `kernel` | `timing_direct` | `behavioral_only` | — |
| [memory_pipeline.analyze](probes-v100-32g.md#memory_pipelineanalyze) | `analysis` | `coupled_inference` | `behavioral_only` | _object_ (effective_outstanding_requests) |
| [global_memory.partition_sweep](probes-v100-32g.md#global_memorypartition_sweep) | `kernel` | `timing_direct` | `behavioral_only` | balanced |
| [global_memory.row_policy_sweep](probes-v100-32g.md#global_memoryrow_policy_sweep) | `kernel` | `timing_direct` | `bounded` | 1.56871 ratio |
| [global_memory.analyze](probes-v100-32g.md#global_memoryanalyze) | `analysis` | `coupled_inference` | `behavioral_only` | _object_ (partition_class, peak_gbps, row_locality_sensitivity) |
| [tensor_core.mma_latency](probes-v100-32g.md#tensor_coremma_latency) | `kernel` | `timing_direct` | `uniquely_identified` | 64.3516 cycles_per_op |
| [tensor_core.mma_throughput](probes-v100-32g.md#tensor_coremma_throughput) | `kernel` | `timing_direct` | `uniquely_identified` | 0.009 mma/cycle |
| [synchronization.fence_latency](probes-v100-32g.md#synchronizationfence_latency) | `kernel` | `timing_direct` | `conditionally_identified` | 163.074 cycles |
| [tma_copy.async_copy_latency](probes-v100-32g.md#tma_copyasync_copy_latency) | `metadata` | `unsupported` | `unsupported` | — |
| [tma_copy.tma_transfer_sweep](probes-v100-32g.md#tma_copytma_transfer_sweep) | `metadata` | `unsupported` | `unsupported` | — |
| [tma_copy.analyze](probes-v100-32g.md#tma_copyanalyze) | `metadata` | `unsupported` | `unsupported` | — |
| [interconnect.injection_rate](probes-v100-32g.md#interconnectinjection_rate) | `kernel` | `timing_direct` | `bounded` | 869.47 GB/s |
| [interconnect.address_mapping](probes-v100-32g.md#interconnectaddress_mapping) | `kernel` | `timing_direct` | `behavioral_only` | periodic_camping |
| [interconnect.analyze](probes-v100-32g.md#interconnectanalyze) | `analysis` | `coupled_inference` | `behavioral_only` | _object_ (address_mapping_class, injection_saturation_gbps) |
