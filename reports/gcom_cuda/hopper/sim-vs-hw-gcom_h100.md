# gcom_cuda sim-vs-HW comparison: gcom_h100

- mapping version: `2026-06-gcom-cuda-v1`
- probes: 36 · with HW value: 20 · with sim value: 6
- categories: approximate=11, comparable=9, unavailable=16

## Validation anchors

- passed: 1 · failed: 4 · unavailable: 4 (tol 50%)
- broad comparison reliable: **False**

## Accuracy rollup (comparable probes)

| group | n | mean |pct err| | median |pct err| |
|---|---|---|---|
| Compute & Scheduling | 2 | 514.8% | 514.8% |
| On-chip Memory | 1 | 15.6% | 15.6% |
| Register, Tensor & Sync | 1 | 199.0% | 199.0% |

## Probe-level comparison

| probe | group | category | state | hw | sim | pct err | anchor |
|---|---|---|---|---|---|---|---|
| arithmetic_latency.dependent_chain | Compute & Scheduling | comparable | comparable | 4.377 | 11.02 | 151.7% | ✓ |
| arithmetic_throughput.independent_chains | Compute & Scheduling | comparable | comparable | 1.147 | 11.22 | 877.8% | ✓ |
| scheduler_policy.ready_warps | Compute & Scheduling | approximate | missing_stat | 16 | — | — |  |
| scheduler_policy.mixed_issue | Compute & Scheduling | unavailable | unsupported | single_issue_like | — | — |  |
| scheduler_policy.analyze | Compute & Scheduling | unavailable | not_applicable | (composite) | — | — |  |
| topology.device_attributes | Compute & Scheduling | unavailable | not_applicable | (composite) | — | — |  |
| topology.occupancy | Compute & Scheduling | unavailable | not_applicable | (composite) | — | — |  |
| topology.persistent_cta | Compute & Scheduling | approximate | missing_stat | 8 | — | — |  |
| register_file.register_bank_sweep | Register, Tensor & Sync | approximate | missing_stat | 16 | — | — |  |
| register_file.register_latency | Register, Tensor & Sync | approximate | missing_stat | 2.361 | — | — |  |
| register_file.analyze | Register, Tensor & Sync | unavailable | not_applicable | (composite) | — | — |  |
| tensor_core.mma_latency | Register, Tensor & Sync | comparable | comparable | 24.46 | 73.14 | 199.0% | ✓ |
| tensor_core.mma_throughput | Register, Tensor & Sync | approximate | approximate | 0.1599 | 1.236e-05 | 100.0% | ✓ |
| synchronization.barrier_latency | Register, Tensor & Sync | comparable | missing_stat | 45.27 | — | — | ✓ |
| synchronization.fence_latency | Register, Tensor & Sync | approximate | missing_stat | 929 | — | — |  |
| shared_memory.pointer_chase | On-chip Memory | comparable | comparable | 29.01 | 33.53 | 15.6% | ✓ |
| shared_memory.bank_stride | On-chip Memory | unavailable | unsupported | 32 | — | — |  |
| shared_memory.analyze | On-chip Memory | unavailable | not_applicable | (composite) | — | — |  |
| l1_cache.pointer_chase | On-chip Memory | comparable | missing_stat | 70.61 | — | — | ✓ |
| l1_cache.working_set | On-chip Memory | approximate | missing_stat | (composite) | — | — |  |
| l1_cache.conflict_sets | On-chip Memory | unavailable | unsupported | — | — | — |  |
| l1_cache.analyze | On-chip Memory | unavailable | not_applicable | (composite) | — | — |  |
| l2_cache.pointer_chase | On-chip Memory | comparable | missing_stat | 329.9 | — | — | ✓ |
| memory_pipeline.outstanding_requests | On-chip Memory | approximate | missing_stat | 4 | — | — |  |
| memory_pipeline.lane_patterns | On-chip Memory | unavailable | proxy_only | 32 | — | — |  |
| memory_pipeline.analyze | On-chip Memory | unavailable | not_applicable | (composite) | — | — |  |
| global_memory.streaming | Global Memory & DRAM | comparable | missing_stat | (composite) | — | — | ✓ |
| global_memory.partition_sweep | Global Memory & DRAM | unavailable | unsupported | balanced | — | — |  |
| global_memory.row_policy_sweep | Global Memory & DRAM | approximate | missing_stat | 1.755 | — | — |  |
| global_memory.analyze | Global Memory & DRAM | unavailable | not_applicable | (composite) | — | — |  |
| tma_copy.async_copy_latency | Transfer & Interconnect | approximate | approximate | 723.2 | 442.7 | 38.8% |  |
| tma_copy.tma_transfer_sweep | Transfer & Interconnect | approximate | missing_stat | 37.48 | — | — |  |
| tma_copy.analyze | Transfer & Interconnect | unavailable | not_applicable | (composite) | — | — |  |
| interconnect.injection_rate | Transfer & Interconnect | comparable | missing_stat | 3075 | — | — |  |
| interconnect.address_mapping | Transfer & Interconnect | unavailable | unsupported | uniform | — | — |  |
| interconnect.analyze | Transfer & Interconnect | unavailable | not_applicable | (composite) | — | — |  |

## Counter-level comparison (GCoM-derived vs NCU)

| probe | logical | fidelity | hw ncu | sim gcom | pct err |
|---|---|---|---|---|---|
| arithmetic_latency.dependent_chain | dram_throughput | proportional | — | 0 | — |
| arithmetic_latency.dependent_chain | inst_executed | direct | — | 3.147e+05 | — |
| arithmetic_latency.dependent_chain | interconnect_latency | proxy | — | 28 | — |
| arithmetic_latency.dependent_chain | l2_hit_rate | proportional | — | 0.3151 | — |
| arithmetic_latency.dependent_chain | l2_sector_hits | proportional | — | 1024 | — |
| arithmetic_latency.dependent_chain | sm_active_cycles | direct | — | 4.512e+04 | — |
| arithmetic_throughput.independent_chains | dram_throughput | proportional | — | 7.6e-05 | — |
| arithmetic_throughput.independent_chains | inst_executed | direct | — | 7.089e+07 | — |
| arithmetic_throughput.independent_chains | interconnect_latency | proxy | — | 71 | — |
| arithmetic_throughput.independent_chains | l2_hit_rate | proportional | — | 0.7091 | — |
| arithmetic_throughput.independent_chains | l2_sector_hits | proportional | — | 3540 | — |
| arithmetic_throughput.independent_chains | sm_active_cycles | direct | — | 4.594e+04 | — |
| shared_memory.pointer_chase | dram_throughput | proportional | — | 1e-06 | — |
| shared_memory.pointer_chase | inst_executed | direct | — | 1.411e+06 | — |
| shared_memory.pointer_chase | interconnect_latency | proxy | — | 43 | — |
| shared_memory.pointer_chase | l2_hit_rate | proportional | — | 0.9846 | — |
| shared_memory.pointer_chase | l2_sector_hits | proportional | — | 7032 | — |
| shared_memory.pointer_chase | sm_active_cycles | direct | — | 1.373e+05 | — |
| tensor_core.mma_latency | dram_throughput | proportional | — | 1.2e-05 | — |
| tensor_core.mma_latency | inst_executed | direct | — | 1.659e+06 | — |
| tensor_core.mma_latency | interconnect_latency | proxy | — | 23 | — |
| tensor_core.mma_latency | l2_hit_rate | proportional | — | 0.4696 | — |
| tensor_core.mma_latency | l2_sector_hits | proportional | — | 1.344e+04 | — |
| tensor_core.mma_latency | sm_active_cycles | direct | — | 3.745e+04 | — |
| tensor_core.mma_throughput | dram_throughput | proportional | — | 1.6e-05 | — |
| tensor_core.mma_throughput | inst_executed | direct | — | 2.285e+06 | — |
| tensor_core.mma_throughput | interconnect_latency | proxy | — | 22 | — |
| tensor_core.mma_throughput | l2_hit_rate | proportional | — | 0.322 | — |
| tensor_core.mma_throughput | l2_sector_hits | proportional | — | 3328 | — |
| tensor_core.mma_throughput | sm_active_cycles | direct | — | 1.293e+04 | — |
| tma_copy.async_copy_latency | dram_throughput | proportional | — | 0.0001 | — |
| tma_copy.async_copy_latency | inst_executed | direct | — | 1.53e+07 | — |
| tma_copy.async_copy_latency | interconnect_latency | proxy | — | 2 | — |
| tma_copy.async_copy_latency | l2_hit_rate | proportional | — | 0.3757 | — |
| tma_copy.async_copy_latency | l2_sector_hits | proportional | — | 1.029e+05 | — |
| tma_copy.async_copy_latency | sm_active_cycles | direct | — | 2.833e+04 | — |

