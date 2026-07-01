# gcom_cuda sim-vs-HW comparison: gcom_h100

- mapping version: `2026-06-gcom-cuda-v1`
- probes: 36 · with HW value: 20 · with sim value: 3
- categories: approximate=11, comparable=9, unavailable=16

## Validation anchors

- passed: 1 · failed: 2 · unavailable: 6 (tol 50%)
- broad comparison reliable: **False**

## Accuracy rollup (comparable probes)

| group | n | mean |pct err| | median |pct err| |
|---|---|---|---|
| Compute & Scheduling | 2 | 5.179 | 5.179 |
| On-chip Memory | 1 | 0.1811 | 0.1811 |

## Probe-level comparison

| probe | group | category | state | hw | sim | pct err | anchor |
|---|---|---|---|---|---|---|---|
| arithmetic_latency.dependent_chain | Compute & Scheduling | comparable | comparable | 4.377 | 11.02 | 1.517 | ✓ |
| arithmetic_throughput.independent_chains | Compute & Scheduling | comparable | comparable | 1.147 | 11.29 | 8.841 | ✓ |
| scheduler_policy.ready_warps | Compute & Scheduling | approximate | — | 16 | — | — |  |
| scheduler_policy.mixed_issue | Compute & Scheduling | unavailable | unsupported | single_issue_like | — | — |  |
| scheduler_policy.analyze | Compute & Scheduling | unavailable | not_applicable | (composite) | — | — |  |
| topology.device_attributes | Compute & Scheduling | unavailable | not_applicable | (composite) | — | — |  |
| topology.occupancy | Compute & Scheduling | unavailable | not_applicable | (composite) | — | — |  |
| topology.persistent_cta | Compute & Scheduling | approximate | — | 8 | — | — |  |
| register_file.register_bank_sweep | Register, Tensor & Sync | approximate | — | 16 | — | — |  |
| register_file.register_latency | Register, Tensor & Sync | approximate | — | 2.361 | — | — |  |
| register_file.analyze | Register, Tensor & Sync | unavailable | not_applicable | (composite) | — | — |  |
| tensor_core.mma_latency | Register, Tensor & Sync | comparable | — | 24.46 | — | — | ✓ |
| tensor_core.mma_throughput | Register, Tensor & Sync | approximate | — | 0.1599 | — | — | ✓ |
| synchronization.barrier_latency | Register, Tensor & Sync | comparable | — | 45.27 | — | — | ✓ |
| synchronization.fence_latency | Register, Tensor & Sync | approximate | — | 929 | — | — |  |
| shared_memory.pointer_chase | On-chip Memory | comparable | comparable | 29.01 | 34.27 | 0.1811 | ✓ |
| shared_memory.bank_stride | On-chip Memory | unavailable | unsupported | 32 | — | — |  |
| shared_memory.analyze | On-chip Memory | unavailable | not_applicable | (composite) | — | — |  |
| l1_cache.pointer_chase | On-chip Memory | comparable | — | 70.61 | — | — | ✓ |
| l1_cache.working_set | On-chip Memory | approximate | — | (composite) | — | — |  |
| l1_cache.conflict_sets | On-chip Memory | unavailable | unsupported | — | — | — |  |
| l1_cache.analyze | On-chip Memory | unavailable | not_applicable | (composite) | — | — |  |
| l2_cache.pointer_chase | On-chip Memory | comparable | — | 329.9 | — | — | ✓ |
| memory_pipeline.outstanding_requests | On-chip Memory | approximate | — | 4 | — | — |  |
| memory_pipeline.lane_patterns | On-chip Memory | unavailable | proxy_only | 32 | — | — |  |
| memory_pipeline.analyze | On-chip Memory | unavailable | not_applicable | (composite) | — | — |  |
| global_memory.streaming | Global Memory & DRAM | comparable | — | (composite) | — | — | ✓ |
| global_memory.partition_sweep | Global Memory & DRAM | unavailable | unsupported | balanced | — | — |  |
| global_memory.row_policy_sweep | Global Memory & DRAM | approximate | — | 1.755 | — | — |  |
| global_memory.analyze | Global Memory & DRAM | unavailable | not_applicable | (composite) | — | — |  |
| tma_copy.async_copy_latency | Transfer & Interconnect | approximate | — | 723.2 | — | — |  |
| tma_copy.tma_transfer_sweep | Transfer & Interconnect | approximate | — | 37.48 | — | — |  |
| tma_copy.analyze | Transfer & Interconnect | unavailable | not_applicable | (composite) | — | — |  |
| interconnect.injection_rate | Transfer & Interconnect | comparable | — | 3075 | — | — |  |
| interconnect.address_mapping | Transfer & Interconnect | unavailable | unsupported | uniform | — | — |  |
| interconnect.analyze | Transfer & Interconnect | unavailable | not_applicable | (composite) | — | — |  |

## Counter-level comparison (GCoM-derived vs NCU)

| probe | logical | fidelity | hw ncu | sim gcom | pct err |
|---|---|---|---|---|---|
| arithmetic_latency.dependent_chain | dram_throughput | proportional | — | 0 | — |
| arithmetic_latency.dependent_chain | inst_executed | direct | — | 3.147e+05 | — |
| arithmetic_latency.dependent_chain | interconnect_latency | proxy | — | 28 | — |
| arithmetic_latency.dependent_chain | l2_hit_rate | proportional | — | 0.512 | — |
| arithmetic_latency.dependent_chain | l2_sector_hits | proportional | — | 1664 | — |
| arithmetic_latency.dependent_chain | sm_active_cycles | direct | — | 4.512e+04 | — |
| arithmetic_throughput.independent_chains | dram_throughput | proportional | — | 0 | — |
| arithmetic_throughput.independent_chains | inst_executed | direct | — | 7.089e+07 | — |
| arithmetic_throughput.independent_chains | interconnect_latency | proxy | — | 71 | — |
| arithmetic_throughput.independent_chains | l2_hit_rate | proportional | — | 0.6899 | — |
| arithmetic_throughput.independent_chains | l2_sector_hits | proportional | — | 3444 | — |
| arithmetic_throughput.independent_chains | sm_active_cycles | direct | — | 4.624e+04 | — |
| shared_memory.pointer_chase | dram_throughput | proportional | — | 0 | — |
| shared_memory.pointer_chase | inst_executed | direct | — | 1.411e+06 | — |
| shared_memory.pointer_chase | interconnect_latency | proxy | — | 43 | — |
| shared_memory.pointer_chase | l2_hit_rate | proportional | — | 0.429 | — |
| shared_memory.pointer_chase | l2_sector_hits | proportional | — | 3064 | — |
| shared_memory.pointer_chase | sm_active_cycles | direct | — | 1.404e+05 | — |

