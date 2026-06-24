# AMORA Summary — nvidia

- Families: 2  ·  SKUs: 2

## SKUs

| family | sku | device | probes | fit_status | report |
| --- | --- | --- | ---: | --- | --- |
| `hopper` | `h100-80g` | NVIDIA H100 80GB HBM3 | 36 | `behavioral_only`=6, `bounded`=9, `conditionally_identified`=4, `direct`=6, `underconstrained`=4, `uniquely_identified`=7 | [probes](hopper/probes-h100-80g.md) |
| `volta` | `v100-32g` | Tesla V100-SXM2-32GB | 8 | `direct`=4, `uniquely_identified`=4 | [probes](volta/probes-v100-32g.md) |

## Measurement trends

Per family, one table per probe group. Rows are SKUs; columns are the scalar probes in that group. Adding a SKU appends one row; adding a probe adds one column within its group.

### hopper

#### Compute & Scheduling

| sku | arithmetic_latency.dependent_chain (cycles_per_op) | arithmetic_throughput.independent_chains (cycles_per_op) | scheduler_policy.mixed_issue | scheduler_policy.ready_warps (warps) | topology.persistent_cta (blocks) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h100-80g` | 4.377 | 1.1471 | single_issue_like | 16 | 8 |

#### Register, Tensor & Sync

| sku | register_file.register_bank_sweep (accumulators) | register_file.register_latency (cycles) | synchronization.barrier_latency (cycles) | synchronization.fence_latency (cycles) | tensor_core.mma_latency (cycles_per_op) | tensor_core.mma_throughput (mma/cycle) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `h100-80g` | 16 | 2.3606 | 45.2683 | 928.961 | 24.4648 | 0.1599 |

#### On-chip Memory

| sku | l1_cache.conflict_sets (ways) | l1_cache.pointer_chase (cycles) | l2_cache.pointer_chase (cycles) | shared_memory.bank_stride (banks) | shared_memory.pointer_chase (cycles) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `h100-80g` | — | 70.6091 | 329.901 | 32 | 29.0146 |

#### Global Memory & DRAM

| sku | global_memory.partition_sweep | global_memory.row_policy_sweep (ratio) | memory_pipeline.lane_patterns (sectors/request) | memory_pipeline.outstanding_requests (loads) |
| --- | ---: | ---: | ---: | ---: |
| `h100-80g` | balanced | 1.75543 | 32 | 4 |

#### Transfer & Interconnect

| sku | interconnect.address_mapping | interconnect.injection_rate (GB/s) | tma_copy.async_copy_latency (cycles) | tma_copy.tma_transfer_sweep (GB/s) |
| --- | ---: | ---: | ---: | ---: |
| `h100-80g` | uniform | 3075 | 723.234 | 37.48 |

### volta

#### Compute & Scheduling

| sku | arithmetic_latency.dependent_chain (cycles_per_op) | arithmetic_throughput.independent_chains (cycles_per_op) | topology.persistent_cta (blocks) |
| --- | ---: | ---: | ---: |
| `v100-32g` | 4.376 | 2.1097 | 13 |

#### On-chip Memory

| sku | shared_memory.bank_stride (banks) | shared_memory.pointer_chase (cycles) |
| --- | ---: | ---: |
| `v100-32g` | 32 | 26.9988 |
