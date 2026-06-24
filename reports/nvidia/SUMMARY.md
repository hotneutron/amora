# AMORA Summary — nvidia

- Families: 2  ·  SKUs: 2

## SKUs

| family | sku | device | probes | fit_status | report |
| --- | --- | --- | ---: | --- | --- |
| `hopper` | `h100-80g` | NVIDIA H100 80GB HBM3 | 8 | `direct`=4, `uniquely_identified`=4 | [probes](hopper/probes-h100-80g.md) |
| `volta` | `v100-32g` | Tesla V100-SXM2-32GB | 8 | `direct`=4, `uniquely_identified`=4 | [probes](volta/probes-v100-32g.md) |

## Measurement trends

Per family, one table per probe group. Rows are SKUs; columns are the scalar probes in that group. Adding a SKU appends one row; adding a probe adds one column within its group.

### hopper

#### Compute & Scheduling

| sku | arithmetic_latency.dependent_chain (cycles_per_op) | arithmetic_throughput.independent_chains (cycles_per_op) | topology.persistent_cta (blocks) |
| --- | ---: | ---: | ---: |
| `h100-80g` | 4.377 | 1.1471 | 8 |

#### On-chip Memory

| sku | shared_memory.bank_stride (banks) | shared_memory.pointer_chase (cycles) |
| --- | ---: | ---: |
| `h100-80g` | 32 | 29.0129 |

### volta

#### Compute & Scheduling

| sku | arithmetic_latency.dependent_chain (cycles_per_op) | arithmetic_throughput.independent_chains (cycles_per_op) | topology.persistent_cta (blocks) |
| --- | ---: | ---: | ---: |
| `v100-32g` | 4.376 | 2.1097 | 13 |

#### On-chip Memory

| sku | shared_memory.bank_stride (banks) | shared_memory.pointer_chase (cycles) |
| --- | ---: | ---: |
| `v100-32g` | 32 | 26.9988 |
