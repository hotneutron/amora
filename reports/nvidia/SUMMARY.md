# AMORA Summary — nvidia

- Families: 1  ·  SKUs: 1

## SKUs

| family | sku | device | probes | fit_status | report |
| --- | --- | --- | ---: | --- | --- |
| `hopper` | `h100-80g` | NVIDIA H100 80GB HBM3 | 8 | `direct`=4, `uniquely_identified`=4 | [probes](hopper/probes-h100-80g.md) |

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
