# TPU v6e architecture profile

Profile: [v6e.json](../../../../amora/backends/tpu/profiles/v6e.json).

These are static JAX 0.12.0.1 architecture facts with scoped units, recorded in
the profile with the source-file SHA-256. Actual runtime mappings, available
memory and feature qualification are recorded per deployed instance.

| Quantity | Source value | Scope |
|---|---:|---|
| Physical TPU TensorCores | 1 | chip |
| Megacore support | false | architecture |
| MXUs | 2 | TPU TensorCore |
| MXU column size | 256 | matrix unit |
| Vector sublanes × lanes | 8 × 128 | vector layout |
| VMEM capacity (bytes) | 134217728 | TPU TensorCore |
| HBM capacity (bytes) | 34400000000 | TPU TensorCore |

Logical JAX device counts can differ from physical core/chip counts. Do not
scale per-core values using an unverified logical-device count. Pallas,
SparseCore and profiler availability require exact runtime/instance admission.
Other generations do not inherit this profile's geometry or memory budgets.

Sources: [Google TPU source index](../tpu/sources.md).
Status: architecture catalogued; hardware qualification is tracked separately.

## Revision History

| Revision | UTC date | Change |
|---|---|---|
| r1 | 2026-11-17 | Recorded scoped, source-backed generation facts and separate runtime qualification. |
