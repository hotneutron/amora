# Google TPU sources

Generation catalog: `amora/backends/tpu/profiles/*.json`. Every sourced numeric
fact records its unit scope, source version and source-file hash; absent facts
remain unknown. Runtime metadata and measurements are separate evidence.

- [Cloud TPU architecture](https://cloud.google.com/tpu/docs/system-architecture-tpu-vm)
- [TPU v4](https://cloud.google.com/tpu/docs/v4)
- [TPU v5e](https://cloud.google.com/tpu/docs/v5e)
- [TPU v5p](https://cloud.google.com/tpu/docs/v5p)
- [TPU v6e](https://cloud.google.com/tpu/docs/v6e)
- [TPU7x](https://cloud.google.com/tpu/docs/tpu7x)
- [JAX installation](https://docs.jax.dev/en/latest/installation.html)
- [JAX TPU core programming](https://docs.jax.dev/en/latest/pallas/tpu/core_map.html)
- [JAX shard_map](https://docs.jax.dev/en/latest/notebooks/shard_map.html)
- [JAX TPU metadata source](https://github.com/jax-ml/jax/blob/main/jax/_src/tpu_info.py)

The initial catalog's numeric values are drawn from JAX 0.12.0.1's static
`tpu_info.py`, SHA-256
`ae2ca34b595a5749cee2f9d9c71078694800cf873352db60289be2d3392da80e`.
They establish source-backed architecture defaults; they are not measurements
of the supplied instance. Device/core modes and actual available memory must be
recorded from the deployed runtime. PCI ID mappings in JAX are identification
hints and require runtime corroboration for hardware qualification.

## Revision History

| Revision | UTC date | Change |
|---|---|---|
| r1 | 2026-11-17 | Established generation/runtime sources and pinned initial numeric-fact provenance. |
