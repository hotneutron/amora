# AMORA Google TPU qualification

The TPU backend is qualified on one TPU7x instance, with exact runtime and
workload scope recorded in the [TPU7x summary](tpu7x/qualification-summary.md).
Generation catalog coverage and hardware qualification are separate.

| Generation | Architecture catalog | Native hardware evidence |
|---|---|---|
| v2 | source-backed profile | not collected |
| v3 | source-backed profile | not collected |
| v4 | source-backed profile | not collected |
| v5e | source-backed profile | not collected |
| v5p | source-backed profile | not collected |
| v6e | source-backed profile | not collected |
| TPU7x | source-backed profile | qualified on `gke-tpu-f20b10b7-2pbv` |

Raw compiler, sample and trace artifacts are local under `out/`. Curated reports
link to their verified source manifests and retain source/runtime/target hashes.
