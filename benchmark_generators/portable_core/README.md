# Portable core20/core24/core28 workloads

A lightweight workload-definition package hosted by AMORA. It provides typed
mathematical programs, deterministic inputs, independent oracles and portable
NumPy/PyTorch/optional-JAX execution. It does **not** contain a predictor or claim
native TPU/Ascend performance qualification.

## Why this belongs here

AMORA already separates benchmark definitions in `benchmark_generators/` from
hardware probes and measurement backends. Workload semantics belong here;
Accorde's predictor fitting, OOD gates and campaign evidence remain separate.
The package is not a new NVIDIA diagnostic probe and is deliberately not wired
into the existing CUDA-template classification/detail runner.

Use one canonical future workload implementation in AMORA, with version/hash
bindings for consumers. Accorde's old campaign snapshots remain frozen historical
sources; do not actively maintain two unversioned copies.

## Install and CLI

From the repository root:

```bash
pip install -e '.[portable-core,test]'
python -m benchmark_generators.portable_core list --tier 28
python -m benchmark_generators.portable_core freeze --tier 24 --out out/portable_core/core24
python -m benchmark_generators.portable_core check \
  --frozen out/portable_core/core24 --out out/portable_core/core24-check \
  --backends numpy,torch,jax
```

The installed dedicated entrypoint is `amora-portable-core`. NumPy is an optional
extra, not a new mandatory dependency of AMORA's basic probe CLI. Install a
compatible PyTorch/JAX/NPU runtime separately if that backend is needed; absence
fails closed rather than falling back to CPU for an accelerator request.

## API

```python
from benchmark_generators import portable_core as kernels
case = kernels.generate_cases(tier=28, profile="smoke")[0]
inputs = kernels.make_inputs(case)
backend = kernels.create_backend("numpy", "cpu")
outputs = kernels.evaluate(case, backend, inputs)  # always a tuple
```

|Tier|Families|Smoke cases|Full-shape specifications|
|---|---:|---:|---:|
|20|20|152|304|
|24|24|168|336|
|28|28|184|368|

Tier24 adds exact dense attention, SwiGLU, gathered-KV attention and grouped/ragged
GEMM. Tier28 adds inclusive scan, deterministic segmented sum, zero-initialized
scatter-add and signed per-channel zero-boundary3x3 stencil. All casts, reduction
axes, group/offset metadata and tuple/empty outputs are explicit.

## Scope and integrity

- BF16-rounded matrix operands, FP32 accumulation/output; PyTorch's current
  implementation promotes BF16 values to FP32 before matrix arithmetic.
- Pure FP32 broadcast is not a reduction; row/column alignment is explicit.
- Prepared execution separates host validation/input staging from the program body.
- Native atomic/carry instructions, compiler fusion and device-only timing remain
  separately admitted facts, not conclusions from array-API correctness.
- Family/case identifiers are metadata, not serving model features.
- Generated manifests and runtime sources are hash-bound and refuse overwrite.
- `SEMANTIC_FINGERPRINTS.json` binds all six tier/profile case payloads to their
  Accorde source. Packaging adapts namespaces, not mathematical contracts.

Only source, generator/test contracts and provenance are packaged. Bulk input
banks, temporary validation runs, historical timings and model checkpoints are
not imported into this package. Regenerate inputs/manifests on demand, or use
consumer-controlled artifact storage with hashes.

## License and provenance

Imported code is **Apache-2.0**, preserved in `LICENSE.Apache-2.0` and `NOTICE`.
AMORA's root MIT license is unchanged and does not relicense these components.
Adapted files contain prominent namespace-change notices. `PROVENANCE.json`
records exact source/packaged hashes and the user requests verbatim; quoted
machine-local request text is kept in JSON, not planning/design Markdown paths.

## Revision History

|Revision|Timestamp|Change|
|---|---|---|
|r1|2026-10-22|Packaged the portable workloads under AMORA's benchmark-definition namespace,with a dedicated API/CLI,optional array dependency,Apache provenance and no bulk campaign/model import or inherited native-performance claim.|
