# Portable-core workloads: legacy tiers and V2 suites

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

## V2 suites

V1 is still selected by `--tier` and uses the unchanged legacy payloads. Select
v2 with `--suite`:

| Suite | Reference | Development |
|---|---|---|
| `operation-gate-v2` | 720 cases from the pinned 240-motif registry | 288 development-only source cases |
| `mechanism-controls` | The same 288 development specs, for independent selection | 9 diagnostic softmax, variance and top-k controls |
| `h3-semantics` | Blocked pending approved case-source export | 40 separately identified semantic and indexed-access controls |
| `h3-programs` | Blocked pending approved case-source export | 50 directed pair/size cases, 32 topology/shape cases and two fork/chain cases |

Each suite has a bounded `smoke` profile. To freeze and check a v2 shard:

```bash
amora-portable-core list --suite operation-gate-v2
amora-portable-core freeze --suite operation-gate-v2 --profile reference \
  --out out/portable_core/v2-gate
amora-portable-core check --frozen out/portable_core/v2-gate \
  --start 0 --stop 48 --out out/portable_core/v2-gate-first
```

V2 checks use NumPy and, for required parity rows, Torch CPU. They compare
node outputs against an independent oracle and fail if a required backend or
row is missing. The [CPU parity lane](../../.github/workflows/portable-core-v2-cpu.yml)
pins Torch 2.8.0 and runs the complete source-available matrix in bounded shards.
A shard pass is not whole-snapshot coverage; aggregate disjoint shards with
`aggregate --frozen ... --shard ...`. There is no v2 JAX/accelerator fallback.

The v2 snapshot carries versioned typed programs, canonical program/case hashes,
source crosswalk and coverage receipts. Graph ports and historical case identity
are preserved separately from new portable input bytes. The archived Gate-v2
FP16 CUDA RNG sequence and native timing are **not** replayed. H3 development
controls do not close exact-reference rows. Consumer-side verification can use
[verify_export.py](../../tools/portable_core_v2/verify_export.py) with a manifest
SHA-256 pinned independently of the export. The [tracker](../../.plan/trackers/20261123-1522-portable-core-h3-operation-coverage.md)
records release evidence and remaining external gates.

## V2 accelerator correctness

The CPU checker above is unchanged. The separate accelerator CLI consumes the
same immutable snapshot and executes one bounded shard on an explicit device:

```bash
amora benchmarks inspect-v2
amora-portable-core freeze --suite h3-programs --profile smoke \
  --out out/portable_core/v2-h3-smoke
amora benchmarks qualify-v2 --frozen out/portable_core/v2-h3-smoke \
  --backend cuda_torch --device cuda:0 --out out/portable_core/v2-cuda-check
amora benchmarks qualify-v2 --frozen out/portable_core/v2-h3-smoke \
  --backend tpu_jax --device tpu:0 --expected-generation tpu7x \
  --out out/portable_core/v2-tpu-check
```

Install a compatible CUDA-enabled Torch separately for the CUDA worker; a CPU or
ROCm build is not accepted. For TPU, run through the existing
[host-specific launch wrapper](../../tools/tpu/instance.py) in a verified
JAX/jaxlib/libtpu environment with its `runtime-lock.txt` and approved ownership.
Use `--case` repeatedly to select at most 32 cases from larger snapshots, and
`--manifest-sha256` to bind a previously approved snapshot. Do not reuse an
output directory. Library-free `inspect-v2` declares adapter capability only;
it does not open a device or assert native support.

Each result requires verified device placement and CPU-oracle parity for the
output-only program and its node-capture control, plus a separate FP16/FP32
rounding witness. Only packed layouts and unique scatter destinations are
admitted. Unsupported rows remain unqualified and the command returns nonzero
if any requested case does not pass. The TPU worker retains separate StableHLO
and optimized HLO artifacts for its two executables, not native ISA/disassembly.
The [device gate tracker](../../.plan/trackers/20261123-1650-portable-core-v2-cuda-tpu-backends.md)
records 1,130 passes and 11 expected denials on each of H100 and TPU7x.
`qualify-v2` does not time workloads. The separate
[V2 accelerator and latency guide](../../docs/development/portable-core-v2-accelerators.md)
explains the verified TPU7x latency dataset and Accorde handoff. CUDA latency
and other TPU generations are not qualified by those measurements.

## Legacy scope and integrity

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
|r2|2026-11-23|Documented the isolated v2 suite selector, frozen Gate-v2 sources, mandatory CPU coverage and source-blocked H3 reference boundary.|
|r3|2026-11-23|Documented bounded CUDA/TPU correctness commands, the packed/unique-index contract and native evidence limits. The CPU checker is unchanged.|
|r4|2026-11-25|Recorded full H100 and TPU7x correctness coverage and linked the separate TPU latency workflow.|
