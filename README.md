# AMORA

Automated Microarchitecture Observation and Reverse-engineering for
Accelerators. The NVIDIA CUDA backend provides 36 registered diagnostic probes,
with a GCoM CUDA simulator adapter and a TPU JAX backend for generation-aware
discovery, compiled portable workloads, and capability-gated measurements.
Runtime and hardware qualification are explicit; unavailable tools or devices
produce structured reasons.

## Install

```bash
pip install -e '.[test]'
```

The console entrypoint is `amora`. The package can also be invoked with
`python -m amora` (no install required as long as the repo root is on
`PYTHONPATH`).

## CLI

The CLI is structured as `amora <backend> <command>`. The `nvidia`, `gcom_cuda`,
and `tpu` backends are wired up.

```bash
amora nvidia list                     # registry of probes (implemented vs. planned)
amora nvidia inspect-capabilities     # discovered toolchain and GPUs
amora nvidia run --probe topology.device_attributes
amora nvidia run --all --output report.json
```

`run` requires either `--probe <id>` or `--all`. Without `--output` the
JSON report is written to stdout. Each report has a `metadata`
section with the shared backend capability snapshot and a `results`
list whose `tool_context` entries reference that snapshot rather than
duplicating it.

## Probes

| probe_id | status |
| --- | --- |
| `topology.device_attributes` | implemented; runtime capability gated |
| `topology.occupancy` | implemented; runtime capability gated |
| `topology.persistent_cta` | implemented; runtime capability gated |
| `arithmetic_latency.dependent_chain` | implemented; runtime capability gated |
| `arithmetic_throughput.independent_chains` | implemented; runtime capability gated |
| `shared_memory.pointer_chase` | implemented; runtime capability gated |
| `shared_memory.bank_stride` | implemented; runtime capability gated |
| `shared_memory.analyze` | implemented; runtime capability gated |
| `l1_cache.pointer_chase` | implemented; runtime capability gated |
| `l1_cache.working_set` | implemented; runtime capability gated |
| `l1_cache.conflict_sets` | implemented; runtime capability gated |
| `l1_cache.analyze` | implemented; runtime capability gated |
| `scheduler_policy.ready_warps` | implemented; runtime capability gated |
| `scheduler_policy.mixed_issue` | implemented; runtime capability gated |
| `scheduler_policy.analyze` | implemented; runtime capability gated |
| `register_file.register_bank_sweep` | implemented; runtime capability gated |
| `register_file.register_latency` | implemented; runtime capability gated |
| `register_file.analyze` | implemented; runtime capability gated |
| `synchronization.barrier_latency` | implemented; runtime capability gated |
| `global_memory.streaming` | implemented; runtime capability gated |
| `l2_cache.pointer_chase` | implemented; runtime capability gated |
| `memory_pipeline.outstanding_requests` | implemented; runtime capability gated |
| `memory_pipeline.lane_patterns` | implemented; runtime capability gated |
| `memory_pipeline.analyze` | implemented; runtime capability gated |
| `global_memory.partition_sweep` | implemented; runtime capability gated |
| `global_memory.row_policy_sweep` | implemented; runtime capability gated |
| `global_memory.analyze` | implemented; runtime capability gated |
| `tensor_core.mma_latency` | implemented; runtime capability gated |
| `tensor_core.mma_throughput` | implemented; runtime capability gated |
| `synchronization.fence_latency` | implemented; runtime capability gated |
| `tma_copy.async_copy_latency` | implemented; runtime capability gated |
| `tma_copy.tma_transfer_sweep` | implemented; runtime capability gated |
| `tma_copy.analyze` | implemented; runtime capability gated |
| `interconnect.injection_rate` | implemented; runtime capability gated |
| `interconnect.address_mapping` | implemented; runtime capability gated |
| `interconnect.analyze` | implemented; runtime capability gated |

## TPU backend

Generation profiles cover v2/v3/v4/v5e/v5p/v6e/tpu7x; each hardware/runtime
combination needs its own qualification. Listing probes is framework-independent:

```bash
python -m amora tpu list
pip install -e '.[portable-core,test,tpu]'
amora tpu inspect-capabilities
amora tpu measure --recipe docs/development/tpu-single-device.json \
  --out-root out/tpu --run-id baseline-001
```

Measurement workers require exclusive accelerator ownership and never fall back
to CPU. They retain compiler IR, correctness, raw synchronized host timing,
source/runtime hashes and immutable manifests. Optional Pallas DMA/memory and
local-mesh collective recipes have independent feature gates. See the
[TPU operator guide](docs/development/tpu-backend.md) and
[execution tracker](.plan/trackers/20261116-1659-tpu-backend.md) for exact evidence
and qualification status. The NVIDIA/GCoM CUDA-template benchmark runner remains
separate from portable TPU execution.

## Tests

```bash
pytest               # full suite
pytest -m "not cuda" # skip GPU-gated tests
```

The `cuda`, `ncu`, and `nvbit` markers gate CUDA hardware and external tools.
TPU hardware markers (`tpu`, `tpu_pallas`, `tpu_profile`) require explicit
`AMORA_TEST_TPU=1` on an exclusive TPU host. CPU tests do not initialize libtpu.

## Benchmarks

Benchmark definitions are separate from the 36 diagnostic probes. AMORA hosts
both the NVIDIA-focused `ppp_canonical` evidence pipeline and portable
mathematical workload definitions.

### PPP canonical

The AMORA-owned `ppp_canonical` generator materializes deterministic
kernel-and-shape case sets without requiring a GPU:

```bash
amora benchmarks list
amora benchmarks inspect ppp_canonical
amora benchmarks materialize ppp_canonical --preset h100_2500
amora benchmarks materialize ppp_canonical --cases 5600 --seed 20260717
amora benchmarks classify ppp_canonical --manifest out/benchmarks/.../manifest.json
amora benchmarks detail ppp_canonical \
  --manifest out/benchmarks/.../manifest.json \
  --classification out/benchmarks/.../classification.json \
  --size-rank small --run-id h100-small-01
amora benchmarks review ppp_canonical \
  --manifest out/benchmarks/.../manifest.json \
  --classification out/benchmarks/.../classification.json \
  --comparison out/benchmarks/.../runs/small/.../h100-small-01/comparison.json \
  --semantic-decision "OD2 scalar accuracy remains deferred"
amora benchmarks detail ppp_canonical \
  --manifest out/benchmarks/.../manifest.json \
  --classification out/benchmarks/.../classification.json \
  --size-rank medium --review-marker out/benchmarks/.../reviews/small/.../review.json \
  --run-id h100-medium-01
```

Materialization writes an immutable manifest under `out/benchmarks/` by
default. Classification collects basic NCU instruction/cycle/duration metrics
and writes a digest-keyed immutable rank overlay. Detailed NCU collection and
GCoM execution run in persisted rank order: `small`, then `medium`, then
`large`. A complete comparison has to be explicitly accepted with
`amora benchmarks review` before the next rank is allowed. Each detail
invocation writes an immutable `runs/<rank>/<classification-digest>/<run-id>/`
directory containing raw hardware/GCoM JSONL, the comparison, and `run.json`;
existing evidence is never overwritten.

### Portable core workloads

`benchmark_generators.portable_core` provides the versioned core20, core24, and
core28 mathematical workload tiers. It includes deterministic inputs,
independent oracles, and NumPy, PyTorch, and optional JAX execution backends:

```bash
pip install -e '.[portable-core,test]'
amora-portable-core list --tier 28
amora-portable-core freeze --tier 24 --out out/portable_core/core24
amora-portable-core check \
  --frozen out/portable_core/core24 \
  --out out/portable_core/core24-check \
  --backends numpy,torch,jax
```

The same CLI is available as `python -m benchmark_generators.portable_core`.
Frozen manifests and runtime sources are hash-bound and are never overwritten.
PyTorch, JAX, and accelerator runtimes must be installed separately when those
backends are needed.

Portable-core checks establish mathematical correctness only. They do not
qualify native lowering, accelerator instructions, compiler fusion, or
performance, and the package is intentionally separate from the existing
CUDA-template classification/detail runner. See the
[portable-core documentation](benchmark_generators/portable_core/README.md) for
the API, tier contents, provenance, licensing, and scope details.

## Layout

- `amora/cli.py` — argparse CLI
- `amora/benchmarking/` — benchmark materialization contracts
- `benchmark_generators/` — canonical PPP and portable-core workload definitions
- `amora/backends/nvidia/` — toolchain discovery, build adapters, CUPTI/NCU/NVBit hooks
- `amora/probes/nvidia/baseline/` — the baseline probe family
- `amora/reports/json_report.py` — JSON renderer
- `amora/schemas/` — `ProbeResult` and evidence enums
- `.plan/` — design notes and methodology
- `docs/` — vendor research and developer guides
