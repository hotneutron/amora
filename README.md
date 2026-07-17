# AMORA

Automated Microarchitecture Observation and Reverse-engineering for
Accelerators. The current implementation is the NVIDIA CUDA *baseline cutline*:
one fully implemented topology probe plus seven planned probes that emit
well-formed `unsupported` results until their kernels and analyzers land.

## Install

```bash
pip install -e '.[test]'
```

The console entrypoint is `amora`. The package can also be invoked with
`python -m amora` (no install required as long as the repo root is on
`PYTHONPATH`).

## CLI

The CLI is structured as `amora <backend> <command>`. Today only the
`nvidia` backend is wired up.

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
| `topology.device_attributes` | implemented (direct nvidia-smi metadata) |
| `topology.occupancy` | implemented (planning sweep, no kernel) |
| `topology.persistent_cta` | planned (CUDA source registered) |
| `arithmetic_latency.dependent_chain` | planned (CUDA source registered) |
| `arithmetic_throughput.independent_chains` | planned (CUDA source registered) |
| `shared_memory.pointer_chase` | planned (CUDA source registered) |
| `shared_memory.bank_stride` | planned (CUDA source registered) |
| `shared_memory.analyze` | planned (needs pointer-chase + bank-stride inputs) |

## Tests

```bash
pytest               # full suite
pytest -m "not cuda" # skip GPU-gated tests
```

The `cuda`, `ncu`, and `nvbit` markers are declared in `pyproject.toml`
and gate tests that require a real CUDA-capable host or external tools.

## Benchmarks

Benchmark definitions are separate from the 36 diagnostic probes. The initial
AMORA-owned `ppp_canonical` generator materializes deterministic
kernel-and-shape case sets without requiring a GPU:

```bash
amora benchmarks list
amora benchmarks inspect ppp_canonical
amora benchmarks materialize ppp_canonical --preset h100_2500
amora benchmarks materialize ppp_canonical --cases 5600 --seed 20260717
amora benchmarks classify ppp_canonical --manifest out/benchmarks/.../manifest.json
```

Materialization writes an immutable manifest under `out/benchmarks/` by
default. Classification collects basic NCU instruction/cycle/duration metrics
and writes a digest-keyed immutable rank overlay. Detailed NCU collection and
GCoM execution are subsequent benchmark phases.

## Layout

- `amora/cli.py` — argparse CLI
- `amora/benchmarking/` — benchmark materialization contracts
- `benchmark_generators/` — local generated benchmark definitions
- `amora/backends/nvidia/` — toolchain discovery, build adapters, CUPTI/NCU/NVBit hooks
- `amora/probes/nvidia/baseline/` — the baseline probe family
- `amora/reports/json_report.py` — JSON renderer
- `amora/schemas/` — `ProbeResult` and evidence enums
- `.plan/` — design notes and methodology
- `docs/` — vendor research and developer guides
