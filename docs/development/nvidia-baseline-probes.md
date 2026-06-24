# NVIDIA Baseline Probe Development

## Current State

The NVIDIA probe suite now implements the full baseline (P0) plus P1 probe
families. All 18 probes are registered and kernel-backed where the methodology
requires hardware timing; CPU-only probes (metadata/planning) and cross-probe
analyzers run without a kernel.

Implemented building blocks:

- package and CLI scaffold (`amora <backend> <command>`),
- layered result schemas (`ProbeResult` with four evidence layers),
- JSON report rendering and a per-vendor Markdown report generator,
- NVIDIA capability discovery (`nvcc`, `nvidia-smi`, `ncu`, `nvdisasm`,
  `cuobjdump`),
- a shared CUDA build + launch helper (`amora/backends/nvidia/runner.py`) that
  compiles each probe's `.cu` driver and parses its JSON stdout,
- registered CUDA source hashes for every kernel-bound probe,
- non-hardware unit tests plus opt-in CUDA smoke tests.

Each kernel-bound probe ships a `.cu` (device kernel + host driver that prints a
single JSON line) and a `.py` wrapper that maps the payload into the four-layer
result with a methodology-faithful fit status. On hosts without CUDA the probe
returns a structured `unsupported` result that still registers its source hash.

## Probe Families

### P0 — baseline

| probe_id | mode | notes |
| --- | --- | --- |
| `topology.device_attributes` | metadata | nvidia-smi device identity |
| `topology.occupancy` | planning | launch-shape cross-product, no kernel |
| `topology.persistent_cta` | kernel | resident CTAs/SM via `%smid` + busy-spin |
| `arithmetic_latency.dependent_chain` | kernel | FP32 FMA dependent latency |
| `arithmetic_throughput.independent_chains` | kernel | FP32 FMA ILP throughput |
| `shared_memory.pointer_chase` | kernel | shared-load latency |
| `shared_memory.bank_stride` | kernel | bank-conflict stride sweep |
| `shared_memory.analyze` | analysis | merges pointer-chase + bank-stride |

### P1 — caches, scheduler, register file

| probe_id | mode | notes |
| --- | --- | --- |
| `l1_cache.pointer_chase` | kernel | L1-hit latency with a DRAM control |
| `l1_cache.working_set` | kernel | capacity knee from a working-set sweep |
| `l1_cache.conflict_sets` | kernel | effective associativity (bounded) |
| `l1_cache.analyze` | analysis | merged L1 cache summary |
| `scheduler_policy.ready_warps` | kernel | issue-scaling saturation point |
| `scheduler_policy.mixed_issue` | kernel | FP32/INT pipe-overlap class |
| `scheduler_policy.analyze` | analysis | scheduler behavioral summary |
| `register_file.register_bank_sweep` | kernel | operand-width plateau (candidate) |
| `register_file.register_latency` | kernel | RAW-distance differential latency |
| `register_file.analyze` | analysis | register-file summary |

Per the P1 methodology, several probes intentionally report bounded,
conditional, or behavioral fit statuses rather than exact scalars (capacity,
associativity, scheduler policy, register-bank count, operand-delivery cost).

## Commands

The console entrypoint is `amora`; `python -m amora` works without an install as
long as the repo root is on `PYTHONPATH`.

List NVIDIA probes (implemented vs. planned):

```bash
amora nvidia list
```

Inspect NVIDIA backend capabilities (toolchain + GPUs):

```bash
amora nvidia inspect-capabilities
```

Run all probes and write a JSON report:

```bash
amora nvidia run --all --output out/nvidia-baseline.json
```

Run a single probe:

```bash
amora nvidia run --probe l1_cache.pointer_chase
```

Render Markdown reports from a JSON run:

```bash
amora nvidia report --input out/nvidia-baseline.json --out-dir reports
```

`run` requires either `--probe <id>` or `--all`; without `--output` the JSON is
written to stdout.

### Build configuration

The CUDA runner honors two environment variables:

- `AMORA_NVCC_ARCH` — target SASS arch (default `sm_80`; use `sm_90` for H100).
- `AMORA_BUILD_ROOT` — cache directory for compiled drivers
  (default `~/.cache/amora/build/nvidia/baseline`).

Build artifacts are cached per source SHA-256, so repeated runs skip
recompilation. `nvcc` must be on `PATH`; e.g.:

```bash
export PATH=/usr/local/cuda-12.8/bin:$PATH
export AMORA_NVCC_ARCH=sm_90
amora nvidia run --all --output out/nvidia-baseline.json
```

## Output Model

Every result preserves four layers:

- `raw_observation` — evidence tier, raw values, metrics, units,
- `normalized_measurement` — hardware-neutral value with fit status and
  uncertainty,
- `backend_interpretation` — NVIDIA-specific concept and downgrade reason,
- `simulator_estimate` — simulator-facing parameter through a mapping contract.

Unsupported probes still emit all four layers, so reports stay stable on
machines without CUDA and missing-tool failures never become silent scalar
estimates.

## Reports

`amora nvidia report` renders an organized tree from a JSON run:

```
reports/
  README.md                          # vendor index
  nvidia/
    SUMMARY.md                       # cross-SKU trend tables, grouped by probe group
    <family>/                        # e.g. hopper
      README.md                      # per-SKU outcome tables
      manifest.json                  # run metadata, keyed by SKU
      environment.md                 # toolchain + devices
      probes-<sku>.md                # all probes for one SKU in one file
```

Probe grouping in the summary is pluggable per vendor and lives in
`amora/reports/probe_groups.py`.

## Tests

```bash
pytest               # full suite
pytest -m "not cuda" # skip GPU-gated tests
```

The `cuda`, `ncu`, and `nvbit` markers are declared in `pyproject.toml` and gate
tests that require a real CUDA-capable host or external tools. The registry test
exercises every probe without a GPU (kernel-bound probes return `unsupported`
with a registered source hash; analyzers degrade cleanly).

## Layout

- `amora/cli.py` — argparse CLI (`list`, `inspect-capabilities`, `run`, `report`)
- `amora/backends/nvidia/` — toolchain discovery, build/launch runner, NCU/NVBit/disasm hooks
- `amora/probes/nvidia/baseline/` — P0 + P1 probe families (`topology`,
  `arithmetic_*`, `shared_memory`, `l1_cache`, `scheduler_policy`,
  `register_file`)
- `amora/reports/json_report.py` — JSON renderer
- `amora/reports/markdown_report.py` — Markdown report tree generator
- `amora/reports/probe_groups.py` — pluggable per-vendor probe grouping
- `amora/schemas/` — `ProbeResult` and evidence enums
- `.plan/` — design notes and methodology (P0–P3)
- `docs/` — vendor research and developer guides

## Next Implementation Steps

1. Add SASS validation with `nvdisasm`/`cuobjdump` so opcode/dependency
   assumptions are enforced rather than assumed.
2. Integrate NCU/CUPTI metric discovery and the metric resolver so counter
   evidence can corroborate or replace timing for probes that support it.
3. Implement the P2 and P3 probe families (memory pipeline, L2, DRAM, tensor
   core, synchronization, TMA/async copy, interconnect).
4. Replace CUDA approximations (e.g. the register-bank sweep proxy) with
   SASS-controlled variants where toolchain support allows.
