# NVIDIA Baseline Probe Development

## Current State

The NVIDIA probe suite implements the full P0 + P1 + P2 + P3 probe set — 36
probes — folded into the `baseline/` tree. All kernel-bound probes are validated
with opcode-level SASS checks, and probes with a direct counter contract collect
NCU counters in a separate profiler pass.

Implemented building blocks:

- package and CLI scaffold (`amora <backend> <command>`),
- layered result schemas (`ProbeResult` with four evidence layers),
- JSON report rendering and a per-vendor Markdown report generator,
- NVIDIA capability discovery (`nvcc`, `nvidia-smi`, `ncu`, `nvdisasm`,
  `cuobjdump`) including the supported NCU metric set,
- a shared CUDA build + launch helper (`amora/backends/nvidia/runner.py`) that
  compiles each probe's `.cu` driver, parses its JSON stdout, and optionally
  validates the kernel's SASS,
- SASS validation (`amora/backends/nvidia/sass.py`): per-probe opcode
  expectations with reject / downgrade / pass gating,
- NCU counter collection (`amora/backends/nvidia/ncu_run.py`): separate
  profiler pass, CSV parsing, metric-resolver-driven requests,
- registered CUDA source + disassembly hashes for every kernel-bound probe,
- non-hardware unit tests plus opt-in CUDA/NCU smoke tests.

Each kernel-bound probe ships a `.cu` (device kernel + host driver that prints a
single JSON line) and a `.py` wrapper that maps the payload into the four-layer
result with a methodology-faithful fit status. On hosts without CUDA the probe
returns a structured `unsupported` result that still registers its source hash.

## Probe Families (36 probes)

- **P0 baseline** — `topology` (device_attributes, occupancy, persistent_cta),
  `arithmetic_latency.dependent_chain`,
  `arithmetic_throughput.independent_chains`,
  `shared_memory` (pointer_chase, bank_stride, analyze).
- **P1 caches / scheduler / register file** — `l1_cache` (pointer_chase,
  working_set, conflict_sets, analyze), `scheduler_policy` (ready_warps,
  mixed_issue, analyze), `register_file` (register_bank_sweep, register_latency,
  analyze).
- **P2 memory / tensor / sync** — `memory_pipeline` (lane_patterns,
  outstanding_requests, analyze), `l2_cache.pointer_chase`, `global_memory`
  (streaming, partition_sweep, row_policy_sweep, analyze), `tensor_core`
  (mma_latency, mma_throughput), `synchronization` (barrier_latency,
  fence_latency).
- **P3 transfer / interconnect** — `tma_copy` (async_copy_latency,
  tma_transfer_sweep, analyze), `interconnect` (injection_rate, address_mapping,
  analyze).

Per the P1–P3 methodologies, many probes intentionally report bounded,
conditional, or behavioral fit statuses rather than exact scalars (cache
capacity/associativity, scheduler policy, register-bank count, partition/row
mapping, async-copy/interconnect behavior).

### Validation

- **SASS**: every kernel-bound probe declares a `SassExpectation` (required and
  forbidden opcodes, optional dependency) matched to its kernel — e.g. FFMA for
  arithmetic, LDS for shared memory, LDG for L1/global, HMMA for tensor core,
  LDGSTS for async copy, BAR/MEMBAR for sync. A missing required opcode or a
  forbidden opcode rejects the measurement; a low count or unconfirmed
  dependency downgrades the fit one notch.
- **NCU**: probes with a direct counter contract (e.g.
  `memory_pipeline.lane_patterns` for sectors-per-request,
  `shared_memory.bank_stride` for shared conflicts) collect counters in a
  separate profiler pass as primary or validation evidence; missing/locked-down
  NCU degrades cleanly to timing-only.

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

## Published Facts & Capability Gating

`amora/backends/nvidia/archinfo.py` holds a curated, per-architecture
published-facts table (compute capability, SM count, L2 size, memory bandwidth,
shared-memory per SM, and feature flags such as `tensor_core` / `async_copy` /
`tma` / `fp8`), keyed by device-name patterns. It serves two purposes:

- **Trust-and-verify anchors**: `topology.device_attributes` attaches the
  matching published facts so runtime metadata can be cross-checked against
  known specs.
- **Capability gating**: `feature_gate(...)` lets architecture-specific probes
  return a clean `unsupported` result on hardware that lacks a feature (e.g. the
  `tma_copy.*` async-copy probes gate out on pre-Ampere parts) with the
  compute-capability reason recorded. Unknown devices are *allowed* (the probe
  falls back to its own evidence) rather than mis-gated.

## Next Implementation Steps

1. Extend the report SUMMARY with cross-SKU trend deltas once a second
   architecture (beyond H100/V100) is profiled.
2. Wire simulator traces as the target side of the mapping contracts so
   bounded/behavioral P3 fits (partition, address mapping) can be promoted to
   uniquely-identified.
3. Query compute capability directly from the CUDA runtime (rather than
   inferring it from the device name) to harden gating on unrecognized SKUs.
4. Broaden the published-facts table as new architectures (Blackwell Ultra,
   Rubin) are validated.
