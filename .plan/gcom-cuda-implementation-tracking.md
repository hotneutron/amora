# Tracking: `gcom_cuda` Backend Implementation

Living progress tracker for implementing the `gcom_cuda` backend. The
authoritative spec is [gcom-cuda-probe-sim-vs-hw-comparison.md](./gcom-cuda-probe-sim-vs-hw-comparison.md);
this file tracks status, decisions, and verification against that plan. Update it
as work lands.

- Branch: `gcom_cuda`
- Plan commit: `f0d68e0` (Generalize GCoM CUDA comparison plan)
- Status legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked

## Environment Snapshot (fill in at first run)

| item | value | source |
|---|---|---|
| GCOM_ROOT | `${GCOM_ROOT}` (default `~/wk/modern-gpu-simulator-micro-2025/simulator-remodeled`) | env / config.py |
| sim binary built? | **yes** (rebuilt 2026-06-30 against system protobuf 3.21) | `build.py::ensure_sim_built` |
| NVBit tracer built? | **yes** (built 2026-06-30, NVBit v1.7.5, tracer_tool.so) | `build.py::ensure_tracer_built` |
| nvcc available? | **yes** — `/usr/local/cuda-12.8/bin/nvcc` (set CUDA_INSTALL_PATH + PATH) | capability discovery |
| real GPU available? | **yes** — 8× NVIDIA H100 80GB HBM3, driver 595.71.05 | nvidia-smi |
| H100 gpgpusim.config present? | **yes** | SKU profile |
| HW baseline JSON | `/tmp/amora_full.json` (H100 nvidia run); regenerate with `amora nvidia run --all` | nvidia backend |

### Build environment (CRITICAL — protobuf version match)

GCoM's generated `*.pb.*` and the simulator link **system protobuf 3.21.12**
(`/usr/bin/protoc`, `libprotobuf.so.32`). A conda env on this host
(`torch21`) ships protobuf 28.2; if its headers leak via `CPATH` or its
`protoc` is first on `PATH`, the tracer/sim build fails (mismatched
`PROTOBUF_NAMESPACE_*` / `runtime_version.h`). Build with:

```bash
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH
export CUDA_INSTALL_PATH=/usr/local/cuda-12.8
export PATH=/usr/bin:$CUDA_INSTALL_PATH/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/sbin:/bin
```

The prebuilt `accel-sim.out` (Apr 17) was stale vs. its own H100 config
(rejected `-memory_shared_memory_extra_latency_stsm_multiple_matrix`); rebuilding
the simulator from source resolved it. No GCoM source/config was edited.

## Open Decisions (from plan §Open Decisions)

- [ ] SKU ID spelling for H100 — default `gcom_h100`; profile may add display name + config hashes.
- [ ] `sim-vs-hw-gcom_h100.md`: link to HW reference report vs. embed a small HW summary table.
- [ ] Anchor pass/fail thresholds: global default vs. per-probe metadata.
- [ ] Minimum GCoM stat schema required before the first smoke comparison.

## Phase 1 — Backend Foundation & Version Metadata

Goal: no-GPU-importable skeleton + capability discovery + version/archival, one smoke probe path.

- [x] `amora/backends/gcom_cuda/__init__.py` (package marker)
- [x] `config.py` — `GCOM_ROOT`, `SIM_BIN`, SKU profile loader (family/sku/hardware_reference/config paths), `OUT_ROOT=out/gcom_cuda/`, clocks; env overrides; **no hardcoded H100 in APIs**
- [x] `gcom.py` — `GcomCapabilities(backend="gcom_cuda", ...)` + `discover_capabilities()` (tracer/sim/nvcc/GPU/config/SKU status); reuse nvidia `_discover_devices`
- [x] `build.py` — `ensure_tracer_built()`, `ensure_sim_built()` (idempotent)
- [x] version contract (`version.py`): AMORA+GCoM commit/dirty/branch, sim binary path+mtime, CUDA/driver versions, GPU model, SKU profile, config paths+hashes, env → run metadata
- [~] raw archival under `out/gcom_cuda/<family>/<sku>/<run_id>/` — paths defined (`config.run_output_dir`); wired fully when trace+sim execution lands
- [x] CLI: `gcom_cuda` subparser (`inspect-capabilities`, `list`, `run`, `compare`) via shared `_cmd_*` refactor (behavior-preserving for nvidia)
- [~] minimal `runner.simulate` + `parse_stats` (done) + one smoke probe — execution path GPU/tracer-gated, returns structured `missing_stat` until tracer built
- [x] **Verify**: `amora gcom_cuda inspect-capabilities` reports status; `pytest -m "not cuda"` passes (48)

## Phase 2 — Probe-Level Comparison

Goal: cycle-derived comparison driven by the canonical probe inventory + `compare`.

- [x] `probes/gcom_cuda/baseline/__init__.py` — `PLANNED_PROBES = tuple(nvidia.baseline.PROBES)` (single source of truth; never re-list)
- [x] `metrics_map.py` — policy-only table (category, derivation kind, required GCoM stat keys, required HW denominator fields, fidelity, architecture_scope, limitations) + load-time drift assertion
- [x] drift unit test: `set(metrics_map) == set(nvidia.baseline.PROBES)` (no GPU)
- [~] generic runner factory: structure + HW-denominator lookup + state handling done; the `gpu_sim_cycle`-based numeric derivation lands with the trace+sim execution path
- [x] `--hw-baseline` flag; comparable probes without it → `missing_stat` (no hardcoded counts)
- [x] `compare.py` — `load_backend_report`, probe-level `compare`, preserve unsupported/proxy/missing states
- [x] **Verify**: `run --all` returns canonical inventory (36) with comparable/approximate/unavailable rows; `compare` writes md+json

## Phase 3 — Stat Schema & Coverage Reporting

- [ ] full GCoM stat parse + schema snapshot (`SUPPORTED_GCOM_STAT_SCHEMA`, `MAPPING_VERSION`)
- [ ] per-run: missing required / missing optional / new unmapped / deprecated stats; fail only on missing core stat (`gpu_sim_cycle`)
- [ ] grouped metric coverage report using the existing report taxonomy (Compute & Scheduling / Register, Tensor & Sync / On-chip Memory / Global Memory & DRAM / Transfer & Interconnect)
- [ ] **Verify**: coverage section present in compare output, grouped per `reports/nvidia/SUMMARY.md` taxonomy

## Phase 4 — Expanded Metric Mapping (GCoM ↔ NCU)

- [ ] `gcom_metrics_map.py` — logical metric → GCoM stat(s) + derivation → nearest NCU metric, fidelity, architecture_scope, limitations (versioned)
- [ ] `runner.derive_logical_metrics(stats)`
- [ ] counter-level comparison in `compare` (separate from probe rows; proxy never upgrades a probe scalar)
- [ ] fold richer comment-doc groups into existing report groups (e.g. warp scheduling → Compute & Scheduling; async copy/TMA → Transfer & Interconnect)
- [ ] **Verify**: counter section diffs sim-derived vs real NCU, fidelity-tagged

## Phase 5 — Accuracy Model

- [ ] per-row `fidelity` / `model_confidence` / `known_limitations` / `expected_error_band` / `calibration_status` / `architecture_scope`
- [ ] validation anchors: flag the 9 anchor probe IDs from the plan; report passed/failed/unavailable + overall reliability
- [ ] accuracy summary section in reports
- [ ] **Verify**: anchor summary + confidence annotations present

## Report Outputs (target tree)

- [ ] `reports/gcom_cuda/SUMMARY.md`
- [ ] `reports/gcom_cuda/hopper/probes-gcom_h100.md`
- [ ] `reports/gcom_cuda/hopper/sim-vs-hw-gcom_h100.{md,json}`
- [ ] `.gitignore`: `out/gcom_cuda/`

## Definition of Done (plan §Verification)

- [ ] inspect-capabilities reports tracer/sim/compiler/GPU/config/SKU status
- [ ] version metadata + config hashes in every JSON report
- [ ] raw sim outputs archived under `out/gcom_cuda/<family>/<sku>/`
- [ ] single-probe run yields finite sim value when sim+trace available
- [ ] `run --all` returns canonical inventory (comparable/approximate/unavailable)
- [ ] `compare` writes Markdown+JSON under `reports/gcom_cuda/hopper/`
- [ ] compare report has: probe-level + counter-level comparison, grouped coverage, anchor summary, version metadata, unavailable-state breakdown
- [ ] non-CUDA tests pass; simulator-dependent tests gated

## Log

- 2026-06-26 — Tracking doc created against plan `f0d68e0`. No code yet.
- 2026-06-26 — Phase 1+2 scaffold landed: backend package (config/gcom/build/trace/runner/version/compare),
  probe registry (inventory derived from nvidia; metrics_map policy table + drift guard),
  gcom_metrics_map, CLI subparser + shared-handler refactor, 9 no-GPU tests. `pytest -m "not cuda"`
  = 48 passed. CLI verified: inspect-capabilities, `run --all` (36 results, structured states:
  not_applicable=14/missing_stat=16/unsupported=5/proxy_only=1), and `compare` writes md+json.
  Categories: comparable=10, approximate=10, unavailable=16 (lane_patterns classed proxy_only, not
  approximate — counter proxy, not a probe scalar). Numeric trace+sim derivation pending GPU+tracer.
- 2026-06-30 (cont.) - Wired the real trace->simulate->derive path into the runner factory: per_op /
  throughput / bandwidth derivations (bandwidth from sim DRAM bytes), per-run archival under
  out/gcom_cuda/<family>/<sku>/<run_id>/, GCoM-derived logical counters attached for the
  counter-comparison layer. Grounded the metrics_map HW-denominator field names against the real
  report (chain, chain_length, barriers, chase_len, steps; mma_throughput -> proxy; bandwidth probes
  use sim DRAM bytes). Sweep/differential probes return honest missing_stat (multi-trace reduction is
  a later phase). Fixed the .cu source path (parents[2]) and absolute -trace/config paths. CLI smoke
  verified end-to-end: tier=simulator_trace, finite cycles_per_fma, logical counters. pytest=49.
- 2026-06-30 (partial run) - Kicked off full `gcom_cuda run --all --hw-baseline` on H100. Cycle-accurate
  sim is very slow on latency-bound probes (shared/L1/L2 pointer-chase: 20-90+ min each), so produced
  a PARTIAL sim-vs-hw report from completed probes: shared_memory.pointer_chase sim=34.3 vs HW=29.0
  (18% - LDS latency tracks well); FFMA latency/throughput are far off (152%/884%) due to launch-shape
  mismatch in the naive single-trace per_op derivation (surfaced, not hidden; anchors reliable=False).
  Fixed compare markdown to collapse composite/behavioral HW values to "(composite)". Full run still
  executing in background. Report: reports/gcom_cuda/hopper/sim-vs-hw-gcom_h100.{md,json}.
