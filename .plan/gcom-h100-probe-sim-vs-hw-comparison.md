# Plan: Add a `gcom_cuda` Backend to AMORA — Simulated-H100 Probes vs Real-H100

## Summary

Model **GCoM** as a first-class AMORA backend, `gcom_cuda`, sitting beside the existing `nvidia`
backend. GCoM is the working-directory simulator (the MICRO-2025 Accel-Sim derivative at
[simulator-remodeled](file:///home/cliu/wk/modern-gpu-simulator-micro-2025/simulator-remodeled);
"GCoM" is the name accorde's [build_gcom.sh](file:///home/cliu/wk/accorde/scripts/build_gcom.sh)
gives it). Because GCoM only consumes NVIDIA CUDA SASS traces, a CUDA-specific backend
`gcom_cuda` is the right abstraction.

The `gcom_cuda` backend reuses the **same probe `.cu` kernels** and emits **the same `ProbeResult`
schema** as the `nvidia` backend — but each probe's value comes from *simulating* the kernel on an
H100 config rather than running it on real hardware. Comparison against the real H100 then becomes a
natural diff of two backend reports (`nvidia` = real, `gcom_cuda` = simulated), plus a dedicated
`compare` command.

User decisions locked in: **real H100 available now** (collect fresh NVBit traces), **cycle-level
metrics only** (NCU/address-mapping/metadata probes → `unsupported`), output **in the amora repo**.

## Why a backend (vs. a standalone module)

AMORA's architecture already supports this cleanly (verified):
- CLI is `amora <backend> <command>` ([cli.py](file:///home/cliu/wk/amora/amora/cli.py)).
- Each backend has `backends/<name>/` (capability discovery) + `probes/<name>/baseline/` (a `PROBES`
  registry exposing `list_probes()`, `run_probe()`, `run_all()`, `PLANNED_PROBES`).
- Probe `run()` functions return `list[ProbeResult]`; the nvidia probes already populate a
  `simulator_param` field — exactly the quantity a simulator should reproduce.
- Reports are backend-agnostic ([json_report.py](file:///home/cliu/wk/amora/amora/reports/json_report.py),
  [markdown_report.py](file:///home/cliu/wk/amora/amora/reports/markdown_report.py)).

So `gcom_cuda` slots in as a peer backend, maximizing reuse and making the comparison symmetric.

## Current State (grounded)

- **Backend contract** ([cuda.py](file:///home/cliu/wk/amora/amora/backends/nvidia/cuda.py)):
  `discover_capabilities() -> Capabilities` with `.to_dict()`; `backend` field string
  (`"nvidia_cuda"`). The `gcom_cuda` analogue reports tool availability of the *simulator + tracer*
  instead of nvcc/nvidia-smi/ncu.
- **Probe registry** ([baseline/__init__.py](file:///home/cliu/wk/amora/amora/probes/nvidia/baseline/__init__.py)):
  `PROBES: dict[str, ProbeRunner]`, `run_probe`, `run_all`, `list_probes`. The `gcom_cuda` baseline
  mirrors the **same 36 probe_ids**.
- **Probe sources**: 30 of 36 probe_ids have a `.cu` kernel under
  [baseline/](file:///home/cliu/wk/amora/amora/probes/nvidia/baseline); the 6 `*.analyze` (and
  `topology.device_attributes`/`topology.occupancy`) are analysis/metadata-only with no kernel.
- **GCoM H100 config already exists** (no new config):
  [gpgpusim.config](file:///home/cliu/wk/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config)
  (`compute_capability 9.0`, `n_clusters 132`, `clock_domains 1800:1800:1800:8000`,
  `shmem_num_banks 32`, `ptx_opcode_latency_fp 4,4,4,4,39`, `ptx_opcode_latency_tesnor 64`) +
  [trace.config](file:///home/cliu/wk/modern-gpu-simulator-micro-2025/simulator-remodeled/gpu-simulator/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/trace.config).
- **Nothing built yet**: no `accel-sim.out`, no NVBit tracer `.so`.
- **Stat parsing precedent**: accorde's
  [run_gcom_ground_truth.sh](file:///home/cliu/wk/accorde/scripts/run_gcom_ground_truth.sh)
  greps `gpu_sim_cycle`, `gpu_tot_sim_insn`, `gpu_ipc`.
- **Real-H100 baseline** to compare against:
  [probes-h100-80g.md](file:///home/cliu/wk/amora/reports/nvidia/hopper/probes-h100-80g.md)
  (per-probe `measurement`, `simulator_param`, `metrics`, raw op counts, SASS histograms).

### Core conceptual constraint
Probes self-time with **clock64()** and read **NCU counters**; GCoM does not replay clock64 as
timing — it computes `gpu_sim_cycle`. So the `gcom_cuda` backend derives each probe's value from
`gpu_sim_cycle` and the probe's known op count (per-op cycles) or byte count (bandwidth), using the
**same denominators** the HW probe used (op counts live in the HW result's raw values). Probes whose
signal is NCU-counter-/address-mapping-/metadata-based have no trace-driven equivalent → emitted as
`ProbeResult.unsupported` with a stated reason.

## Multi-trace sweep (promotes the 11 "approximate" → "comparable")

The single-trace MVP can only place **one point** on a probe's curve, so sweep/knee/plateau/ratio
probes are "approximate". The fix needs **no simulator change**: trace the *same sweep the HW probe
runs* — one trace per launch configuration — and have the `gcom_cuda` runner reconstruct the curve
from the simulated points exactly as the HW probe does.

### Design
- `trace.py` gains `trace_probe_sweep(probe_id, src, variants) -> list[trace_dir]`, where `variants`
  is a list of `(label, compile_defines|argv)` taken from the probe's sweep spec. Each variant is its
  own compile (when the sweep is a `#define`, e.g. `-DAMORA_WORKING_SET_KIB=N`) or its own argv (when
  the kernel takes a runtime arg), then its own trace.
- `metrics_map.py` entries for sweep probes carry a `sweep` spec: the parameter name, the list of
  points, whether it varies by `define` or `argv`, and a `reducer` (`knee` / `plateau` / `ratio` /
  `min_max`) that turns the per-point simulated values into the probe's scalar — the **same reducer
  the HW probe's analyzer uses**, so HW and sim are computed identically.
- The generic runner: for a sweep probe, run all variant traces, derive a per-point value
  (`gpu_sim_cycle / op_count` etc., op_count per point from the HW result), apply the reducer, and
  emit one `ProbeResult` whose `simulator_param` is the reduced scalar (knee/plateau/ratio).
- Multi-kernel/baseline probes (`register_file.register_latency`, `synchronization.fence_latency`)
  are the degenerate 2-variant case: trace both sub-kernels, then subtract/difference.

### Promotions (11 → comparable), with residual caveats
`l1_cache.working_set`, `scheduler_policy.ready_warps`, `register_file.register_bank_sweep`,
`memory_pipeline.outstanding_requests`, `global_memory.row_policy_sweep`,
`tma_copy.tma_transfer_sweep` (sweeps); `register_file.register_latency`,
`synchronization.fence_latency` (multi-kernel); `topology.persistent_cta` (residency stat).
Residual caveats kept in the note column, not hidden:
- `register_bank_sweep` plateau is bounded by the sim's register-bank model (config = 8 banks);
- `tma_copy.*` stay semantically caveated (LDGSTS modeled, native TMA not);
- any sweep whose variants the tracer cannot actually build/trace auto-downgrades to `unsupported`
  with a recorded reason (never a fabricated point).

This raises the clean comparable count from **10 → ~21**. The remaining unsupported set is the one
that genuinely needs simulator features AMORA may not modify (see the GCoM↔NCU mapping for what the
sim *does* expose).

## GCoM internal-stats ↔ NCU/profiler metric mapping

GCoM dumps a **much larger** internal stat set than NCU exposes (cycles, occupancy, per-level cache
accesses/misses, DRAM commands/bytes, interconnect/latency, register-bank conflicts, and a detailed
issue-stage stall taxonomy). Several probes AMORA marked `unsupported` *for NCU reasons* therefore
have a **simulator-side equivalent** — so we add a `gcom_metrics_map.py` translating GCoM stat keys
into AMORA's existing logical metric names (`amora/backends/nvidia/metrics.py::MetricResolver`), and
`compare` can diff sim-derived counters against the real NCU counters the `nvidia` backend already
records in each result's `metric_resolver` / `raw_observation.metrics`.

> Grounding: the GCoM stat keys below were extracted from the simulator source at
> `gpu-simulator/gpgpu-sim/src` (printed stat strings). Exact key spelling is re-confirmed against a
> real `accel-sim.out` run during the smoke test (open item); the resolver tolerates absent keys.

### Mapping table (AMORA logical metric → GCoM stat → NCU counter it mirrors)

| AMORA logical (metrics.py) | GCoM stat key(s) | NCU counter (HW side) | mapping note / fidelity |
|---|---|---|---|
| `sm_active_cycles` | `gpu_sim_cycle` / `gpu_tot_sim_cycle` | `sm__cycles_active.avg` | sim core cycles — direct |
| `inst_executed` | `gpu_tot_sim_insn`, `gpgpu_n_tot_w_icount` | `smsp__inst_executed.sum` | thread vs warp granularity noted — direct |
| `global_load_requests` | `total_dl1_accesses` (LD portion) | `l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` | L1 access count ≈ requests — proportional |
| `global_load_sectors` | `total_dl1_accesses` × sectors/access, `sector_mask_count` | `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` | sectors/request derivable → unblocks `lane_patterns` as counter-comparable — proportional |
| `shared_conflicts` | `gpu_reg_bank_conflict_stalls`, `total_accesses_per_shared_instruction` | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | sim bank-conflict stalls; units differ — proxy |
| `dram_bytes_read` | `gpgpu_n_dram_reads` × DRAM atom (32 B) | `dram__bytes_read.sum` | reads×atom — proportional |
| `dram_bytes_write` | `gpgpu_n_dram_writes` × atom | `dram__bytes_write.sum` | writes×atom — proportional |
| `dram_throughput` | `bwutil`, `DRAM_BW_total` | `dram__throughput.avg.pct_of_peak_sustained_elapsed` | BW utilization % ≈ pct-of-peak — proportional |
| `l2_sector_hits` | `L2_total_cache_accesses` − `L2_total_cache_misses` | `lts__t_sectors_lookup_hit.sum` | sim L2 hit count — proportional |
| (L1 hit rate, extra) | `1 − total_dl1_miss_rate` | `l1tex__t_sector_hit_rate.pct` | extra — proportional |
| (L2 hit rate, extra) | `1 − L2_total_cache_miss_rate` | derive from sectors | extra — proportional |
| `tensor_pipe_active` | (tensor-pipe inst count if exposed) | `sm__pipe_tensor_cycles_active...` | sim may not separate tensor pipe → may stay unsupported |
| `stall_long_scoreboard` | `..._waiting_scoreboard`, `per_cyc_sche_stall_war_scoreboard_dependencies` | `..._stalled_long_scoreboard...pct` | scoreboard ≈ mem-dependency stall — proxy |
| `stall_barrier` | `..._waiting_inst_barrier`, `..._waiting_wait_barrier` | `..._stalled_barrier...pct` | barrier-wait — proxy |
| `stall_mio_throttle` / `stall_lg_throttle` | `..._issue_port_busy`, `..._with_fu_occupied`, `..._mem_dispatch_full_and_ldst_unit_stalled` | `..._stalled_*_throttle...pct` | pipe/port-busy ≈ throttle — proxy |
| `stall_not_selected` | `..._stall_no_warps_ready`, `per_cyc_sched_stall_idle` | `..._stalled_not_selected...pct` | no-ready-warp ≈ not-selected — proxy |
| `stall_wait` | `per_cyc_sched_stall_dependencies` | `..._stalled_wait...pct` | dependency wait — proxy |
| (interconnect latency, extra) | `avg_icnt2mem_latency`, `avg_mrq_latency`, `averagemflatency` | (no direct NCU logical) | sim-only extra for memory_pipeline |
| (occupancy, extra) | `gpu_occupancy`, `gpu_tot_sms_occupancy` | `sm__warps_active.avg.pct...` | achieved occupancy — proportional |

### How the mapping is used
1. `gcom_metrics_map.py`: `GCOM_TO_LOGICAL: dict[str, GcomDerivation]` — each entry names the GCoM
   stat key(s), a small derivation (identity / `×atom_bytes` / `1−rate` / `sum_of`), the AMORA logical
   name, and a `fidelity` tag (`direct` / `proportional` / `proxy`).
2. `runner.parse_stats` already keeps all `=`-delimited keys; a new `derive_logical_metrics(stats)`
   produces `{logical_name: value}` using the map (absent keys skipped).
3. `compare` gains a **counter section**: where the `nvidia` result recorded an NCU value
   (`metric_resolver`/`metrics`) *and* GCoM exposes the analogue, emit a row
   `{logical, hw_ncu, sim_gcom, fidelity, pct_error}`, reported separately from the cycle-level probe
   comparison and labelled by `fidelity` (proxy rows are informational).
4. Effect on categories: counter-only probes gain a **secondary comparison** even when their probe
   value stays `unsupported` — e.g. `memory_pipeline.lane_patterns` (sectors/request) and the
   scheduler stall metrics become *counter-comparable* via the sim, surfacing GCoM's richer internals
   against NCU without faking the probe's own scalar.

> Constraint respected: reads GCoM's existing printed stats only — **no simulator edits**. Where GCoM
> lacks an analogue (true address-mapping/partition-camping, native TMA), the row is omitted with a
> reason rather than approximated.

## Proposed Changes

New code mirrors the existing backend layout. Only one existing file is edited (`cli.py` wiring).

### New package `amora/backends/gcom_cuda/`

1. `__init__.py` — package marker.
2. `config.py` — paths + constants:
   - `GCOM_ROOT` (env override; default the working-dir simulator path), `SIM_BIN`
     (`gpu-simulator/bin/release/accel-sim.out`), `H100_GPGPUSIM_CONFIG`, `H100_TRACE_CONFIG`,
     `TRACER_DIR` (`util/tracer_nvbit`), `OUT_ROOT` (`out/gcom_cuda/`, gitignored), `NVCC_ARCH=sm_90`,
     `CORE_CLOCK_HZ=1800e6`, `DRAM_CLOCK_HZ=8000e6`, `HW_BOOST_CLOCK_HZ=1830e6`.
3. `gcom.py` — capability discovery (peer of nvidia `cuda.py`):
   - `@dataclass GcomCapabilities(backend="gcom_cuda", ...)` with `.to_dict()` reporting:
     tracer built?, simulator built?, nvcc available?, GPU available? (tracing needs a real GPU),
     H100 config present?, `unsupported_reasons`.
   - `discover_capabilities() -> GcomCapabilities`: check `SIM_BIN.exists()`, tracer `.so` presence,
     `shutil.which("nvcc")`, nvidia-smi device list (reuse nvidia `_discover_devices` via import),
     config file existence.
4. `build.py` — one-time build helpers:
   - `ensure_tracer_built()`: run `util/tracer_nvbit/install_nvbit.sh` + `make -C util/tracer_nvbit/`
     (cwd=GCOM_ROOT, env CUDA_INSTALL_PATH/PATH). Idempotent.
   - `ensure_sim_built()`: `bash -c "source gpu-simulator/setup_environment_no_git.sh &&
     make -j$(nproc) -C gpu-simulator/"` (env IS_SERT=0). Returns `SIM_BIN`.
5. `trace.py` — produce a trace per probe kernel (needs real GPU):
   - `compile_probe(src) -> Path`: `nvcc -arch sm_90 -std=c++14 -O2 <src> -o <bin>` (static).
   - `trace_probe(probe_id, src) -> Path`: run the compiled binary under the NVBit tracer (replicate
     `util/tracer_nvbit/run_hw_trace.py`'s LD_PRELOAD + env mechanism — verify exact env var names
     by reading that script at execution time); output to `OUT_ROOT/traces/<probe_id>/traces/...`;
     assert `dynamic_trace.pb` exists.
6. `runner.py` — run GCoM + parse stats:
   - `simulate(probe_id, trace_dir) -> dict`: `SIM_BIN -trace <pb> -config <gpgpusim> -config
     <trace.config>`, env LD_LIBRARY_PATH (sim lib + CUDA lib64) and OMP tuning (8/close/cores per
     accorde), tee to `gcom_sim.log`.
   - `parse_stats(stdout) -> dict`: last value per key matching
     `^\s*([A-Za-z0-9_:\.\[\]]+)\s*=\s*([-\d.eE+]+)`; keep `gpu_sim_cycle`, `gpu_tot_sim_cycle`,
     `gpu_tot_sim_insn`, `gpu_ipc`, and any `*l1*`/`*l2*`/`*dram*`/`*hit_rate*`.

### New package `amora/probes/gcom_cuda/baseline/`

> **Single source of truth (decision).** The set of probe IDs lives in exactly one place —
> `amora/probes/nvidia/baseline.PROBES` — and `gcom_cuda` **derives** from it. We do *not* re-list the
> 36 IDs in the new backend. Likewise, each probe's op/byte **denominators**, `concept`, and `unit`
> are read from the real-HW `ProbeResult` at derive/compare time, not hardcoded a second time. Only
> the genuinely-new knowledge — each probe's **category** and **sim-derivation policy** — is declared
> in `metrics_map.py`. See "Single source of truth" below.

7. `__init__.py` — registry derived from nvidia's (no second inventory):
   - `PLANNED_PROBES = tuple(amora.probes.nvidia.baseline.PROBES)` — derived, never hand-listed.
   - `PROBES = {pid: _make_runner(pid) for pid in PLANNED_PROBES}` via a single generic runner factory.
   - `list_probes()`, `run_probe(probe_id, caps)`, `run_all(caps)` — same signatures so the CLI is
     identical to nvidia's.
   - Each runner: ensure trace+sim for the kernel, derive the value using the policy from
     `metrics_map.py` and the denominator from the HW result, wrap in a `ProbeResult` whose
     `simulator_param` is the simulated value and whose `concept`/`unit` match the nvidia probe
     (so reports align 1:1), or `ProbeResult.unsupported(...)` with a stated reason.
8. `metrics_map.py` — declarative table of **policy only** (category + sim-derivation), keyed by the
   probe IDs from the nvidia registry. It carries `category`
   (`comparable`/`approximate`/`unsupported`), a `derivation` kind (`per_op`/`bandwidth`/`throughput`),
   and the **name of the HW raw-value key** to use as the denominator (e.g. `"chain_length"`) — not
   the literal count. A module-load assertion enforces it covers exactly the nvidia IDs:
   ```python
   assert set(METRICS_MAP) == set(NVIDIA_PROBES), (
       f"gcom_cuda metrics_map drifted from nvidia registry: "
       f"missing={set(NVIDIA_PROBES) - set(METRICS_MAP)}, "
       f"extra={set(METRICS_MAP) - set(NVIDIA_PROBES)}")
   ```
   A unit test asserts the same so drift fails CI without a GPU.

### `amora/backends/gcom_cuda/compare.py`
- `load_backend_report(json_path) -> {probe_id: result}` for both the `nvidia` (real) and
  `gcom_cuda` (sim) JSON reports (produced by `amora <backend> run --all --output ...`, schema from
  [json_report.py](file:///home/cliu/wk/amora/amora/reports/json_report.py)).
- `compare(real, sim, table) -> rows` with `{probe_id, category, concept, unit, hw_value, sim_value,
  abs_error, pct_error, note}`; `unsupported` rows carry `sim_value=None` + reason.
- `write_outputs`: `reports/nvidia/hopper/sim_vs_hw/comparison.{json,md}` — table + per-category
  rollup (mean/median |pct_error| over `comparable`) + clock-delta note (sim 1800 vs HW 1830 MHz).

### EDIT `amora/cli.py` (only change to existing code)
Add a `gcom_cuda` backend subparser mirroring the `nvidia` one, sourcing
`discover_capabilities`, `baseline.list_probes/run_all/run_probe` from the new packages, plus a
`compare` command:
```python
gcom = subparsers.add_parser("gcom_cuda")
gcom_sub = gcom.add_subparsers(dest="command")
# list / inspect-capabilities / run (--probe|--all, --output) -> mirror nvidia handlers,
#   but import from amora.backends.gcom_cuda.gcom and amora.probes.gcom_cuda.baseline
# compare --real <nvidia.json> --sim <gcom.json> --out-dir reports/nvidia/hopper/sim_vs_hw
```
Refactor the three nvidia `_cmd_*` helpers to accept injected
`(discover_capabilities, baseline)` so both backends share them (small, local refactor — no behavior
change for `nvidia`).

> **`run` and HW denominators.** Because denominators are read from the real-HW result (single source
> of truth), `gcom_cuda run` takes an optional `--hw-baseline real_h100.json`. With it, `comparable`
> probes derive numeric values; without it they emit `ProbeResult.unsupported("HW baseline required
> for denominator")` rather than hardcoding a count. `compare` always has both reports, so the
> derivation there is unconditional.

### `pyproject.toml`
Ensure the new packages are included by the package finder (verify current `packages`/`find`
config; add `amora.backends.gcom_cuda*` / `amora.probes.gcom_cuda*` if explicit lists are used).

### Full 36-probe mapping table (category + sim derivation), grounded in probes-h100-80g.md

> This table is the **policy reference** behind `metrics_map.py`. In code the probe IDs are derived
> from the nvidia registry (not copied) and the literal denominators below (chain_length=4096,
> steps=4096, …) are shown for grounding only — at runtime they are read from the HW result's raw
> values, so the table encodes only *which* raw key and *which* derivation kind to use.

| probe_id | category | HW metric (value) | sim derivation | note |
|---|---|---|---|---|
| topology.device_attributes | unsupported | identity metadata | — | metadata, not a sim output |
| topology.occupancy | unsupported | planning sweep | — | planning artifact, no kernel |
| topology.persistent_cta | approximate | peak_resident_blocks_per_sm=8 | sim residency stat / max_resident_ctas | report if sim exposes it, else unsupported |
| arithmetic_latency.dependent_chain | comparable | cycles_per_fma=4.377 | gpu_sim_cycle / chain_length(4096) | dependent FFMA; sim FP lat cfg=4 |
| arithmetic_throughput.independent_chains | comparable | cycles_per_fma_per_thread=1.1471 | gpu_sim_cycle / (chain_length*indep_chains) | ILP-saturated FP32 |
| shared_memory.pointer_chase | comparable | cycles_per_load=29.0146 | gpu_sim_cycle / chase_len(4096) | LDS dep latency |
| shared_memory.bank_stride | unsupported | bank_count=32 | — | bank count is a sim *input* (shmem_num_banks 32), NCU-validated |
| shared_memory.analyze | unsupported | summary | — | analysis-only |
| l1_cache.pointer_chase | comparable | l1_hit_cycles_per_load=70.6091 | gpu_sim_cycle / steps(4096) | L1-hit dep load |
| l1_cache.working_set | approximate | capacity knee | per-op cycles, single point | knee needs sweep → likely unsupported at runtime |
| l1_cache.conflict_sets | unsupported | associativity | — | knee not identifiable from one trace |
| l1_cache.analyze | unsupported | summary | — | analysis-only |
| scheduler_policy.ready_warps | approximate | saturation_warps=16 | ops/cycle (single point) | sweep needed → may downgrade |
| scheduler_policy.mixed_issue | unsupported | overlap class | — | behavioral, NCU-coupled |
| scheduler_policy.analyze | unsupported | summary | — | analysis-only |
| register_file.register_bank_sweep | approximate | plateau=16 | per-op cycles vs width | sim reg banks cfg=8 |
| register_file.register_latency | approximate | differential=2.3606 | diff of two cycles_per_op | needs two sub-kernels |
| register_file.analyze | unsupported | summary | — | analysis-only |
| synchronization.barrier_latency | comparable | cycles_per_barrier=45.2683 | gpu_sim_cycle / barriers(4096) | BAR latency |
| global_memory.streaming | comparable | peak_gbps=3135.34 | dram_bytes / (gpu_sim_cycle/CORE_CLOCK) | sim DRAM bytes + core clock |
| l2_cache.pointer_chase | comparable | l2_hit_cycles_per_load=329.901 | gpu_sim_cycle / steps(4096) | L2-hit dep load |
| memory_pipeline.outstanding_requests | approximate | eff_outstanding=4 | bytes/cycle (single point) | sweep needed → may downgrade |
| memory_pipeline.lane_patterns | unsupported | sectors/request=32 (NCU) | — | NCU counter only |
| memory_pipeline.analyze | unsupported | summary | — | analysis-only |
| global_memory.partition_sweep | unsupported | camping class | — | partition behavior not comparable |
| global_memory.row_policy_sweep | approximate | row_locality=1.755 | best/worst bandwidth across strides | needs sweep traces |
| global_memory.analyze | unsupported | summary | — | analysis-only |
| tensor_core.mma_latency | comparable | cycles_per_mma=24.4648 | gpu_sim_cycle / chain(512) | HMMA dep; sim tensor lat=64 |
| tensor_core.mma_throughput | comparable | mma_per_cycle_per_warp=0.1599 | HMMA_count / gpu_sim_cycle | independent HMMA |
| synchronization.fence_latency | approximate | net_cycles_per_fence=928.961 | gpu_sim_cycle/fences − empty baseline | fence semantics differ |
| tma_copy.async_copy_latency | approximate | cycles_per_tile=723.234 | gpu_sim_cycle / tiles(64) | LDGSTS modeled; TMA not |
| tma_copy.tma_transfer_sweep | approximate | peak_gbps=37.48 | bandwidth per tile size | cp.async modeled |
| tma_copy.analyze | unsupported | summary | — | analysis-only |
| interconnect.injection_rate | comparable | saturation_gbps=3075 | dram_bytes / (gpu_sim_cycle/CORE_CLOCK) | aggregate injection |
| interconnect.address_mapping | unsupported | mapping class | — | not comparable |
| interconnect.analyze | unsupported | summary | — | analysis-only |

Counts: comparable=10, approximate=11, unsupported=15 (=36). Approximate probes that need
multi-launch sweeps the single trace cannot provide auto-downgrade to `unsupported` at runtime with a
recorded reason.

## Resulting UX

```bash
amora gcom_cuda inspect-capabilities          # tracer/sim/nvcc/GPU/config status
amora gcom_cuda list                          # same 36 probe_ids, sim-implemented vs unsupported
amora nvidia   run --all --output real_h100.json   # real HW (existing)
amora gcom_cuda run --all --output sim_h100.json   # simulated H100 via GCoM
amora gcom_cuda compare --real real_h100.json --sim sim_h100.json \
      --out-dir reports/nvidia/hopper/sim_vs_hw
```

## Assumptions & Decisions

1. `gcom_cuda` is a peer backend emitting the standard `ProbeResult` schema; its `simulator_param` is
   the simulated value, `concept` matches the nvidia probe so reports/compare align 1:1.
2. GCoM = working-dir simulator; use existing H100 config as-is (no ~300-param retuning; errors are
   reported, not calibrated away).
3. Trace arch `sm_90`; cycle-level only; 15 probes `unsupported`, 10 comparable, 11 approximate.
4. Sim values derived per-op from `gpu_sim_cycle` with HW-matched denominators (no clock64 replay).
5. One generic runner factory (table-driven) instead of 36 hand-written probe files, to avoid bloat.
6. Only existing-code edit: `cli.py` backend wiring + small shared-handler refactor; no
   simulator/probe-kernel edits.
7. `out/gcom_cuda/` gitignored; committed deliverable under `reports/nvidia/hopper/sim_vs_hw/`.
8. **Single source of truth (supersedes "copy the list" in the handoff).** The 36 probe IDs are
   derived from `amora.probes.nvidia.baseline.PROBES`; `gcom_cuda` never re-lists them.
   Denominators/`concept`/`unit` come from the real-HW `ProbeResult` at derive/compare time.
   `metrics_map.py` declares only category + derivation policy, keyed by those IDs, with a load-time
   assertion + unit test that it covers exactly the nvidia set (drift fails fast, no GPU needed).
   Consequence: `gcom_cuda run` takes `--hw-baseline`; comparable probes without it emit
   `unsupported("HW baseline required for denominator")`.

## Verification

1. `amora gcom_cuda inspect-capabilities` reports tracer+sim build status accurately.
2. Build: tracer `.so` + `accel-sim.out` produced.
3. Smoke: `amora gcom_cuda run --probe arithmetic_latency.dependent_chain` yields a `ProbeResult`
   with a finite simulated `cycles_per_fma`; trace + `gcom_sim.log` with nonzero `gpu_sim_cycle`.
4. Full: `amora gcom_cuda run --all` returns 36 results (10 with numeric sim values, 11
   approximate/possibly-downgraded, 15 unsupported).
5. `compare` produces `reports/nvidia/hopper/sim_vs_hw/comparison.md` with table + per-category
   accuracy rollup; magnitudes sanity-checked (FMA latency near HW 4.377 cyc band; streaming
   bandwidth same order as ~3.1 TB/s) with divergences reported.
6. `pytest -m "not cuda"` in amora still passes (new backend is additive and sim/CUDA-gated; the
   nvidia handler refactor is behavior-preserving).
