# NVIDIA NCU / CUPTI Integration Plan

## Scope

Add Nsight Compute (NCU) counter collection as a first-class evidence source for
NVIDIA probes, so that counter metrics can act as **primary** evidence where the
metric contract is direct (e.g. shared-memory transactions/conflicts, issued
instructions, DRAM bytes) and as **validation** evidence otherwise. This is the
highest design-risk item on the roadmap and a prerequisite for most P2/P3
probes.

## Methodology anchor

From the kernel methodologies: counters are *primary only when the metric
contract has direct semantics; otherwise they validate or constrain fits.*
Timing must never be collected in the same pass as NCU (replay perturbs it).
Every counter-derived value carries a metric-resolver record.

## Existing building blocks

- `amora/backends/nvidia/ncu.py` — `NcuCommand` (argv builder with
  `--target-processes all`, `--metrics`, `--export`) and `list_metrics`
  (`ncu --query-metrics`).
- `amora/backends/nvidia/metrics.py` — `MetricResolver` mapping logical names
  (`sm_active_cycles`, `inst_executed`, `shared_transactions`,
  `shared_conflicts`) to ordered candidate counter names, with availability
  resolution and a structured `MetricResolution` record.
- `amora/backends/nvidia/runner.py` — build + launch + JSON parse for the host
  driver executables.
- capability discovery already detects `ncu`.

## Key design decisions

### 1. Separate profiler mode, never alongside timing

NCU replays each kernel many times to collect counters, which destroys timing
fidelity. Therefore:

- timing runs and counter runs are **distinct executions of the same binary**;
- the existing `run_kernel` (timing) is unchanged;
- a new `run_kernel_profiled(source, metrics=...)` builds the same cached binary
  and launches it under `ncu`, parsing counter output instead of (or in addition
  to) the driver's JSON.

Probes that want counter evidence call both and keep the two payloads in
separate evidence layers/values; they are never averaged together.

### 2. Output format: `--csv --page raw` (not `.ncu-rep`)

`--export` produces a binary `.ncu-rep` requiring `ncu-ui`/`ncu --import` to
read. Instead collect machine-readable CSV:

```
ncu --target-processes all --csv --page raw \
    --metrics <resolved,counter,names> --launch-count 1 \
    ./probe_driver <args>
```

Parse the CSV (stdlib `csv`) keyed by `("Kernel Name", "Metric Name")`. Extend
`NcuCommand` with `csv: bool` and `page: str` and a `--launch-count`/
`--kernel-name` filter. This avoids a binary-format dependency and is stable
across CUDA versions.

### 3. Metric resolution drives the request

Probes declare *logical* metric names; the resolver maps them to the first
supported candidate using the host's `--query-metrics` set (captured once at
capability-discovery time and threaded through `NvidiaCapabilities`). The
collection request uses only resolved, available counters; unresolved logicals
are recorded as `available=false` with a reason and never silently dropped.
Expand `MetricResolver.CANDIDATES` as probes need new logicals (DRAM bytes, L2
hit rate, tensor-pipe utilization, issue/eligible-warp metrics, etc.).

### 4. Primary-vs-validation policy per probe

A per-probe declaration states, for each logical metric, whether it is
`primary` or `validation` for that probe's target:

- **Primary** (direct contract): the counter value populates
  `raw_observation` with `evidence_tier=direct_counter` and may set the
  normalized scalar. Examples: `shared_conflicts` for `shared_memory.bank_stride`,
  `inst_executed`/issue metrics for throughput, DRAM bytes for streaming.
- **Validation**: the counter only corroborates a timing-derived value; on
  disagreement beyond a tolerance the probe **downgrades** (lowers fit_status,
  sets `downgrade_reason`) rather than overriding the scalar.

This keeps the existing fit-status discipline intact and prevents counters from
manufacturing false precision.

## Permissions and environment

- NCU needs perf-counter access; on locked-down hosts it fails with a
  permissions error. Treat this as a clean capability gate: discovery records
  `ncu_counters_available=false` with the error, and counter-dependent probes
  fall back to timing-only with a recorded reason (never crash).
- Honor `AMORA_NCU` / PATH discovery like `nvcc`. Document that counter
  collection may require `--target-processes all` and elevated `perf_event`
  settings.

## Data model

- New `amora/backends/nvidia/ncu_run.py`: `NcuResult(metrics: dict[str,float],
  raw_rows, returncode, stderr)` and `run_kernel_profiled(...)`.
- New evidence on results: `raw_observation.metrics` may carry counter values
  (units from the resolver); `backend_interpretation.metric_resolver` records
  the `MetricResolution` list (field already exists on the schema).
- `evidence_tier=direct_counter` / `tool_derived_counter` used where
  appropriate (enums already exist).

## Implementation steps

1. Capture supported metric set during capability discovery
   (`ncu --query-metrics`), store on `NvidiaCapabilities`, expose to probes.
2. Extend `NcuCommand` for `--csv --page raw --launch-count --kernel-name`; add
   a CSV parser in `ncu_run.py`.
3. Implement `run_kernel_profiled()` in `ncu_run.py` (reuse the cached binary
   from `runner.build_executable`).
4. Add a per-probe metric declaration (logical names + primary/validation role).
5. Wire one pilot probe end-to-end first: `shared_memory.bank_stride` with
   `shared_conflicts` as primary — validates the whole path on real hardware.
6. Roll out to the remaining counter-friendly P0/P1 probes (throughput, issue
   scaling), then make counters available for P2/P3.
7. Render metric-resolver + counter evidence in the Markdown report.

## Tests

- Unit: CSV parser against committed `ncu --csv --page raw` fixtures (no GPU).
- Unit: `MetricResolver` fallback/expansion (already partially covered).
- Unit: primary/validation gating with synthetic counter values
  (agreement -> pass, disagreement -> downgrade, missing -> timing fallback).
- CUDA+NCU-gated (`-m "cuda and ncu"`): pilot probe collects `shared_conflicts`
  and agrees with the stride sweep.

## Risks

- Metric name churn across architectures/driver versions — mitigated by the
  candidate-list resolver and capturing the supported set at runtime.
- Replay sensitivity for stateful probes (caches warmed differently under
  replay) — restrict primary-counter use to metrics whose semantics are
  replay-stable; keep latency/state probes timing-primary.
- Permission failures in CI/cloud — capability-gated, never fatal.
- Long runtimes under NCU — keep `--launch-count` minimal and scope
  `--kernel-name` to the probe kernel.

## Dependencies / sequencing

- Independent of SASS validation (can proceed in parallel), but pairs well with
  it: SASS confirms the instruction stream, NCU confirms the counter behavior.
- **Blocks most P2/P3 probes**, which are counter-heavy (see the P2/P3
  sequencing plan).
