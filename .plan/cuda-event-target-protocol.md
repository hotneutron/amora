# CUDA-event and device-interval target protocol

AMORA launches each command as an argv sequence without a shell. The target must emit
one compact JSON object on its final non-empty stdout line. Diagnostic logging may
precede that line.

## CUDA-event timing

The unprofiled lane must emit:

```json
{
  "schema_version": 1,
  "kind": "cuda_event_timing",
  "timing": {
    "unit": "us_per_launch",
    "samples": [21.31, 21.29],
    "warmup_launches": 20,
    "launches_per_sample": 100,
    "cache_protocol": "warm_reuse",
    "clock_policy": "application_clocks"
  },
  "device": {
    "uuid": "GPU-...",
    "name": "NVIDIA H100 80GB HBM3",
    "sm_clock_mhz": "1980"
  },
  "subject_identity": {
    "kernel_name": "...",
    "ttgir_sha256": "...",
    "ptx_sha256": "...",
    "sass_sha256": "...",
    "cubin_sha256": "..."
  },
  "subject_metadata": {
    "registers_per_thread": 64,
    "shared_memory_bytes": 32768,
    "spill_count": 0
  },
  "measurement_axes": {
    "cta_load": 132,
    "independent_register_work_gap_ns": 16
  },
  "tool_versions": {
    "python": "3.10.16",
    "torch": "2.1.0",
    "triton": "3.7.0",
    "cuda_runtime": "12.1"
  }
}
```

Valid cache protocols are `warm_reuse`, `disjoint_rotation`, and `cold_flush`.
The target receives `AMORA_MEASUREMENT_LANE=cuda_event_timing`,
`AMORA_PROCESS_REPEAT=<zero-based index>`, and, when the recipe declares one,
`AMORA_CACHE_PROTOCOL=<protocol>`. The same cache-protocol signal is passed to the
aggregate and SourceCounters target processes so all lanes can reproduce the declared
semantic cache state; NCU's separate `--cache-control` option remains independently
recorded.
`measurement_axes` must exactly equal the frozen recipe point so runtime-only shape,
load, queue, and cache interventions cannot silently drift across evidence lanes.
`tool_versions` is required and must contain non-empty strings; it is also checked
across timing, interval, aggregate, and SourceCounters lanes.

## Device intervals

An optional independently launched instrumented target must emit:

```json
{
  "schema_version": 1,
  "kind": "cuda_device_intervals",
  "clock": "globaltimer_ns",
  "intervals": [
    {"name": "wgmma_completion", "start_ns": 1000, "end_ns": 1120, "duration_ns": 120}
  ],
  "instrumentation": {
    "unprofiled_event_overhead_percent": 1.4,
    "overhead_threshold_percent": 2.0,
    "unprofiled_event_slope_change_percent": 0.5,
    "slope_threshold_percent": 2.0,
    "sass_bracketing_verified": true,
    "instruction_sequence_unchanged": true
  },
  "device": {"uuid": "GPU-...", "name": "NVIDIA H100 80GB HBM3"},
  "subject_identity": {
    "ttgir_sha256": "...",
    "ptx_sha256": "...",
    "sass_sha256": "...",
    "cubin_sha256": "..."
  },
  "subject_metadata": {
    "registers_per_thread": 64,
    "shared_memory_bytes": 32768,
    "spill_count": 0
  },
  "measurement_axes": {
    "cta_load": 132,
    "independent_register_work_gap_ns": 16
  },
  "tool_versions": {
    "python": "3.10.16",
    "torch": "2.1.0",
    "triton": "3.7.0",
    "cuda_runtime": "12.1"
  }
}
```

The target receives `AMORA_MEASUREMENT_LANE=device_intervals`. Invalid core schema
fields are rejected. Unverified bracketing, instruction drift, non-monotonic intervals,
or excess timing/slope overhead retain the evidence but mark it `diagnostic`.

## Recipe execution

Run a validated JSON recipe with:

```text
amora nvidia measure \
  --recipe path/to/recipe.json \
  --run-id unique-run-id \
  --out-root out/measurements/nvidia \
  --cwd path/to/target/repository \
  --env CUDA_VISIBLE_DEVICES=0
```

Every point is executed in a deterministic randomized order. The output directory is
created exclusively and contains a pre-execution `recipe.json`, one
`points/<point-id>/bundle.json`, one `mediation.json`, and a content-digested
`manifest.json`. Reusing a run ID fails rather than overwriting evidence.

Each recipe declares `mechanism` as `wgmma_fixed_completion` or
`tma_route_offered_load`, a calibration/held-out split chosen before execution, and
controlled `interaction_groups`. Every requested physical axis must be the sole varied
axis in at least one group. Set `requires_same_subject_identity` to `false` only when
the intervention necessarily recompiles the kernel; runtime interventions default to
requiring identical TTGIR/PTX/SASS/cubin identity.

Each point also declares `evidence_bindings`: dotted paths from its bundle, axes, or
subject metadata into physical causal nodes. A path may select a named interval, for
example:

```json
{
  "wgmma_completion": [
    "bundle.device_intervals.intervals.name=wgmma_completion.duration_ns"
  ],
  "memory_route": [
    "bundle.aggregate_counters.metrics.dram__bytes_read.sum"
  ],
  "total_latency": [
    "bundle.timing.median_us"
  ]
}
```

When SourceCounters are enabled, at least one causal node must bind the PC-sampling
evidence. A point is diagnostic if its PC records do not all join independently to a
target-declared SASS or cubin artifact. `subject_artifacts.sass` and
`subject_artifacts.cubin` accept either a path string or `{"path": ...,
"sha256": ...}`. Relative paths resolve against `--cwd`.

The recipe validator requires all five WGMMA axes and all eight TMA axes to have
controlled groups. WGMMA recipes must collect `wait`, `math_pipe_throttle`, `barrier`,
and `warpgroup_arrive`; TMA recipes must include route/traffic and occupancy/launch-wave
counters. The mediation output reports measured co-change, falsified links, and missing
links. It emits no direct regression on elapsed duration, K, or working-set size.

The WGMMA knee test binds a distinct `wgmma_exposed_wait` interval and evaluates
`independent_register_work_gap_ns + exposed_wait_ns` for stability. The separately
bound `wgmma_completion` node is used for CTA-load sensitivity.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r2 | 2026-08-31 18:57 -0700 | Added runtime axes, target tool versions, frozen recipe semantics, direct PC-to-SASS evidence, complete controlled-axis requirements, and physical mediation outputs. |
| r1 | 2026-08-31 17:46 -0700 | Documented the target protocols, instrumentation downgrade contract, environment signals, and immutable recipe CLI. |
