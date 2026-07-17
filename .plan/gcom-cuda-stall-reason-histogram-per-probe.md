# Plan: `gcom_cuda` Stall-Reason Histogram Per Probe

## Summary

The `tma-impl` branch in `~/wk/modern-gpu-simulator-micro-2025` added an
NCU-aligned stall taxonomy to GCoM. AMORA should consume that taxonomy and attach
a stall-reason histogram to **every `gcom_cuda` probe result**.

This belongs in AMORA, not the simulator repo, because the user-visible object is
an AMORA `ProbeResult` and the report/compare pipeline in `~/wk/amora` decides
what evidence each probe carries. The simulator remains the source of raw GCoM
stats; AMORA parses and normalizes them.

## Source Contract

Treat the simulator's `ncu_stall_*` lines as the input contract. The current
GCoM-side taxonomy plan is:

- `~/wk/modern-gpu-simulator-micro-2025/.plan/NCU_STALL_TAXONOMY_METRICS_IMPL.md`

Expected reason set, in fixed order:

```text
selected
not_selected
dispatch_stall
warpgroup_arrive
long_scoreboard
short_scoreboard
barrier
wait
mio_throttle
math_pipe_throttle
mma
no_instructions
imc_miss
sleeping
branch_resolving
membar
drain
lg_throttle
tex_throttle
misc
```

Expected raw lines in simulator stdout:

```text
ncu_stall_<reason> = <count>
ncu_stall_<reason>_pct = <percentage>
total_num_cycles_issue_stage_evaluated = <denominator>
```

`selected` is not a stall, but it is part of the same NCU warp-state view. Keep
it in the histogram so AMORA can compare the full NCU-shaped stack. Consumers
that need a stall-only vector can drop `selected`.

## Output Contract

Every successful `gcom_cuda` `ProbeResult` should include:

```python
raw_observation.metrics["gcom_stall_reason_histogram"] = {
    "selected": {"count": ..., "pct": ...},
    "not_selected": {"count": ..., "pct": ...},
    ...
}
raw_observation.metrics["gcom_stall_reason_denominator"] = ...
raw_observation.metrics["gcom_stall_reason_schema"] = "ncu-stall-v1"
```

For unavailable/unsupported probes, omit the histogram and keep the existing
structured state (`missing_stat`, `unsupported`, `not_applicable`, etc.).

The histogram is probe-level because each AMORA probe runs a separate
trace/simulation, so the simulator's kernel-level `ncu_stall_*` stack is already
the histogram for that probe run.

## Phase 1 - Parser Support

Add a small parser layer in `amora/backends/gcom_cuda/runner.py`.

1. Define the canonical reason tuple:

   ```python
   STALL_REASON_SCHEMA = "ncu-stall-v1"
   STALL_REASON_KEYS = (
       "selected",
       "not_selected",
       ...
       "misc",
   )
   ```

2. Keep `parse_stats(stdout)` as the low-level numeric parser. It already accepts
   `ncu_stall_<reason>` and `ncu_stall_<reason>_pct` because `_STAT_LINE`
   permits underscores.

3. Add:

   ```python
   def extract_stall_reason_histogram(stats: dict[str, float]) -> dict[str, Any] | None:
       ...
   ```

   Return `None` only when no `ncu_stall_*` keys are present. If some are present
   but the reason set is incomplete, return a structured object with:

   - `schema`
   - `complete: False`
   - `missing_reasons`
   - any parsed reason entries

   Do not silently treat missing reasons as zero. The simulator is supposed to
   emit explicit zero lines for residual reasons.

4. Normalize percentages from counts when `_pct` lines are absent but the
   denominator exists. Prefer simulator-provided `_pct` when present, but keep
   both count and denominator so reports can recompute if needed.

5. Use `total_num_cycles_issue_stage_evaluated` as the denominator. If it is
   missing, still parse counts but mark `denominator_missing: True`.

## Phase 2 - Attach To Probe Results

Update `_result()` in `amora/probes/gcom_cuda/baseline/__init__.py`.

Current behavior:

```python
derived_metrics = {
    "gpu_sim_cycle": stats.get("gpu_sim_cycle"),
    "gpu_tot_sim_insn": stats.get("gpu_tot_sim_insn"),
    "gpu_ipc": stats.get("gpu_ipc"),
}
```

Extend it to:

```python
from amora.backends.gcom_cuda.runner import extract_stall_reason_histogram

stall_hist = extract_stall_reason_histogram(stats)
if stall_hist is not None:
    derived_metrics["gcom_stall_reason_schema"] = stall_hist["schema"]
    derived_metrics["gcom_stall_reason_denominator"] = stall_hist.get("denominator")
    derived_metrics["gcom_stall_reason_histogram"] = stall_hist["reasons"]
    derived_metrics["gcom_stall_reason_complete"] = stall_hist["complete"]
    if not stall_hist["complete"]:
        derived_metrics["gcom_stall_reason_missing"] = stall_hist["missing_reasons"]
```

Keep this in `raw_observation.metrics`, not `normalized_measurement`, because it
is supporting evidence for the probe run rather than the probe's scalar result.

## Phase 3 - Logical Metric Mapping

Extend `amora/probes/gcom_cuda/baseline/gcom_metrics_map.py` so the counter-level
comparison can expose important stall reasons as logical metrics.

Add direct/proportional entries for the common NCU reasons:

| AMORA logical | GCoM stat key | NCU metric family |
|---|---|---|
| `stall_selected_pct` | `ncu_stall_selected_pct` | `smsp__average_warps_issue_stalled_selected*` / selected warp-state view |
| `stall_not_selected_pct` | `ncu_stall_not_selected_pct` | `smsp__average_warps_issue_stalled_not_selected*` |
| `stall_dispatch_pct` | `ncu_stall_dispatch_stall_pct` | `smsp__average_warps_issue_stalled_dispatch_stall*` |
| `stall_long_scoreboard_pct` | `ncu_stall_long_scoreboard_pct` | `smsp__average_warps_issue_stalled_long_scoreboard*` |
| `stall_short_scoreboard_pct` | `ncu_stall_short_scoreboard_pct` | `smsp__average_warps_issue_stalled_short_scoreboard*` |
| `stall_barrier_pct` | `ncu_stall_barrier_pct` | `smsp__average_warps_issue_stalled_barrier*` |
| `stall_wait_pct` | `ncu_stall_wait_pct` | `smsp__average_warps_issue_stalled_wait*` |
| `stall_mio_throttle_pct` | `ncu_stall_mio_throttle_pct` | `smsp__average_warps_issue_stalled_mio_throttle*` |
| `stall_mma_pct` | `ncu_stall_mma_pct` | `smsp__average_warps_issue_stalled_mma*` |
| `stall_math_pipe_throttle_pct` | `ncu_stall_math_pipe_throttle_pct` | `smsp__average_warps_issue_stalled_math_pipe_throttle*` |

Keep these in the counter-level layer. They should not upgrade or replace the
probe scalar.

## Phase 4 - Reports And Compare

Update `amora/backends/gcom_cuda/compare.py` and report rendering only enough to
surface the histogram clearly.

1. In JSON reports, no renderer change should be required because
   `raw_observation.metrics` already serializes nested dictionaries.

2. In `compare_counters()`, the new `gcom_metrics_map.py` entries will produce
   rows for stall percentages when the HW baseline contains matching NCU logical
   metrics. If HW lacks them, rows should still show `sim_gcom` with `hw_ncu`
   empty.

3. In Markdown, add a compact optional section per report:

   ```text
   ## GCoM Stall-Reason Coverage

   - probes with complete stall histogram: N / total successful sim probes
   - probes with partial histogram: ...
   - missing reasons observed: ...
   ```

4. Do not print all 20 histogram columns in the main probe table. Keep detailed
   histograms in JSON and the counter-comparison table.

## Phase 5 - Tests

Add no-GPU tests in `tests/backends/test_gcom_cuda.py`.

1. `test_parse_stats_extracts_ncu_stall_keys`

   Input:

   ```text
   ncu_stall_selected = 10
   ncu_stall_selected_pct = 12.5
   total_num_cycles_issue_stage_evaluated = 80
   ```

   Assert `parse_stats()` preserves these keys.

2. `test_extract_stall_reason_histogram_complete`

   Build a synthetic stats dict with all 20 reasons, counts, `_pct`, and the
   denominator. Assert:

   - schema is `ncu-stall-v1`
   - complete is true
   - all reasons are present
   - counts and percentages are preserved

3. `test_extract_stall_reason_histogram_partial_is_marked`

   Provide only one or two `ncu_stall_*` keys. Assert:

   - complete is false
   - missing reasons are listed
   - no missing reason is fabricated as zero

4. `test_gcom_result_attaches_stall_histogram`

   Unit-test `_result()` or a small helper by passing synthetic stats and
   asserting the histogram lands under `raw_observation.metrics`.

5. Extend `test_derive_logical_metrics_from_stats` with one stall percentage key
   once `gcom_metrics_map.py` includes stall entries.

## Verification

- `pytest -m "not cuda"` passes.
- A GCoM run whose simulator stdout contains `ncu_stall_*` lines produces JSON
  where every successful probe has `raw_observation.metrics.gcom_stall_reason_*`.
- Existing probe scalar comparison output is unchanged.
- Existing reports remain readable when the simulator is older and emits no
  `ncu_stall_*` lines; histogram fields are simply absent.
- Counter comparison shows simulator-side stall percentages when available.

## Risks

- **Older simulator output.** Some GCoM builds may not emit `ncu_stall_*`.
  Mitigation: histogram extraction returns `None` when no stall keys are present.
- **Partial simulator taxonomy.** A build may emit only some reasons. Mitigation:
  mark `complete: False` and list missing reasons; do not synthesize zeroes.
- **NCU metric naming drift.** Hardware report logical names may differ from the
  simulator's reason names. Mitigation: keep the simulator histogram in JSON
  regardless; counter comparison can be expanded as HW metric resolver names are
  finalized.
- **Main table bloat.** Twenty reasons per probe would overwhelm Markdown.
  Mitigation: JSON carries full detail; Markdown only reports coverage and
  selected counter rows.

## Acceptance Criteria

- Every successful `gcom_cuda` probe can carry a stall-reason histogram when the
  simulator emits the NCU stall taxonomy.
- Histogram schema is fixed, versioned, and complete/partial status is explicit.
- Unsupported/missing probes keep existing behavior.
- No simulator source changes are required for AMORA ingestion.
- The GCoM compare report can use stall histograms as counter-level evidence
  without confusing them with the probe scalar.
