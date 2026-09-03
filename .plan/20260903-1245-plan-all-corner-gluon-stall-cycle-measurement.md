# Plan: All-Corner Gluon Warp-Stall Cycle Measurement

Status: Complete — AMORA implementation and bounded GH100 validation complete
Date: 2026-09-03 12:45 -0700
Owner: AMORA NVIDIA backend
Request: `.plan/20260903-1233-request-all-corner-gluon-stall-cycle-measurement.md`
Tracker: `.plan/trackers/20260903-1245-tracker-all-corner-gluon-stall-cycle-measurement.md`

## Objective

Extend AMORA's existing arbitrary-command measurement framework so an external finite
Gluon corner registry can collect dimensionally matched hardware warp-stall cycles for
every ordered compiled launch. The implementation must preserve each launch ordinal as
an independently observed record, select whole launch rows across three independent NCU
reports, and keep replay-derived mechanism cycles separate from CUDA-event latency.

## Non-Negotiable Measurement Boundary

For one selected row from one NCU report:

```text
active_warp_cycles = smsp__warps_active.sum
reason_warp_stall_cycles[r]
  = active_warp_cycles * normalized_per_warp_active_ratio[r]
```

The active-warp numerator and every issue-state ratio must come from that same row. A
`.pct` input is divided by 100; a `.ratio` input is used directly. Neither CUDA-event
duration nor NCU duration may enter the conversion. The raw derived unit is
`diagnostic_profiler_replay_warp_cycle`.

## Work Packages

### AM1: Capability and Metric Contract

- Add a stall-cycle metric resolver for the
  `smsp__warp_issue_stalled_<reason>_per_warp_active` family, preferring `.ratio` and
  falling back coherently to `.pct`.
- Require `smsp__warps_active.sum`, `smsp__cycles_active.sum`, and
  `smsp__warps_eligible.sum`.
- Preserve resolved metric names, input unit, missing issue states, and a capability
  result suitable for live GH100 preflight.
- Reject mixed ratio/percentage reason families within one record.

### AM2: Same-Row Stall-Cycle Reduction

- Parse one whole NCU row into active warp cycles, active SMSP cycles, eligible warp
  cycles, raw issue-state ratios, and absolute reason warp-stall cycles.
- Emit the full issue-state vector, the structural vector excluding `selected` and
  `not_selected`, and structural fractions.
- Quantify observed issue-state coverage and missing-state coverage; qualify conservation
  only when observed ratios sum to `1.0 +/- 0.05`, or expose the incomplete coverage
  explicitly without inventing zero-valued reasons.
- Reject negative/non-finite ratios and reason cycles that exceed active warp cycles
  outside tolerance.

### AM3: Three-Report Whole-Row Selection

- Run three independent aggregate NCU reports for every launch corner.
- Within each report, select a whole row using `median_total_stall`, restricted to the
  exact kernel/launch identity supplied by the target and NCU filters.
- Across the three selected vectors, select the median-total-stall observed record as the
  reduced record; retain all three raw selected records and report paths/hashes.
- Validate GPU, subject identity, operation identity, contract/corner IDs, launch ordinal,
  kernel, grid, cache/clock policy, runtime axes, and tool versions across reports.
- Never compose component-wise maxima or average reason fields across rows.

### AM4: Ordered Multi-Launch and Corner Contract

- Define an external target payload for exact composite operation identity and ordered
  launch descriptors.
- Keep every launch ordinal as a separate stall-cycle record.
- Form operation totals only by summing independently reduced launch records; never
  merge kernels into a synthetic selection row.
- Treat registry size and corner IDs as coverage; `launch_count` remains per-operation
  topology metadata.

### AM5: Conditional Repeated SourceCounters

- Reuse the existing repeated SourceCounters collector with three independent reports.
- Accept a consumer-supplied selection flag/reason for sentinel, compile-time corner,
  warm/rotating pair, failed gate, or dominant-reason transition points.
- Preserve raw sample counts, repeat support, exact offsets and SASS joins, report hashes,
  and lane identity. Never relabel PC support as cycles.

### AM6: Compact Artifacts and Immutable Run

- Add `aggregate_stall_cycles.csv`, `aggregate_stall_cycle_repeats.csv`, and
  `aggregate_stall_cycle_findings.json` to an immutable all-corner run manifest.
- Keep the existing `pc_samples_by_offset.csv` when SourceCounters were selected.
- Include every requested identity, unit, row-selection, repeat, conservation, and
  qualification column.
- Hash every compact artifact and validate hashes on manifest load.

### AM7: Verification and Live Preflight

- Unit-test ratio and percentage conversion, conservation, incomplete coverage, row
  selection, repeat reduction, identity drift, multi-launch totals, and CSV schemas.
- Add a CPU-only synthetic multi-launch/corner target for end-to-end orchestration.
- Run focused tests, full suite, mypy, compileall, CLI help, and diff checks.
- Query live GH100 metrics and, when the target/tool environment allows, execute a small
  H100 smoke with three aggregate reports and selected SourceCounters.

## Acceptance Gates

- A0: one coherent per-warp-active reason family plus all three cycle metrics resolve.
- A1: each repeat and final reduction references an observed whole launch row.
- A2: exactly three reports qualify with identical point and launch identity.
- A3: units, conservation, missing coverage, and structural exclusions are explicit.
- A4: selected SourceCounters retain three raw reports and exact SASS joins for
  quote-worthy rows.
- A5: CUDA events remain the latency oracle and profiler duration remains diagnostic.

## Ownership Boundary

AMORA owns metric resolution, collection, whole-row selection, dimensional reduction,
repeat validation, PC localization, immutable artifacts, and hardware-side findings.
The external Accorde workflow owns exhaustive corner generation, Gluon compilation,
target payload production, P3 stall-cycle prediction, and measured-versus-predicted
model interpretation.

## Execution Outcome

- The focused NVIDIA/all-corner suite passes: `108 passed`.
- The complete AMORA suite passes: `183 passed, 3 skipped`.
- Scoped mypy passes for the eight changed NVIDIA/CLI source modules with imported
  modules skipped; compileall, CLI help, manifest validation, and `git diff --check`
  also pass.
- The final qualifying GH100 aggregate smoke has run digest
  `a80796a5cea254fe0f6181bce76f547bb67dd3534f1fd7ff691b848e3eca2783`: three
  retained aggregate reports, selected row indices `[0, 1, 2]`, selected repeat index
  `2`, issue-state ratio sum `1.0`, and 19 resolved states with `mma` explicitly
  unavailable.
- The repeated SourceCounters diagnostic smoke has run digest
  `46ade588d21a8085e1a11bbdcdec4e954a8e09c8fe346179ac6fb37aec43bce8`: three
  aggregate plus three SourceCounters reports, 19 offsets, and 5,379 raw samples. The
  generic PyTorch target lacks an independent cubin/SASS artifact, so its joins remain
  diagnostic and are not quote-worthy.
- The 5,699-corner Accorde preregistry is not executed here because its compile
  preflight, strict target parameterization, realization-matched P3, and predicted
  stall-cycle gates remain incomplete. Its current CSV and manifest agree on SHA-256
  `76892da7158dc5f78ca0e1025ac808be13290ca8afcdcbc6d2e081b66c465107`; the
  Accorde tracker still contains an older 5,707-count log entry.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r7 | 2026-09-03 14:53 -0700 | Preserved specific multiline tool versions in manifests and reran both finalized-schema GH100 smokes. |
| r6 | 2026-09-03 14:53 -0700 | Required semantic cache/clock axes, distinguished capability-missing from selected-row-missing reasons, and reran validation and hardware smokes. |
| r5 | 2026-09-03 14:42 -0700 | Added independent expected corner and launch-record cardinality and reran both GH100 smoke paths. |
| r4 | 2026-09-03 14:34 -0700 | Reconciled the test counts and immutable GH100 evidence, and made the external Accorde campaign gate explicit. |
| r3 | 2026-09-03 14:24 -0700 | Finalized after independent audit: selection is based on structural blocked ratios while retaining the full row; row filtering precedes selection; final cross-lane and PC qualification propagates to CSVs and operation totals; duplicate ordinals, unsafe IDs, missing report hashes, cache or clock drift, and pct-label ambiguity are rejected or explicit. Added exact registry cardinality and launch-context and tool-version validation. |
| r2 | 2026-09-03 14:22 -0700 | Completed the AMORA implementation: same-row ratio-to-warp-cycle conversion, structural row selection, three-report reduction, exact ordered-launch context, conditional repeated SourceCounters, immutable compact artifacts, CLI, audit fixes, and GH100 validation. The full external campaign remains gated by Accorde compile preflight and target/prediction completion. |
| r1 | 2026-09-03 12:45 -0700 | Converted the request into capability, same-row reduction, repeat selection, multi-launch, SourceCounters, artifact, and validation work packages. |
