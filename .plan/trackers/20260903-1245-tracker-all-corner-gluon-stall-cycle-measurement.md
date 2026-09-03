# Tracker: All-Corner Gluon Warp-Stall Cycle Measurement

Status: Complete — AMORA implementation and bounded GH100 validation complete
Plan: `.plan/20260903-1245-plan-all-corner-gluon-stall-cycle-measurement.md`
Request: `.plan/20260903-1233-request-all-corner-gluon-stall-cycle-measurement.md`
Started: 2026-09-03 12:45 -0700
Completed: 2026-09-03 14:53 -0700

## Checklist

### Metric Contract

- [x] Resolve one per-warp-active ratio/pct reason family.
- [x] Resolve active-warp, active-SMSP, and eligible-warp cycle metrics.
- [x] Record live capability coverage and missing reasons.

### Same-Row Reduction

- [x] Derive absolute reason warp-stall cycles from one whole row.
- [x] Emit full, structural, and structural-fraction views.
- [x] Enforce or quantify dimensional conservation and missing coverage.
- [x] Keep replay-cycle units separate from CUDA-event and PC-sample units.

### Repeats and Launches

- [x] Collect three independent aggregate reports.
- [x] Select one whole launch row per report with `median_total_stall`.
- [x] Select the median-total observed repeat without component aggregation.
- [x] Validate exact repeat identity, runtime axes, launch ordinal, and tools.
- [x] Validate registry corner cardinality separately from ordered launch records.
- [x] Require semantic cache and clock axes and expose supported-row omissions.
- [x] Preserve ordered multi-launch records and additive operation totals.

### SourceCounters

- [x] Collect three reports for selected sentinel/residual corners.
- [x] Preserve raw support, repeat support, offsets, joins, and report hashes.
- [x] Keep PC sample counts distinct from derived warp-stall cycles.

### Artifacts

- [x] Emit aggregate stall-cycle summary CSV.
- [x] Emit repeat-level aggregate stall-cycle CSV.
- [x] Emit findings JSON and optional pooled PC CSV.
- [x] Hash and validate every immutable artifact.

### Validation

- [x] Add focused unit/integration coverage.
- [x] Execute a synthetic all-corner/multi-launch campaign.
- [x] Run live GH100 capability preflight and feasible hardware smoke.
- [x] Run full suite, mypy, compileall, CLI help, and diff checks.
- [x] Close tracker with evidence and external handoff boundary.

## Baseline

- `main` is aligned with `origin/main` at `a795f78`.
- The measurement framework already supports process-isolated CUDA-event timing,
  aggregate NCU, repeated SourceCounters, runtime-axis validation, immutable artifacts,
  and barrier-topology reduction.
- The all-corner request and a separate exact-GEMM request are untracked inputs. This
  execution uses only the all-corner request and leaves the exact-GEMM request untouched.
- `traces/` is unrelated untracked data and remains untouched.

## Execution Log

| Timestamp | State | Evidence |
|---|---|---|
| 2026-09-03 12:45 -0700 | Started | Read the complete 256-line request and audited the current stall, bundle, repeated-PC, topology-artifact, and test paths. |
| 2026-09-03 12:45 -0700 | Planned | Established same-row absolute-cycle derivation and observed-whole-row repeat reduction as the central dimensional invariant. |
| 2026-09-03 13:20 -0700 | Implemented | Added the dedicated per-warp-active metric resolver, same-row cycle reducer, three-report observed-row selector, launch context schema, multi-launch totals, conditional repeated SourceCounters, immutable outputs, and CLI. |
| 2026-09-03 13:45 -0700 | Audit corrected | Independent same-model audit found structural-row selection, pre-selection filtering, final qualification propagation, duplicate ordinals, unit labeling, mandatory report hashes, cache/clock evidence, safe IDs, and replay-count gaps. Corrected all findings and added regressions. |
| 2026-09-03 14:00 -0700 | GH100 A0 | NCU 2026.2.1 resolved the per-warp-active ratio family plus active-warp, active-SMSP, and eligible-warp cycle counters. All exposed issue states resolve except unavailable `mma`. |
| 2026-09-03 14:10 -0700 | GH100 A1-A3 | Initial immutable smoke runs established three-report collection, whole-row selection, and qualifying identity/conservation. |
| 2026-09-03 14:15 -0700 | GH100 A4 | Initial sentinel runs established repeated SourceCounters collection and confirmed that the generic PyTorch target remains diagnostic without an independent cubin/SASS artifact path. |
| 2026-09-03 14:24 -0700 | Final audit | Corrected structural-total row selection, pre-selection launch filtering, final qualification propagation, duplicate launch ordinal detection, normalized pct labeling, mandatory report hashes, cache/clock identity, path-safe IDs, and replay-count collection. Added registry cardinality, ordered launch descriptors, and exact context/tool-version checks. |
| 2026-09-03 14:42 -0700 | Final schema | Split the explicit finite-registry corner count from ordered launch-record count, added SourceCounters cache/clock-control drift diagnostics, and covered both with regressions. |
| 2026-09-03 14:53 -0700 | Provenance corrected | Tool discovery now retains the specific version-bearing output line; finalized manifests identify `/usr/local/cuda-13.3/bin/ncu` as `Version 2026.2.1.0 (build 38283040) (public-release)`. |
| 2026-09-03 14:53 -0700 | Complete | Focused suite (`108 passed`), complete suite (`183 passed, 3 skipped`), scoped mypy, compileall, CLI help, finalized-schema smoke manifest validation, and whitespace checks passed. External registry readiness was audited without starting the gated full campaign. |

## Verification

- Focused all-corner and shared-backend tests pass.
- Focused all-corner/shared-backend suite: `108 passed`.
- Full suite: `183 passed, 3 skipped`.
- `mypy --follow-imports=skip` passes for the eight affected NVIDIA/CLI source
  modules. The repository's default import-following mypy invocation is not currently a
  clean global gate: it reports 71 errors in 33 imported, out-of-scope modules.
- `python -m compileall -q amora tests`, CLI help, and `git diff --check` pass.
- GH100 capability: 19 issue states resolved in the coherent per-warp-active ratio
  family; `mma` is explicitly missing; all three cycle counters resolve.
- Qualifying aggregate smoke:
  - run digest `a80796a5cea254fe0f6181bce76f547bb67dd3534f1fd7ff691b848e3eca2783`;
  - campaign digest `f7cfc597d7139c843ba124a6bc5d20cf6e2f8f7e47212ca5e2ed05ff6160da89`;
  - one expected corner and one expected launch record;
  - three aggregate reports; selected rows `[0, 1, 2]`;
  - selected repeat index `2`; ratio sum `1.0`; input and normalized units `ratio`;
    raw unit `diagnostic_profiler_replay_warp_cycle`;
  - profiler replay/pass count `8`;
  - exact timing/aggregate identity and context; no SourceCounters requested.
- Sentinel SourceCounters smoke:
  - run digest `46ade588d21a8085e1a11bbdcdec4e954a8e09c8fe346179ac6fb37aec43bce8`;
  - campaign digest `97665ec5cf5cc4a0a5afed749ba9d397f7b23b2b63755be7a7092938135feb62`;
  - one expected corner and one expected launch record;
  - three aggregate reports plus three SourceCounters reports;
  - 19 pooled offsets and 5,379 raw samples;
  - retained as diagnostic because the generic PyTorch smoke target does not expose an
    independent SASS/cubin artifact for exact joining.

## External Campaign Gate

The Accorde preregistry manifest declares 5,699 corners, and its CSV has 5,699 data rows
with matching SHA-256
`76892da7158dc5f78ca0e1025ac808be13290ca8afcdcbc6d2e081b66c465107`. The
Accorde tracker still contains a stale 5,707-count log entry and still marks compile
preflight, strict all-corner target parameterization, realization-matched P3, and
predicted stall cycles incomplete. The full H100 campaign was therefore not started:
launch-time coverage must be based on compile-qualified points and the exact
ordered-launch target protocol, not on the preregistry's pending points.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r7 | 2026-09-03 14:53 -0700 | Corrected multiline tool-version provenance, regenerated the immutable H100 smokes, and closed at 108 focused plus 183 full-suite passing tests. |
| r6 | 2026-09-03 14:50 -0700 | Required semantic cache/clock axes, separated capability and row-level missing reasons, recorded the final-schema GH100 digests, and closed at 107 focused plus 182 full-suite passing tests. |
| r5 | 2026-09-03 14:42 -0700 | Added independent corner-versus-launch cardinality and SourceCounters control-drift gates, reran the finalized-schema GH100 smokes, and recorded the final 178-test result. |
| r4 | 2026-09-03 14:34 -0700 | Revalidated 100 focused and 175 full-suite tests, corrected final GH100 digests/counts/row selections from the immutable manifests, and documented the scoped mypy boundary. |
| r3 | 2026-09-03 14:24 -0700 | Recorded all independent-audit corrections, ordered launch/context and registry completeness, and final validation intent. |
| r2 | 2026-09-03 14:22 -0700 | Closed every AMORA checklist item, recorded audit corrections, synthetic and GH100 evidence, and the external compile/target/prediction gate for the full 5,699-corner campaign. |
| r1 | 2026-09-03 12:45 -0700 | Created actionable checklist, baseline, execution log, and acceptance-gate structure. |
