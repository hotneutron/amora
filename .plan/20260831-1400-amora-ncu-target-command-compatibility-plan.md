# Plan: reusable NCU collection for external and JIT targets

Status: Complete
Owner: AMORA NVIDIA backend
Source: Accorde's completed 52-configuration Gluon/P3 stall-gap study

## User Prompt, Verbatim

> write a plan to add support for the requested feature, and execute it with a tracker.

## Goal

Make AMORA's NVIDIA evidence collector directly reusable for Python/JIT launchers
and portable across the legacy and Nsight Compute 2026.2 SourceCounters schemas.
AMORA owns collection, normalization, evidence identity, and explicit capability
status. Consumer-specific Gluon/P3 interpretation remains outside AMORA.

## Scope

1. Add aggregate and SourceCounters entry points that profile an arbitrary target
   argv without compiling a CUDA source file. Preserve exact target and NCU commands
   in provenance. Keep the existing source-building entry points as wrappers.
2. Detect whether the installed profiler accepts `--sampling-interval` or
   `--warp-sampling-interval` using `ncu --help`, and record the selected option.
3. Parse instruction text from either `SASS` or `Source` source-page columns.
4. Parse both raw PC-sampling metric names and display aliases such as
   `stall_long_sb (Not Issued)`, preferring not-issued columns when both forms exist.
5. Detect loaded absolute PCs and normalize them to function-relative offsets before
   SASS joining. Retain the raw address and normalization base in every sample/result.
6. Extend SASS parsing to uniform predicates including `@UP2`, `@!UP2`, `@PT`, and
   `@!PT`.
7. Resolve Hopper's aggregate `gmma` metric as logical `warpgroup_arrive` within the
   same coherent metric family.

## Implementation Slices

### 1. NCU command and execution APIs

- Extend `amora/backends/nvidia/ncu.py` with a pure sampling-option parser and a
  best-effort help probe.
- Add `run_command_profiled` and `run_command_pc_sampling` in
  `amora/backends/nvidia/ncu_run.py`.
- Refactor `run_kernel_profiled` and `run_kernel_pc_sampling` to compile as before,
  then delegate target execution to the command APIs.
- Reject empty target commands and unresolved aggregate metrics explicitly.

### 2. SourceCounters compatibility

- Centralize source-page reason aliases in `amora/backends/nvidia/ncu_run.py`.
- Select `(Not Issued)` or `_not_issued` columns when available; only fall back to
  all-sample columns when no not-issued family exists.
- Normalize absolute addresses per function when function identity is present and per
  result otherwise. Keep relative `/*offset*/` values unchanged.
- Expose parser metadata including address mode/base, column mode, present reasons,
  and missing reasons.

### 3. Hopper aliases and SASS parsing

- Add `gmma` as a metric-family alias for logical `warpgroup_arrive` in
  `amora/backends/nvidia/stall_metrics.py`.
- Extend `amora/backends/nvidia/sass.py` predicate recognition without changing the
  opcode-family or offset contract.

### 4. Verification and integration

- Add GPU-independent regression fixtures for legacy and 2026.2 CSV shapes, absolute
  PC normalization, display aliases, sampling-option negotiation, arbitrary command
  construction, uniform predicates, and Hopper metric resolution.
- Run focused backend tests, then the full test suite.
- Record commands, outcomes, deviations, and file-level completion in the tracker.

## Acceptance Criteria

- A Python/JIT argv can be passed directly to aggregate and PC-sampling APIs.
- The generated source-profile command uses the option exposed by the installed NCU.
- Both known source-page schemas produce non-empty raw-count records.
- Absolute and relative PC fixtures join to the same function-relative SASS offsets.
- Uniformly predicated Hopper instructions remain in the SASS offset map.
- `smsp__warp_issue_stalled_gmma_per_warp_active` resolves to
  `warpgroup_arrive` without mixing metric families.
- Existing CUDA-source callers and their provenance remain compatible.
- Focused and full tests pass, or any environment-only skip/blocker is recorded.

## Risks and Guards

- An absolute-PC minimum is only a safe proxy for function base when records are
  grouped by function. Normalize per function and expose the inferred base rather than
  hiding the heuristic.
- Do not compare aggregate percentages with raw PC counts. The parser retains raw
  counts and labels its column mode.
- Do not add Gluon fixture knowledge or P3 causal mappings to AMORA.
- Preserve the user's pre-existing dirty working-tree changes and avoid unrelated
  formatting or generated artifacts.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r2 | 2026-08-31 14:12 -0700 | Marked complete after unit, type, full-suite, dual-NCU help, and retained 52-case PC-to-SASS validation passed. |
| r1 | 2026-08-31 14:00 -0700 | Created from the seven AMORA compatibility requests in the completed Gluon/P3 stall-gap report; added API, provenance, compatibility, testing, and ownership-boundary acceptance criteria. |
