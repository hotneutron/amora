# Tracker: reusable NCU collection for external and JIT targets

Plan: `.plan/20260831-1400-amora-ncu-target-command-compatibility-plan.md`
Status: Complete
Started: 2026-08-31 14:00 -0700
Completed: 2026-08-31 14:12 -0700

## Checklist

- [x] Read the complete upstream report and identify the seven AMORA requests.
- [x] Audit the current dirty working tree and preserve the existing stall/PC work.
- [x] Write the implementation plan and acceptance criteria.
- [x] Add aggregate arbitrary-target profiling API.
- [x] Add arbitrary-target SourceCounters profiling API.
- [x] Detect and record the installed NCU sampling option.
- [x] Parse `SASS` and `Source` instruction columns.
- [x] Parse display and raw not-issued reason columns.
- [x] Normalize absolute loaded PCs while preserving raw addresses and bases.
- [x] Parse Hopper uniform predicates.
- [x] Map aggregate `gmma` to logical `warpgroup_arrive`.
- [x] Add regression tests for all compatibility paths.
- [x] Run focused backend tests.
- [x] Run the full test suite.
- [x] Review the final diff for scope and provenance quality.

## Baseline

- Branch: `main` tracking `origin/main`.
- The working tree already contains uncommitted NVIDIA stall coverage and PC-sampling
  changes, including `stall_metrics.py`; these are treated as prerequisite user work.
- Existing source sampling hardcodes `--sampling-interval`, builds only `.cu` targets,
  and recognizes raw `issue_stalled_*` columns.
- Existing SASS parsing retains offsets but recognizes only `P<number>` predicates.

## Execution Log

| Timestamp | State | Evidence |
|---|---|---|
| 2026-08-31 14:00 -0700 | Started | Read the report, prior AMORA request documents, current backend, tests, and dirty-tree diff. |
| 2026-08-31 14:00 -0700 | Planned | Defined four slices and explicit backward-compatibility/provenance acceptance criteria. |
| 2026-08-31 14:06 -0700 | Implemented | Added command-target APIs, automatic sampling-option detection, 2026.2 source aliases, PC normalization metadata, uniform predicates, and `gmma` resolution. |
| 2026-08-31 14:08 -0700 | Focused verification | NVIDIA parser, command, SASS, and metric tests passed: 27 passed. |
| 2026-08-31 14:09 -0700 | Tool compatibility | NCU 2026.2.1 advertised `--warp-sampling-interval`; NCU 2022.3 advertised `--sampling-interval`. |
| 2026-08-31 14:10 -0700 | Real-artifact verification | Parsed all 52 retained NCU 2026.2 source reports into 2,485 nonzero PC rows. Counts and support matched the prior reducer in 52/52 cases; normalized offsets joined retained cubin SASS in 52/52 cases. |
| 2026-08-31 14:12 -0700 | Complete | Full suite, type checks, bytecode compilation, and whitespace checks passed. |

## Verification

- `PYTHONPATH=. pytest -q tests/backends/test_ncu_run.py tests/backends/test_sass.py tests/backends/test_nvidia_cuda.py`
  - Result: `27 passed in 0.08s`.
- `PYTHONPATH=. pytest -q`
  - Result: `102 passed, 3 skipped in 8.70s`.
- `PYTHONPATH=. mypy amora/backends/nvidia/ncu.py amora/backends/nvidia/ncu_run.py amora/backends/nvidia/sass.py amora/backends/nvidia/stall_metrics.py amora/probes/nvidia/baseline/_sources.py`
  - Result: success, no issues in five source files.
- `python -m compileall -q amora tests`
  - Result: passed.
- `git diff --check`
  - Result: passed.
- Local NCU help probes:
  - Nsight Compute 2026.2.1 exposes `--warp-sampling-interval`.
  - Nsight Compute 2022.3.0 exposes `--sampling-interval`.
- Retained real-evidence replay under `/tmp`:
  - 52/52 source CSV files parsed with NCU 2026.2 display aliases.
  - 2,485 nonzero sampled-PC records recovered.
  - 52/52 cases exactly matched the prior reducer's PC count and total support.
  - 52/52 cases achieved complete normalized PC-to-SASS joins against retained cubins.

Live counter recollection was not required: the compatibility path was tested against
the actual 52-report NCU 2026.2 corpus, while command execution itself is covered by
subprocess-isolated tests.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r2 | 2026-08-31 14:12 -0700 | Closed all checklist items and recorded focused, full-suite, type, tool-version, and retained-real-artifact verification. |
| r1 | 2026-08-31 14:00 -0700 | Created execution tracker with baseline, checklist, and evidence log. |
