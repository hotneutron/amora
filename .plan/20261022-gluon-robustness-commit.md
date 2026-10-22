# Gluon robustness changes — separate commit

Status: **70 focused tests pass; separate from portable workload packaging**

## User prompt, verbatim

```text
also commit amora's gluon changes in a separate commit
```

## Source inputs and commit scope

These changes were already present before portable-kernel packaging:

- `amora/backends/nvidia/ncu_run.py`.
- `amora/backends/nvidia/stall_cycle_measurement.py`.
- `tests/backends/test_ncu_run.py`.
- `tests/backends/test_stall_cycle_measurement.py`.
- `.plan/trackers/20260903-1245-tracker-all-corner-gluon-stall-cycle-measurement.md`.

The commit preserves optional zero-PC-support SourceCounters as diagnostic
results while strict callers still fail closed by default. All-corner timing
consumes at most one retry when process CV exceeds2%; selection keeps the entire
second attempt, validates unchanged subject/measurement context, and retains
both attempts as provenance. Manifest validation rejects unstable legacy rows
without the required retry evidence. The all-executable-corner selection reason
is accepted.

No kernel, latency gate or benchmark model is changed by the commit operation.
The existing patch is committed as inspected, with no new hardware collection.
The separate exact-TMA request and raw `traces/` artifacts remain uncommitted.

## Validation

```bash
PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 -m pytest --import-mode=importlib \
  tests/benchmarks/test_portable_core.py tests/benchmarks/test_materialize.py \
  tests/backends/test_ncu_run.py tests/backends/test_stall_cycle_measurement.py -q
```

Result: **70 passed**. Packaging and Gluon code are staged into separate commits
on the new feature branch; no Co-Authored-By trailer or push is requested.

## Revision History

|Revision|Timestamp|Change|
|---|---|---|
|r1|2026-10-22T04:17:37Z|Recorded the existing five-file Gluon diff,zero-support and bounded retry behavior,70 focused passing tests,separate-commit scope and exclusions of raw traces/exact-TMA request.|
