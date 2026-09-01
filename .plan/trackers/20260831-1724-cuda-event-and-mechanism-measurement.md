# Tracker: CUDA-event and physical-mechanism measurement

Plan: `.plan/20260831-1724-plan-cuda-event-and-mechanism-measurement.md`
Request: `.plan/20260831-1719-request-cuda-event-and-mechanism-measurement.md`
Status: Complete
Started: 2026-08-31 17:24 -0700
Completed: 2026-08-31 18:57 -0700

## Checklist

- [x] Read the complete request and audit current command-target NCU support.
- [x] Write the actionable implementation plan.
- [x] Implement CUDA-event payload parsing and process-isolated repetitions.
- [x] Implement device-interval validation and diagnostic downgrade rules.
- [x] Propagate and record NCU cache control through all four entry points.
- [x] Preserve target identity from profiled runs.
- [x] Implement separate-lane bundle orchestration and identity validation.
- [x] Implement physical-mechanism recipe validation.
- [x] Encode coupled causal nodes/edges and same-case physical evidence roles.
- [x] Validate controlled interaction groups and WGMMA CTA-load sensitivity.
- [x] Generate edge-level mediation output without direct intervention regressors.
- [x] Implement immutable run layout, bundle files, and content-digested manifest.
- [x] Add focused unit and integration tests.
- [x] Run focused and full validation.
- [x] Attempt or disposition the requested GPU smoke.
- [x] Review scope and close the tracker.

## Baseline

- Branch `main` was aligned with `origin/main` at task start.
- The request document and `traces/` are pre-existing untracked paths and will be
  preserved.
- Arbitrary-command aggregate and SourceCounters APIs already exist from the preceding
  compatibility implementation.
- No CUDA-event target protocol, device-interval contract, cross-lane bundle, or
  mechanism recipe currently exists.
- NCU exposes cache-control values `all` and `none` on both installed versions.

## Execution Log

| Timestamp | State | Evidence |
|---|---|---|
| 2026-08-31 17:24 -0700 | Started | Read the request, current NCU APIs, evidence schemas, statistics helpers, immutable benchmark writers, and installed NCU help. |
| 2026-08-31 17:24 -0700 | Planned | Defined protocol, interval, orchestration, recipe/persistence, and verification slices. |
| 2026-08-31 17:27 -0700 | Corrected scope | Re-read all 474 request lines after interruption; revision r2 adds a coupled-bottleneck causal graph, CTA-load discrimination, interaction controls, and mediation reporting. Updated the plan before code implementation. |
| 2026-08-31 17:38 -0700 | Protocols complete | Added strict CUDA-event timing and device-interval payloads, independent process repetition, robust summaries, cache/clock provenance, and diagnostic downgrade reasons. |
| 2026-08-31 17:52 -0700 | Bundle complete | Added independent timing, interval, aggregate NCU, and SourceCounters lanes with cross-lane GPU, identity, and subject-metadata checks. NCU duration is isolated as diagnostic profiler evidence. |
| 2026-08-31 18:12 -0700 | Physical contract complete | Added pre-frozen recipes, deterministic randomized order, complete-axis controlled groups, causal evidence bindings, WGMMA completion-knee and CTA-load assessments, TMA route/occupancy requirements, spill fail-closed handling, and edge-level mediation. |
| 2026-08-31 18:20 -0700 | Artifact contract complete | Added independent target-declared SASS/cubin joins, report/artifact hashes, immutable point bundles and run manifests, and the `amora nvidia measure` CLI. |
| 2026-08-31 18:30 -0700 | H100 smoke | Ran CUDA-event timing plus aggregate NCU plus SourceCounters on GPU 0. Identity passed; CUDA-event latency remained the oracle; SourceCounters returned nonzero PC evidence. |
| 2026-08-31 18:35 -0700 | Hardened | Added exact runtime-axis validation, pre-execution frozen recipes, complete controlled-axis sweeps, independent SASS/cubin joins, and distinct exposed-wait versus completion evidence. |
| 2026-08-31 18:30 -0700 | Final H100 smoke | Two process repeats produced ten CUDA-event samples with a 6.0336 us median, within 1.5894% of a separate direct invocation; NCU duration was 3520 ns and remained diagnostic; SourceCounters parsed 17 PCs and 1,738 raw samples; identity, subject metadata, runtime axes, and tool-version capture passed. |
| 2026-08-31 18:35 -0700 | Complete | Focused tests, full suite, mypy, bytecode compilation, CLI help, and whitespace validation passed. |
| 2026-08-31 18:57 -0700 | Final review | Required target tool versions and exact runtime axes across lanes, required direct physical evidence for causal nodes, strengthened the TMA route counter groups, and added opcode-grouped SourceCounters for instruction-localized bindings. |

## Verification

- Focused implementation tests:
  - `PYTHONPATH=. pytest -q tests/backends/test_cuda_event_run.py tests/backends/test_measurement.py tests/backends/test_mechanism_measurement.py tests/backends/test_ncu_run.py tests/test_cli.py`
  - Result: all focused tests passed; the final focused backend set has 46 tests.
- Full suite:
  - `PYTHONPATH=. pytest -q`
  - Result: `137 passed, 3 skipped in 8.98s`.
- Static and syntax validation:
  - Mypy passed for all six changed/new NVIDIA measurement modules.
  - `python -m compileall -q amora tests` passed.
  - `git diff --check` passed.
- H100 live smoke with a protocol-compliant PyTorch CUDA target:
  - two independent timing processes and ten total CUDA-event samples;
  - timing median `6.0336 us`, within `1.5894%` of a separate direct invocation;
  - aggregate NCU profiler duration `3520 ns`, retained only as diagnostic;
  - aggregate/timing identity check passed;
  - SourceCounters parsed 17 PCs with 1,738 raw not-issued samples;
  - independent PC-to-SASS join correctly remained unavailable because the generic
    PyTorch smoke target did not declare a cubin or SASS artifact path;
  - NCU cache and clock controls were both recorded as `none`.
- Independent SASS-join replay over retained Gluon evidence:
  - 52/52 reports and 2,485/2,485 nonzero PC rows joined to retained cubin SASS.
- CLI smoke:
  - `PYTHONPATH=. python -m amora nvidia measure --help` passed.

## External Validation Boundary

The repository has no Gluon target that emits the new `cuda_event_timing` JSON and no
instrumented Gluon target that emits `cuda_device_intervals`. Therefore the exact
requested three-lane Gluon smoke and the full WGMMA/TMA matrices cannot be run solely
inside AMORA. The live smoke validates AMORA's timing, aggregate, SourceCounters, and
identity plumbing on H100. Accorde must provide the two target-side adapters before a
full physical-mechanism campaign can produce qualifying interval evidence.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r6 | 2026-08-31 18:57 -0700 | Finalized same-case runtime-axis and tool-version validation, direct causal-node bindings, complete TMA route counter requirements, opcode-localized PC evidence, and final test counts. |
| r5 | 2026-08-31 18:35 -0700 | Added exact runtime-axis and target tool-version evidence, complete per-axis intervention enforcement, stricter TMA route metric requirements, and opcode-grouped SourceCounters; refreshed the final H100 smoke. |
| r4 | 2026-08-31 18:36 -0700 | Updated final verification after runtime-axis and independent SASS-join hardening; recorded the final H100 comparison and its expected generic-target join limitation. |
| r3 | 2026-08-31 18:35 -0700 | Closed implementation and recorded unit, full-suite, static, real-H100, retained-cubin join, CLI, and external-producer evidence. |
| r2 | 2026-08-31 17:27 -0700 | Added revision-r2 coupled-bottleneck, controlled-ablation, CTA-load, and mediation checklist items. |
| r1 | 2026-08-31 17:24 -0700 | Created with baseline, implementation checklist, execution log, and verification section. |
