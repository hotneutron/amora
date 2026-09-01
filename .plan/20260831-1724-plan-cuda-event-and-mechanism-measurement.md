# Plan: CUDA-event and physical-mechanism measurement for command targets

Status: Complete
Owner: AMORA NVIDIA backend
Source request: `.plan/20260831-1719-request-cuda-event-and-mechanism-measurement.md`

## Request

Implement `.plan/20260831-1719-request-cuda-event-and-mechanism-measurement.md`
with an actionable plan and a tracker under `.plan/trackers/`.

## Outcome

AMORA will expose one evidence workflow for arbitrary Python/JIT commands with
strictly separated measurement lanes:

1. repeated, unprofiled target-emitted CUDA-event timing as the only latency oracle;
2. optional target-emitted `%globaltimer` intervals as diagnostic mechanism evidence;
3. NCU aggregate counters and SourceCounters as profiler-perturbed mechanism evidence.

The workflow will validate identity across lanes, preserve raw evidence and execution
provenance, and write each run once to a unique immutable directory. It will not add
Gluon-specific launch logic or P3 model equations to AMORA.

## Implementation

### 1. CUDA-event target protocol

- Add `amora/backends/nvidia/cuda_event_run.py`.
- Define immutable process-sample and aggregate-result dataclasses with `to_dict()`
  serialization.
- Parse the target's final non-empty stdout line as a versioned
  `cuda_event_timing` JSON payload.
- Validate finite positive `us_per_launch` samples, warmup and batch counts, declared
  cache protocol, GPU UUID, and required subject identities.
- Run the target in independent processes, retaining every process payload, exact argv,
  cwd, explicit environment overrides, stdout, stderr, and return code.
- Reject identity, device, cache-protocol, unit, and launch-batch drift across repeats.
- Compute pooled median/p05/p95 and process-median coefficient of variation.

### 2. Device-interval protocol

- Define and parse versioned `cuda_device_intervals` payloads.
- Preserve named raw start/end/duration values and instrumentation metadata.
- Mark results `diagnostic` when timestamps are non-monotonic, durations disagree,
  SASS bracketing is unverified, instruction-sequence equivalence is false, overhead
  exceeds its declared threshold, or the clock is undocumented.
- Keep these intervals out of the primary latency fields regardless of status.

### 3. NCU parity and bundle orchestration

- Add validated `cache_control` support (`all` or `none`) to `NcuCommand`,
  `run_command_profiled`, `run_command_pc_sampling`, and both source-building wrappers.
- Record cache control in aggregate and SourceCounters provenance.
- Preserve a target-emitted subject identity from NCU runs without treating its timing
  fields as unprofiled latency.
- Add `collect_command_measurement_bundle` in a dedicated orchestration module.
- Execute timing, optional device intervals, aggregate NCU, and optional PC sampling as
  separate target invocations.
- Require matching configured identity fields and GPU UUID across available lanes.
- Serialize `latency_oracle=cuda_events` and
  `ncu_duration_role=diagnostic_profiler_perturbation_only`; expose NCU duration only
  under `profiler_duration`.

### 4. Physical-mechanism recipe and immutable evidence

- Define a generic command-target recipe with named points, physical axes, held-out
  designation, commands for each lane, expected identity, and subject metadata.
- Validate the WGMMA fixed-completion axes and TMA route/offered-load axes requested by
  the source document.
- Reject spill-containing points as qualifying evidence while retaining them with
  `diagnostic` status.
- Execute recipe points through the bundle API and write canonical `bundle.json` files
  plus a content-digested `manifest.json`.
- Create the run directory with exclusive semantics and reject reuse of an existing
  run ID. Never overwrite an evidence file.

### 5. Coupled-bottleneck and mediation contract

- Encode the request's causal nodes and directed edges: footprint/occupancy, producer
  eligibility, issue, queue/partition pressure, route traffic/service, completion,
  barrier release, matrix completion, and finite-stage release/backpressure.
- Require each recipe to assign collected counters, intervals, metadata, and axes to
  those physical roles instead of returning an unstructured table of correlations.
- Require WGMMA intervention groups to include CTA-load variation; report a completion
  knee as `shared_resource_or_scheduling_interaction` when it changes with CTA load.
- Require TMA intervention groups to retain route traffic, completion/release intervals,
  queue capacity, footprint/occupancy, and total CUDA-event latency in the same case.
- Represent interaction groups explicitly with one varied axis and declared held-constant
  axes. Reject groups whose declared controls drift.
- Generate a mediation artifact that reports, edge by edge, which measured intermediate
  changed under each intervention. Use `not_measured` and `falsified` statuses rather
  than filling missing arrows with a fitted correction.
- Preserve `K`, elapsed duration, and working-set size only as interventions. Never
  expose them as direct additive regressors or endorse a portable law from them.

### 6. Verification

- Add GPU-independent unit tests for valid timing payloads and every requested reject
  condition.
- Test device-interval downgrade reasons and serialization boundaries.
- Test cache-control propagation through all four NCU entry points.
- Test cross-lane identity success/failure and verify NCU duration never becomes the
  measured latency.
- Test recipe validation, held-out preservation, spill downgrade, manifest digest, and
  immutable-write refusal.
- Test that controlled interaction groups reject drift, CTA-load sensitivity prevents a
  fixed-latency qualification, and mediation output follows the causal graph rather than
  emitting independent fitted corrections.
- Run focused tests, static type checking, the full suite, and `git diff --check`.
- Attempt a GPU smoke only if a target implementing the new JSON protocol is locally
  available; otherwise record the precise missing producer-side prerequisite.

## Acceptance Criteria

- All validation rules in the source request have direct unit coverage.
- Timing results retain process-level sample sets and expose robust pooled summaries.
- Device intervals can never silently become qualifying evidence after a diagnostic
  condition.
- Timing and NCU are distinct subprocess invocations and their roles are explicit in
  serialized output.
- Bundle construction fails closed on identity or GPU drift.
- Both supported NCU cache-control values are emitted and recorded.
- Recipe artifacts are unique, content-digested, immutable, and distinguish calibration
  from held-out points.
- Same-case output groups measurements by causal node and edge, and full recipes include
  a mediation report plus controlled interaction ablations.
- No API emits direct `a + b*K`, elapsed-duration, or raw-working-set regressors.
- The existing test suite remains green.

## Execution Order

The payload validators and NCU cache-control plumbing are independent and can be
implemented in parallel conceptually. Bundle orchestration follows their stable
contracts. Recipe persistence follows the bundle schema. Verification is added with
each slice and rerun across the full suite at the end.

## Revision History

| Revision | Timestamp | Change |
|---|---|---|
| r6 | 2026-08-31 18:57 -0700 | Finalized exact runtime-axis and tool-version matching, direct evidence requirements for causal nodes, complete TMA route counter coverage, and instruction-localized SourceCounters summaries. |
| r5 | 2026-08-31 18:35 -0700 | Added target-emitted runtime-axis and tool-version checks, required an actual controlled group for every physical axis, strengthened TMA route/occupancy metric requirements, and added opcode-grouped SourceCounters evidence. |
| r4 | 2026-08-31 18:36 -0700 | Added exact runtime-axis matching across lanes, pre-execution frozen recipes, complete controlled-axis coverage, independently verified target SASS joins, and separate exposed-wait/completion semantics for WGMMA. |
| r3 | 2026-08-31 18:35 -0700 | Marked implementation complete after adding target protocols, cross-lane identity and metadata checks, NCU cache/clock control, independently verified PC-to-SASS joins, complete controlled-axis recipe validation, WGMMA knee/CTA-load assessments, immutable artifacts, CLI execution, and H100 smoke validation. |
| r2 | 2026-08-31 17:27 -0700 | Re-read request revision r2 and added the coupled resource/dependency graph, same-case physical roles, CTA-load sensitivity, controlled interaction groups, edge-level mediation, and explicit prohibition on direct intervention regressors. |
| r1 | 2026-08-31 17:24 -0700 | Converted the request into five implementation slices with explicit ownership, fail-closed validation, immutable evidence, and verification criteria. |
