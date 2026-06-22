# NVIDIA Baseline Probe Implementation Plan

## Scope

This document turns the internal priority-track methodology into an execution
plan for building the first runnable NVIDIA baseline probe stack.

Baseline implementation covers:

- CUDA/backend capability discovery,
- layered result schemas,
- topology and occupancy probes,
- arithmetic latency probes,
- arithmetic throughput probes,
- shared-memory latency probes,
- shared-memory bank and broadcast probes,
- report generation and validation gates.

The first implementation target is a usable local probe runner that can emit
layered JSON records even on machines without an NVIDIA GPU. GPU-dependent
probe execution should be capability-gated, skipped cleanly when unsupported,
and testable through mocked tool outputs.

## Revision History

### 2026-06-18: Initial Implementation Plan

Source inputs:

- `.plan/probing-suite-microarchitecture-plan.md`
- `.plan/nvidia-probe-semantic-measurement-gap-plan.md`
- Internal priority-track methodology
- `RULES.md`

Major decisions:

- Build the baseline NVIDIA probe stack as a hardware-first probe stack with
  simulator-assisted mapping contracts, not as a direct simulator-parameter
  emitter.
- Preserve `raw_observation`, `normalized_measurement`,
  `backend_interpretation`, and `simulator_estimate` as separate layers.
- Treat CUDA metadata and direct NCU/CUPTI metrics as primary evidence when
  their semantic contract is direct.
- Treat timing as primary only when timing behavior is the target, otherwise as
  validation or fallback evidence.
- Start with a committed source scaffold because the current branch contains
  planning docs and generated packaging artifacts but no committed `amora/` or
  `tests/` source tree.

## Implementation Principles

- Keep every emitted value attached to its evidence tier, fit status,
  uncertainty category, variance summary, and downgrade reason.
- Never collapse hardware observations and simulator estimates into the same
  field.
- Prefer direct metadata and direct counters over timing when their semantics
  match the target behavior.
- Record tool versions, source hashes, binary hashes, disassembly hashes, launch
  descriptors, and metric resolver decisions.
- Make every GPU-facing operation capability-gated so CI and development can
  run without CUDA hardware.
- Keep baseline probes narrow and reproducible. Avoid later-priority behavior
  except as explicit validation hooks.

## Proposed Source Layout

```text
amora/
  __init__.py
  cli.py
  schemas/
    __init__.py
    evidence.py
    results.py
  reports/
    __init__.py
    json_report.py
  backends/
    __init__.py
    nvidia/
      __init__.py
      cuda.py
      ncu.py
      cupti.py
      nvbit.py
      disasm.py
      metrics.py
  probes/
    __init__.py
    nvidia/
      __init__.py
      baseline/
        __init__.py
        topology/
          __init__.py
          device_attributes.py
          occupancy.py
          persistent_cta.cu
        arithmetic_latency/
          __init__.py
          dependent_chain.py
          dependent_chain.cu
        arithmetic_throughput/
          __init__.py
          independent_chains.py
          independent_chains.cu
        shared_memory/
          __init__.py
          pointer_chase.py
          pointer_chase.cu
          bank_stride.py
          bank_stride.cu
          analyze.py
tests/
  schemas/
  backends/
  probes/
```

CUDA files are probe kernels. Python files own planning, build orchestration,
tool execution, parsing, normalization, and reporting.

## Milestone 0: Package And Test Scaffold

### Deliverables

- `pyproject.toml` with package metadata, dependencies, and `amora` CLI entry.
- `amora/` Python package scaffold.
- `tests/` scaffold with unit tests that run without CUDA.
- `.gitignore` entries for generated artifacts if missing.

### CLI Shape

```text
amora nvidia list
amora nvidia inspect-capabilities
amora nvidia run --probe topology.device_attributes --output out/nvidia-baseline.json
amora nvidia run --all --output out/nvidia-baseline.json
```

### Acceptance Gate

- `python -m pytest tests -q` passes without NVIDIA hardware.
- `amora nvidia list` shows all planned baseline probes.
- Source package does not rely on generated `amora.egg-info/`.

## Milestone 1: Common Schemas And Evidence Model

### Deliverables

- Enum-like constants for evidence tiers:
  `published_fact`, `direct_metadata`, `direct_counter`,
  `tool_derived_counter`, `instrumented_stream`, `timing_direct`,
  `simulator_trace`, `coupled_inference`, `unsupported`.
- Enum-like constants for fit status:
  `direct`, `uniquely_identified`, `bounded`,
  `conditionally_identified`, `underconstrained`, `behavioral_only`,
  `unsupported`.
- Enum-like constants for uncertainty categories:
  `stable_scalar`, `bounded_range`, `conditional_scalar`, `multi_fit`,
  `behavioral_class`, `indeterminate`.
- Dataclasses or Pydantic models for:
  - `ProbeIdentity`
  - `ToolContext`
  - `LaunchDescriptor`
  - `RawObservation`
  - `NormalizedMeasurement`
  - `BackendInterpretation`
  - `SimulatorEstimate`
  - `ProbeResult`

### Required Result Invariants

- Every `ProbeResult` has all four layers, even when a layer is empty or
  unsupported.
- Unsupported probes emit structured `unsupported` records instead of raising
  unhandled runtime errors.
- Simulator estimates always reference a mapping contract and source evidence.
- Scalar estimates require explicit scalar policy approval.

### Tests

- JSON serialization round trip.
- Unsupported result construction.
- Validation that simulator estimates cannot be emitted without evidence,
  assumptions, fit status, and uncertainty category.

## Milestone 2: NVIDIA Backend Capability Discovery

### Deliverables

- `amora/backends/nvidia/cuda.py`
  - discover `nvcc`,
  - discover CUDA runtime and driver through `nvidia-smi` or CUDA APIs when
    available,
  - query device list when CUDA is available,
  - report clean unsupported state when CUDA is unavailable.
- `amora/backends/nvidia/ncu.py`
  - discover `ncu`,
  - list supported metric names when available,
  - expose command builder without executing by default.
- `amora/backends/nvidia/disasm.py`
  - discover `nvdisasm` and `cuobjdump`,
  - hash binaries and disassembly output,
  - parse minimal opcode summaries.
- `amora/backends/nvidia/metrics.py`
  - define logical baseline metric names,
  - map logical names to candidate NCU/CUPTI names,
  - record resolver decision, unit, and fallback reason.

### Capability Record

The NVIDIA backend should emit:

- CUDA availability,
- GPU availability,
- compiler availability,
- disassembler availability,
- NCU availability,
- CUPTI availability if detectable,
- NVBit availability if configured,
- supported timing sources,
- supported profiler metrics,
- unsupported reasons.

### Tests

- Mocked tool discovery.
- Command builder tests.
- Metric resolver fallback tests.
- Unsupported capability record tests.

## Milestone 3: Build And Artifact Manager

### Deliverables

- CUDA build helper that compiles probe kernels into an ignored artifact
  directory such as `out/build/nvidia/baseline/`.
- Source hash, compile command, binary hash, and optional disassembly hash
  capture.
- Build configuration object:
  - architecture target,
  - optimization level,
  - include paths,
  - register-pressure options,
  - debug/disassembly options.

### Rules

- Build artifacts must not be committed.
- Failed compilation emits a structured unsupported or rejected result.
- Every timing-capable binary must have a source hash and compile command.

### Tests

- Build command construction.
- Hash calculation.
- Failed compile handling through mocked subprocess output.

## Milestone 4: Topology Metadata Probe

### Files

- `amora/probes/nvidia/baseline/topology/device_attributes.py`

### Implementation Steps

1. Query CUDA-visible device identity and resource limits.
2. Parse stable metadata:
   - SM count,
   - warp size,
   - max threads per block,
   - max threads per SM,
   - max blocks per SM,
   - registers per SM/block,
   - shared memory per SM/block,
   - clock metadata,
   - memory clock metadata,
   - compute capability,
   - UUID when available.
3. Attach published facts when a curated table exists.
4. Emit direct metadata results.
5. Mark cluster decomposition and physical scheduler counts unsupported unless
   backed by a table or later evidence.

### Acceptance Gate

- Runs on CUDA systems.
- Emits structured unsupported output on non-CUDA systems.
- Produces `direct_metadata`, `direct`, `stable_scalar` records for exposed
  metadata.

## Milestone 5: Occupancy Cross-Check

### Files

- `amora/probes/nvidia/baseline/topology/occupancy.py`
- `amora/probes/nvidia/baseline/topology/persistent_cta.cu`

### Implementation Steps

1. Implement occupancy API wrapper for representative kernels.
2. Add persistent CTA CUDA kernel that records CTA entry, `%smid` when
   available, timestamp, and live residency.
3. Sweep:
   - block size,
   - dynamic shared memory,
   - register-pressure variant,
   - CTA count.
4. Compare observed residency with metadata and occupancy API predictions.
5. Emit runtime-observed results as validation evidence and conditional
   simulator estimates.

### Rejection Rules

- Reject if `%smid` capture fails when SM-level attribution is required.
- Reject if watchdog or timeout truncates the residency window.
- Downgrade if atomic contention prevents stable maximum residency.

### Acceptance Gate

- Metadata-backed occupancy emits direct records.
- Persistent CTA emits `conditionally_identified` records only when stable.
- Report keeps metadata, runtime observation, and simulator estimate separate.

## Milestone 6: Arithmetic Latency

### Files

- `amora/probes/nvidia/baseline/arithmetic_latency/dependent_chain.py`
- `amora/probes/nvidia/baseline/arithmetic_latency/dependent_chain.cu`

### Implementation Steps

1. Generate dependent chains for supported semantic classes:
   - FP32 add, mul, fma,
   - INT add, mul, logic,
   - SFU operations,
   - FP64 add, mul, fma where supported.
2. Compile variants with stable flags.
3. Disassemble and verify:
   - expected opcode count,
   - destination-to-source dependency chain,
   - no memory operations inside the timed loop,
   - no unexpected opcode substitution.
4. Run timing mode with enough iterations to dominate overhead.
5. Run profiler mode when direct instruction and cycle metrics are available.
6. Normalize to cycles per operation in the SM clock domain.
7. Emit one record per opcode semantic class.

### Acceptance Gate

- Invalid SASS pattern is rejected.
- Empty-loop overhead is recorded.
- Latency scalar is emitted only when opcode validation, variance, and metric
  checks pass.

## Milestone 7: Arithmetic Throughput

### Files

- `amora/probes/nvidia/baseline/arithmetic_throughput/independent_chains.py`
- `amora/probes/nvidia/baseline/arithmetic_throughput/independent_chains.cu`

### Implementation Steps

1. Generate independent chains for the same arithmetic semantic classes used by
   latency probes.
2. Sweep:
   - independent chains per warp,
   - active warps,
   - CTAs per SM,
   - unroll factor.
3. Collect instruction counts, pipe utilization, active cycles, and issue
   metrics where direct counters are available.
4. Run timing mode separately to avoid profiler replay effects.
5. Fit throughput plateau before inferring any simulator unit or width.
6. Keep raw throughput as first-class measurement even when functional-unit
   decomposition is underconstrained.

### Acceptance Gate

- Throughput plateau has variance summary.
- Functional-unit count is marked conditional or coupled unless independently
  validated.
- Result explains whether the bottleneck is throughput, scheduler, operand
  delivery, clock instability, or unsupported.

## Milestone 8: Shared-Memory Latency

### Files

- `amora/probes/nvidia/baseline/shared_memory/pointer_chase.py`
- `amora/probes/nvidia/baseline/shared_memory/pointer_chase.cu`

### Implementation Steps

1. Build shared-memory pointer-chase lists.
2. Use dependency chains so each load chooses the next address.
3. Sweep:
   - list size,
   - stride,
   - active lanes,
   - active warps,
   - access width.
4. Verify shared-load SASS.
5. Collect shared-memory transaction/conflict counters where available.
6. Emit conflict-free minimum latency only when access pattern and counters are
   stable.

### Acceptance Gate

- Conflict-free baseline is separated from conflict-heavy controls.
- Scalar latency is emitted only for stable, verified access patterns.
- Bank-conflict contamination produces bounded or conditional output.

## Milestone 9: Shared-Memory Bank And Broadcast Behavior

### Files

- `amora/probes/nvidia/baseline/shared_memory/bank_stride.py`
- `amora/probes/nvidia/baseline/shared_memory/bank_stride.cu`
- `amora/probes/nvidia/baseline/shared_memory/analyze.py`

### Implementation Steps

1. Generate warp-level lane-address patterns:
   - uniform address,
   - contiguous,
   - power-of-two strides,
   - prime strides,
   - half-warp patterns,
   - quarter-warp patterns.
2. Run 32-bit, 64-bit, and vector-width variants.
3. Collect timing and shared-memory transaction/conflict metrics.
4. Fit periodic conflict peaks.
5. Infer bank-count candidates and warp-partition candidates.
6. Classify broadcast and multicast behavior as behavioral unless direct
   metrics strongly support scalar claims.
7. Preserve alternative fits when several mappings explain the curves.

### Acceptance Gate

- Bank-count scalar requires stable periodicity across repeats and widths.
- Broadcast behavior defaults to `behavioral_class`.
- Analyzer records `alternative_fits` and `coupled_with` when needed.

## Milestone 10: Report Runner And Aggregation

### Deliverables

- `amora/reports/json_report.py`
- baseline aggregate runner.
- Stable JSON output format for a full baseline run.

### Required Report Sections

- backend capability record,
- probe list and skipped probes,
- raw observations,
- normalized measurements,
- NVIDIA backend interpretations,
- simulator estimates,
- metric resolver decisions,
- SASS validation records,
- timing variance summaries,
- downgrade and rejection reasons.

### Acceptance Gate

- Full baseline report can be generated with mixed supported and unsupported probes.
- Report schema is stable enough to become input for later P1/P2/P3 analysis.
- No report field requires a hidden global context to interpret units or
  evidence.

## Milestone 11: Test Strategy

### Unit Tests

- schema construction and validation,
- JSON serialization,
- capability discovery with mocked tools,
- metric resolver mapping and fallback,
- command construction,
- parser behavior for mocked CUDA, NCU, and disassembly outputs,
- downgrade and rejection paths.

### Golden Tests

- fixture-based topology metadata output,
- fixture-based NCU metric output,
- fixture-based disassembly summaries,
- fixture-based baseline report JSON.

### Hardware Smoke Tests

Hardware tests should be opt-in and skipped by default unless CUDA is detected.

Suggested markers:

```text
pytest -m "cuda"
pytest -m "ncu"
pytest -m "nvbit"
```

### Acceptance Gate

- Non-hardware tests pass in normal development environments.
- Hardware tests emit device identity and tool versions.
- Hardware tests never silently convert unsupported evidence into scalar
  estimates.

## Milestone 12: Documentation And Developer Workflow

### Deliverables

- `docs/development/nvidia-baseline-probes.md`
- CLI examples for capability discovery and one-probe execution.
- Explanation of output layers and evidence tiers.
- Troubleshooting section for missing CUDA, missing NCU, missing disassembler,
  and unsupported metrics.

### Acceptance Gate

- A developer can run one metadata-only probe without a GPU.
- A developer with CUDA can run topology and occupancy probes.
- A developer with NCU can run counter-backed arithmetic and shared-memory
  probes.

## Implementation Order

1. Milestone 0: package and test scaffold.
2. Milestone 1: common schemas and evidence model.
3. Milestone 2: NVIDIA backend capability discovery.
4. Milestone 3: build and artifact manager.
5. Milestone 4: topology metadata probe.
6. Milestone 5: occupancy cross-check.
7. Milestone 6: arithmetic latency.
8. Milestone 7: arithmetic throughput.
9. Milestone 8: shared-memory latency.
10. Milestone 9: shared-memory bank and broadcast behavior.
11. Milestone 10: report runner and aggregation.
12. Milestone 11: test strategy hardening.
13. Milestone 12: developer documentation.

## Initial Cutline

The first useful merge should include:

- package scaffold,
- result schemas,
- NVIDIA capability discovery,
- topology metadata probe,
- JSON report output,
- mocked unit tests,
- clean unsupported behavior on non-CUDA machines.

This cutline gives AMORA a runnable baseline spine before adding CUDA kernels.

## Full Baseline Completion Criteria

The baseline implementation is complete when AMORA can:

- discover NVIDIA backend capabilities,
- compile and hash baseline CUDA probe binaries,
- verify relevant SASS patterns,
- run topology metadata probes,
- run occupancy cross-check probes,
- run arithmetic latency probes,
- run arithmetic throughput probes,
- run shared-memory latency probes,
- run shared-memory bank/broadcast probes,
- emit layered JSON records for every probe,
- mark unsupported or weak evidence explicitly,
- preserve raw observations separately from simulator estimates,
- pass non-hardware tests by default,
- pass opt-in CUDA smoke tests on a supported NVIDIA machine.

## Open Decisions

- Whether to use dataclasses only or add Pydantic for schema validation.
- Whether CUDA metadata should be queried first through Python bindings, a small
  compiled helper, or `nvidia-smi` plus CUDA sample binaries.
- Whether NCU integration should begin as CLI parsing or CUPTI Range Profiling
  should be introduced early.
- Where to store curated published NVIDIA facts for trust-and-verify metadata
  anchors.
- How much simulator trace schema should be implemented during the baseline cutline versus left
  as a declared mapping contract for later simulator integration.
