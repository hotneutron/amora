# NVIDIA Baseline Kernel Methodology

## Scope

This document defines the baseline NVIDIA probe methodology under the current AMORA
hardware-first, simulator-assisted validation model.

Baseline probes cover the fastest, lowest-risk path to useful NVIDIA hardware profiles:

- topology and occupancy,
- arithmetic latency,
- arithmetic throughput,
- shared-memory latency,
- shared-memory bank and broadcast behavior.

Baseline is not just a collection of kernels. It is the first end-to-end test of the
AMORA evidence pipeline:

```text
published/CUDA facts
-> raw tool and timing observations
-> normalized hardware measurements
-> NVIDIA backend interpretation
-> simulator mapping contract
-> simulator-equivalent estimate with fit metadata
```

## Revision History

### 2026-06-18: Layered Evidence Refresh

Source inputs:

- `.plan/probing-suite-microarchitecture-plan.md`
- `.plan/nvidia-probe-semantic-measurement-gap-plan.md`
- Previous baseline methodology document

Major changes:

- Replaced absolute workspace paths with repo-relative paths.
- Updated baseline methodology to follow the hardware-first, simulator-assisted methodology.
- Made CUDA metadata, published facts, NCU/CUPTI counters, microkernel timing,
  NVBit streams, disassembly, and simulator traces separate evidence layers.
- Changed the old rule "NCU/CUPTI counters are supporting evidence" to the
  current rule: counters are primary only when the metric contract has direct
  semantics; otherwise they validate or constrain fits.
- Added explicit layered output requirements for every baseline probe.
- Added fit status, uncertainty category, scalar-output policy, rejection rules,
  downgrade rules, and simulator trace hooks.

Superseded assumptions:

- Superseded: Baseline outputs should mainly be raw measurements plus confidence.
  Replacement: Baseline outputs must preserve raw observations, normalized
  measurements, backend interpretations, and simulator estimates separately.

- Superseded: microkernel timing is the default primary evidence.
  Replacement: primary evidence is selected by semantic match. CUDA metadata or
  direct counters can be primary; timing is workload evidence, cross-check, or
  fallback unless timing behavior itself is the target.

## Common Baseline Contract

Every baseline probe must define:

- hardware-neutral concept,
- NVIDIA-specific interpretation,
- target simulator parameters,
- required backend capabilities,
- primary evidence,
- validation evidence,
- timing and profiler execution modes,
- required SASS/disassembly pattern where applicable,
- clock-domain policy,
- scalar-output policy,
- fit status and uncertainty category,
- rejection and downgrade rules,
- fallback behavior.

Every baseline result must emit these layers:

- `raw_observation`
- `normalized_measurement`
- `backend_interpretation`
- `simulator_estimate`

Required shared fields:

- probe ID,
- source hash,
- binary hash when available,
- disassembly hash when available,
- launch configuration,
- CUDA device identity,
- driver/runtime/tool versions,
- clock domain and clock source,
- metric names and units,
- variance summary,
- assumptions,
- `coupled_with`,
- unsupported or downgrade reason.

## Evidence And Risk Policy

Evidence tiers:

- `published_fact`
- `direct_metadata`
- `direct_counter`
- `tool_derived_counter`
- `instrumented_stream`
- `timing_direct`
- `simulator_trace`
- `coupled_inference`
- `unsupported`

Fit status values:

- `direct`
- `uniquely_identified`
- `bounded`
- `conditionally_identified`
- `underconstrained`
- `behavioral_only`
- `unsupported`

Uncertainty categories:

- `stable_scalar`
- `bounded_range`
- `conditional_scalar`
- `multi_fit`
- `behavioral_class`
- `indeterminate`

Risk scale:

- Low: published fact, CUDA metadata, or direct metric with a close semantic
  match.
- Medium: stable timing/counter behavior with limited coupling and verified
  instruction stream.
- High: result depends on hidden scheduler, operand, memory, compiler, or tool
  behavior.

## Probe: `topology/device_attributes.py`

### Concept

Topology limits and runtime-visible occupancy resources.

### Target Parameters

- `gpgpu_sim_config::num_shader()`
- `shader_core_config::warp_size`
- `shader_core_config::max_warps_per_shader`
- `shader_core_config::max_cta_per_core`
- `shader_core_config::gpgpu_shader_registers`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`

### Primary Evidence

- CUDA runtime and driver device attributes.
- Published NVIDIA device specifications when available.

### Validation Evidence

- NCU launch metadata.
- CUDA occupancy API predictions.
- `topology/persistent_cta.cu` for selected launch shapes.
- Simulator occupancy-state trace.

### Methodology

1. Query CUDA device identity, compute capability, UUID, driver version, and
   runtime version.
2. Query SM count, warp size, max threads per block, max threads per SM, max
   blocks per SM, registers per SM/block, shared memory per SM/block, clock
   metadata, memory clock metadata, and opt-in shared-memory limits.
3. Query occupancy helper APIs for representative kernels.
4. Query NCU launch metadata where available.
5. Record every attribute's source and unit.
6. Emit metadata-backed simulator estimates only through mapping contracts.

### Scalar Policy

Allowed:

- SM count,
- warp size,
- metadata-backed resident limits,
- metadata-backed register/shared-memory limits.

Not directly allowed:

- SIMT cluster decomposition unless table-backed,
- physical scheduler/subcore counts unless independently supported.

### Fit And Uncertainty

- Expected fit status: `direct` for direct metadata.
- Expected uncertainty: `stable_scalar`.
- Cluster decomposition: `conditionally_identified`, `bounded`, or
  `unsupported`.

### Rejection And Downgrade

Reject an attribute if the CUDA query fails or returns zero/invalid values.
Downgrade if runtime metadata conflicts with NCU launch metadata or if a device
mode changes available resources.

### Risk

Low for exposed limits; medium for inferred topology decomposition.

## Probe: `topology/persistent_cta.cu`

### Concept

Runtime-observed resident CTA capacity under controlled launch shapes.

### Target Parameters

- `shader_core_config::max_cta_per_core`
- `shader_core_config::max_warps_per_shader`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`
- `shader_core_config::gpgpu_shader_registers`

### Primary Evidence

- Runtime-observed residency from controlled persistent CTA kernels.

### Validation Evidence

- CUDA occupancy API predictions.
- CUDA metadata.
- NCU launch metadata.
- Simulator occupancy trace.

### Methodology

1. Launch more CTAs than can reside concurrently.
2. Have each CTA record entry timestamp, block ID, and SM ID when `%smid` is
   available.
3. Hold CTAs in a bounded spin window to observe maximum simultaneous residency.
4. Sweep threads per block, dynamic shared memory, and register-pressure
   variants.
5. Record `ptxas` resource usage and disassembly hash for every variant.
6. Fit resource cliffs against metadata limits.

### Required SASS/Metadata Checks

- Register-pressure variants must preserve intended register usage.
- Shared-memory allocation must match launch metadata.
- Spin loop must not be optimized away.

### Scalar Policy

Allowed as cross-check or conditionally identified runtime limit. Prefer direct
metadata for final scalar when metadata and persistent CTA agree.

### Fit And Uncertainty

- Expected fit status: `direct` for matching metadata/runtime observations;
  `conditionally_identified` when resource cliffs infer the limiting resource.
- Expected uncertainty: `stable_scalar` or `conditional_scalar`.

### Rejection And Downgrade

Reject if watchdog, timeout, failed SM ID capture, or invalid synchronization
corrupts the run. Downgrade if atomic contention or scheduling order prevents a
stable maximum.

### Risk

Medium. This measures runtime admission behavior, not raw physical resource
counts.

## Probe: `topology/occupancy.py`

### Concept

Occupancy-resource fitting across launch dimensions.

### Target Parameters

- `shader_core_config::max_warps_per_shader`
- `shader_core_config::max_cta_per_core`
- `shader_core_config::gpgpu_shader_registers`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`

### Primary Evidence

- CUDA metadata and occupancy API models.

### Validation Evidence

- Persistent CTA measurements.
- `ptxas` resource reports.
- NCU launch metadata.

### Methodology

1. Generate launch configurations across block sizes and dynamic shared-memory
   sizes.
2. Compile register-pressure variants and record resource usage.
3. Predict occupancy using CUDA occupancy APIs.
4. Run persistent CTA checks at selected boundaries.
5. Fit the tightest limiting resource per launch point.
6. Preserve metadata estimates separately from runtime-observed estimates.

### Scalar Policy

Allowed for direct metadata limits; conditional for fitted resource explanations.

### Fit And Uncertainty

- Expected fit status: `direct` or `conditionally_identified`.
- Expected uncertainty: `stable_scalar` or `conditional_scalar`.

### Risk

Low to medium. Compiler version and resource allocation can shift boundaries.

## Probe: `arithmetic_latency/dependent_chain.cu`

### Concept

Dependent instruction latency for scalar arithmetic semantic classes.

### Target Parameters

- `shader_core_config::max_sp_latency`
- `shader_core_config::max_int_latency`
- `shader_core_config::max_sfu_latency`
- `shader_core_config::max_dp_latency`

### Primary Evidence

- Direct NCU/CUPTI instruction and active-cycle metrics when a logical metric
  maps directly to the target SASS opcode class.
- Otherwise validated dependent-chain timing.

### Validation Evidence

- Disassembly dependency-chain verification.
- NVBit opcode histogram or dynamic instruction stream.
- Timing baseline subtraction.
- Active-warp sweep.
- Simulator pipeline and scoreboard trace.

### Methodology

1. Generate dependent chains for FP32, INT, SFU, and FP64 where supported.
2. Use inline PTX or constrained source patterns, then verify emitted SASS.
3. Keep memory operations outside the timed hot loop.
4. Use enough iterations/unrolling to dominate timer overhead.
5. Measure timer overhead with empty-chain baselines.
6. Run separate counter, timing, and NVBit validation modes.
7. Normalize in the SM clock domain and record clock source.
8. Emit one backend interpretation per opcode semantic class before mapping to
   simulator latency fields.

### Required SASS Pattern

- Target opcode appears with expected count.
- Destination feeds next source.
- No unexpected memory operation inside the timed region.
- No replacement by alternate opcode without semantic record update.

### Scalar Policy

Allow scalar latency only when opcode, counters, timing, and variance agree.
Otherwise emit `conditional_scalar` or `bounded_range`.

### Fit And Uncertainty

- Expected fit status: `direct` or `conditionally_identified`.
- Expected uncertainty: `stable_scalar` or `conditional_scalar`.

### Rejection And Downgrade

Reject if the dependency chain is optimized away, broken, or compiled to an
unexpected opcode. Downgrade if latency changes strongly with occupancy or stall
metrics show unrelated bottlenecks.

### Risk

Medium. The instruction stream can be verified, but scheduler and scoreboard
effects can still contaminate timing.

## Probe: `arithmetic_latency/independent_chains.cu`

### Concept

Reciprocal throughput and effective functional-unit/pipeline throughput.

### Target Parameters

- `gpgpu_num_sp_units`
- `gpgpu_num_int_units`
- `gpgpu_num_sfu_units`
- `gpgpu_num_dp_units`
- `shader_core_config::pipe_widths`
- `shader_core_config::pipeline_widths_string`

### Primary Evidence

- NCU/CUPTI instruction counts, active cycles, pipe utilization, and issue
  metrics when metric contracts are direct.
- Independent-chain saturation microkernels.

### Validation Evidence

- NVBit opcode histogram.
- Disassembly opcode count.
- Timing plateau.
- Simulator pipeline trace.

### Methodology

1. Generate independent streams for each operation class.
2. Sweep chains per warp, active warps, CTAs per SM, and unroll factor.
3. Collect counters in profiler runs and timing in separate low-overhead runs.
4. Fit the throughput plateau before inferring simulator units or widths.
5. Record clock domain, active SM count, and throttling indicators.
6. Map measured throughput to simulator parameters only through coupling-aware
   contracts.

### Scalar Policy

Throughput plateau can be a stable scalar. Functional-unit count and pipeline
width are conditional or coupled unless published or independently validated.

### Fit And Uncertainty

- Expected fit status: `conditionally_identified` for unit counts.
- Expected uncertainty: `conditional_scalar` or `multi_fit`.

### Rejection And Downgrade

Reject if opcode mix differs from the intended semantic class. Downgrade if
plateau is absent, clocks vary substantially, or memory/operand stalls dominate.

### Risk

Medium to high. Throughput couples with scheduler issue, operand delivery, and
clock behavior.

## Probe: `shared_memory/pointer_chase.cu`

### Concept

Dependent shared-memory load latency.

### Target Parameters

- `gpgpu_smem_latency`
- `memory_shared_memory_minimum_latency`

### Primary Evidence

- Direct shared-memory NCU/CUPTI metrics if they map to the access pattern.
- Validated pointer-chase timing when no direct latency counter exists.

### Validation Evidence

- SASS shared-load verification.
- NCU/CUPTI shared-memory transactions and conflicts.
- Conflict-free and conflict-heavy controls.
- Simulator shared-memory trace.

### Methodology

1. Build a linked list in shared memory.
2. Use previous load value as the next address to enforce dependency.
3. Sweep list size, access stride, active lanes, active warps, and access width.
4. Time only the hot pointer-chase loop.
5. Collect shared-memory transaction/conflict metrics in separate profiler runs.
6. Compare against known bank-conflict patterns from `shared_memory/bank_stride.cu`.

### Scalar Policy

Allow scalar minimum latency only for conflict-free, stable, verified access
patterns. Emit range or conditional scalar when bank conflicts or warp pressure
affect latency.

### Fit And Uncertainty

- Expected fit status: `direct`, `bounded`, or `conditionally_identified`.
- Expected uncertainty: `stable_scalar`, `bounded_range`, or
  `conditional_scalar`.

### Risk

Medium. Shared-memory latency is coupled with bank mapping, access width, and
warp scheduling.

## Probe: `shared_memory/bank_stride.cu`

### Concept

Shared-memory bank count, bank mapping periodicity, warp partitioning, and
broadcast/multicast behavior.

### Target Parameters

- `shader_core_config::gpgpu_shmem_num_banks`
- `gpgpu_shmem_limited_broadcast`
- `gpgpu_shmem_warp_parts`

### Primary Evidence

- NCU/CUPTI shared-memory transaction/conflict metrics when direct.
- Bank-stride microkernel curves.

### Validation Evidence

- Timing impact.
- SASS shared instruction and access-width verification.
- Simulator shared-memory bank trace.

### Methodology

1. Generate warp-level shared-memory accesses with controlled lane addresses.
2. Sweep stride in bytes and elements.
3. Include uniform-address, contiguous, power-of-two, prime-stride, half-warp,
   and quarter-warp patterns.
4. Run 32-bit, 64-bit, and vector-width variants.
5. Fit periodic conflict peaks.
6. Compare periodicity across access widths and active-lane masks.

### Scalar Policy

Allow bank-count scalar if periodicity is stable across repeats and access
widths. Emit broadcast and warp-partition behavior as `behavioral_class` unless
counter evidence is strong.

### Fit And Uncertainty

- Expected fit status: `uniquely_identified` or `bounded` for bank count;
  `behavioral_only` for policy details.
- Expected uncertainty: `stable_scalar`, `bounded_range`, or
  `behavioral_class`.

### Rejection And Downgrade

Reject scalar bank count if conflict peaks are unstable or incompatible across
access widths. Downgrade if counters report transactions but cannot isolate
conflicts.

### Risk

Low to medium for bank count; medium for broadcast and partition policy.

## Probe: `shared_memory/analyze.py`

### Concept

Shared-memory fitting and simulator mapping.

### Primary Evidence

- Raw pointer-chase and bank-stride curves.
- Shared-memory counters.
- Disassembly records.

### Methodology

1. Compute conflict-free baseline latency.
2. Detect stride-periodic peaks.
3. Fit candidate bank counts and warp partitions.
4. Compare uniform, contiguous, and conflict-heavy patterns.
5. Emit raw curves, normalized measurements, backend interpretation, and
   simulator estimates separately.
6. Record `alternative_fits` and `coupled_with` when multiple explanations fit.

### Scalar Policy

Emit scalar only when the scalar-output policy of the source probes allows it.

### Risk

Medium. The analyzer is where false precision is most likely, so it must keep
raw behavior and simulator mapping separate.

## Baseline Implementation Order

1. Implement `topology/device_attributes.py` first because CUDA metadata and
   published facts provide the lowest-risk direct evidence.
2. Add `topology/occupancy.py` and `topology/persistent_cta.cu` to cross-check
   runtime-visible resident limits.
3. Add arithmetic latency probes with strict SASS validation.
4. Add arithmetic throughput probes after latency baselines are stable.
5. Add `shared_memory/pointer_chase.cu` for conflict-free latency baselines.
6. Add `shared_memory/bank_stride.cu` and `shared_memory/analyze.py` for bank
   and broadcast behavior.

## Required Simulator Trace Hooks

Baseline needs simulator instrumentation for:

- resident CTA and warp state,
- register and shared-memory resource allocation,
- scheduler-visible active warp state,
- arithmetic pipeline issue and completion,
- shared-memory bank selection,
- shared-memory conflict and broadcast events.

Simulator traces are direct observations of simulator state. They are not proof
that NVIDIA hardware has the same internal state; they define the target side
of the mapping contract.

## Reporting Requirements

Every baseline report must include:

- evidence tier,
- fit status,
- uncertainty category,
- variance summary,
- metric resolver record when profiler metrics are used,
- SASS validation record when instruction behavior matters,
- launch and occupancy descriptor,
- clock-domain record,
- simulator mapping contract,
- rejection or downgrade reason when applicable.

## Baseline Acceptance Criteria

Baseline is complete when AMORA can report:

- metadata-backed topology limits,
- persistent-CTA occupancy cross-checks,
- arithmetic latency for supported FP32, INT, SFU, and FP64 semantic classes,
- arithmetic throughput plateaus for at least FP32 and INT,
- shared-memory minimum latency,
- shared-memory bank-count estimate or explicit indeterminate status,
- layered outputs for every probe,
- fit status, uncertainty, variance, assumptions, and `coupled_with` for every
  simulator estimate,
- explicit unsupported, rejected, or downgraded status where evidence is weak.
