# Probing Suite Microarchitecture Plan

## Summary

Build AMORA as a portable probing suite that turns real accelerator observations
into trustworthy simulator-facing calibration data. The suite should support
NVIDIA first, but its core design must generalize across GPUs, NPUs, TPUs, and
other accelerators with different levels of ISA visibility, profiler support,
timer semantics, memory hierarchy detail, and public documentation.

The core design principle is layered evidence:

```text
published facts
-> backend capabilities
-> raw observations
-> normalized hardware-neutral measurements
-> backend-specific interpretations
-> simulator mapping contracts
-> fitted simulator-equivalent estimates
```

The suite must not collapse these layers into one "parameter estimate" too
early. A measurement can be useful even when no safe simulator scalar can be
emitted.

## Revision History

### 2026-06-18: Full Measurement-Contract Rewrite

Source inputs:

- Original umbrella plan: `.plan/probing-suite-microarchitecture-plan.md`
- Reaction document:
  `.plan/20260618_probing-suite-microarchitecture-plan_comments.md`
- NVIDIA-specific canonical plan:
  `.plan/nvidia-probe-semantic-measurement-gap-plan.md`
- Follow-up design decisions from the generalized-plan discussion.

Major changes:

- Rewrote the plan from scratch around measurement contracts instead of a flat
  probe-family roadmap.
- Applied the NVIDIA-plan decisions here as general AMORA policy:
  NCU/CUPTI-like counters may be primary evidence when semantically direct;
  microkernel timing is essential but not universally primary; published
  parameters are trust-and-verify anchors; simulator states are directly
  observable inside the simulator; fitted estimates need fit metadata.
- Separated portable measurement concepts from backend-specific implementation
  and simulator-specific mapping.
- Added explicit handling for different levels of information availability:
  published specs, ISA-visible execution, profiler-visible metrics, binary-only
  execution, runtime-only metadata, and black-box timing.
- Added an ISA semantics layer that records instruction availability, semantic
  equivalence, architectural block relationships, and confidence in mapping an
  instruction to a hardware subsystem.
- Added a profiler/tool support strategy that is vendor layered. NVIDIA NCU,
  CUPTI, and NVBit are one backend instance, not global abstractions.
- Replaced the old `ISABackend`-only framing with a richer backend model:
  capability discovery, ISA semantics, profiler metrics, launch/runtime control,
  binary validation, and simulator trace integration.
- Added capability-gated probes so unsupported or weakly supported accelerator
  targets produce honest `unsupported`, `behavioral_only`, or `underconstrained`
  results instead of misleading simulator scalars.
- Added layered output schemas:
  `raw_observation`, `normalized_measurement`, `backend_interpretation`, and
  `simulator_estimate`.
- Reframed acceptance criteria away from requiring every backend to emit
  GPGPU-Sim-specific parameters. Portable milestones require honest measurement
  and mapping status; simulator scalars are emitted only when the mapping
  contract allows them.

### Superseded Assumptions From The Original Plan

- Superseded: "ISA-level microkernels are the primary basis for reverse
  engineering."
  Replacement: Use the evidence source with the closest semantic match. Direct
  counters, published parameters, runtime metadata, ISA microkernels, binary
  instrumentation, and simulator traces each have a role.

- Superseded: "`simulator_parameter_map.yaml` can be a family-to-parameter
  coverage list."
  Replacement: It must become a mapping-contract registry with observability,
  evidence, formula/fit, scalar-output policy, fallback, and known mismatches.

- Superseded: "One backend abstraction around assemble/launch/timer is enough."
  Replacement: Backends must declare trust-critical capabilities, ISA semantics,
  profiler support, validation routes, timer domains, memory hierarchy concepts,
  and mapping constraints.

- Superseded: "All accelerator backends should attempt the same simulator
  parameter outputs."
  Replacement: Backends emit portable measurements first. Simulator mapping is
  optional, capability-gated, and classified as direct, fitted, behavioral,
  underconstrained, or unsupported.

## Goals

AMORA should:

- discover accelerator and tool capabilities,
- identify available instruction semantics and their architectural relationships,
- generate or select probes appropriate for the backend capability level,
- collect raw observations from timing, profilers, metadata, instrumentation,
  and published sources,
- normalize those observations into hardware-neutral measurements,
- interpret them using backend-specific architecture knowledge,
- map them to simulator parameters only through explicit contracts,
- emit uncertainty, variance, fitting metadata, and rejection/downgrade status,
- keep unsupported or unidentifiable parameters explicit.

Non-goal:

- Claim exact hidden hardware structures when only behavioral evidence exists.

## Architecture

The suite has five layers.

```text
Layer 1: Hardware-neutral measurement concepts
Layer 2: Backend capabilities and ISA/profiler support
Layer 3: Backend-specific probe implementations
Layer 4: Backend interpretation and evidence fusion
Layer 5: Simulator mapping and fitting
```

### Layer 1: Hardware-Neutral Measurement Concepts

This layer defines concepts that can exist across accelerators without assuming
NVIDIA SIMT terminology or GPGPU-Sim parameter names.

Initial concepts:

- topology limits
- residency and occupancy-like capacity
- scalar instruction latency
- scalar instruction reciprocal throughput
- vector/SIMD/SIMT issue behavior
- tensor/matrix operation latency and throughput
- local/scratchpad/shared-memory latency
- local/scratchpad/shared-memory conflict behavior
- cache or cache-like capacity knees
- cache or cache-like transaction granularity
- global/external-memory latency
- global/external-memory bandwidth plateau
- memory coalescing or request formation behavior
- synchronization cost
- DMA/async-copy transfer behavior
- interconnect/fabric saturation behavior

These are measurements, not simulator parameters.

### Layer 2: Backend Capabilities And ISA/Profiler Support

Each backend must describe what can be trusted on the target.

Minimum capability schema:

```yaml
backend: nvidia_cuda
target:
  vendor: nvidia
  device_name: string
  architecture: string
  driver_version: string
  runtime_version: string
instruction_control:
  can_emit_low_level_isa: true
  can_verify_disassembly: true
  can_control_register_assignment: partial
  can_prevent_compiler_fusion: true
  can_validate_dynamic_instruction_stream: true
timing:
  has_device_timer: true
  timer_domain: sm_cycles
  timer_scope: per_sm_or_global
  timer_overhead_measurable: true
profiling:
  has_per_kernel_counters: true
  has_sampling_counters: true
  has_pc_sampling: true
  has_dynamic_binary_instrumentation: true
memory_control:
  can_select_address_space: true
  can_control_cache_policy: partial
  can_allocate_local_shared_memory: true
  can_control_alignment: true
simulator_mapping:
  target_simulator: gpgpu_sim_like
  simulator_state_trace_available: true
```

Capability values should support `true`, `false`, `partial`, and `unknown`.

### Layer 3: Backend-Specific Probe Implementations

Each hardware-neutral concept is implemented by one or more backend-specific
probes. A probe declares required and optional capabilities.

Example:

```yaml
probe: arithmetic_latency.dependent_chain
concept: scalar_instruction_latency
requires:
  - controllable_instruction_sequence
  - device_timer
  - disassembly_or_binary_verification
optional:
  - direct_instruction_counter
  - dynamic_instruction_counter
  - active_cycle_counter
```

Example:

```yaml
probe: register_file.bank_sweep
concept: operand_delivery_conflict_behavior
requires:
  - controllable_instruction_sequence
  - controllable_register_assignment
  - disassembly_or_binary_verification
optional:
  - stall_counters
  - dynamic_instruction_counter
```

If requirements are not met, the runner skips or downgrades the probe before
execution.

### Layer 4: Backend Interpretation And Evidence Fusion

Backend interpretation maps a normalized measurement onto a backend-specific
architectural fact or behavior.

Examples:

- NVIDIA SASS `FFMA` throughput interpreted as FP32 pipe behavior.
- AMD GCN/CDNA `v_add_f32` latency interpreted in wavefront execution context.
- Intel Xe metric interpreted against EU/Xe-core organization.
- TPU matrix-unit throughput interpreted against systolic or MXU semantics.

The backend interpretation must record:

- instruction or operation semantics,
- architectural block relationship,
- tool metrics used,
- timer domain,
- observed variance,
- known confounders,
- validation result.

### Layer 5: Simulator Mapping And Fitting

Simulator mapping converts backend-interpreted measurements into simulator
parameters only through mapping contracts.

A mapping contract defines:

- `parameter`
- `simulator_component`
- `hardware_behavior`
- `backend_interpretation_required`
- `observability`
- `primary_evidence`
- `validation_evidence`
- `formula_or_fit`
- `fit_status_required`
- `scalar_output_allowed`
- `known_mismatches`
- `fallback`

Example:

```yaml
parameter: shader_core_config::max_sp_latency
simulator_component: shader_core
hardware_behavior: dependent scalar FP32 operation latency
observability: direct_or_conditionally_identified
primary_evidence: backend_instruction_latency
validation_evidence:
  - direct_instruction_counter_if_available
  - disassembly_hash
  - dynamic_instruction_stream_if_available
formula_or_fit: cycles_per_instruction_after_overhead_subtraction_or_counter_normalization
fit_status_required: direct_or_conditionally_identified
scalar_output_allowed: true
known_mismatches:
  - instruction fusion
  - operand delivery stalls
  - scheduler effects at low occupancy
fallback: report normalized latency measurement only
```

Example:

```yaml
parameter: routing_delay
simulator_component: interconnect
hardware_behavior: latency increase under controlled injection pressure
observability: behavioral_only
primary_evidence: injection_rate_curve
validation_evidence:
  - fabric_or_partition_counters_if_available
  - simulator_trace
formula_or_fit: effective_latency_curve_fit
fit_status_required: behavioral_only_or_bounded
scalar_output_allowed: false
known_mismatches:
  - real fabric topology not represented by simulator router model
  - downstream memory backpressure can mimic routing delay
fallback: report saturation curve and no scalar estimate
```

## Evidence Policy

The evidence source with the closest semantic match should be primary.

| Evidence Source | Primary When | Validation When | Main Risk |
|---|---|---|---|
| Published parameter | Vendor exposes stable hardware or architectural value | Checking inferred values | Specs may omit mode-specific behavior |
| Runtime metadata | Runtime-visible limit maps directly to a concept | Checking profiler launch metadata | May expose policy, not physical structure |
| Direct profiler counter | Metric directly matches the behavior | Checking timing or fitted model | Naming, derivation, replay, permission |
| Derived profiler metric | Formula and normalization are clear | Sanity-checking trends | Hidden derivation and architecture dependence |
| Sampling profiler | Phase behavior matters | Explaining time-varying behavior | Attribution and sampling granularity |
| ISA microkernel timing | Timing behavior is the target or no direct metric exists | Cross-checking counters | Runtime fog and clock variation |
| Binary/dynamic instrumentation | Need executed instruction or memory stream | Validating microkernel design | Instrumentation overhead |
| Simulator trace | Need simulator-internal state | Validating simulator mapping | Simulator may miss hardware mechanisms |

This policy intentionally differs from a "microkernels first" rule. Microkernels
are essential for generating controlled workloads, but timing alone is affected
by runtime fog. Direct profiler metrics are stronger when their semantics match
the measurement contract.

## ISA Semantics Strategy

ISA support is not binary. AMORA should classify the information level available
for each accelerator.

### ISA Information Levels

| Level | Description | Probe Strategy |
|---|---|---|
| `isa_public` | ISA syntax and semantics are public enough for handwritten kernels | Generate low-level probes and verify binary/disassembly |
| `isa_disassemblable` | ISA can be inspected but not reliably authored | Generate high-level kernels, verify output, reject unstable patterns |
| `compiler_only` | Only compiler-level control is available | Use constrained source patterns and profiler validation; lower confidence |
| `runtime_only` | No useful ISA or disassembly path | Use metadata, profiler counters, and black-box timing only |
| `published_only` | Only public specs and docs exist | Trust-and-verify against any observable runtime data |
| `black_box` | Very limited docs/tools | Emit behavioral observations and unsupported mappings |

### ISA Semantic Records

Every backend should maintain instruction semantic records:

```yaml
instruction: FFMA
backend: nvidia_sass
semantic_class: fp32_fma
architectural_block:
  primary: fp32_pipe
  secondary:
    - scheduler
    - register_file
required_operands:
  sources: 3
  destinations: 1
control_level: handwritten_sass_or_inline_ptx
verification:
  disassembly_required: true
  dynamic_instruction_count_optional: true
mapping_confidence: high
known_mismatches:
  - compiler may generate alternate opcode from high-level source
```

This record is the bridge between instruction-level probing and architectural
blocks. Without it, a portable probe cannot safely claim it measured a specific
hardware subsystem.

## Profiler And Tool Support Strategy

NCU, CUPTI, and NVBit are NVIDIA-specific. The generic framework must represent
them as one backend's tool stack, then provide equivalent layered support for
other vendors where possible.

### Tool Layers

| Tool Layer | Purpose | NVIDIA Examples | Other Vendor Examples |
|---|---|---|---|
| runtime metadata | device limits, launch metadata, memory properties | CUDA runtime/driver | HIP/ROCm runtime, Level Zero, Metal, vendor SDKs |
| exact counters | per-kernel or per-range metrics | NCU, CUPTI Range Profiling | rocprof/rocprofiler-sdk, Intel metrics discovery, Xcode GPU counters, vendor profilers |
| sampling counters | time-varying PM state | CUPTI PM Sampling | rocprof sampling, VTune timelines, vendor trace tools |
| PC/stall attribution | instruction/source stall attribution | CUPTI PC Sampling, SASS Metrics | AMD profiler attribution, VTune GPU hotspots where available |
| dynamic instrumentation | executed instruction/memory stream | NVBit | vendor DBI if available, binary trace tools, emulator/simulator traces |
| disassembly/binary validation | verify generated code | nvdisasm, cuobjdump | llvm-objdump, roc-objdump, Intel tools, vendor disassemblers |

### Backend Tool Registry

Each backend defines:

- available tools,
- supported metric concepts,
- candidate metric names,
- normalization formulas,
- permission requirements,
- replay/sampling behavior,
- unsupported reasons,
- fallback tools.

Example:

```yaml
backend: nvidia_cuda
logical_metric: fp32_instruction_count
tool_candidates:
  - ncu
  - cupti_range
metric_candidates:
  - smsp__sass_thread_inst_executed_op_fadd_pred_on.sum
  - smsp__sass_thread_inst_executed_op_ffma_pred_on.sum
normalization: divide by active lanes or expected dynamic instruction count as contract requires
fallback:
  - nvbit_opcode_histogram
  - disassembly_static_count
```

## Layered Output Schema

The output must preserve the transition from raw observation to simulator
estimate.

```yaml
raw_observation:
  source: cupti_range
  raw_values:
    smsp__cycles_active.avg: 100000
    smsp__inst_executed.sum: 25000
  probe_id: arithmetic_latency.fp32_add.nvidia
  binary_hash: string

normalized_measurement:
  concept: scalar_instruction_latency
  value: 4
  unit: backend_core_cycles
  variance:
    count: 20
    median: 4
    mad: 0
    min: 4
    max: 5

backend_interpretation:
  backend: nvidia_cuda
  instruction_semantics: fp32_add
  architectural_block: fp32_pipe
  clock_domain: SM
  primary_evidence: direct_counter
  validation_evidence:
    - disassembly_hash
    - timing_cross_check

simulator_estimate:
  parameter: shader_core_config::max_sp_latency
  value: 4
  unit: cycles
  evidence_tier: direct_counter
  fit_status: conditionally_identified
  uncertainty_category: conditional_scalar
  assumptions:
    - metric maps to the intended opcode class
    - active cycle normalization is valid
  coupled_with:
    - operand_delivery
    - scheduler_issue
```

## Fitting, Identifiability, And Uncertainty

Every non-direct simulator estimate should carry fitting metadata:

- `fit_status`
- `fit_residual`
- `lower_bound`
- `upper_bound`
- `alternative_fits`
- `assumptions`
- `coupled_with`
- `identifiability`

Fit status values:

- `direct`
- `uniquely_identified`
- `bounded`
- `conditionally_identified`
- `underconstrained`
- `behavioral_only`
- `unsupported`

Compact uncertainty categories:

- `stable_scalar`: low variance and direct semantic match.
- `bounded_range`: lower/upper bounds are more defensible than one exact value.
- `conditional_scalar`: scalar valid only under stated assumptions.
- `multi_fit`: multiple parameter sets explain the data.
- `behavioral_class`: emit class or curve, not hardware scalar.
- `indeterminate`: evidence insufficient or contradictory.

Variance fields:

- sample count,
- median,
- MAD,
- min,
- max,
- coefficient of variation,
- per-pass/per-run counter variance when available.

## Clock-Domain Policy

Every timing-derived or rate-derived measurement must record:

- `clock_domain`
- `clock_source`
- `clock_locked`
- `observed_clock_range`
- `native_unit`
- `conversion_method`
- `clock_assumptions`

Clock domains include:

- `core`
- `simd_or_simt_lane_group`
- `l1_or_local_memory`
- `l2_or_shared_cache`
- `dram_or_hbm`
- `fabric`
- `copy_engine`
- `host`
- `mixed`
- `unknown`

Keep native units unless a simulator mapping contract explicitly defines a
conversion.

## Capability-Gated Probe Families

### Topology And Residency

Concepts:

- compute-unit count,
- lane/warp/wave/subgroup size,
- max resident workgroups,
- register and scratch/shared-memory limits.

Primary evidence:

- published specs,
- runtime metadata,
- launch/profiler metadata.

Fallback:

- persistent-workgroup probe where safe.

Simulator mapping:

- scalar allowed for direct metadata-backed limits,
- topology decomposition may be architecture-specific or fitted.

### Instruction Latency

Concept:

- dependent operation latency for a specific instruction semantic class.

Requires:

- controllable instruction sequence or verified compiler output,
- device timer or direct counter,
- binary/disassembly verification.

Primary evidence:

- direct profiler counters if semantic match exists,
- otherwise validated dependent-chain timing.

Fallback:

- normalized behavioral latency with no simulator scalar.

### Instruction Throughput And Issue

Concept:

- reciprocal throughput and issue saturation for an instruction semantic class.

Requires:

- independent instruction stream or counter-supported workload,
- active-cycle/rate measurement,
- instruction stream validation.

Simulator mapping:

- throughput plateau can be stable,
- unit count and issue width are conditionally identified unless published.

### Operand Delivery And Register File

Concept:

- register-bank, operand-port, collector, or operand-delivery conflict behavior.

Requires:

- controllable register assignment for scalar claims.

Fallback:

- emit behavior under register pressure or mark unsupported.

Simulator mapping:

- scalar output only with repeated periodicity and independent validation.

### Local / Shared / Scratchpad Memory

Concept:

- explicitly managed local memory latency, bandwidth, and conflict behavior.

Backend names:

- NVIDIA shared memory,
- AMD LDS,
- Intel SLM,
- accelerator scratchpad SRAM,
- unsupported if no comparable structure exists.

Simulator mapping:

- map only after backend-specific equivalence is declared.

### Cache-Like Structures

Concept:

- latency plateaus, capacity knees, transaction granularity, conflict behavior.

Primary evidence:

- profiler cache metrics if direct,
- pointer-chase and working-set probes for behavior.

Simulator mapping:

- capacity often bounded,
- associativity/MSHR/replacement often fitted or behavioral only.

### Global / External Memory

Concept:

- global memory latency, bandwidth, partitioning, row/bank behavior.

Primary evidence:

- profiler memory metrics where direct,
- published bandwidth specs as trust-and-verify anchors,
- streaming and pointer-chase probes.

Simulator mapping:

- bandwidth plateau can constrain parameters,
- DRAM timing internals usually underconstrained or multi-fit.

### Tensor / Matrix Engines

Concept:

- matrix operation latency, throughput, supported shapes, and data types.

Requires:

- operation semantic record for shape/datatype/layout.

Primary evidence:

- direct tensor/matrix profiler metrics when available,
- validated instruction streams or kernel libraries otherwise.

Simulator mapping:

- tensor rate may be stable,
- unit count and shape-specific latency are conditional or fitted unless
  published.

### DMA / Async Copy / TMA-Like Engines

Concept:

- command issue, transfer bandwidth, in-flight capacity, completion semantics.

Primary evidence:

- backend-specific copy-engine metrics and feature checks,
- transfer microkernels,
- phase sampling when exact replay perturbs overlap.

Simulator mapping:

- internal queue names are simulator equivalents unless direct evidence exists.

### Synchronization

Concept:

- barrier, fence, event, and completion latency under controlled arrival and
  traffic patterns.

Primary evidence:

- direct stall/attribution metrics when available,
- synchronization microkernels.

Simulator mapping:

- report per-scope/per-pattern behavior, not one universal scalar.

### Interconnect / Fabric

Concept:

- saturation behavior, routing/partitioning effects, injection pressure.

Primary evidence:

- fabric/partition counters if available,
- balanced and imbalanced traffic probes,
- simulator traces for mapping.

Simulator mapping:

- router microparameters are usually behavioral-only or bounded effective
  simulator values.

## Backend Families

### NVIDIA

Tool stack:

- CUDA runtime/driver metadata,
- NCU,
- CUPTI Range Profiling,
- CUPTI Host Profiling,
- CUPTI PM Sampling,
- CUPTI PC Sampling,
- CUPTI SASS Metrics,
- NVBit,
- nvdisasm/cuobjdump.

Strategy:

- Use NCU/CUPTI direct metrics as primary when metric contracts are direct.
- Use microkernels to generate controlled workloads and cross-check.
- Use NVBit for dynamic instruction and memory streams in separate runs.

### AMD

Expected tool stack:

- ROCm/HIP runtime metadata,
- rocprof,
- rocprofiler-sdk,
- rocm-smi where useful,
- LLVM/ROCm disassembly tools,
- architecture ISA docs where public.

Strategy:

- Map NVIDIA-like concepts to AMD-specific wavefront, CU, SIMD, LDS, cache, and
  matrix-core semantics before simulator mapping.
- Treat LDS/shared-memory mappings as backend-specific equivalence, not a global
  assumption.

### Intel

Expected tool stack:

- Level Zero runtime and metrics,
- oneAPI/SYCL metadata where used,
- VTune GPU Hotspots,
- Intel metrics discovery tools,
- disassembly tools where available.

Strategy:

- Map concepts through EU/Xe-core/subslice or tile organization as documented.
- Avoid forcing SIMT-specific parameters before backend interpretation.

### Apple, Mobile GPUs, TPUs, NPUs, And Less-Documented Accelerators

Expected support may range from runtime metadata and profilers to black-box
timing only.

Strategy:

- Start with capability discovery and published facts.
- Emit hardware-neutral measurements first.
- Add backend-specific mappings only when architecture docs or tool evidence
  justify them.
- Prefer `unsupported`, `behavioral_only`, or `underconstrained` over forced
  GPGPU-style scalar estimates.

## Proposed Repository Structure

Use the existing `amora/` package rather than a separate `tools/probe_suite/`
implementation tree.

```text
amora/
  core/
    capabilities.py
    statistics.py
    fitting.py
    clock.py
    measurement.py
    parameter_model.py
    runner.py
  backends/
    base.py
    nvidia/
      cuda_tools.py
      metrics.py
      ncu.py
      cupti.py
      nvbit.py
      isa_semantics.yaml
    amd/
      rocm_tools.py
      metrics.py
      isa_semantics.yaml
    intel/
      level_zero.py
      metrics.py
      isa_semantics.yaml
  probes/
    concepts/
      arithmetic_latency.yaml
      throughput.yaml
      local_memory.yaml
      cache_behavior.yaml
      global_memory.yaml
      tensor_matrix.yaml
      synchronization.yaml
      interconnect.yaml
    nvidia/
    amd/
    intel/
  schemas/
    backend_capability.schema.json
    isa_semantics.schema.json
    metric_mapping.schema.json
    measurement_contract.schema.json
    probe_result.schema.json
    hardware_profile.schema.json
    simulator_parameter_map.yaml
  reports/
    json_report.py
    markdown.py
```

## Implementation Phases

### Phase 0: Schema And Contracts

Deliver:

- backend capability schema,
- ISA semantics schema,
- metric mapping schema,
- measurement contract schema,
- layered result schema,
- simulator mapping contract registry.

Acceptance:

- A parameter cannot be emitted without a contract.
- A probe cannot run unless required capabilities are present or explicitly
  downgraded.

### Phase 1: NVIDIA Backend As Reference

Deliver:

- CUDA capability discovery,
- NCU/CUPTI metric resolver,
- NVBit/disassembly validation hooks,
- P0 probes updated to layered evidence output.

Acceptance:

- NVIDIA P0 produces raw observations, normalized measurements, backend
  interpretations, and simulator estimates with fit metadata.

### Phase 2: Add AMD Or Second Backend

Deliver:

- second backend capability discovery,
- at least topology, arithmetic latency, throughput, and local-memory concept
  support where available.

Acceptance:

- The same hardware-neutral concepts can be represented with backend-specific
  semantics.
- Unsupported simulator mappings are explicit.

### Phase 3: Fitting And Simulator Trace Integration

Deliver:

- simulator trace backend,
- queue/cache/scheduler/pipeline trace contracts,
- fitting models with alternatives and residuals.

Acceptance:

- Hardware observations can be compared against simulator dynamic state.
- Coupled simulator parameters report identifiability status.

### Phase 4: Broader Vendor Tooling

Deliver:

- profiler/tool registry entries for AMD, Intel, Apple/mobile, and selected AI
  accelerators,
- documentation-backed published-fact ingestion,
- backend-specific metric mapping candidates.

Acceptance:

- A new backend can start in `published_only`, `runtime_only`, or `black_box`
  mode and still produce honest reports.

## Acceptance Criteria

The generalized suite is ready for implementation when:

- hardware-neutral measurement concepts are defined separately from simulator
  parameters,
- every backend declares trust-critical capabilities,
- every backend records ISA semantics availability and instruction-to-block
  relationships,
- every backend has a profiler/tool registry, even if sparse,
- every probe declares required and optional capabilities,
- every result preserves raw observation, normalized measurement, backend
  interpretation, and simulator estimate layers,
- every simulator parameter estimate references a mapping contract,
- direct counters can be primary evidence when semantically direct,
- microkernel timing is used as controlled workload evidence, validation, or
  fallback according to contract,
- published parameters are used as trust-and-verify anchors,
- simulator dynamic states are traceable for calibration,
- fitted estimates include fit status, variance, bounds or alternatives,
  assumptions, and coupled parameters,
- clock domains and conversion methods are explicit,
- unsupported or unsafe mappings are reported honestly.

## First Concrete Milestone

The first implementation milestone should not require all GPGPU-Sim parameters
to be estimated. It should require the framework to show the full evidence flow
for a small set of concepts:

- topology metadata,
- scalar FP32 or integer instruction latency,
- scalar FP32 or integer throughput,
- local/shared/scratchpad memory latency or unsupported status,
- global memory bandwidth plateau,
- one simulator mapping contract with a direct scalar,
- one simulator mapping contract with `bounded` or `conditionally_identified`,
- one simulator mapping contract with `unsupported` or `behavioral_only`.

This milestone proves AMORA can measure, interpret, and map without pretending
that every accelerator exposes every simulator parameter.
