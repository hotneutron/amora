# Comments on Probing Suite Microarchitecture Plan

## Summary

`probing-suite-microarchitecture-plan.md` is a useful umbrella plan for a
portable real-hardware probing suite. Compared with the NVIDIA-specific plan, it
is stronger at the framework level because it introduces backend abstraction,
ISA-level probing, calibration, reporting, and accelerator portability.

However, it shares several measurement-methodology problems with the NVIDIA
plan. It also has additional issues specific to its generalization goal. The
main unresolved question is still how the suite moves from:

```text
we can run a probe
```

to:

```text
this simulator parameter estimate is trustworthy
```

The document should separate common measurement trust issues from the additional
problems introduced by trying to support multiple architectures.

## 1. Common Methodological Problems

These concerns are shared with the NVIDIA-specific probing plan.

### 1.1 Measurements Are Not Hardware Facts

The plan correctly avoids making performance counters mandatory and emphasizes
ISA-level microkernels. This is a good direction, but ISA microkernel results are
still workload observations, not direct hardware facts.

Probe results can still be affected by:

- runtime stalls
- scheduler behavior
- occupancy
- register allocation
- instruction fetch behavior
- cache state
- memory subsystem interference
- clock variation
- compiler or assembler transformations
- hidden arbitration and throttling

Recommended mitigation:

- Treat counters as validation evidence, not primary truth.
- Treat microkernel timing as primary behavioral evidence only after ISA/binary
  validation.
- Require disassembly or binary verification for each probe where applicable.
- Reject or downgrade results when timing, expected instruction behavior, and
  validation counters disagree.
- Record known confounders for each estimate.

### 1.2 The Parameter Map Is Not A Mapping Contract

The proposed `simulator_parameter_map.yaml` maps probe families to simulator
parameters. This is useful as a coverage list, but it is not enough for a
measurement plan.

For each parameter, the map should define:

- simulator parameter name
- intended hardware behavior
- observability classification
- primary probe
- validation sources
- formula or fitting method
- expected evidence tier
- required fit status
- expected confidence
- known semantic mismatches
- whether scalar output is allowed
- fallback behavior

Without this, the suite may collect useful measurements but still fail to
produce trustworthy simulator parameter estimates.

### 1.3 Microkernel Designs Are Under-Specified

The document says to build dependent chains, independent streams, stride sweeps,
pointer chases, and working-set sweeps. These are the right probe families, but
the plan does not yet define exact measurement contracts.

Each probe should specify:

- target simulator parameter
- hardware behavior being isolated
- kernel structure
- required ISA/SASS pattern
- register allocation constraints
- active warp/CTA policy or architecture equivalent
- timing method
- overhead subtraction method
- warmup policy
- repetition count
- variance threshold
- validation counters or traces
- rejection conditions
- downgrade conditions

Example requirements for arithmetic latency:

- Use a dependent instruction chain.
- Avoid memory operations inside the timed loop.
- Use enough unrolling or iterations to dominate timer overhead.
- Subtract timing overhead using an empty-chain baseline.
- Verify the emitted ISA/SASS dependency pattern.
- Verify dynamic instruction count where possible.
- Sweep active work units to check whether latency is stable.
- Reject if measured latency changes strongly with occupancy.
- Reject or downgrade if unrelated stalls dominate validation metrics.

### 1.4 Fitting And Identifiability Are Not Formalized

Confidence scores and coupled estimates are useful, but not sufficient. Many
simulator parameters are not uniquely identifiable from one measurement.

Examples:

- DRAM latency curves may be explained by different combinations of `tRCD`,
  `tRP`, `tRAS`, queueing, scheduler policy, and partition mapping.
- FP32 throughput may depend on functional-unit count, issue width, scheduler
  behavior, operand delivery, and clock rate.
- Cache capacity may appear as a gradual knee rather than a clean boundary.
- Queue depth may only be visible as a saturation range.

The schema should include fitting and identifiability metadata:

- `fit_status`
- `fit_residual`
- `lower_bound`
- `upper_bound`
- `alternative_fits`
- `assumptions`
- `coupled_with`
- `identifiability`

Suggested `fit_status` values:

- `direct`
- `uniquely_identified`
- `bounded`
- `conditionally_identified`
- `underconstrained`
- `behavioral_only`
- `unsupported`

### 1.5 Clock Domains Need Explicit Handling

The document says raw timings should be normalized to cycles when a cycle
counter exists. This is useful but incomplete.

Different subsystems may effectively operate under different clock domains:

- SM/core clock
- memory clock
- L2/fabric behavior
- copy/TMA engines
- PCIe/NVLink/storage paths if included later

Every timing-derived estimate should record:

- clock source
- clock domain
- observed clock range
- whether clocks were locked
- timing unit
- conversion method to simulator cycles
- clock-domain assumptions

## 2. Issues Specific To This Generalized Plan

These issues come from the document's goal of supporting multiple GPUs and AI
accelerators, not just NVIDIA.

### 2.1 The Generalization Model Is Promising But Too Thin

The `ISABackend` abstraction is a good start, but portability requires more than
abstracting `assemble`, `launch`, and `read_timer`.

Different accelerators may differ in:

- ISA availability
- binary injection support
- timer semantics
- memory hierarchy
- cache controls
- synchronization model
- tensor/matrix instruction semantics
- performance counter availability
- address-space and memory consistency behavior

Simple example:

```text
A dependent FP32 add latency probe is straightforward on NVIDIA if the backend
can generate and verify SASS such as a dependent FADD chain.

On another accelerator, the ISA may not expose the same instruction form, the
compiler may fuse operations, the timer may report global nanoseconds instead of
core cycles, register assignment may not be controllable, and the execution model
may be vector/SIMD rather than SIMT warp-based.

The concept "arithmetic latency" is portable. The exact probe implementation and
the simulator mapping are not.
```

Another example:

```text
Shared-memory bank count is meaningful for NVIDIA-style shared memory. On another
accelerator, the closest structure may be scratchpad SRAM, LDS, SLM, or there may
be no explicitly managed shared memory at all.

The backend should not force this probe to produce
shader_core_config::gpgpu_shmem_num_banks. It should report unsupported or map to
an architecture-specific equivalent before any simulator-specific mapping is
attempted.
```

### 2.2 Metrics Must Be Divided By Architecture And Backend

The framework can define hardware-neutral measurement concepts, but each backend
should define which concrete metrics, instructions, timers, and validation
signals implement those concepts.

Examples:

- NVIDIA may use SASS, CUDA launch metadata, NCU/CUPTI metrics, and NVBit.
- AMD may use different ISA syntax, wavefront semantics, rocprof-style counters,
  and LDS behavior.
- Intel or custom accelerators may expose different timer and memory hierarchy
  semantics.

The plan should not assume one global metric vocabulary. It should separate
generic measurement concepts from backend-specific metric implementations.

Recommended structure:

```text
hardware-neutral measurement concept
-> backend-specific metric/instruction/timer implementation
-> backend-specific interpretation
-> simulator-specific parameter mapping
```

### 2.3 Backend Capabilities Should Focus On Trust-Critical Features

The initial backend capability model does not need to be exhaustive. A large
capability taxonomy could become unnecessary overhead early in the project.

However, the backend should capture the capabilities that directly determine
whether a probe result can be trusted.

Minimum useful capability fields:

```yaml
capabilities:
  can_control_instruction_sequence: true
  can_verify_disassembly: true
  has_device_timer: true
  timer_domain: sm_cycles
  can_control_register_assignment: partial
  has_per_kernel_counters: true
```

Why these matter:

- If instruction sequence cannot be controlled, latency probes are weak.
- If disassembly cannot be verified, compiler or assembler changes may invalidate
  results.
- If timer domain is unknown, cycle normalization is unsafe.
- If register assignment cannot be controlled, register-bank probes are unsafe.
- If counters are unavailable, confidence may be lower, but some probes can still
  run.

### 2.4 Probe Families Need Capability Requirements

Each probe should declare required and optional backend capabilities. This lets
the runner skip probes that cannot produce trustworthy evidence on a given
backend.

Example:

```yaml
probe: arithmetic_latency.dependent_chain
requires:
  - controllable_instruction_sequence
  - device_timer
  - disassembly_or_binary_verification
optional:
  - dynamic_instruction_counter
  - performance_counters
```

Example:

```yaml
probe: register_file.bank_sweep
requires:
  - controllable_instruction_sequence
  - controllable_register_assignment
  - device_timer
  - disassembly_or_binary_verification
optional:
  - stall_counters
```

### 2.5 Output Schema Should Use Layered Results

The current schema groups `measurements`, `repo_parameter_estimates`,
`confidence`, and `raw_results`. For a generalized framework, this should be
more explicitly layered.

Recommended flow:

```text
raw observation
-> normalized measurement
-> backend-specific interpretation
-> simulator parameter estimate
```

Example:

```yaml
raw_observation:
  elapsed_ticks: 4000
  iterations: 1000

normalized_measurement:
  name: fp32_add_dependent_latency
  value: 4
  unit: backend_core_cycles

backend_interpretation:
  backend: nvidia_sass
  instruction: FADD
  clock_domain: SM
  value: 4
  unit: sm_cycles

simulator_estimate:
  parameter: shader_core_config::max_sp_latency
  value: 4
  unit: cycles
  evidence_tier: timing_direct
  fit_status: direct
```

This separation prevents portable measurements from being confused with
simulator-specific mappings.

### 2.6 Acceptance Criteria Are Too Simulator-Specific

The acceptance criteria require estimates for many GPGPU-Sim-style parameters,
including cache associativity, pipeline widths, memory bus width, burst length,
and DRAM latency.

That is reasonable for an NVIDIA/AMORA target, but too specific for a portable
probing framework milestone.

For the generalized framework, the first milestone should require:

- backend capability discovery works
- probes are selected based on capabilities
- raw and normalized measurements are emitted
- backend-specific interpretations are emitted where supported
- simulator mapping is optional and explicitly classified
- unsupported mappings are reported honestly

The framework should emit direct values where available, fitted values only when
identifiable, behavioral observations for hidden structures, and unsupported
status when a mapping is not safe.

## 3. Recommended Reorganization

The plan should separate the design into three layers.

### 3.1 Hardware-Neutral Measurement Layer

This layer defines portable measurement concepts, such as:

- instruction latency
- reciprocal throughput
- memory latency
- cache capacity-like knees
- bandwidth plateau
- synchronization cost
- tensor operation throughput

These are not simulator parameters yet. They are normalized measurement concepts.

### 3.2 Backend-Specific Probe Layer

This layer implements those measurements for each backend.

Examples:

- NVIDIA SASS/CUDA implementation
- AMD ISA/HIP implementation
- Intel implementation
- custom accelerator implementation

Each backend should declare capabilities and supported probes.

### 3.3 Simulator Mapping Layer

This layer maps backend-interpreted measurements to simulator parameters.

It should include:

- GPGPU-Sim/AMORA parameter mapping
- observability classification
- evidence tier
- fit status
- scalar-output rules
- fallback behavior
- known semantic mismatches

Example mapping contract:

```yaml
parameter: cache_config::m_assoc
hardware_behavior: effective conflict behavior under controlled access patterns
observability: indirect
primary_probe: l1_cache/conflict_sets
validation_sources:
  - timing_curve
  - cache_traffic_counters_if_available
formula_or_fit: conflict_knee_model
fit_status_required: bounded_or_conditionally_identified
scalar_output_allowed: only_if_clear_conflict_periodicity
known_mismatches:
  - adaptive replacement
  - sectorized cache
  - unified cache paths
fallback: report_behavioral_observation_only
```

Example for a hidden interconnect parameter:

```yaml
parameter: routing_delay
hardware_behavior: latency increase under controlled injection pressure
observability: behavioral_only
primary_probe: interconnect/injection_rate
scalar_output_allowed: false
fallback: report_effective_latency_curve_and_saturation_point
```

## Overall Assessment

The generalization direction is good, but incomplete. `ISABackend` is a useful
starting point, and the document makes good choices by emphasizing ISA-level
microkernels, calibration, robust statistics, optional counters, and explicit
unsupported handling.

However, the document still mixes portable measurement design with
GPGPU-Sim-specific parameter mapping. Before implementation, the plan should
separate measurement, backend interpretation, and simulator mapping. It should
also add capability-gated probes, layered output schema, per-probe measurement
contracts, and per-parameter mapping contracts.

Without these additions, the suite may produce plausible-looking simulator
parameters that are not actually identifiable from portable hardware
measurements.
