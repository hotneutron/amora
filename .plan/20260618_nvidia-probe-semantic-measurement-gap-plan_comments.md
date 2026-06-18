# Comments on NVIDIA Probe Semantic and Measurement Gap Plan

## Summary

The current plan is a useful high-level roadmap for building NVIDIA GPU probing
support in AMORA, but it is not yet concrete enough as a measurement
methodology. The main concern is that it identifies probe families and tooling
sources, but does not sufficiently define how raw measurements become
trustworthy simulator parameter estimates.

The most important missing layer is a per-parameter and per-probe measurement
contract.

## Major Concerns

### 1. Hardware Metrics Are Not Hardware Facts

NCU/CUPTI metrics are observations of kernel execution, not direct disclosure of
hardware structure. They are affected by runtime stalls, scheduling, occupancy,
compiler output, cache state, replay behavior, clock state, and hidden
arbitration.

The current evidence-tier model is useful, but trust labels alone do not
mitigate unreliable evidence. For non-directly-readable data, the plan needs
concrete mitigation rules.

Recommended mitigation:

- Treat NCU/CUPTI as supporting evidence unless the metric has a direct semantic
  match.
- Use microkernel timing as the primary source for behavioral measurements.
- Use disassembly and NVBit to validate the executed instruction stream.
- Reject or downgrade results when timing, counters, and expected instruction
  behavior disagree.
- Define per-metric normalization formulas, expected ranges, and fallback
  behavior.

The desired model should be:

```text
carefully designed microkernel
+ disassembly verification
+ low-overhead timing
+ repeated runs and statistics
+ NCU/CUPTI counter sanity checks
+ NVBit dynamic instruction validation where needed
```

### 2. Fitting Status Is Required

Many simulator parameters are not independently observable. A single measured
curve can often be explained by multiple parameter combinations.

Examples:

- DRAM latency behavior may be explained by different combinations of `tRCD`,
  `tRP`, `tRAS`, scheduler policy, queueing, and partition mapping.
- FP32 throughput may depend on functional-unit count, issue width, scheduler
  behavior, operand delivery, and clock rate.
- Cache capacity or associativity may appear as a gradual behavioral knee rather
  than a clean structural value.
- Queue depth may only be visible as a saturation range, not an exact count.

The result schema should include fitting metadata such as:

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

This is needed to avoid reporting fitted or ambiguous parameters as if they were
direct hardware facts.

Example for a DRAM timing-like parameter:

```yaml
parameter: memory_config::tRCD
value: 18
unit: cycles
fit_status: underconstrained
confidence: low
evidence_tier: coupled_inference
fit_residual: 0.11
alternative_fits:
  - tRCD: 16
    tRP: 20
    residual: 0.12
  - tRCD: 20
    tRP: 16
    residual: 0.12
coupled_with:
  - memory_config::tRP
  - memory_config::tRAS
  - memory_config::scheduler_type
  - memory_partition_mapping
notes: Multiple parameter sets explain the observed latency curve. This is an effective fitted parameter, not a literal vendor DRAM timing.
```

Example for functional-unit count:

```yaml
parameter: gpgpu_num_sp_units
value: 128
unit: units_per_sm
fit_status: conditionally_identified
confidence: medium
evidence_tier: coupled_inference
assumptions:
  - one FP32 FADD maps to one SP operation
  - SM clocks were stable during the timing window
  - measured throughput reached a stable plateau
  - no operand-delivery bottleneck was observed
coupled_with:
  - scheduler_issue_width
  - fp32_pipe_initiation_interval
  - operand_delivery
```

### 3. Microkernel Design Is Under-Specified

The plan depends heavily on custom CUDA/SASS microkernels, but does not yet
define enough detail about how those kernels isolate the intended behavior.

Microkernel results can be corrupted by:

- compiler optimization
- unexpected instruction selection
- broken dependency chains
- register bank conflicts
- scoreboard stalls
- scheduler effects
- instruction fetch effects
- cache state
- memory subsystem interference
- clock variation
- profiling overhead

Stall metrics can help, but they do not fully explain all root causes.
Therefore, each microkernel should have an explicit design and validation
contract.

Each probe should define:

- target simulator parameter
- hardware behavior being isolated
- kernel structure
- required SASS pattern
- timing method
- warmup policy
- repetition count
- variance threshold
- counter checks
- rejection conditions
- downgrade conditions

Example requirements for arithmetic latency:

- Use a dependent instruction chain.
- Avoid memory operations inside the timed loop.
- Subtract timing overhead with an empty-chain baseline.
- Verify the SASS dependency pattern.
- Verify dynamic instruction count.
- Reject if measured latency changes strongly with occupancy.
- Reject if unrelated stalls dominate.

Example requirements for L1/cache latency:

- Use a pointer chase to enforce dependency.
- Use working sets that intentionally fit or exceed the target cache level.
- Separate warm-cache and cold-cache phases.
- Sweep working-set sizes to detect latency knees.
- Verify the generated load path in SASS.
- Reject if the latency curve has no stable plateau.
- Downgrade if NCU/CUPTI counters show unexpected L2 or DRAM traffic.

### 4. Hidden Simulator Parameters Need Stronger Mitigation

Some simulator parameters do not correspond to directly observable NVIDIA
hardware facts.

Examples:

- scheduler policy names
- operand collector counts
- register file port counts
- MSHR depth
- DRAM scheduler internals
- interconnect router delay
- VC allocation delay
- TMA internal queue state

The plan should not emit exact scalar values for these unless they are
identifiable from independent evidence.

Recommended rule:

A hidden or simulator-specific parameter may be emitted as a scalar only if:

- at least two independent probe families support the estimate
- alternative explanations are ruled out or reported
- assumptions and coupled parameters are recorded
- the report clearly marks the value as an effective simulator-equivalent
  parameter

Otherwise, AMORA should report behavioral observations rather than exact hardware
parameters.

Example:

```yaml
parameter: shader_core_config::gpgpu_scheduler_string
value: loose_round_robin_like
confidence: low
evidence_tier: coupled_inference
fit_status: behavioral_only
notes: This is not a vendor-exposed scheduler policy. It is the closest simulator behavior class under the tested workloads.
```

### 5. Clock Domains Need Explicit Handling

Timing-derived estimates must specify the relevant clock domain.

Different subsystems may effectively operate under different rates:

- SM/core clock
- memory clock
- L2/fabric behavior
- copy or async engines
- PCIe/NVLink/storage paths if added later

The plan should record:

- clock source
- observed clock range
- whether clocks were locked
- timing unit
- conversion method to simulator cycles
- clock-domain assumptions

Without this, latency and throughput estimates can be incorrectly normalized.

Example for SM latency:

```yaml
parameter: shader_core_config::max_sp_latency
value: 4
unit: sm_cycles
clock_domain: SM
clock_source: device_clock64
clock_locked: false
observed_sm_clock_mhz:
  median: 1830
  min: 1815
  max: 1845
```

Example for DRAM latency:

```yaml
parameter: dram_latency
value: 310
unit: ns
clock_domain: memory_effective
observed_memory_clock_mhz:
  median: 9501
notes: Converted to simulator cycles using the selected simulator core/memory clock ratio.
```

### 6. Profiling Replay Can Perturb Small Kernels

NCU/CUPTI replay usually may not dominate large kernels, but it can affect small
microkernels, cache tests, and phase-sensitive behavior.

Mitigation:

- Separate timing-only runs from profiler runs.
- Use profiler metrics primarily for validation.
- Batch small kernels or use long inner loops.
- Correlate separate runs offline using probe ID, binary hash, launch config, and
  disassembly hash.

### 7. Metric Naming Is Mostly an Implementation Issue

Metric naming varies across architectures, CUDA versions, and tools. This is
real but not the main conceptual risk.

The plan should include a metric resolver layer:

- logical metric name
- architecture-specific candidate metrics
- validation rule
- fallback source
- unsupported reason

### 8. Validation Against Known Data Is Useful But Secondary

Validation against public or well-known hardware properties is useful for sanity
checking, but it does not solve the semantic gap.

Useful checks include:

- warp size
- SM count
- shared memory size
- approximate FP32 peak throughput
- approximate memory bandwidth
- known shared-memory bank behavior

This should be included as a validation layer, not treated as the core
measurement methodology.

### 9. Uncertainty and Error Bounds Are Needed

For fitted or noisy estimates, confidence labels are not enough. The profile
should report uncertainty.

Examples:

- Cache capacity should include a range if the latency knee is gradual.
- Queue depth should include lower and upper bounds if saturation occurs across a
  range.
- DRAM timing-like fields should include alternative fits and residuals.

This prevents false precision in the generated simulator profile.

Example for effective cache capacity:

```yaml
parameter: effective_l1_capacity
value: 112
unit: KiB
confidence: medium
fit_status: bounded
lower_bound: 96
upper_bound: 128
fit_residual: 0.07
notes: Capacity knee is gradual. Value is effective capacity under this access pattern.
```

Example for queue depth:

```yaml
parameter: memory_sm_prt_size
value: 32
unit: requests
confidence: low
fit_status: range_estimate
lower_bound: 24
upper_bound: 32
coupled_with:
  - l1_miss_latency
  - l2_latency
  - warp_scheduler
  - memory_partition_mapping
```

### 10. Exact Parameter Mapping Must Be Defined Up Front

The plan should define exact mapping contracts before implementation. Otherwise,
implementation may discover too late that many simulator parameters cannot be
meaningfully mapped.

For every simulator parameter, the plan should define:

- simulator parameter name
- intended hardware behavior
- direct, indirect, or unobservable classification
- primary probe
- validation probe
- formula or fitting method
- evidence tier
- expected confidence
- known semantic mismatches
- scalar output allowed or not
- fallback behavior

Example for shared-memory bank count:

```text
Parameter:
  shader_core_config::gpgpu_shmem_num_banks

Hardware behavior:
  Periodicity of shared-memory bank conflicts under stride sweep.

Primary probe:
  shared_memory/bank_stride.cu

Formula:
  bank_count = conflict_period_from_latency_or_transaction_curve

Validation:
  NCU shared-memory bank conflict metrics, repeatability across block sizes.

Known mismatch:
  Bank width, multicast, and generation-specific behavior may affect periodicity.

Emit scalar:
  yes, if clean periodicity is detected.
```

Example for scheduler policy:

```text
Parameter:
  shader_core_config::gpgpu_scheduler_string

Hardware behavior:
  Behavioral scheduling class under controlled ready-warp tests.

Primary probe:
  scheduler_policy/ready_warps.cu

Formula:
  No direct formula. Classify based on issue distribution and stall response.

Validation:
  smsp issue/stall metrics, PC sampling.

Known mismatch:
  Simulator scheduler labels are not NVIDIA hardware facts.

Emit scalar:
  no exact hardware scalar. Emit behavioral classification only.
```

## Recommended Additions To The Plan

### Add A Parameter Mapping Contract

Each target parameter should have a mapping contract.

Required fields:

- `parameter`
- `hardware_behavior`
- `observability`
- `primary_probe`
- `validation_sources`
- `formula`
- `evidence_tier`
- `fit_status`
- `known_mismatches`
- `scalar_output_allowed`
- `fallback`

### Add A Per-Probe Measurement Protocol

Each probe should define:

- kernel design
- required SASS pattern
- timing method
- metric checks
- clock-domain policy
- warmup/repetition policy
- variance threshold
- rejection rules
- downgrade rules

### Add A Fitting And Identifiability Policy

For inferred parameters, require:

- fit status
- residual/error
- bounds or alternatives
- assumptions
- coupled parameters
- identifiability classification

## Overall Assessment

The current plan is strong as a roadmap, but not yet strong enough as an
executable measurement methodology.

It should be strengthened before implementation by adding:

- exact metric-to-parameter mapping contracts
- concrete microkernel designs
- validation and rejection rules
- fitting status and uncertainty
- clock-domain handling
- stronger distinction between hardware facts, measured behavior, and
  simulator-equivalent parameters

Without these additions, AMORA risks producing plausible-looking simulator
parameters that are not actually trustworthy.
