# NVIDIA Probe Semantic and Measurement Gap Plan

## Summary

This revised plan updates
`.plan/nvidia-probe-semantic-measurement-gap-plan.md`
using the review comments in
`.plan/20260618_nvidia-probe-semantic-measurement-gap-plan_comments.md`
and the follow-up decisions:

- Design microkernels carefully, but do not treat end-to-end microkernel timing
  as the universal primary evidence source.
- Prefer NCU/CUPTI counters as primary evidence when their semantics directly
  match the intended behavior, because they can expose execution facts that
  application-level timing can obscure.
- Use microkernel timing as a controlled workload generator, cross-check, and
  fallback for behaviors not directly exposed by counters.
- Trust-and-verify published hardware parameters where NVIDIA documentation,
  public specifications, or stable tool metadata expose them.
- Treat simulator parameters and dynamic simulator states as directly observable
  inside the simulator. Simulator queue lengths, occupancy state, scheduler
  state, cache state, and pipeline state can be printed or traced. The hard
  problem is mapping real-hardware observations to simulator-equivalent
  parameters, not observing simulator internals.
- Add fitting metadata, clock-domain policy, metric mapping, and compact
  uncertainty/error categories.

The goal is an executable methodology that distinguishes:

1. published hardware facts,
2. tool-observed hardware execution metrics,
3. controlled microkernel behavior,
4. NVBit dynamic instruction or memory streams,
5. simulator-internal state traces,
6. fitted simulator-equivalent parameters.

## Revision History

### 2026-06-18: Measurement-Contract Revision

Source inputs:

- Original plan:
  `.plan/nvidia-probe-semantic-measurement-gap-plan.md`
- Review comments:
  `.plan/20260618_nvidia-probe-semantic-measurement-gap-plan_comments.md`
- Follow-up decisions from the AMORA design discussion on 2026-06-18.

Major changes:

- Replaced the original two-track framing, which treated tooling adapters and
  microkernels as peer implementation tracks, with an evidence-by-semantic-match
  model.
- Accepted the need for careful microkernel design, but rejected the idea that
  microkernel timing should always be the primary source of truth.
- Promoted NCU/CUPTI counters to primary evidence when a metric has a direct
  semantic match to the target behavior.
- Clarified that microkernel timing is still essential as controlled workload
  generation, corroboration, and fallback evidence, but is itself affected by
  runtime fog.
- Added trust-and-verify treatment for published NVIDIA parameters and stable
  CUDA metadata.
- Added simulator tracing as a first-class evidence source because simulator
  parameters and dynamic states are directly observable inside the simulator.
- Added a measurement-contract model for every parameter estimate.
- Added fitting metadata, identifiability states, uncertainty categories,
  clock-domain policy, metric resolver requirements, and rejection/downgrade
  rules.
- Recast hidden or simulator-specific parameters as simulator-equivalent fitted
  parameters unless direct evidence or simulator tracing justifies a scalar.
- Compressed uncertainty reporting into one-line categories with pointers to
  detailed variance, fit, counter, timing, and disassembly records.

### Superseded Assumptions From The Original Plan

- Superseded: "Use microkernel timing as the primary source for behavioral
  measurements."
  Replacement: Use the evidence source with the closest semantic match. NCU/CUPTI
  direct counters are primary when their metric contract is direct; timing is
  primary only when timing behavior itself is the target or no direct metric
  exists.

- Superseded: "Simulator parameters are hidden in the same way hardware
  parameters are hidden."
  Replacement: Simulator parameters and dynamic states are directly observable
  by instrumentation. The hard part is mapping hardware evidence to
  simulator-equivalent knobs and state trajectories.

- Superseded: "Confidence labels are enough for ambiguity."
  Replacement: Every fitted or indirect estimate needs fit status, variance or
  residuals, bounds or alternative fits, assumptions, and coupled parameters.

- Superseded: "Metric naming is a secondary implementation detail."
  Replacement: Metric naming is still implementation detail, but it requires an
  explicit resolver layer with logical names, candidate metrics, normalization,
  validation, and fallback behavior.

### Compatibility With Existing P0-P3 Methodology Files

The split methodology files remain useful but should be updated to conform to
this canonical plan:

- `.plan/nvidia-p0-kernel-methodology.md`
- `.plan/nvidia-p1-kernel-methodology.md`
- `.plan/nvidia-p2-kernel-methodology.md`
- `.plan/nvidia-p3-kernel-methodology.md`

Each should inherit the measurement-contract fields, explicit primary evidence
selection, clock-domain policy, and compact uncertainty categories defined here.

## Evidence Philosophy

The evidence stack should be ordered by semantic match, not by tool category.

| Evidence Source | Use As Primary When | Use As Validation When | Main Risk |
|---|---|---|---|
| Published NVIDIA parameters | Parameter is explicitly documented or exposed as stable device metadata | Checking inferred values | Documentation may omit generation-specific operating modes |
| CUDA runtime / driver metadata | Runtime-visible limit directly maps to a simulator limit | Cross-checking NCU launch metadata | Metadata may describe runtime policy rather than physical structure |
| NCU/CUPTI direct counters | Metric semantics closely match the target behavior | Sanity-checking timing or fitted models | Replay, metric derivation, and architecture-specific naming |
| CUPTI PM Sampling | Time-varying state matters more than exact replayed kernel metrics | Validating phase behavior and throttling | Sampling granularity and attribution ambiguity |
| CUPTI PC Sampling / SASS Metrics | Need source/SASS attribution of stalls or instruction metrics | Explaining anomalous counters | Sampling and patching overhead |
| NVBit dynamic stream | Need exact executed instruction, opcode, register, or memory-reference stream | Verifying generated SASS and dynamic mix | Instrumentation overhead and injection conflicts |
| Microkernel timing | No direct metric exists, or timing behavior is the target | Cross-checking direct metrics and fitted models | Runtime fog: clocks, launch overhead, scheduling, cache state, interference |
| Simulator tracing | Need ground truth for simulator parameter/state behavior | Validating mapping from hardware observations to simulator knobs | Simulator model may not represent all real hardware mechanisms |

## Measurement Contract Model

Each parameter estimate must be produced by a measurement contract. A contract
states what is being observed, why the evidence is meaningful, and when the
result must be rejected, downgraded, or reported as a range.

Required fields:

- `parameter`
- `hardware_behavior`
- `simulator_behavior`
- `observability`: `published|metadata|direct_counter|tool_derived|timing|instrumented_stream|simulator_trace|fitted|unobservable`
- `primary_evidence`
- `validation_evidence`
- `probe_workload`
- `metric_mapping`
- `formula_or_fit`
- `clock_domain`
- `fit_status`
- `uncertainty_category`
- `scalar_output_allowed`
- `rejection_rules`
- `downgrade_rules`
- `fallback`

## Evidence Tiers

V2 keeps the original tier idea but makes the primary-source rule explicit:

- `published_fact`: documented hardware or architecture value.
- `direct_metadata`: CUDA/runtime/device metadata with direct semantic mapping.
- `direct_counter`: NCU/CUPTI metric with close semantic mapping.
- `tool_derived_counter`: NCU/CUPTI derived metric requiring normalization or
  interpretation.
- `instrumented_stream`: NVBit dynamic instruction, memory, or register stream.
- `timing_direct`: microkernel timing with one dominant behavior.
- `simulator_trace`: direct simulator-internal observation.
- `coupled_inference`: fitted estimate involving multiple hidden or interacting
  behaviors.
- `unsupported`: probe skipped with explicit reason.

Counters are primary evidence only when the metric mapping contract says the
counter has a close semantic match. Otherwise counters are validation or
fitting inputs.

## Fitting Metadata

Every non-direct estimate should carry fitting metadata.

Suggested `fit_status` values:

- `direct`: no fitting required.
- `uniquely_identified`: one model explains the observations within tolerance.
- `bounded`: estimate is a range with lower and upper bounds.
- `conditionally_identified`: scalar is valid only under stated assumptions.
- `underconstrained`: multiple plausible parameter sets remain.
- `behavioral_only`: report behavior class, not hardware scalar.
- `unsupported`: unavailable on the target.

Required fitting fields when applicable:

- `fit_residual`
- `lower_bound`
- `upper_bound`
- `alternative_fits`
- `assumptions`
- `coupled_with`
- `identifiability`

## Compact Uncertainty Categories

Use short uncertainty categories in the main report, with pointers to detailed
raw data and fit records.

| Category | One-Line Meaning | Detail Pointer |
|---|---|---|
| `stable_scalar` | Low variance and direct semantic match; scalar output is acceptable. | Link to metric samples, disassembly hash, and variance summary. |
| `bounded_range` | Data supports a lower/upper range better than one exact value. | Link to fitted curve and confidence interval. |
| `conditional_scalar` | Scalar is valid under explicit assumptions. | Link to assumptions and counter/timing agreement checks. |
| `multi_fit` | Multiple parameter sets explain the data. | Link to alternative fits and residual table. |
| `behavioral_class` | Only a behavior class should be emitted. | Link to classification rules and traces. |
| `indeterminate` | Evidence is insufficient or contradictory. | Link to rejection/downgrade reason. |

Variance should be part of these categories:

- Store sample count, median, MAD, min, max, and coefficient of variation.
- For counter values, record per-pass/per-run variance when available.
- For fitted curves, report residual and bounds.
- Promote scalar confidence only when variance and residual thresholds pass.

## Clock-Domain Policy

Every timing-derived or rate-derived estimate must state its clock domain.

Required fields:

- `clock_domain`: `SM|L1TEX|L2|DRAM|fabric|copy_engine|host|mixed|unknown`
- `clock_source`: `clock64|globaltimer|NCU metric|CUPTI metric|published|host_timer`
- `clock_locked`: `true|false|unknown`
- `observed_clock_range`
- `unit`: `cycles|sm_cycles|ns|bytes_per_cycle|inst_per_cycle|requests_per_cycle`
- `conversion_method`
- `clock_assumptions`

Rules:

- Keep native units when possible.
- Convert to simulator cycles only with an explicit simulator clock ratio.
- Keep separate SM, memory, and fabric/copy-engine normalization paths.

## Runtime Fog And Tool Fog

Both microkernel timing and profiler metrics have distortion modes.

Runtime fog affecting end-to-end timing:

- launch overhead
- clock variation
- OS scheduling and power state
- cache and TLB state
- warp scheduling and occupancy interactions
- compiler instruction selection
- memory-system interference

Tool fog affecting NCU/CUPTI:

- replay behavior
- metric derivation
- pass scheduling
- sampling granularity
- profiling permission/mode
- architecture-specific metric availability
- instrumentation or patching overhead

Mitigation rule:

- Do not ask one evidence source to explain everything. Use the source with the
  closest semantic match as primary, then require agreement or an explicit
  discrepancy record from the others.

## Probe Execution Modes

Each probe should define separate execution modes:

1. `capability`: discover device, tool, metric, and permission support.
2. `counter`: collect NCU/CUPTI metrics with replay allowed when safe.
3. `timing`: run low-overhead workload timing without profiler attachment.
4. `instrumented`: run NVBit validation separately.
5. `sim_trace`: run simulator with internal state tracing for calibration.
6. `fit`: combine evidence and emit estimates.

Small-kernel policy:

- Keep timing-only runs separate from profiler runs.
- Batch or lengthen inner loops for tiny kernels.
- Correlate separate runs by probe ID, binary hash, launch config, metric set,
  disassembly hash, and device ID.

## Metric Resolver Layer

Metric naming is an implementation detail, but it must be explicit.

Each logical metric should define:

- `logical_name`
- `candidate_metrics_by_arch`
- `preferred_metric`
- `normalization`
- `validation_rule`
- `fallback_metric`
- `unsupported_reason`

Example:

```yaml
logical_name: sm_active_cycles
candidate_metrics_by_arch:
  default:
    - sm__cycles_active.avg
    - sm__cycles_elapsed.avg
normalization: use active cycles for latency/issue fits; use elapsed cycles for wall-throughput fits
validation_rule: reject if active cycles are zero or if launch metadata reports no executed kernel
fallback_metric: device clock timing
unsupported_reason: metric unavailable in NCU/CUPTI query
```

## Parameter Mapping Contracts

### Topology And Occupancy

Primary evidence:

- Published specs and CUDA metadata.
- NCU launch metadata as validation.
- Persistent CTA kernel for runtime-residency cross-check.

Direct scalar allowed:

- SM count
- warp size
- max threads per SM
- max blocks per SM
- shared memory limits
- register limits when exposed

Mapping notes:

- `gpgpu_sim_config::num_shader()` maps to CUDA SM count.
- `shader_core_config::warp_size` maps to CUDA warp size.
- `shader_core_config::max_cta_per_core` maps to max resident blocks per SM,
  with persistent-CTA validation.
- Cluster decomposition remains architecture/fitting dependent unless published
  or table-backed.

### Arithmetic Latency

Primary evidence:

- NCU/CUPTI instruction counters and active-cycle counters when the metric has a
  direct match to the SASS opcode class.
- Dependent-chain timing as a controlled cross-check and fallback.
- Disassembly and NVBit opcode stream for validation.

Contract:

- Required SASS dependency chain must be verified.
- Dynamic instruction count must match expected loop structure.
- Reject if the target opcode is optimized away or replaced.
- Downgrade if unrelated stalls dominate or latency changes strongly with
  occupancy.

Output:

- Scalar allowed for stable opcode-class latency only with matching counter,
  SASS, and timing evidence.
- Otherwise emit `conditional_scalar` or `bounded_range`.

### Arithmetic Throughput And Functional Units

Primary evidence:

- NCU/CUPTI pipe utilization, instruction counts, active cycles, and issue
  metrics with explicit normalization.
- Independent-chain microkernels generate saturation workloads.
- NVBit validates dynamic opcode mix.

Contract:

- Fit plateau throughput before inferring unit count.
- Functional-unit count is `conditionally_identified`, not a raw hardware fact,
  unless published.
- Record coupling with scheduler issue width, operand delivery, clocks, and
  instruction initiation interval.

### Shared Memory

Primary evidence:

- NCU/CUPTI shared-memory transaction/conflict metrics when available.
- Bank-stride microkernels generate controlled conflict patterns.
- Timing confirms behavioral impact.

Contract:

- Verify access width and SASS shared-memory instruction.
- Sweep stride, active lanes, and access width.
- Reject bank-count scalar if conflict periodicity is unstable.
- Downgrade broadcast/multicast policy to behavioral class unless direct metric
  support is strong.

### L1, Constant, Texture, And Instruction Caches

Primary evidence:

- NCU/CUPTI `l1tex__*` counters for requests, sectors, hits, misses, and
  throughput.
- Pointer-chase and working-set microkernels generate access patterns.
- Disassembly validates load path and cache modifiers.

Contract:

- Report capacity and line/sector size as `bounded_range` when knees are
  gradual.
- Treat associativity, MSHR, and replacement policy as fitted simulator
  equivalents.
- Downgrade if counters indicate unexpected L2/DRAM traffic in an L1-hit probe.

### Scheduler And Issue

Primary evidence:

- NCU/CUPTI `smsp__*` issue, eligible-warps, active-warps, and stall metrics.
- PC Sampling for stall attribution.
- Ready-warp and mixed-issue kernels generate controlled pressure.

Contract:

- Simulator scheduler strings are behavioral classes, not NVIDIA policy names.
- Emit exact scalar only for directly observed counts or published values.
- Emit `behavioral_class` for scheduler policy.
- Couple with arithmetic throughput and register/operand-delivery probes.

### Register File And Operand Delivery

Primary evidence:

- SASS-controlled register-number sweeps plus NCU stall/issue metrics.
- NVBit register/instruction validation when needed.

Contract:

- Bank count may be scalar if periodicity is stable across register-number
  hypotheses.
- Port counts and operand collector counts require at least two independent
  probe families or remain `multi_fit`/`behavioral_only`.
- Record coupling with scheduler and arithmetic pipe behavior.

### SM Memory Pipeline And Coalescing

Primary evidence:

- NCU/CUPTI L1TEX sector/request/replay metrics.
- NVBit memory stream validates lane addresses.
- Lane-pattern kernels generate controlled coalescing cases.

Contract:

- Coalescing behavior can be scalar or class-like if sector counts match the
  model.
- Queue depths and PRT limits are fitted simulator equivalents.
- Downgrade if L2/DRAM saturation explains the same cliff.

### L2, DRAM, And Memory Partitions

Primary evidence:

- NCU/CUPTI `lts__*`, `dram__*`, memory-partition metrics where available, and
  PM Sampling for phase behavior.
- Streaming, pointer-chase, partition-camping, and row-policy microkernels
  generate controlled workloads.

Contract:

- Published memory bandwidth/spec values are trust-and-verify anchors.
- L2 hit latency and bandwidth may be bounded or conditional.
- DRAM timing-like fields are usually `underconstrained` or `multi_fit`.
- Emit alternative fits for timing parameters such as `tRCD`, `tRP`, `tRAS`,
  scheduler policy, and bank/partition mapping.

### Tensor Core

Primary evidence:

- NCU/CUPTI tensor instruction, tensor pipe, active-cycle, and utilization
  metrics.
- MMA kernels generate shape/datatype-specific workloads.
- NVBit validates dynamic MMA opcode mix.

Contract:

- Tensor throughput is primary counter-derived when tensor metrics are direct.
- Tensor unit count is published or fitted; mark assumptions.
- Shape-specific extra latency is conditional on opcode shape, datatype, layout,
  and register pressure.

### Synchronization And Barriers

Primary evidence:

- PC Sampling stall attribution and NCU/CUPTI barrier/scheduler stall metrics.
- Barrier/fence microkernels generate controlled arrival and traffic patterns.

Contract:

- Barrier latency is a behavior under a specific arrival pattern.
- Fence latency must be reported as traffic-dependent, not one fixed scalar.
- Unsupported barrier classes must be explicit by architecture.

### TMA, DMA, And Async Copy

Primary evidence:

- Architecture feature checks, SASS verification, NCU/CUPTI async/copy-related
  metrics where available, PM Sampling for phases.
- Async/TMA kernels generate issue, transfer, and completion workloads.

Contract:

- Internal queue names are simulator structures.
- Emit queue/request values only as simulator-equivalent fitted parameters.
- Record coupling with L2/DRAM, shared memory, and synchronization.

### Interconnect And Address Mapping

Primary evidence:

- NCU/CUPTI partition/fabric/L2/DRAM metrics where available.
- Address-pattern and injection-rate kernels generate balanced and imbalanced
  traffic.
- Simulator tracing provides internal queue/router state for calibration.

Contract:

- Router fields such as VC allocation, switch allocation, and routing delay are
  simulator-equivalent parameters.
- Exact physical address mapping should be a candidate model with confidence,
  not a claimed fact unless independently verified.
- Emit `behavioral_class`, `bounded_range`, or `multi_fit` by default.

## Result Schema Additions

Extend estimate records with:

```yaml
parameter: string
value: scalar_or_range_or_class
unit: string
evidence_tier: string
primary_evidence: string
validation_evidence: [string]
fit_status: string
uncertainty_category: string
confidence: low|medium|high
variance:
  count: int
  median: number
  mad: number
  min: number
  max: number
  coefficient_of_variation: number
bounds:
  lower: number
  upper: number
fit_residual: number
alternative_fits: []
assumptions: []
coupled_with: []
clock_domain: string
clock_source: string
clock_locked: bool_or_unknown
observed_clock_range: {}
scalar_output_allowed: bool
rejection_status: accepted|rejected|downgraded|unsupported
notes: string
detail_refs: []
```

Keep compact report rows one or two lines long, and link each row to the detailed
fit/counter/timing/disassembly records.

## Implementation Plan

### 1. Add Measurement Contract Schema

Files:

- `amora/schemas/measurement_contract.schema.json`
- `amora/schemas/fit_metadata.schema.json`
- `amora/schemas/metric_mapping.schema.json`

Purpose:

- Make parameter mapping explicit before probe implementation.
- Prevent unsupported or fitted values from looking like direct facts.

### 2. Add Parameter Contract Registry

Files:

- `amora/schemas/simulator_parameter_map.yaml`
- `amora/probes/nvidia/contracts/*.yaml`

Purpose:

- One contract per simulator parameter or tightly coupled parameter group.
- Include primary evidence, validation evidence, formula/fit, scalar-output
  policy, and fallback behavior.

### 3. Add Metric Resolver

Files:

- `amora/backends/nvidia/metrics.py`
- `amora/backends/nvidia/ncu.py`
- `amora/backends/nvidia/cupti.py`

Purpose:

- Resolve logical metrics to architecture/tool-specific names.
- Store normalization and unsupported reasons.
- Separate metric discovery from probe logic.

### 4. Add Fit And Variance Model

Files:

- `amora/core/statistics.py`
- `amora/core/fitting.py`
- `amora/core/parameter_model.py`

Purpose:

- Compute robust variance summaries.
- Emit fit status, bounds, residuals, alternatives, and assumptions.
- Produce compact uncertainty categories.

### 5. Add Clock-Domain Recording

Files:

- `amora/core/clock.py`
- `amora/backends/nvidia/cuda_tools.py`
- `amora/backends/nvidia/ncu.py`

Purpose:

- Record clock sources and observed clock ranges.
- Keep native units and conversion methods explicit.

### 6. Update P0-P3 Probe Methodology Files

Files:

- `.plan/nvidia-p0-kernel-methodology.md`
- `.plan/nvidia-p1-kernel-methodology.md`
- `.plan/nvidia-p2-kernel-methodology.md`
- `.plan/nvidia-p3-kernel-methodology.md`

Purpose:

- Add per-probe measurement contracts.
- Mark primary evidence source per probe.
- Add rejection/downgrade rules.
- Add clock-domain and uncertainty categories.

### 7. Add Simulator Trace Contract

Files:

- `amora/backends/simulator_trace.py`
- `amora/probes/simulator_state/*.py`

Purpose:

- Treat simulator dynamic state as directly observable.
- Define trace points for queue length, scheduler state, cache state, pipeline
  state, and memory-partition state.
- Use simulator traces to calibrate and validate fitted hardware-to-simulator
  mappings.

## Acceptance Criteria

V2 is implemented when:

- Every simulator parameter family has a mapping contract.
- Every probe identifies its primary evidence source and validation sources.
- NCU/CUPTI direct metrics are used as primary evidence where semantically
  appropriate.
- Microkernel timing is separated from profiler runs and used as workload,
  validation, or fallback according to contract.
- Published parameters are recorded as trust-and-verify anchors.
- Simulator dynamic states have trace contracts.
- Every fitted or indirect estimate reports fit status, uncertainty category,
  variance summary, assumptions, and coupled parameters.
- Every timing/rate estimate records clock domain and conversion method.
- Unsupported, downgraded, and rejected measurements are explicit in the report.

## Key Decision Record

- Decision: NCU/CUPTI counters can be primary evidence when the metric has a
  direct semantic match.
  Reason: low-level counters can expose execution behavior that end-to-end
  timing may hide behind runtime fog.

- Decision: Microkernels remain essential but are not automatically primary
  evidence.
  Reason: they generate controlled workloads and validate behavior, but timing
  alone is also affected by runtime fog.

- Decision: Published parameters are trusted and verified.
  Reason: documented values are stronger anchors than reverse-engineered fits,
  but still need runtime/tool consistency checks.

- Decision: Simulator internal parameters and dynamic states are directly
  observable inside the simulator.
  Reason: simulator instrumentation can print queue lengths, scheduler state,
  and cache/pipeline state; the semantic gap is in hardware-to-simulator
  mapping.

- Decision: Uncertainty should be compact in the top-level report.
  Reason: main reports should stay readable, while detailed variance, fit, and
  alternative-model records remain linked for auditability.
