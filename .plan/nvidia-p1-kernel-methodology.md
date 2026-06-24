# NVIDIA P1 Kernel Methodology

## Scope

This document defines NVIDIA P1 probe methodology under the AMORA
hardware-first, simulator-assisted validation model.

P1 probes are executable early but have larger semantic gaps than the baseline probes. They map
observable NVIDIA behavior onto simulator-equivalent cache, scheduler, and
operand-delivery parameters. Most P1 outputs should be fitted, bounded, or
behavioral unless a published fact or direct metric supports a scalar.

P1 covers:

- L1, constant, texture, and instruction-cache behavior,
- scheduler and issue behavior,
- register file and operand collector behavior.

## Revision History

### 2026-06-18: Layered Evidence Refresh

Source inputs:

- `.plan/probing-suite-microarchitecture-plan.md`
- `.plan/nvidia-probe-semantic-measurement-gap-plan.md`
- Previous `.plan/nvidia-p1-kernel-methodology.md`
- baseline methodology refresh in `.plan/nvidia-baseline-kernel-methodology.md`

Major changes:

- Replaced absolute workspace paths with repo-relative paths.
- Updated P1 to depend explicitly on baseline layered outputs.
- Reframed cache, scheduler, and register/operand results as backend
  interpretations before simulator mapping.
- Added primary-evidence selection by semantic match.
- Added simulator trace validation hooks for cache state, scheduler state, and
  operand-collector/register behavior.
- Added scalar-output policies, fit status, uncertainty categories, rejection
  rules, and downgrade rules per probe.

Superseded assumptions:

- Superseded: P1 probes can report simulator cache/scheduler/register fields
  primarily through fitted confidence labels.
  Replacement: P1 reports must preserve measurement, interpretation, and
  simulator mapping layers with explicit scalar-output policy.

- Superseded: scheduler policy inference can produce a simulator policy string.
  Replacement: NVIDIA scheduler behavior should be reported as a behavioral
  class unless direct evidence justifies a more specific mapping.

## Common P1 Contract

P1 requires baseline probe outputs:

- topology and occupancy,
- clock and device identity,
- arithmetic latency,
- arithmetic throughput,
- shared-memory latency and bank behavior.

Every P1 probe must emit:

- `raw_observation`,
- `normalized_measurement`,
- `backend_interpretation`,
- `simulator_estimate`.

Every P1 simulator estimate must include:

- fit status,
- uncertainty category,
- variance summary,
- assumptions,
- `coupled_with`,
- primary evidence and validation evidence,
- rejection/downgrade state,
- fallback behavior.

Risk scale:

- Medium: stable curve and matching counters, but simulator mapping is indirect.
- High: multiple mechanisms can explain the same behavior.
- Very high: target parameter is a simulator-only abstraction with weak external
  evidence.

## Probe: `l1_cache/pointer_chase.cu`

### Concept

L1-like path hit latency for data, read-only, constant, texture, and instruction
paths where supported.

### Target Parameters

- `l1d_cache_config::l1_latency`
- `m_L1D_config`
- `m_L1C_config`
- `m_L0C_config`
- `m_L1T_config`

### Primary Evidence

- NCU/CUPTI `l1tex__*` and path-specific metrics when direct.
- Validated dependent pointer-chase timing when no direct latency metric exists.

### Validation Evidence

- SASS load opcode and cache modifier verification.
- L2/DRAM traffic checks.
- baseline shared-memory and arithmetic latency baselines.
- Simulator L1/cache path trace.

### Methodology

1. Generate dependent pointer-chase kernels for each supported path:
   ordinary global, cache-hinted global, read-only, constant, texture, and
   instruction-footprint variants.
2. Keep hit-latency working sets below the candidate L1 capacity.
3. Use larger working sets as L2/DRAM controls.
4. Sweep active warps to distinguish raw hit latency from scheduler hiding.
5. Collect NCU/CUPTI hit, request, sector, and throughput metrics in profiler
   mode.
6. Run timing mode separately.
7. Record exact opcode path and disassembly hash.

### Scalar Policy

Allow scalar latency only for verified path-specific hit behavior with stable
variance and matching counters. Otherwise emit `bounded_range` or
`conditional_scalar`.

### Fit And Uncertainty

- Expected fit status: `direct`, `bounded`, or `conditionally_identified`.
- Expected uncertainty: `stable_scalar`, `bounded_range`, or
  `conditional_scalar`.

### Rejection And Downgrade

Reject if the load path differs from the intended path. Downgrade if L2/DRAM
traffic appears in a supposed L1-hit run or if path metrics cannot isolate the
target.

### Risk

Medium. NVIDIA cache paths do not map one-to-one to simulator cache objects.

## Probe: `l1_cache/working_set.cu`

### Concept

Effective capacity knees and line/sector granularity for L1-like paths.

### Target Parameters

- `cache_config::m_line_sz`
- `cache_config::m_nset`
- `m_L1D_config`
- `m_L1I_L1_half_C_cache_config`
- `m_L0I_config`
- `m_L1C_config`
- `m_L0C_config`
- `m_L1T_config`

### Primary Evidence

- Working-set latency and miss-rate curves.
- NCU/CUPTI sector, hit, miss, and request metrics.

### Validation Evidence

- Disassembly path checks.
- Stride sweeps.
- Simulator cache-state trace.

### Methodology

1. Sweep working-set size using pointer-chase or randomized strided loads.
2. Use access patterns that reduce prefetch and streaming artifacts.
3. Sweep access stride to detect line/sector transitions.
4. Repeat for data, read-only, constant, texture, and instruction variants when
   practical.
5. Fit latency and counter knees using segmented regression or derivative
   thresholds.
6. Preserve full curves and alternative knee candidates.

### Scalar Policy

Line or sector size can be scalar when stride/counter evidence is direct.
Capacity should usually be a bounded effective range.

### Fit And Uncertainty

- Expected fit status: `bounded` or `conditionally_identified`.
- Expected uncertainty: `bounded_range` or `conditional_scalar`.

### Rejection And Downgrade

Reject a capacity scalar if knees are broad or non-repeatable. Downgrade if
sectorized behavior cannot be represented by one simulator line size.

### Risk

Medium. Replacement, sectoring, and shared/L1 partitioning blur structural
interpretation.

## Probe: `l1_cache/conflict_sets.cu`

### Concept

Effective cache associativity, index conflict behavior, MSHR pressure, and
miss-queue saturation.

### Target Parameters

- `cache_config::m_assoc`
- `cache_config::m_nset`
- `cache_config::m_mshr_entries`
- `cache_config::m_mshr_max_merge`
- `cache_config::m_miss_queue_size`
- `l1d_cache_config::l1_banks`

### Primary Evidence

- Conflict-set latency and miss curves.
- NCU/CUPTI L1 sector, miss, replay, and request metrics.

### Validation Evidence

- Multiple address-index hypotheses.
- Simulator cache and MSHR trace.
- P2 outstanding-request probes for queue-like behavior.

### Methodology

1. Construct candidate same-index address sets under multiple mapping
   hypotheses.
2. Sweep number of lines in each conflict set.
3. Measure latency and cache metrics as set size grows.
4. Add parallel miss streams to stress MSHR and miss-queue behavior.
5. Compare candidate hypotheses and preserve alternatives.
6. Do not collapse associativity, replacement, and queue depth into one scalar.

### Scalar Policy

Associativity-like behavior may be bounded. MSHR and miss-queue values are
`coupled_inference` or `behavioral_only` unless multiple probes agree.

### Fit And Uncertainty

- Expected fit status: `bounded`, `underconstrained`, or `behavioral_only`.
- Expected uncertainty: `bounded_range`, `multi_fit`, or `behavioral_class`.

### Risk

High. Cache indexing, replacement, hashing, and downstream latency can mimic one
another.

## Probe: `l1_cache/analyze.py`

### Concept

Cache-path analysis and simulator-equivalent cache mapping.

### Primary Evidence

- Pointer-chase latency curves.
- Working-set curves.
- Conflict curves.
- NCU/CUPTI hit/miss/sector/replay metrics.

### Methodology

1. Normalize raw curves by clock domain and launch metadata.
2. Detect hit-latency plateaus, capacity knees, sector/line transitions, and
   conflict knees.
3. Generate separate backend interpretations by access path.
4. Map to simulator cache fields only through mapping contracts.
5. Emit alternatives and residuals for ambiguous fits.

### Scalar Policy

Emit scalar only when source probe policy permits it; otherwise emit bounded or
behavioral outputs.

### Risk

Medium to high. The analyzer must prevent false precision from noisy knees.

## Probe: `scheduler_policy/ready_warps.cu`

### Concept

Ready-warp scaling, scheduler issue capacity, and broad scheduling behavior.

### Target Parameters

- `shader_core_config::num_subcores_in_SM`
- `shader_core_config::gpgpu_num_sched_per_core`
- `shader_core_config::gpgpu_scheduler_string`
- `shader_core_config::gpgpu_max_insn_issue_per_warp`

### Primary Evidence

- NCU/CUPTI `smsp__*` issue, eligible-warps, active-warps, and stall metrics.

### Validation Evidence

- Controlled ready-warp kernels.
- CUPTI PC Sampling stall attribution.
- baseline arithmetic throughput coupling.
- Simulator scheduler trace.

### Methodology

1. Generate kernels with controlled numbers of ready warps per SM.
2. Use independent arithmetic instructions to avoid dependency stalls.
3. Sweep active warps, CTAs per SM, and readiness patterns.
4. Measure issue throughput and per-warp progress.
5. Collect issue and eligible-warp metrics.
6. Use PC Sampling when stall attribution is required.
7. Classify behavior rather than naming proprietary scheduler policy.

### Scalar Policy

Allow exact counts only with direct or strongly corroborated evidence. Emit
`gpgpu_scheduler_string` as `behavioral_class`.

### Fit And Uncertainty

- Expected fit status: `conditionally_identified` or `behavioral_only`.
- Expected uncertainty: `conditional_scalar`, `multi_fit`, or
  `behavioral_class`.

### Rejection And Downgrade

Reject runs where arithmetic or memory stalls dominate intended readiness
behavior. Downgrade if different readiness patterns imply different scheduler
classes.

### Risk

High. Scheduler behavior is coupled with subpartition placement, pipeline
availability, and scoreboard state.

## Probe: `scheduler_policy/mixed_issue.cu`

### Concept

Mixed-pipeline issue and overlap behavior.

### Target Parameters

- `shader_core_config::gpgpu_dual_issue_diff_exec_units`
- `shader_core_config::pipeline_widths_string`
- `shader_core_config::pipe_widths`
- `shader_core_config::gpgpu_max_insn_issue_per_warp`

### Primary Evidence

- NCU/CUPTI pipe utilization, issue metrics, and instruction counts.

### Validation Evidence

- Mixed independent instruction streams.
- NVBit opcode mix.
- baseline single-pipe baselines.
- Simulator pipeline trace.

### Methodology

1. Generate independent mixed streams such as FP32+INT, FP32+SFU, FP32+memory,
   and FP32+tensor when tensor support is available.
2. Compare measured throughput to additive, max-only, and partially overlapped
   models.
3. Sweep active warps and independent chains.
4. Validate opcode mix and disassembly.
5. Record one-warp and many-warp behavior separately.

### Scalar Policy

Mixed-issue capability should be a behavioral classification unless multiple
streams and counters support a conditional scalar.

### Fit And Uncertainty

- Expected fit status: `conditionally_identified` or `behavioral_only`.
- Expected uncertainty: `conditional_scalar`, `multi_fit`, or
  `behavioral_class`.

### Risk

Medium to high. Apparent overlap may come from latency hiding, not true dual
issue.

## Probe: `scheduler_policy/analyze.py`

### Concept

Scheduler and issue-behavior classification.

### Methodology

1. Fit throughput scaling versus ready warps.
2. Detect saturation points and per-SMSP-like issue capacity where visible.
3. Compare mixed-stream data against overlap models.
4. Classify behavior as round-robin-like, greedy-like, two-level-like,
   dual-issue-like, partial-overlap, unknown, or coupled.
5. Emit simulator estimates with explicit `coupled_with` links to baseline arithmetic
   throughput and P1 register behavior.

### Scalar Policy

Counts can be conditional; policies are behavioral classes.

### Risk

High. Multiple simulator policies can explain the same curves.

## Probe: `register_file/register_bank_sweep.sass`

### Concept

Register-bank periodicity and operand-delivery conflict behavior.

### Target Parameters

- `shader_core_config::gpgpu_num_reg_banks`
- `shader_core_config::reg_file_port_throughput`
- `gpgpu_reg_bank_use_warp_id`
- `num_regular_register_file_read_ports_per_bank`
- `num_regular_register_file_write_ports_per_bank`

### Primary Evidence

- SASS-controlled register-number sweeps.
- NCU/CUPTI issue and scoreboard stall metrics.

### Validation Evidence

- Disassembly register-number verification.
- NVBit register/instruction validation in narrow windows.
- Simulator register-bank and operand-collector trace.

### Methodology

1. Generate SASS or inline-PTX variants with explicit source/destination
   register numbers where toolchain support allows.
2. Test candidate bank mappings by varying register stride, operand count,
   destination pattern, and warp ID.
3. Measure throughput degradation relative to candidate conflict-free cases.
4. Collect issue and scoreboard metrics.
5. Score candidate bank mappings and preserve alternatives.

### Scalar Policy

Register-bank count may be scalar if periodicity is stable. Port counts and
warp-ID mapping are conditional or multi-fit unless independently validated.

### Fit And Uncertainty

- Expected fit status: `uniquely_identified`, `bounded`, or `underconstrained`.
- Expected uncertainty: `stable_scalar`, `bounded_range`, or `multi_fit`.

### Rejection And Downgrade

Reject if register numbers are not preserved. Downgrade if periodicity changes
with unroll factor, active warps, or unrelated instruction alignment.

### Risk

High. Toolchain control and hidden mapping are fragile.

## Probe: `register_file/register_latency.cu`

### Concept

Differential register read-after-write and operand-delivery cost.

### Target Parameters

- `max_latency_regular_register_file_latency`
- `shader_core_config::reg_file_port_throughput`
- `gpgpu_operand_collector_num_units_*`

### Primary Evidence

- Differential latency/throughput under controlled register reuse patterns.

### Validation Evidence

- baseline arithmetic latency baseline.
- NCU/CUPTI scoreboard/issue stall metrics.
- Disassembly checks.
- Simulator operand-collector trace.

### Methodology

1. Use the same arithmetic opcode while varying register reuse and candidate
   bank patterns.
2. Compare same-register, rotating-register, candidate same-bank, and candidate
   distributed-bank variants.
3. Attribute only differential excess cost to operand delivery.
4. Prefer SASS-controlled variants for high-confidence claims.
5. Report absolute arithmetic latency separately from differential penalties.

### Scalar Policy

Register-file latency and operand collector parameters are usually conditional,
multi-fit, or behavioral. Do not emit precise collector counts without multiple
independent supports.

### Fit And Uncertainty

- Expected fit status: `conditionally_identified`, `underconstrained`, or
  `behavioral_only`.
- Expected uncertainty: `conditional_scalar`, `multi_fit`, or
  `behavioral_class`.

### Risk

High. Arithmetic latency, scoreboard behavior, bank conflicts, and operand
collectors are entangled.

## Probe: `register_file/analyze.py`

### Concept

Register-bank and operand-delivery analysis.

### Methodology

1. Detect periodic throughput penalties across register-number strides.
2. Score candidate bank mappings.
3. Compare operand-count sweeps for port-like saturation.
4. Separate bank-count evidence from port/collector evidence.
5. Emit alternatives, residuals, and `coupled_with` references to arithmetic and
   scheduler probes.

### Scalar Policy

Bank count may be scalar; port and collector fields should usually be bounded,
multi-fit, or behavioral.

### Risk

High. This is one of the first places P1 can overfit simulator internals.

## P1 Implementation Order

1. Implement `l1_cache/pointer_chase.cu` and `l1_cache/analyze.py` first
   because cache-regime classification is needed by later P1/P2 probes.
2. Add `l1_cache/working_set.cu` and `l1_cache/conflict_sets.cu` after the L1
   hit-latency path is stable.
3. Add `scheduler_policy/ready_warps.cu` to measure issue scaling under simple
   instruction streams.
4. Add `scheduler_policy/mixed_issue.cu` only after baseline throughput and P1
   ready-warp baselines are available.
5. Add `register_file/register_bank_sweep.sass` where SASS-level register
   control is supported.
6. Add `register_file/register_latency.cu` and `register_file/analyze.py` last
   because register and operand-collector behavior is highly coupled.

## Required Simulator Trace Hooks

P1 needs simulator instrumentation for:

- L1/TEX cache hit, miss, fill, and eviction events,
- constant and texture path events where modeled,
- instruction-cache access and miss events,
- scheduler ready-warp sets,
- issued instruction and functional-unit selection,
- scoreboard dependency state,
- register-bank selection,
- operand-collector allocation and conflicts.

Simulator traces are direct observations of simulator state. They are not proof
that NVIDIA hardware has the same internal state; they define the target side
of the mapping contract.

## Reporting Requirements

Every P1 report must include:

- evidence tier,
- fit status,
- uncertainty category,
- variance summary,
- metric resolver record,
- SASS validation record,
- access-pattern or instruction-mix descriptor,
- launch and occupancy descriptor,
- clock-domain record,
- simulator mapping contract,
- rejection or downgrade reason when applicable.

## P1 Acceptance Criteria

P1 is complete when AMORA can report:

- at least one path-specific L1 hit-latency estimate or bounded range,
- line/sector granularity estimate where counters support it,
- effective L1 capacity range,
- associativity/conflict behavior or explicit indeterminate status,
- scheduler issue-scaling curve,
- mixed-issue behavior classification,
- register-bank periodicity result or explicit unsupported/indeterminate status,
- layered outputs and mapping contracts for every result,
- explicit separation between direct measurements, NVIDIA interpretations, and
  simulator-equivalent estimates.
