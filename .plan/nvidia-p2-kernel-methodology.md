# NVIDIA P2 Kernel Methodology

## Scope

This document defines the P2 NVIDIA probe methodology under the current AMORA
hardware-first, simulator-assisted validation model.

P2 probes target hardware behavior that is important for simulator fidelity but
more coupled than baseline and P1 behavior:

- SM global-memory pipeline and coalescing,
- L2 cache behavior,
- DRAM/HBM throughput and partition behavior,
- tensor-core latency and throughput,
- synchronization and fence behavior.

P2 is where AMORA must be strict about separating what the hardware directly
exposes from what the simulator wants as a compact parameter. P2 results should
not collapse coupled memory, scheduler, cache, and fabric effects into a single
scalar unless the fit metadata proves that scalar is identifiable for the
stated workload class.

```text
published/CUDA facts
-> raw tool, instruction-stream, and timing observations
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
- Previous `.plan/nvidia-p2-kernel-methodology.md`

Major changes:

- Rewrote P2 around hardware-first evidence selection and simulator-assisted
  validation.
- Replaced any absolute workspace references with repo-relative paths.
- Treated NCU/CUPTI metrics as primary evidence only when the metric contract
  directly matches the behavior under study.
- Clarified that memory-pipeline, L2, DRAM, tensor-core, and synchronization
  parameters are frequently coupled fits rather than direct scalars.
- Added explicit result layering, capability gates, clock-domain handling,
  rejection rules, downgrade rules, and simulator trace hooks.
- Added risk estimates that distinguish direct throughput observations from
  underconstrained simulator-equivalent decomposition.

Superseded assumptions:

- Superseded: P2 microkernels can directly infer simulator memory hierarchy
  constants from timing curves.
  Replacement: P2 probes emit raw observations and normalized measurements
  first; simulator constants are fitted only through declared mapping contracts.

- Superseded: end-to-end timing is the default primary evidence for memory and
  tensor behavior.
  Replacement: direct NCU/CUPTI metrics, CUPTI sampling, NVBit streams,
  disassembly, and timing each become primary only for behaviors they directly
  observe.

## Common P2 Contract

P2 requires baseline and P1 baselines:

- topology and occupancy,
- clock and device identity,
- arithmetic latency and throughput,
- shared-memory behavior,
- L1 and instruction-cache behavior where relevant,
- scheduler and issue constraints where relevant.

Every P2 probe must define:

- hardware-neutral concept,
- NVIDIA-specific interpretation,
- target simulator parameters,
- required backend capabilities,
- primary evidence,
- validation evidence,
- timing and profiler execution modes,
- required SASS/disassembly pattern,
- clock-domain policy,
- scalar-output policy,
- fit status and uncertainty category,
- rejection and downgrade rules,
- fallback behavior.

Every P2 result must emit:

- `raw_observation`
- `normalized_measurement`
- `backend_interpretation`
- `simulator_estimate`

Required shared fields:

- probe ID,
- source hash,
- binary hash,
- disassembly hash,
- launch configuration,
- CUDA device identity,
- driver/runtime/tool versions,
- metric names and units,
- clock domain and clock source,
- working-set and access-pattern descriptors,
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

- Low: direct profiler metric or published/CUDA metadata with a close semantic
  match.
- Medium: stable counter/timing behavior after SASS and access-pattern
  validation.
- High: decomposition across hidden coalescing, MSHR, partition, row-buffer,
  fabric, scheduler, or tensor-pipeline effects.

## Probe: `memory_pipeline/lane_patterns.cu`

### Concept

Warp-level global-memory request formation, lane coalescing, sector behavior,
and replay behavior for controlled address patterns.

### Target Parameters

- coalescing and segment/sector-equivalent behavior,
- replay or reissue behavior for non-coalesced accesses,
- `memory_config` request-size and sector assumptions,
- simulator memory coalescer rules,
- simulator load/store unit issue and replay rules.

### Primary Evidence

- NCU/CUPTI direct metrics for global load/store requests, sectors, bytes, and
  replay/stall behavior when the metric resolver maps them directly.
- NVBit memory-instruction stream for executed load/store opcodes and effective
  address patterns when dynamic address validation is required.

### Validation Evidence

- SASS disassembly to verify load/store width and cache operators.
- Microkernel timing across lane masks, strides, and alignment offsets.
- CUPTI PC/SASS metrics for source/SASS attribution.
- Simulator coalescer trace for request grouping.

### Methodology

1. Generate one warp per CTA and multi-warp variants for lane masks,
   per-lane strides, vector widths, alignment offsets, and cache operators.
2. Verify SASS contains the intended global load/store instructions and does
   not introduce unexpected vectorization or local-memory spills.
3. Collect NCU/CUPTI request, sector, byte, replay, and stall metrics when
   available.
4. Use NVBit to record dynamic instruction and address streams for selected
   patterns, especially when metric semantics are ambiguous.
5. Normalize observations to requests per warp, sectors per request, bytes per
   request, and replay events per memory instruction.
6. Compare normalized measurements against simulator coalescer traces for the
   same lane pattern.

### Scalar Policy

Allowed:

- request/sector counts for a named lane pattern and cache operator,
- behavioral coalescing class for a pattern family,
- bounded replay count for a named pattern.

Not directly allowed:

- a universal coalescing scalar across all widths, masks, alignments, and cache
  operators,
- hidden hardware queue capacity unless isolated by a separate fit.

### Fit And Uncertainty

- Expected fit status: `direct` for direct request/sector metrics after
  validation; `bounded` or `conditionally_identified` for replay behavior.
- Expected uncertainty: `stable_scalar` for canonical aligned patterns;
  `bounded_range` or `behavioral_class` for irregular patterns.

### Rejection And Downgrade

Reject if SASS, NVBit, or profiler evidence shows unintended instructions,
spills, vector-width changes, or cache-operator mismatch. Downgrade to
`behavioral_only` if direct request/sector metrics are unavailable and timing is
the only observation.

### Risk

Medium for direct request/sector behavior; high for replay decomposition.

## Probe: `memory_pipeline/outstanding_requests.cu`

### Concept

Effective outstanding global-memory work per SM or warp under controlled
latency-hiding pressure.

### Target Parameters

- simulator load/store queue capacity,
- simulator memory pipeline issue constraints,
- MSHR-like effective capacity where modeled,
- warp-level memory dependency and scoreboard behavior.

### Primary Evidence

- NCU/CUPTI metrics for active warps, memory-pipeline stalls, outstanding
  memory behavior, and achieved memory throughput when semantically direct.
- Timing-throughput saturation curves for controlled independent memory chains.

### Validation Evidence

- NVBit instruction stream to confirm independent memory instructions.
- SASS disassembly to verify dependencies and cache operators.
- CUPTI PM sampling to detect phase behavior.
- Simulator queue-length and load/store-unit traces.

### Methodology

1. Sweep independent memory streams per warp, warps per CTA, CTAs per SM, and
   dependency distance.
2. Pin the access pattern to a known cache regime using P2 L2/DRAM probes.
3. Collect direct stall, throughput, and occupancy metrics when available.
4. Fit saturation curves only after checking that scheduler occupancy and cache
   hit-rate changes are not the dominant explanation.
5. Emit an effective outstanding-work estimate with the workload class and
   coupled parameters attached.

### Scalar Policy

Allowed:

- effective outstanding-request bound for a declared access class,
- saturation knee for a declared occupancy and cache regime.

Not directly allowed:

- physical MSHR or queue entry count unless supported by direct simulator trace
  or architecture documentation.

### Fit And Uncertainty

- Expected fit status: `bounded` or `conditionally_identified`.
- Expected uncertainty: `bounded_range` or `multi_fit`.

### Rejection And Downgrade

Reject fits with multiple saturation knees, unstable clocks, compiler
dependency changes, or large cache-hit-rate drift. Downgrade to
`underconstrained` when scheduler, cache, and memory-partition explanations
cannot be separated.

### Risk

High. This is an effective behavioral capacity, not a direct physical counter
on current public tooling.

## Probe: `memory_pipeline/analyze.py`

### Concept

Analysis layer for converting P2 memory-pipeline observations into normalized
coalescing and outstanding-work records.

### Target Parameters

- normalized coalescing records,
- effective outstanding-work estimates,
- simulator memory-pipeline mapping contracts.

### Primary Evidence

- Structured outputs from `memory_pipeline/lane_patterns.cu`.
- Structured outputs from `memory_pipeline/outstanding_requests.cu`.

### Validation Evidence

- Metric resolver records.
- NVBit address-stream samples.
- Simulator coalescer and queue traces.

### Methodology

1. Load raw profiler, timing, SASS, NVBit, and simulator trace artifacts.
2. Normalize metric names through the NVIDIA metric resolver.
3. Join evidence by probe ID, binary hash, launch configuration, and access
   pattern.
4. Emit separate hardware measurements and simulator estimates.
5. Mark unsupported, downgraded, or underconstrained rows explicitly.

### Scalar Policy

The analyzer may emit scalar summaries only when all source rows share the same
pattern class, cache policy, launch shape, and fit status.

### Fit And Uncertainty

- Expected fit status: inherited from source probes.
- Expected uncertainty: inherited from source probes plus aggregation variance.

### Rejection And Downgrade

Reject rows with missing metric units, unmatched hashes, or inconsistent access
descriptors. Downgrade aggregation if the source rows have mixed fit statuses.

### Risk

Medium. The main risk is accidental aggregation across incompatible regimes.

## Probe: `l2_cache/pointer_chase.cu`

### Concept

L2 hit latency, miss latency, capacity regime, and conflict behavior using
dependency chains that bypass or overwhelm L1 where possible.

### Target Parameters

- simulator L2 latency,
- simulator L2 capacity and associativity-equivalent behavior,
- cache-line or sector behavior as seen by L2,
- L2 hit/miss policy knobs where modeled.

### Primary Evidence

- NCU/CUPTI L2 hit-rate, sector, request, and throughput metrics when direct.
- Timing of dependent pointer chains when latency behavior is the target.

### Validation Evidence

- SASS cache operators and load width verification.
- Working-set sweeps across expected L1, L2, and DRAM regimes.
- CUPTI PM sampling for phase stability.
- Simulator L2 cache traces.

### Methodology

1. Build pointer-chase arrays with controlled footprint, stride, and
   randomization.
2. Use cache operators and working-set size to minimize L1 ambiguity.
3. Sweep footprints across below-L2, near-L2, and above-L2 regimes.
4. Collect L2 hit/miss/request metrics and timing for each footprint.
5. Fit latency and capacity-equivalent behavior only when the curve has stable
   regimes and direct metrics support the regime assignment.

### Scalar Policy

Allowed:

- L2 hit latency for a validated cache-hit regime,
- L2 capacity range or knee for a declared access pattern,
- L2 miss-latency range when DRAM and partition effects are controlled.

Not directly allowed:

- physical associativity unless conflict-set experiments and direct metrics
  identify it uniquely,
- a universal L2 latency independent of access pattern and clock state.

### Fit And Uncertainty

- Expected fit status: `direct` for direct hit-rate metrics; `bounded` or
  `conditionally_identified` for latency/capacity estimates.
- Expected uncertainty: `stable_scalar` for clean hit latency;
  `bounded_range` or `multi_fit` for capacity/associativity.

### Rejection And Downgrade

Reject runs with unstable hit-rate regimes, unintended local-memory traffic, or
clock throttling. Downgrade to `behavioral_only` when metrics cannot separate
L2 from DRAM behavior.

### Risk

Medium for hit-rate regimes; high for associativity-equivalent inference.

## Probe: `global_memory/streaming.cu`

### Concept

Sustained DRAM/HBM bandwidth and memory-system throughput under simple streaming
loads, stores, and copy patterns.

### Target Parameters

- simulator DRAM bandwidth-equivalent throughput,
- memory clock and bus-width-derived published bound,
- read/write/copy throughput classes,
- memory scheduler throughput constraints.

### Primary Evidence

- NCU/CUPTI DRAM byte, sector, and throughput metrics when direct.
- Published memory bandwidth and memory clock facts as trust-and-verify anchors.

### Validation Evidence

- Kernel timing throughput.
- CUPTI PM sampling for steady-state phases.
- NVBit instruction stream to verify dynamic memory instruction mix.
- Simulator memory-controller throughput traces.

### Methodology

1. Run streaming load, store, and copy kernels with varied vector widths,
   occupancy, CTA shape, and working-set size.
2. Ensure footprints exceed cache capacity for DRAM-dominant measurements.
3. Collect DRAM byte/sector/throughput metrics and kernel timing.
4. Normalize to bytes per second, bytes per cycle where clock data is
   available, and fraction of published peak.
5. Emit simulator-equivalent bandwidth only with the traffic class, clock
   domain, and achieved-utilization regime.

### Scalar Policy

Allowed:

- achieved sustained bandwidth for a named traffic class,
- fraction of published peak under a named launch shape,
- simulator-equivalent bandwidth range for a steady-state regime.

Not directly allowed:

- physical bus-width or channel count unless published or directly documented,
- row-buffer or partition-policy parameters from streaming alone.

### Fit And Uncertainty

- Expected fit status: `direct` for DRAM byte metrics; `bounded` for
  simulator-equivalent bandwidth.
- Expected uncertainty: `stable_scalar` for stable sustained bandwidth;
  `bounded_range` when clocks vary.

### Rejection And Downgrade

Reject if the working set remains cache-resident, clocks throttle without a
recorded correction, or profiler replay changes the traffic class. Downgrade if
only wall-time bandwidth is available.

### Risk

Low to medium for sustained bandwidth; high for physical decomposition.

## Probe: `global_memory/partition_sweep.cu`

### Concept

Memory partition or slice distribution behavior inferred from address sweeps
and throughput variation.

### Target Parameters

- simulator memory partition count or partition-equivalent behavior,
- address-to-partition mapping class,
- partition camping sensitivity,
- interleaving granularity where identifiable.

### Primary Evidence

- Timing and throughput variation across controlled address strides and base
  offsets.
- NCU/CUPTI partition, L2 slice, or memory-subpartition metrics when available
  and semantically direct.

### Validation Evidence

- NVBit effective-address streams.
- DRAM/L2 byte and sector metrics.
- Simulator partition traces with known mapping.

### Methodology

1. Sweep base offsets, strides, page placements, and concurrent streams.
2. Detect periodic throughput or latency variation that indicates partition
   camping or imbalance.
3. Use direct partition/slice metrics when the tool stack exposes them for the
   target architecture.
4. Fit mapping classes rather than exact proprietary hash functions unless the
   pattern is uniquely identified.
5. Validate candidate mappings against simulator partition traces.

### Scalar Policy

Allowed:

- behavioral partition-camping class,
- partition-count lower bound or candidate set,
- address-period candidate set for a declared allocation mode.

Not directly allowed:

- exact proprietary hash function unless uniquely supported by observations.

### Fit And Uncertainty

- Expected fit status: `bounded`, `conditionally_identified`, or
  `behavioral_only`.
- Expected uncertainty: `bounded_range`, `multi_fit`, or `behavioral_class`.

### Rejection And Downgrade

Reject exact mapping claims if multiple hash candidates explain the data.
Downgrade when virtual allocation, page coloring, compression, or cache effects
cannot be excluded.

### Risk

High. NVIDIA partition mapping is architecture-specific and often not directly
exposed.

## Probe: `global_memory/row_policy_sweep.cu`

### Concept

DRAM row-buffer locality and row-policy-equivalent behavior under controlled
strides and access orders.

### Target Parameters

- simulator DRAM timing-equivalent behavior,
- row-hit and row-miss behavioral classes,
- memory-controller scheduling policy sensitivity.

### Primary Evidence

- Timing curves and DRAM throughput metrics for carefully controlled access
  patterns.
- Published memory technology facts where available.

### Validation Evidence

- DRAM byte/sector metrics.
- CUPTI PM sampling for phase behavior.
- Simulator DRAM row-hit, row-miss, and scheduler traces.

### Methodology

1. Sweep strides and access permutations designed to vary row locality.
2. Keep occupancy, instruction mix, and coalescing behavior constant.
3. Compare timing and throughput changes against direct DRAM metrics.
4. Fit only behavioral classes unless simulator traces and hardware
   observations identify a unique timing-equivalent parameter set.

### Scalar Policy

Allowed:

- row-locality sensitivity class,
- bounded timing penalty between access classes,
- simulator-equivalent parameter set only when fit is unique under declared
  assumptions.

Not directly allowed:

- physical DRAM timing constants from public GPU timing alone.

### Fit And Uncertainty

- Expected fit status: `behavioral_only`, `bounded`, or
  `underconstrained`.
- Expected uncertainty: `behavioral_class`, `bounded_range`, or `multi_fit`.

### Rejection And Downgrade

Reject if coalescing, partition mapping, or cache residency changes between
classes. Downgrade to `underconstrained` if several DRAM timing models explain
the same curves.

### Risk

High. Row-policy behavior is deeply coupled with partition mapping and memory
controller scheduling.

## Probe: `global_memory/analyze.py`

### Concept

Analysis layer for bandwidth, partition, and row-policy experiments.

### Target Parameters

- DRAM/HBM throughput records,
- partition-mapping candidate records,
- row-policy behavioral records,
- simulator memory-system mapping contracts.

### Primary Evidence

- Structured outputs from `global_memory/streaming.cu`,
  `global_memory/partition_sweep.cu`, and
  `global_memory/row_policy_sweep.cu`.

### Validation Evidence

- Metric resolver output.
- NVBit address streams.
- Simulator memory-partition and DRAM traces.

### Methodology

1. Normalize bytes, sectors, requests, cycles, and clock domains.
2. Classify each row by cache regime, traffic class, partition pattern, and
   clock stability.
3. Fit throughput and mapping candidates separately.
4. Preserve all candidate mappings when no unique solution exists.
5. Emit simulator estimates only when the mapping contract states the supported
   assumptions.

### Scalar Policy

Scalar summaries are allowed for sustained bandwidth regimes. Partition and
row-policy outputs should default to candidate sets or behavioral classes.

### Fit And Uncertainty

- Expected fit status: `direct` for direct byte metrics; inherited or
  `underconstrained` for partition/row-policy fits.
- Expected uncertainty: `stable_scalar`, `bounded_range`, `multi_fit`, or
  `behavioral_class`.

### Rejection And Downgrade

Reject aggregation across mixed clock domains, mixed cache regimes, or
incompatible allocation modes.

### Risk

Medium for analysis mechanics; high for overconfident physical interpretation.

## Probe: `tensor_core/mma_latency.cu`

### Concept

Tensor-core dependent-operation latency for specific MMA instruction shapes and
data types.

### Target Parameters

- tensor-core pipeline latency-equivalent behavior,
- simulator tensor-core functional-unit latency,
- instruction-shape and data-type latency classes.

### Primary Evidence

- Timing of dependent MMA chains when latency behavior is the target.
- NCU/CUPTI tensor instruction counts and tensor-pipe utilization when direct.

### Validation Evidence

- SASS disassembly to verify exact MMA instructions.
- NVBit dynamic instruction stream for instruction count validation.
- CUPTI PC/SASS metrics for attribution.
- Simulator tensor-pipeline trace.

### Methodology

1. Generate dependent MMA chains for each target instruction shape and data
   type.
2. Fix register allocation as much as possible and record operand-layout
   assumptions.
3. Verify exact SASS opcode, data type, and instruction count.
4. Collect timing and tensor instruction metrics.
5. Convert elapsed cycles to latency only after subtracting loop overhead and
   validating dependency preservation.

### Scalar Policy

Allowed:

- latency for a specific SASS MMA opcode, shape, data type, and dependency
  pattern.

Not directly allowed:

- one tensor-core latency scalar across all MMA shapes and data types.

### Fit And Uncertainty

- Expected fit status: `uniquely_identified` when dependency and SASS are
  verified; otherwise `conditionally_identified`.
- Expected uncertainty: `stable_scalar` or `conditional_scalar`.

### Rejection And Downgrade

Reject if the compiler changes MMA shape, inserts spills, changes dependencies,
or if loop overhead dominates. Downgrade if only high-level CUDA source can be
verified.

### Risk

Medium. Latency is measurable, but exact instruction control is architecture
and compiler sensitive.

## Probe: `tensor_core/mma_throughput.cu`

### Concept

Tensor-core issue throughput and sustained tensor-pipe utilization for
independent MMA instructions.

### Target Parameters

- simulator tensor-core initiation interval,
- tensor functional-unit throughput,
- per-SM tensor throughput class,
- issue constraints coupled with scheduler behavior.

### Primary Evidence

- NCU/CUPTI tensor instruction counts, tensor-pipe utilization, and achieved
  tensor throughput when direct.
- Timing of independent MMA groups when throughput behavior is the target.

### Validation Evidence

- SASS disassembly and NVBit dynamic instruction count.
- Scheduler/issue metrics from P1.
- Simulator tensor-pipeline and scheduler traces.

### Methodology

1. Generate independent MMA instruction groups with varied unroll factors,
   register pressure, and resident warp counts.
2. Verify SASS opcode, instruction count, and absence of memory bottlenecks.
3. Collect tensor-pipe utilization, instruction counts, issue/stall metrics,
   and timing.
4. Normalize to MMA instructions per cycle per SM and fraction of published or
   tool-reported peak where available.
5. Attribute throughput limits to tensor pipe, scheduler, register pressure, or
   occupancy only when supporting evidence separates them.

### Scalar Policy

Allowed:

- sustained throughput for a specific MMA opcode, data type, launch shape, and
  occupancy regime,
- initiation-interval estimate when tensor-pipe saturation is isolated.

Not directly allowed:

- physical tensor-core count unless published or otherwise directly supported.

### Fit And Uncertainty

- Expected fit status: `direct` for direct instruction/utilization metrics;
  `conditionally_identified` for simulator initiation interval.
- Expected uncertainty: `stable_scalar`, `conditional_scalar`, or
  `multi_fit`.

### Rejection And Downgrade

Reject if memory traffic, spills, scheduler stalls, or occupancy changes explain
the throughput. Downgrade to `bounded` if tensor and scheduler limits cannot be
separated.

### Risk

Medium to high because tensor throughput is coupled with scheduler, register,
and occupancy constraints.

## Probe: `synchronization/barrier_latency.cu`

### Concept

CTA-level barrier cost and scaling behavior under controlled thread, warp, and
shared-memory pressure.

### Target Parameters

- simulator barrier latency,
- barrier throughput or serialization class,
- CTA synchronization behavior under occupancy pressure.

### Primary Evidence

- Timing of repeated barriers when barrier behavior is the target.
- NCU/CUPTI synchronization or stall metrics when direct.

### Validation Evidence

- SASS/control-flow verification.
- baseline and P1 occupancy and scheduler baselines.
- Simulator barrier-state trace.

### Methodology

1. Sweep threads per CTA, active warps, CTAs per SM, and work between barriers.
2. Use repeated barriers with controlled pre/post work to amortize timer
   overhead.
3. Collect timing and direct stall metrics where available.
4. Normalize to cycles per barrier per CTA and report scaling class.
5. Map to simulator barrier latency only for the launch and occupancy class
   represented by the experiment.

### Scalar Policy

Allowed:

- cycles per barrier for a named CTA shape and occupancy regime,
- scaling class across CTA sizes.

Not directly allowed:

- a universal barrier latency independent of CTA shape, occupancy, and
  scheduler state.

### Fit And Uncertainty

- Expected fit status: `uniquely_identified` or `conditionally_identified`.
- Expected uncertainty: `stable_scalar` or `conditional_scalar`.

### Rejection And Downgrade

Reject if compiler motion, divergent control flow, or timer overhead dominates.
Downgrade if direct stall metrics are missing and timing variance is high.

### Risk

Medium. The behavior is measurable but occupancy-coupled.

## Probe: `synchronization/fence_latency.cu`

### Concept

Memory fence cost and ordering behavior for selected scopes and traffic
conditions.

### Target Parameters

- simulator fence latency,
- memory ordering behavior class,
- store-drain or visibility-equivalent cost where modeled.

### Primary Evidence

- Timing of repeated fence sequences when fence cost is the target.
- NCU/CUPTI memory dependency, store, or stall metrics when direct.

### Validation Evidence

- SASS verification of fence instructions and scope.
- NVBit instruction stream.
- Memory traffic metrics for pre-fence and post-fence work.
- Simulator memory-ordering and queue-drain traces.

### Methodology

1. Generate fence sequences with varied scope, preceding stores, following
   loads, and traffic volume.
2. Verify SASS fence instruction and memory instruction sequence.
3. Collect timing and memory-stall metrics.
4. Separate empty-fence overhead from queue-drain behavior using traffic-volume
   sweeps.
5. Emit scope-specific and traffic-specific records rather than one global
   fence number.

### Scalar Policy

Allowed:

- empty-fence overhead for a verified scope,
- bounded queue-drain cost for a declared traffic class,
- behavioral ordering class.

Not directly allowed:

- one universal fence latency across scopes and memory states.

### Fit And Uncertainty

- Expected fit status: `conditionally_identified`, `bounded`, or
  `behavioral_only`.
- Expected uncertainty: `conditional_scalar`, `bounded_range`, or
  `behavioral_class`.

### Rejection And Downgrade

Reject if the compiler removes or reorders the fence sequence. Downgrade when
memory traffic state cannot be separated from the fence instruction cost.

### Risk

High. Fence behavior is stateful and scope-dependent.

## P2 Implementation Order

1. Implement `memory_pipeline/lane_patterns.cu` and
   `memory_pipeline/analyze.py` first because direct request/sector metrics make
   them the lowest-risk P2 probes.
2. Add `global_memory/streaming.cu` to establish DRAM/HBM throughput baselines.
3. Add `l2_cache/pointer_chase.cu` after cache-regime classification is wired
   into reports.
4. Add tensor-core latency and throughput probes with strict SASS validation.
5. Add synchronization barrier/fence probes.
6. Add outstanding-request, partition, and row-policy sweeps last because they
   are the most underconstrained and depend on earlier baselines.

## Required Simulator Trace Hooks

P2 needs simulator instrumentation for:

- coalescer request grouping,
- load/store queue occupancy,
- memory-pipeline issue and replay events,
- L2 hit/miss state,
- memory partition selection,
- DRAM scheduler state,
- tensor-pipeline issue and completion,
- barrier state,
- fence and memory-ordering events.

Simulator traces are direct observations of simulator state. They are not proof
that the hardware has the same internal state; they define the target side of
the mapping contract.

## Reporting Requirements

Every P2 report must include:

- evidence tier,
- fit status,
- uncertainty category,
- variance summary,
- metric resolver record,
- SASS validation record,
- access-pattern descriptor,
- clock-domain record,
- simulator mapping contract,
- rejection or downgrade reason when applicable.

P2 reports should prefer ranges, candidate sets, and behavioral classes over
unsupported scalar claims.
