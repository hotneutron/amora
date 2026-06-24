# NVIDIA P3 Kernel Methodology

## Scope

This document defines the P3 NVIDIA probe methodology under the current AMORA
hardware-first, simulator-assisted validation model.

P3 probes target behavior with the largest semantic and measurement gaps:

- TMA, asynchronous copy, and DMA-like transfer behavior,
- interconnect injection and contention behavior,
- address mapping across memory partitions, cache slices, and fabric paths.

P3 outputs are expected to be more often bounded, behavioral, or
underconstrained than baseline through P2 outputs. The methodology should still be useful:
it records what can be observed on hardware, what the NVIDIA backend can
interpret, what the simulator exposes directly, and where the mapping is not
identifiable.

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
- Previous `.plan/nvidia-p3-kernel-methodology.md`

Major changes:

- Rewrote P3 as a high-gap methodology instead of a list of ambitious kernel
  ideas.
- Replaced any absolute workspace references with repo-relative paths.
- Made published facts, CUDA metadata, NCU/CUPTI metrics, CUPTI sampling,
  NVBit streams, disassembly, timing, and simulator traces separate evidence
  layers.
- Added explicit capability gates because TMA, async-copy, and interconnect
  metrics vary substantially by architecture and tool version.
- Changed expected outputs from precise physical parameters to ranges,
  candidate sets, behavioral classes, and simulator-equivalent estimates with
  fit metadata.
- Added downgrade and unsupported reporting rules to avoid overclaiming hidden
  interconnect and address-mapping behavior.

Superseded assumptions:

- Superseded: P3 probes should recover exact proprietary interconnect or
  address-mapping parameters.
  Replacement: P3 probes should recover direct observations first, then emit
  exact mappings only when the evidence uniquely identifies them.

- Superseded: asynchronous-copy timing alone can define simulator copy-engine
  parameters.
  Replacement: timing must be joined with instruction-stream validation,
  profiler metrics, memory traffic metrics, synchronization state, and simulator
  traces before emitting simulator-equivalent estimates.

## Common P3 Contract

P3 requires baseline through P2 baselines:

- topology, occupancy, clocks, and device identity,
- arithmetic and shared-memory baselines,
- L1/L2/DRAM regime classification,
- scheduler and issue baselines,
- memory coalescing and partition-sensitivity baselines,
- tensor and synchronization baselines where relevant.

Every P3 probe must define:

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

Every P3 result must emit:

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
- transfer descriptor or address-pattern descriptor,
- synchronization descriptor,
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

- Low: direct metadata or direct counter with a documented semantic match.
- Medium: stable behavior after disassembly, NVBit, profiler, and timing
  validation agree.
- High: any claim about hidden copy engines, fabric topology, address hashing,
  queue capacity, arbitration, or physical routing.

## Probe: `tma_copy/async_copy_latency.cu`

### Concept

Latency and completion behavior for asynchronous copy paths, including
instruction issue, transfer progress, wait/synchronization, and data visibility.

### Target Parameters

- simulator async-copy issue latency,
- simulator async-copy completion latency,
- copy-engine or transfer-pipeline throughput where modeled,
- wait-group or synchronization latency,
- state-machine behavior for transfer visibility.

### Primary Evidence

- Timing of issue, wait, and dependent-use sequences when latency behavior is
  the target.
- NCU/CUPTI async-copy, memory-traffic, and stall metrics when the metric
  resolver identifies direct semantics.
- SASS disassembly for exact async-copy/TMA instructions.

### Validation Evidence

- NVBit dynamic instruction stream where supported.
- CUPTI PC/SASS metrics for attribution.
- Shared-memory and global-memory traffic metrics.
- P2 DRAM/L2 baselines.
- Simulator async-copy state-machine trace.

### Methodology

1. Generate minimal async-copy sequences with explicit issue, wait, and
   dependent-use phases.
2. Sweep transfer size, alignment, source/destination layout, wait distance,
   resident warps, and concurrent CTAs.
3. Verify the intended SASS instructions and synchronization sequence.
4. Collect timing for issue-to-wait, wait-to-use, and full transfer completion.
5. Collect direct profiler metrics for async-copy activity and memory traffic
   when available.
6. Separate instruction issue overhead, transfer latency, memory-system
   throughput, and synchronization overhead only when the evidence supports the
   decomposition.

### Scalar Policy

Allowed:

- latency for a specific instruction sequence, transfer size, alignment, wait
  policy, and occupancy regime,
- bounded completion latency for a declared transfer class,
- behavioral state-machine class.

Not directly allowed:

- physical copy-engine count,
- universal async-copy latency independent of transfer size and wait policy,
- hidden queue capacity unless a saturation fit is identifiable.

### Fit And Uncertainty

- Expected fit status: `conditionally_identified`, `bounded`, or
  `behavioral_only`.
- Expected uncertainty: `conditional_scalar`, `bounded_range`, or
  `behavioral_class`.

### Rejection And Downgrade

Reject if SASS does not contain the intended async-copy/TMA sequence, if
compiler motion invalidates phase timing, or if memory-system bottlenecks
dominate a latency claim. Downgrade to `underconstrained` when issue,
transfer, and wait components cannot be separated.

### Risk

High. Async-copy behavior is stateful and architecture-dependent.

## Probe: `tma_copy/tma_transfer_sweep.cu`

### Concept

Bulk TMA or TMA-like transfer throughput, shape sensitivity, and concurrency
behavior for multidimensional transfers.

### Target Parameters

- simulator TMA transfer throughput,
- simulator TMA setup overhead,
- transfer granularity or alignment-equivalent behavior,
- queue-depth or concurrency-equivalent behavior where modeled.

### Primary Evidence

- NCU/CUPTI TMA, async-copy, memory-byte, sector, and stall metrics when direct.
- Timing-throughput curves when throughput or saturation behavior is the
  target.

### Validation Evidence

- SASS disassembly and source-level transfer descriptor capture.
- NVBit instruction stream where supported.
- P2 DRAM/L2 throughput baselines.
- CUPTI PM sampling for phase stability.
- Simulator TMA queue and transfer traces.

### Methodology

1. Sweep transfer shape, total bytes, alignment, source/destination layout,
   multicast or cluster mode where supported, and concurrent transfer count.
2. Measure setup-dominated, latency-dominated, and throughput-dominated
   regimes separately.
3. Normalize to bytes per second, bytes per cycle, transfers per cycle, and
   fraction of P2 streaming bandwidth.
4. Use direct TMA/async-copy metrics as primary only when metric semantics are
   verified.
5. Fit concurrency and queue-depth-equivalent behavior only after memory
   bandwidth and synchronization bottlenecks are ruled out.

### Scalar Policy

Allowed:

- sustained throughput for a named transfer descriptor and concurrency regime,
- setup overhead for a minimal validated descriptor,
- candidate queue-depth range for a declared workload class.

Not directly allowed:

- exact hardware queue depth,
- exact engine count,
- a single TMA bandwidth independent of shape, alignment, and memory regime.

### Fit And Uncertainty

- Expected fit status: `direct` for direct byte/transfer metrics;
  `bounded` or `conditionally_identified` for setup/concurrency estimates.
- Expected uncertainty: `stable_scalar`, `bounded_range`, or `multi_fit`.

### Rejection And Downgrade

Reject if transfer descriptors differ from the reported class, if clocks drift,
or if the measured regime is cache-resident when DRAM behavior is claimed.
Downgrade concurrency fits when memory bandwidth, synchronization, and queueing
effects cannot be separated.

### Risk

High for queue and engine inference; medium for measured sustained throughput.

## Probe: `tma_copy/analyze.py`

### Concept

Analysis layer for async-copy and TMA observations.

### Target Parameters

- transfer latency records,
- setup-overhead records,
- transfer-throughput records,
- concurrency candidate records,
- simulator async-copy/TMA mapping contracts.

### Primary Evidence

- Structured outputs from `tma_copy/async_copy_latency.cu`.
- Structured outputs from `tma_copy/tma_transfer_sweep.cu`.

### Validation Evidence

- Metric resolver records.
- SASS and NVBit validation records.
- P2 memory-regime records.
- Simulator async-copy/TMA traces.

### Methodology

1. Join timing, profiler, disassembly, NVBit, and simulator trace artifacts by
   probe ID, binary hash, transfer descriptor, and launch configuration.
2. Normalize timing into issue, wait, completion, and throughput records.
3. Attach memory-regime classification from P2.
4. Preserve separate setup, transfer, synchronization, and memory-system
   components when identifiable.
5. Emit bounded or behavioral outputs when decomposition is not identifiable.

### Scalar Policy

Scalar summaries are allowed only for a single transfer descriptor, clock
domain, memory regime, and fit status.

### Fit And Uncertainty

- Expected fit status: inherited from source probes.
- Expected uncertainty: inherited from source probes plus aggregation variance.

### Rejection And Downgrade

Reject records with missing transfer descriptors, mismatched binary hashes, or
ambiguous SASS validation. Downgrade aggregation across mixed memory regimes.

### Risk

Medium. The analysis is straightforward, but component attribution can be
underconstrained.

## Probe: `interconnect/address_mapping.cu`

### Concept

Address-to-fabric, address-to-cache-slice, or address-to-memory-partition
behavior inferred from controlled placement and access-pattern sweeps.

### Target Parameters

- simulator address-to-partition mapping,
- simulator cache-slice or subpartition mapping where modeled,
- fabric-path candidate class,
- partition-camping sensitivity.

### Primary Evidence

- Throughput, latency, and imbalance changes across base-address, stride,
  allocation, and page-placement sweeps.
- Direct partition/slice/fabric metrics when available and semantically direct.

### Validation Evidence

- NVBit effective-address streams.
- NCU/CUPTI L2, DRAM, partition, or subpartition metrics where available.
- P2 partition-sweep baselines.
- Simulator address-mapping and partition traces.

### Methodology

1. Generate controlled address sets with varied base offsets, strides, page
   sizes, allocation order, and concurrent stream placement.
2. Collect timing, throughput, L2/DRAM traffic, and partition metrics where
   available.
3. Use NVBit to verify the effective address stream for candidate cases.
4. Search mapping candidates but retain all candidates that explain the data.
5. Compare candidate mappings with simulator traces and report the mapping
   contract assumptions explicitly.

### Scalar Policy

Allowed:

- partition-camping behavioral class,
- candidate mapping set,
- lower/upper bound on partition or slice count for a declared allocation mode.

Not directly allowed:

- exact proprietary hash function unless uniquely identified,
- physical fabric path or topology without direct evidence,
- mapping claims that ignore virtual allocation and page-placement effects.

### Fit And Uncertainty

- Expected fit status: `bounded`, `conditionally_identified`,
  `behavioral_only`, or `underconstrained`.
- Expected uncertainty: `bounded_range`, `multi_fit`, `behavioral_class`, or
  `indeterminate`.

### Rejection And Downgrade

Reject exact mapping claims if multiple candidates fit the observations.
Downgrade if page placement, compression, migration, cache residency, or
partition metrics are unavailable.

### Risk

High. Exact address mapping is intentionally opaque and architecture-specific.

## Probe: `interconnect/injection_rate.cu`

### Concept

Effective injection bandwidth and contention sensitivity for traffic entering
the on-chip fabric or memory system from SMs.

### Target Parameters

- simulator interconnect injection bandwidth,
- fabric contention or arbitration-equivalent behavior,
- per-SM or per-cluster traffic injection class,
- simulator network queue or link-throughput estimates where modeled.

### Primary Evidence

- NCU/CUPTI throughput, stall, L2/DRAM traffic, and interconnect/fabric metrics
  when direct.
- Timing-throughput saturation curves for controlled multi-SM traffic.

### Validation Evidence

- NVBit instruction and address streams.
- P2 coalescing, L2, DRAM, and partition baselines.
- CUPTI PM sampling for phase stability.
- Simulator interconnect queue, link, and arbitration traces.

### Methodology

1. Generate traffic with controlled source SM count, CTA placement where
   possible, access direction, stride, vector width, and working-set regime.
2. Sweep offered load by varying active CTAs, active warps, instruction mix, and
   independent memory streams.
3. Collect throughput, stall, L2/DRAM, and fabric-related metrics when
   available.
4. Normalize to bytes per cycle per SM, aggregate bytes per cycle, and
   saturation knee for the declared traffic class.
5. Attribute bottlenecks to injection, L2, DRAM, partition camping, scheduler,
   or occupancy only when supporting evidence separates them.

### Scalar Policy

Allowed:

- effective injection-rate bound for a named traffic class,
- saturation knee for a declared source-SM and memory-regime class,
- contention behavioral class.

Not directly allowed:

- physical link width,
- physical crossbar topology,
- exact arbitration policy,
- universal fabric bandwidth independent of traffic shape.

### Fit And Uncertainty

- Expected fit status: `bounded`, `conditionally_identified`, or
  `underconstrained`.
- Expected uncertainty: `bounded_range`, `multi_fit`, or `behavioral_class`.

### Rejection And Downgrade

Reject injection-rate claims when L2/DRAM throughput, partition camping, or
coalescing explains the saturation. Downgrade if direct fabric metrics are
unavailable and timing is the only signal.

### Risk

High. Injection behavior is strongly coupled with memory hierarchy and
scheduler state.

## Probe: `interconnect/analyze.py`

### Concept

Analysis layer for address mapping, partition behavior, and interconnect
injection experiments.

### Target Parameters

- address-mapping candidate records,
- injection-rate bound records,
- contention behavioral records,
- simulator interconnect mapping contracts.

### Primary Evidence

- Structured outputs from `interconnect/address_mapping.cu`.
- Structured outputs from `interconnect/injection_rate.cu`.

### Validation Evidence

- Metric resolver records.
- NVBit address-stream records.
- P2 memory-regime and partition records.
- Simulator partition and interconnect traces.

### Methodology

1. Join raw observations by probe ID, binary hash, allocation descriptor,
   address-pattern descriptor, and launch configuration.
2. Normalize throughput, latency, partition metrics, and clock domains.
3. Fit candidate mapping classes and injection bounds separately.
4. Preserve candidate sets instead of selecting arbitrary single solutions.
5. Emit simulator estimates only with explicit assumptions and unsupported
   reasons.

### Scalar Policy

Scalar summaries are allowed for measured throughput bounds in one declared
traffic class. Mapping and topology outputs should default to candidate sets,
behavioral classes, or unsupported records.

### Fit And Uncertainty

- Expected fit status: inherited from source probes, often `bounded`,
  `behavioral_only`, or `underconstrained`.
- Expected uncertainty: `bounded_range`, `multi_fit`, `behavioral_class`, or
  `indeterminate`.

### Rejection And Downgrade

Reject rows with incompatible allocation modes, mixed cache regimes, unstable
clock domains, or missing address descriptors. Downgrade exact mapping claims
unless uniqueness is proven.

### Risk

Medium for preserving observations; high for physical interpretation.

## P3 Implementation Order

1. Implement `tma_copy/async_copy_latency.cu` for minimal instruction-sequence
   validation and latency decomposition.
2. Implement `tma_copy/tma_transfer_sweep.cu` for throughput and descriptor
   sensitivity.
3. Implement `tma_copy/analyze.py` with strict descriptor and SASS joins.
4. Implement `interconnect/address_mapping.cu` as a candidate-set generator,
   not an exact mapper.
5. Implement `interconnect/injection_rate.cu` only after P2 memory-regime and
   partition baselines are stable.
6. Implement `interconnect/analyze.py` to preserve candidate mappings, bounds,
   and unsupported records.

## Required Simulator Trace Hooks

P3 needs simulator instrumentation for:

- async-copy issue, wait, progress, and completion state,
- TMA setup, transfer, queue, and completion state,
- transfer descriptor interpretation,
- memory partition selection,
- cache-slice selection where modeled,
- interconnect injection queues,
- interconnect link utilization,
- arbitration events,
- memory-system backpressure.

Simulator traces are direct observations of simulator state. They are not proof
that NVIDIA hardware has the same internal state; they define the target side
of the mapping contract.

## Reporting Requirements

Every P3 report must include:

- evidence tier,
- fit status,
- uncertainty category,
- variance summary,
- metric resolver record,
- SASS validation record,
- transfer or address-pattern descriptor,
- clock-domain record,
- simulator mapping contract,
- candidate set when uniqueness is not proven,
- rejection or downgrade reason when applicable.

P3 reports should prefer bounded ranges, candidate sets, behavioral classes,
and explicit `unsupported` records over unsupported exact physical claims.
