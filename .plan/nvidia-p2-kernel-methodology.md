# NVIDIA P2 Kernel Methodology

## Scope

This document defines probing methodology for P2 probes listed in
`/Users/bytedance/wk/amora/.plan/nvidia-probe-semantic-measurement-gap-plan.md`.
P2 probes target strongly coupled subsystems:

- SM memory pipeline and coalescing
- L2 and DRAM
- tensor core behavior
- synchronization and fence behavior

P2 probes should run after P0 and P1 because their interpretation depends on
known occupancy, arithmetic throughput, shared-memory behavior, cache behavior,
and scheduler behavior.

## Common Methodology Rules

1. Treat P2 outputs as effective behavioral parameters unless a metric is a
   direct match.
2. Always preserve raw curves and full launch metadata.
3. Use NCU/CUPTI Range Profiling for exact per-kernel metrics where replay is
   acceptable.
4. Use CUPTI PM Sampling when long-running phase behavior matters.
5. Use NVBit memory or opcode streams in separate validation runs only.
6. Record clocks, active SM count, memory clock, and throttling evidence when
   available.
7. Explicitly populate `coupled_with` for every memory-system, tensor-unit, and
   synchronization estimate.

Risk scale:

- Low: direct metadata or direct counter close to the simulator field.
- Medium: stable timing behavior with limited coupling.
- High: multiple hidden subsystems can explain the same observation.
- Very high: simulator parameter is an internal model construct with no direct
  real-hardware analogue.

## Probe: `memory_pipeline/lane_patterns.cu`

### Goal

Measure warp-level memory coalescing behavior and lane-address effects.

### Parameters

- `memory_maximum_coalescing_cycles`
- `memory_l1d_max_lookups_per_cycle_per_bank`
- `memory_num_scalar_units_per_subcore`
- `memory_subcore_link_to_sm_byte_size`

### Methodology

1. Generate global-memory load/store instructions with controlled lane
   addresses.
2. Sweep address patterns:
   - contiguous lanes
   - fixed stride
   - one active lane
   - half-warp contiguous
   - scattered random lanes
   - all lanes same cache line
   - lanes crossing cache-line and sector boundaries
3. Sweep access widths:
   - 32-bit
   - 64-bit
   - 128-bit vectorized where supported
4. Measure latency and throughput under low occupancy and saturation occupancy.
5. Collect NCU/CUPTI `l1tex__*` request, sector, transaction, replay, and
   throughput metrics.
6. Run NVBit memory tracing on selected windows to verify dynamic addresses and
   access widths.

### Reasoning

Coalescing rules determine how lane addresses become memory sectors and
transactions. Controlled lane patterns expose transaction growth and replay
behavior, which can be mapped to simulator coalescing and SM memory-pipeline
parameters.

### Risk Estimate

Risk: High.

Main risks:

- Coalescing is entangled with L1 sectoring, cache hits, memory dependency
  stalls, and scheduler behavior.
- NVBit memory tracing perturbs timing heavily and cannot be used for timing
  estimates.
- Cache state can hide coalescing penalties.

Mitigation:

- Use NVBit only to validate address streams.
- Separate cold, warm, and cache-bypassed variants where possible.
- Report coalescing outputs as effective behavior, not direct queue internals.

## Probe: `memory_pipeline/outstanding_requests.cu`

### Goal

Infer outstanding request limits, queue saturation points, and memory-pipeline
backpressure.

### Parameters

- `memory_subcore_queue_size`
- `memory_intermidiate_stages_subcore_unit`
- `memory_sm_prt_size`
- `memmory_max_concurrent_requests_standard_per_sm`
- `memory_subcore_link_to_sm_byte_size`

### Methodology

1. Generate kernels with increasing independent memory instructions per warp.
2. Sweep active warps and CTAs per SM.
3. Use pointer arrays to create controlled cache-hit and cache-miss variants.
4. Measure bandwidth and latency as memory-level parallelism increases.
5. Detect throughput cliffs or latency inflection points.
6. Collect NCU/CUPTI metrics for memory dependency stalls, L1TEX requests, L2
   requests, and active warps.
7. Fit outstanding-request saturation models.

### Reasoning

Outstanding queue or request limits often show up as a plateau: more independent
requests no longer improve throughput, and additional pressure increases stall
time. The exact internal queue name is simulator-specific, so the fitted
parameter should be an effective limit.

### Risk Estimate

Risk: High.

Main risks:

- Saturation may occur in the SM, L1, L2, interconnect, or DRAM.
- Scheduler and occupancy effects can mimic queue limits.
- Cache misses and TLB behavior can add unrelated cliffs.

Mitigation:

- Compare hit-heavy and miss-heavy variants.
- Use P1 cache estimates and P0 occupancy estimates in the model.
- Mark queue-size estimates as `coupled_inference`.

## Probe: `memory_pipeline/analyze.py`

### Goal

Fit coalescing and memory-pipeline effective parameters from lane-pattern and
outstanding-request results.

### Methodology

1. Convert lane patterns into expected sectors under candidate coalescing models.
2. Compare expected sectors with NCU/CUPTI sector metrics.
3. Detect saturation points from outstanding-request sweeps.
4. Fit simple models first:
   - sectors per instruction
   - bytes per cycle
   - outstanding requests at plateau
   - stall fraction versus request depth
5. Emit fitted parameters with `coupled_with` links to cache, scheduler, and
   L2/DRAM probes.

### Reasoning

The analyzer must distinguish what is actually measured from what is inferred
for simulator compatibility. Memory-pipeline internals are not directly exposed
by public tooling.

### Risk Estimate

Risk: High.

Main risks:

- Over-attribution of L2/DRAM saturation to SM queue limits.
- Architecture-specific coalescing behavior not represented by the simulator.

Mitigation:

- Keep separate estimates for coalescing, L1 path, and downstream saturation.
- Emit unresolved coupling instead of forcing a single queue-depth value.

## Probe: `l2_cache/pointer_chase.cu`

### Goal

Measure L2 hit latency and L2 capacity behavior.

### Parameters

- `memory_config::m_L2_config`
- `gpgpu_cache:dl2`
- `gpgpu_l2_rop_latency`
- `cache_config::m_line_sz`
- `cache_config::m_assoc`

### Methodology

1. Build dependent pointer-chase lists in global memory.
2. Use working sets larger than L1 and smaller than expected L2 for L2-hit
   latency.
3. Use larger working sets to cross L2 capacity.
4. Use access patterns that reduce DRAM row-buffer and prefetch effects.
5. Sweep:
   - working-set size
   - stride
   - active SM count
   - cache operators where available
6. Collect NCU/CUPTI `lts__*` hit-rate, sector, request, and throughput metrics.

### Reasoning

Dependent global-memory pointer chasing suppresses memory-level parallelism. If
the working set is outside L1 but inside L2, the measured latency approximates
L2-hit service time plus pipeline overhead.

### Risk Estimate

Risk: Medium to High.

Main risks:

- L1 bypass/caching behavior may not be fully controlled.
- TLB misses can contaminate large working-set latency.
- L2 is partitioned and banked; a single pointer chain may not exercise all
  slices uniformly.

Mitigation:

- Use cache controls and disassembly checks.
- Use page-aware allocation and huge-page variants if available.
- Report L2 latency as effective latency under the tested access pattern.

## Probe: `global_memory/streaming.cu`

### Goal

Measure sustained global-memory bandwidth and infer effective DRAM service
parameters.

### Parameters

- `memory_config::busW`
- `memory_config::BL`
- `dram_data_command_freq_ratio`
- `memory_config::m_n_mem`
- `memory_config::m_n_sub_partition_per_memory_channel`

### Methodology

1. Generate streaming read, write, and copy kernels.
2. Sweep:
   - bytes per thread
   - vector width
   - active CTAs per SM
   - total grid size
   - read/write mix
3. Measure steady-state bandwidth with device timing and host-side wall time.
4. Collect NCU/CUPTI `dram__*`, `lts__*`, and memory throughput metrics.
5. Normalize by measured or reported memory clock when available.
6. Fit bandwidth plateau and compare to theoretical metadata if available.

### Reasoning

Peak streaming bandwidth constrains possible bus width, burst length, memory
partition count, and memory clock ratio. It cannot uniquely identify all of
those fields, but it provides a strong bound.

### Risk Estimate

Risk: High.

Main risks:

- Bandwidth depends on clocks, throttling, ECC, memory compression, access
  pattern, and partition balance.
- Multiple simulator parameter combinations can produce the same bandwidth.

Mitigation:

- Treat bus/burst/partition estimates as coupled.
- Store raw bandwidth plateaus and clock evidence.
- Run partition-balanced and intentionally imbalanced variants.

## Probe: `global_memory/partition_sweep.cu`

### Goal

Infer memory partition count and address-bit mapping behavior.

### Parameters

- `memory_config::m_n_mem`
- `memory_config::m_n_sub_partition_per_memory_channel`
- `gpgpu_mem_addr_mapping`
- `gpgpu_mem_address_mask`

### Methodology

1. Allocate large global-memory buffers.
2. Generate address streams that vary one or more address bits while holding
   other fields constant.
3. Measure bandwidth/latency for each candidate stride and base offset.
4. Detect partition camping or periodic bandwidth drops.
5. Collect L2 slice or memory partition metrics where available.
6. Fit candidate address-to-partition hash/mapping models.

### Reasoning

If specific address-bit patterns concentrate traffic on fewer partitions,
bandwidth falls and latency rises. Sweeping address bits can reveal periodicity
associated with partition selection.

### Risk Estimate

Risk: High to Very High.

Main risks:

- Modern GPUs may hash address bits in undocumented ways.
- Virtual-to-physical mapping can obscure intended address patterns.
- L2 slice mapping and DRAM partition mapping may differ.

Mitigation:

- Use many base offsets and large sample sets.
- Report candidate mappings, not a single definitive mapping, unless evidence
  is strong.
- Mark unresolved mapping as high-risk coupled inference.

## Probe: `global_memory/row_policy_sweep.cu`

### Goal

Estimate row-buffer and DRAM scheduler effects as effective timing parameters.

### Parameters

- `memory_config::scheduler_type`
- `memory_config::nbk`
- `memory_config::nbkgrp`
- `memory_config::tCCD`
- `memory_config::tRCD`
- `memory_config::tRAS`
- `memory_config::tRP`
- `memory_config::CL`
- `memory_config::WL`
- `dram_latency`

### Methodology

1. Generate address sequences intended to create:
   - row-hit-like reuse
   - row-miss-like alternation
   - bank-conflict-like access
   - bank-parallel access
2. Sweep stride and working-set geometry.
3. Measure dependent latency and saturated throughput variants.
4. Collect `dram__*` metrics where available.
5. Fit effective row-hit/row-miss penalty and scheduler behavior.

### Reasoning

Simulator DRAM timing fields have no clean public direct measurement path. Row
policy sweeps can reveal effective latency penalties and scheduling preferences
that constrain simulator calibration.

### Risk Estimate

Risk: Very High.

Main risks:

- Physical address mapping is unknown.
- DRAM scheduler, L2, memory partitions, and row policy are deeply coupled.
- Vendor memory compression or controller behavior may invalidate simple models.

Mitigation:

- Report effective timing deltas, not literal DRAM timing truth.
- Couple estimates with partition-sweep and streaming probes.
- Keep confidence low unless multiple independent patterns agree.

## Probe: `global_memory/analyze.py`

### Goal

Fit L2/DRAM effective parameters from latency, bandwidth, partition, and row
policy sweeps.

### Methodology

1. Fit L2 latency and capacity separately from DRAM latency.
2. Fit streaming bandwidth plateau.
3. Score candidate partition-count and address-mapping models.
4. Fit row-hit/row-miss effective penalties.
5. Emit parameter groups rather than isolated values when coupling is high.
6. Preserve unresolved alternatives in notes.

### Reasoning

L2 and DRAM fields are the most coupled P2 memory outputs. The analyzer should
report what the data constrains and avoid false precision.

### Risk Estimate

Risk: High to Very High.

Main risks:

- Many parameter combinations fit the same curves.
- Counter availability varies strongly by architecture and permission.

Mitigation:

- Use confidence ranges.
- Require cross-validation between timing and counters.
- Mark simulator-specific fields as effective calibration values.

## Probe: `tensor_core/mma_latency.cu`

### Goal

Measure dependent tensor/MMA instruction latency by shape and datatype.

### Parameters

- `tensor_latency`
- `shader_core_config::max_tensor_core_latency`
- `tensor_extra_latency_16816_fp32_1688_fp32`

### Methodology

1. Generate dependent MMA chains where the accumulator output of one MMA feeds
   the next MMA.
2. Sweep:
   - MMA shape
   - input datatype
   - accumulator datatype
   - layout
   - active warps
3. Time hot loops with device cycle counters.
4. Disassemble to verify MMA opcodes.
5. Collect NCU/CUPTI tensor instruction and tensor-pipe metrics.
6. Use NVBit opcode histograms to validate dynamic MMA counts.

### Reasoning

Dependent MMA chains expose tensor instruction latency in a similar way to P0
arithmetic latency probes. Shape-specific differences map to simulator tensor
latency and extra-latency fields.

### Risk Estimate

Risk: Medium to High.

Main risks:

- Compiler and WMMA abstractions may generate unexpected instruction sequences.
- Operand layout and register pressure can dominate observed latency.
- Tensor instructions may have pipeline behavior not modeled by one latency.

Mitigation:

- Prefer explicit inline PTX MMA where practical.
- Record shape-specific estimates.
- Separate baseline tensor latency from shape-specific penalties.

## Probe: `tensor_core/mma_throughput.cu`

### Goal

Measure tensor core throughput and infer effective tensor unit count/rate.

### Parameters

- `gpgpu_tensor_core_avail`
- `gpgpu_num_tensor_core_units`
- `tensor_rate_per_cycle`

### Methodology

1. Generate independent MMA streams with enough chains to hide latency.
2. Sweep active warps and CTAs per SM.
3. Sweep shapes and datatypes.
4. Measure steady-state MMA operations per cycle per SM.
5. Collect NCU/CUPTI tensor-pipe utilization and instruction counts.
6. Validate dynamic opcode mix with NVBit.
7. Fit throughput plateau and infer effective tensor rate.

### Reasoning

Tensor throughput plateaus provide the best public evidence for simulator tensor
rate. Unit count is inferred from throughput and should be treated as effective,
not direct.

### Risk Estimate

Risk: High.

Main risks:

- Tensor throughput is coupled with scheduler issue, register file, operand
  delivery, and data layout.
- Tensor unit count is not directly exposed.
- DVFS and power limits can affect sustained throughput.

Mitigation:

- Record clocks and throttling where possible.
- Compare multiple shapes and occupancy levels.
- Report `tensor_rate_per_cycle` with higher confidence than unit count.

## Probe: `synchronization/barrier_latency.cu`

### Goal

Measure CTA barrier latency and active-warp scaling.

### Parameters

- `gpgpu_num_cta_barriers`
- `BARRIER_OP`
- `MBARRIER_OP`
- `CLUSTER_BARRIER_OP`

### Methodology

1. Generate kernels with repeated barriers inside a timed loop.
2. Sweep:
   - active warps per CTA
   - CTAs per SM
   - work before barrier
   - divergent arrival patterns where legal
3. Subtract baseline loop overhead.
4. Collect NCU/CUPTI stall metrics and PC Sampling barrier stalls.
5. Add architecture-specific variants for mbarrier or cluster barriers when
   supported.

### Reasoning

Barrier cost matters for tiled kernels and async-copy completion. Repeated
barriers amplify the cost enough to measure while sweeps reveal whether latency
is fixed or scales with active warps/CTAs.

### Risk Estimate

Risk: Medium.

Main risks:

- Barrier cost depends on warp arrival skew and scheduler state.
- Some barrier types are unavailable on older architectures.
- Compiler may move independent work around barriers unless constrained.

Mitigation:

- Use volatile memory or inline assembly fences around timed regions.
- Report separate estimates by barrier type.
- Mark unsupported barrier classes explicitly.

## Probe: `synchronization/fence_latency.cu`

### Goal

Measure memory fence cost after different memory traffic patterns.

### Parameters

- `MEMORY_BARRIER_OP`
- `BARRIER_OP`
- `MBARRIER_OP`

### Methodology

1. Generate kernels with timed fence instructions after controlled memory
   operations.
2. Sweep preceding traffic:
   - no memory
   - shared stores
   - global stores
   - global loads
   - mixed reads/writes
3. Sweep memory footprint and active warps.
4. Measure fence latency with device cycle counters.
5. Collect NCU/CUPTI memory dependency, barrier, and stall metrics.
6. Use PC Sampling to attribute stalls to fence or memory dependency reasons.

### Reasoning

Fence latency is not a fixed instruction latency; it depends on outstanding
memory work. The probe should measure both baseline fence cost and
traffic-dependent completion cost.

### Risk Estimate

Risk: High.

Main risks:

- Fence latency is coupled with store queues, L2, DRAM, and ordering scope.
- CUDA-level fences may compile to different SASS sequences by scope and
  architecture.
- Timing can include outstanding memory work rather than fence execution only.

Mitigation:

- Generate scope-specific variants.
- Disassemble fence instructions.
- Report traffic-dependent latency curves instead of one global fence value.

## P2 Output Requirements

Every P2 result should include:

- dependency references to P0 and P1 baselines
- full raw curves
- counter metrics and metric availability
- clock/throttling evidence where available
- NVBit validation stream paths where used
- fitted parameter groups
- confidence ranges
- `coupled_with` metadata
- explicit architecture/feature support notes

## P2 Acceptance Criteria

P2 is complete when AMORA can report:

- memory coalescing behavior for representative lane patterns
- outstanding memory request saturation curves
- L2 effective hit latency and capacity range
- global-memory bandwidth plateau
- candidate partition/address-mapping evidence
- effective DRAM row-policy timing deltas or an explicit indeterminate result
- tensor latency and throughput for at least one MMA shape
- barrier and fence latency curves
- clear risk and coupling notes for every inferred simulator parameter
