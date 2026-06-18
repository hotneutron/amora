# NVIDIA P0 Kernel Methodology

## Scope

This document defines probing methodology for P0 probes listed in
`/Users/bytedance/wk/amora/.plan/nvidia-probe-semantic-measurement-gap-plan.md`.
P0 probes are selected because they provide the fastest path to useful AMORA
hardware profiles with the smallest semantic gap:

- topology and occupancy
- arithmetic latency and throughput
- shared memory latency and bank behavior

Each probe should emit raw measurements, tool evidence, inferred simulator
parameters, confidence, and risk notes.

## Common Methodology Rules

All P0 kernels should follow these rules:

1. Use device-side timing for microkernel hot loops.
2. Run enough repetitions to report median, MAD, min, max, and sample count.
3. Compile with fixed flags and record the full command line.
4. Disassemble generated code with `nvdisasm` or equivalent and store a
   disassembly hash.
5. Use NCU/CUPTI counters as supporting evidence, not as the only source.
6. Use NVBit opcode histograms for instruction-stream validation when possible.
7. Mark every estimate with one evidence tier:
   - `direct_metadata`
   - `direct_counter`
   - `timing_direct`
   - `instrumented_stream`
   - `coupled_inference`
   - `unsupported`

Risk scale:

- Low: direct metadata or a robust single-dominant-parameter timing probe.
- Medium: result depends on generated instruction form, counters, or occupancy.
- High: result is strongly coupled with hidden scheduling, cache, or tool
  behavior.

## Probe: `topology/device_attributes.py`

### Goal

Collect direct CUDA-visible device and launch limits.

### Parameters

- `gpgpu_sim_config::num_shader()`
- `shader_core_config::warp_size`
- `shader_core_config::max_warps_per_shader`
- `shader_core_config::max_cta_per_core`
- `shader_core_config::gpgpu_shader_registers`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`

### Methodology

1. Query CUDA device attributes:
   - SM count
   - warp size
   - max threads per block
   - max threads per SM
   - max blocks per SM
   - registers per block and per SM where available
   - shared memory per block and per SM
   - clock rate and memory clock metadata
2. Query runtime occupancy helper APIs for a small set of simple kernels.
3. Record target identity:
   - CUDA runtime version
   - driver version
   - device name
   - compute capability
   - UUID if available
4. Emit direct estimates for metadata-backed limits.

### Reasoning

Many simulator occupancy fields correspond to architectural or runtime-visible
limits. These are the best first measurements because they do not require
fragile timing interpretation.

### Risk Estimate

Risk: Low.

Main risks:

- CUDA attributes expose runtime limits, not always physical resources.
- Some attributes differ by opt-in shared-memory carveout or driver mode.
- Cluster topology such as `n_simt_clusters` is not directly exposed.

Mitigation:

- Tag exposed limits as `direct_metadata`.
- Tag cluster decomposition as `coupled_inference` unless an architecture source
  or device-specific table is added.

## Probe: `topology/persistent_cta.cu`

### Goal

Estimate effective resident CTA capacity and cross-check CUDA occupancy metadata.

### Parameters

- `shader_core_config::max_cta_per_core`
- `shader_core_config::max_warps_per_shader`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`
- `shader_core_config::gpgpu_shader_registers`

### Methodology

1. Launch many CTAs running a persistent loop.
2. Each CTA atomically claims a global slot on entry and records:
   - block ID
   - SM ID if obtainable through inline PTX special register
   - timestamp at entry
   - live-slot count
3. Hold CTAs at a barrier-like spin point long enough to observe maximum
   concurrently resident CTAs.
4. Sweep:
   - threads per block
   - dynamic shared memory bytes
   - artificial register pressure variants
5. Derive residency cliffs as resources become limiting.

### Reasoning

Metadata says what should be possible. Persistent CTAs test what the runtime
actually admits concurrently for specific launch shapes. The sweep separates
thread, block, register, and shared-memory occupancy constraints.

### Risk Estimate

Risk: Medium.

Main risks:

- Atomic contention can perturb entry timing.
- CTA scheduling order is not guaranteed.
- Register pressure may be optimized away unless enforced by disassembly.
- Watchdog or timeout behavior can affect long spin loops on display GPUs.

Mitigation:

- Keep the spin window short and bounded.
- Use volatile state or inline PTX to preserve register-pressure variants.
- Compare against CUDA occupancy API predictions.
- Treat results as runtime-policy evidence, not direct physical proof.

## Probe: `topology/occupancy.py`

### Goal

Drive occupancy sweeps and fit resource-limit explanations.

### Parameters

- `shader_core_config::max_warps_per_shader`
- `shader_core_config::max_cta_per_core`
- `shader_core_config::gpgpu_shader_registers`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`

### Methodology

1. Generate launch configurations across block sizes and shared-memory sizes.
2. Compile register-pressure variants.
3. Run CUDA occupancy API predictions.
4. Run `persistent_cta.cu` for selected sweep points.
5. Fit the tightest limiting resource for each point.
6. Emit direct metadata and runtime-observed estimates separately.

### Reasoning

Occupancy is a multidimensional constraint. A sweep is necessary because a
single kernel launch cannot isolate the limiting resource.

### Risk Estimate

Risk: Low to Medium.

Main risks:

- Occupancy APIs expose CUDA runtime modeling, not measured execution.
- Register allocation can vary with compiler version.

Mitigation:

- Record `ptxas` resource reports.
- Store disassembly/resource metadata beside measurements.
- Cross-check occupancy API predictions with persistent-CTA observations.

## Probe: `arithmetic_latency/dependent_chain.cu`

### Goal

Measure dependent instruction latency for scalar arithmetic classes.

### Parameters

- `shader_core_config::max_sp_latency`
- `shader_core_config::max_int_latency`
- `shader_core_config::max_sfu_latency`
- `shader_core_config::max_dp_latency`

### Methodology

1. Generate kernels with a single warp or controlled active-warps count.
2. Build a dependency chain:
   - `x_i = op(x_{i-1})`
3. Use enough unrolled operations to dominate timer overhead.
4. Time with a device-side cycle counter around the hot loop.
5. Subtract baseline timing for loop/timer overhead.
6. Repeat for operation classes:
   - FP32 add/mul/fma
   - INT add/mul/logic
   - SFU operations such as reciprocal/sqrt/sin where available
   - FP64 add/mul/fma where supported
7. Disassemble and reject runs where the chain is optimized, folded, or
   scheduled into an unexpected opcode.
8. Validate dynamic opcode count with NVBit.

### Reasoning

A dependent chain serializes operations so the per-operation slope approximates
latency. This is the cleanest first timing probe for simulator latency fields.

### Risk Estimate

Risk: Medium.

Main risks:

- Compiler may rewrite operations or break the intended dependency.
- Hardware may use different SASS opcodes for the same CUDA source expression.
- Scoreboard and issue effects can add overhead when too few or too many warps
  are active.

Mitigation:

- Use inline PTX/SASS templates for critical operations.
- Verify exact SASS.
- Run active-warp sweeps to detect scheduler effects.
- Report one estimate per opcode, then map opcode classes to simulator fields.

## Probe: `arithmetic_latency/independent_chains.cu`

### Goal

Measure reciprocal throughput and infer effective functional-unit throughput.

### Parameters

- `gpgpu_num_sp_units`
- `gpgpu_num_int_units`
- `gpgpu_num_sfu_units`
- `gpgpu_num_dp_units`
- `shader_core_config::pipe_widths`
- `shader_core_config::pipeline_widths_string`

### Methodology

1. Generate kernels with multiple independent chains per warp.
2. Sweep:
   - chains per warp
   - active warps per SM
   - CTAs per SM
   - operation class
3. Measure steady-state cycles per instruction.
4. Use NCU/CUPTI instruction and pipe-utilization metrics.
5. Use NVBit opcode histograms to confirm dynamic instruction mix.
6. Fit saturation point and plateau throughput.
7. Normalize by clock and SM count to estimate effective per-SM throughput.

### Reasoning

Independent chains hide latency and expose issue/throughput limits. The plateau
is the useful signal for simulator pipeline width and unit-rate calibration.

### Risk Estimate

Risk: Medium to High.

Main risks:

- Throughput is coupled with scheduler count, issue policy, clocks, and operand
  delivery.
- Unit count is not directly measured; it is inferred from a throughput model.
- DVFS can distort normalized throughput.

Mitigation:

- Collect clock metrics where available.
- Run single-pipe and mixed-pipe variants.
- Mark functional-unit counts as `coupled_inference`.
- Keep raw throughput as a first-class measurement even when unit count is
  uncertain.

## Probe: `shared_memory/pointer_chase.cu`

### Goal

Measure shared-memory dependent load latency.

### Parameters

- `shader_core_config::gpgpu_smem_latency`
- `memory_shared_memory_minimum_latency`

### Methodology

1. Build a linked list in shared memory.
2. Each load uses the previous load result as the next address.
3. Use one warp and controlled active-lane masks for baseline latency.
4. Sweep:
   - list size within shared memory
   - access stride
   - active lanes
   - active warps per SM
5. Time the dependent chain with device cycle counters.
6. Use NCU/CUPTI shared-memory metrics for transaction/conflict evidence.

### Reasoning

Pointer chasing prevents memory-level parallelism and exposes dependent shared
load latency. Sweeps separate baseline latency from bank conflicts and warp
contention.

### Risk Estimate

Risk: Medium.

Main risks:

- Shared memory latency can vary with bank conflicts and access width.
- Compiler may cache values in registers if the list is not volatile enough.
- Barrier/setup cost can leak into timing.

Mitigation:

- Use volatile or inline PTX shared loads.
- Time only the hot pointer-chase loop.
- Include zero-conflict and known-conflict controls.

## Probe: `shared_memory/bank_stride.cu`

### Goal

Infer shared-memory bank count, bank mapping periodicity, and broadcast behavior.

### Parameters

- `shader_core_config::gpgpu_shmem_num_banks`
- `gpgpu_shmem_limited_broadcast`
- `gpgpu_shmem_warp_parts`

### Methodology

1. Generate warp-level shared-memory accesses with controlled lane addresses.
2. Sweep stride in bytes and elements.
3. Measure latency and throughput per stride.
4. Include patterns:
   - all lanes same address
   - contiguous lanes
   - power-of-two strides
   - prime strides
   - half-warp and quarter-warp participation
5. Use NCU/CUPTI bank-conflict metrics where available.
6. Fit periodic conflict peaks to infer bank count and warp partitioning.

### Reasoning

Bank conflicts produce periodic slowdowns as stride aliases lanes onto the same
bank. Broadcast-like behavior can be detected by comparing uniform-address
access against conflict-heavy many-address patterns.

### Risk Estimate

Risk: Low to Medium.

Main risks:

- Bank width or mapping may vary by operation width and architecture.
- Metrics may report transactions rather than direct conflict count.
- Multicast and broadcast policies may hide some conflicts.

Mitigation:

- Run 32-bit, 64-bit, and vector-width variants.
- Use both timing and counter evidence.
- Report mapping as observed for each access width.

## Probe: `shared_memory/analyze.py`

### Goal

Fit latency, bank periodicity, and broadcast-policy estimates from shared-memory
raw results.

### Methodology

1. Compute baseline latency from lowest-conflict pointer-chase results.
2. Detect stride-periodic peaks.
3. Estimate bank count from the smallest stable conflict periodicity.
4. Compare uniform-address, contiguous, and conflict-heavy patterns.
5. Assign confidence:
   - high for stable bank-count periodicity across repeats
   - medium for broadcast/multicast policy
   - low when stride curves are noisy or counter support is absent

### Reasoning

The analysis code is where raw timing curves become simulator parameter
estimates. It must preserve uncertainty instead of forcing a single direct
mapping.

### Risk Estimate

Risk: Medium.

Main risks:

- Multiple bank mappings can explain similar curves.
- Shared-memory and scheduler effects can overlap.

Mitigation:

- Store raw curves.
- Emit `coupled_with` when several explanations fit.
- Require consistency across access widths before raising confidence.

## P0 Output Requirements

Every P0 result should include:

- kernel or probe name
- source hash and disassembly hash
- launch configuration
- raw timing samples
- NCU/CUPTI metric names and values when available
- NVBit opcode histogram path or `unsupported_reason`
- inferred simulator parameters
- confidence and evidence tier
- risk notes

## P0 Acceptance Criteria

P0 is complete when AMORA can produce a report with:

- direct topology metadata
- measured occupancy cross-checks
- arithmetic latency estimates for FP32, INT, SFU, and FP64 when supported
- throughput plateaus for at least FP32 and INT
- shared-memory latency
- shared-memory bank-count estimate
- explicit unsupported/coupled notes for parameters that are not directly
  observable
