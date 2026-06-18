# NVIDIA P1 Kernel Methodology

## Scope

This document defines probing methodology for P1 probes listed in
`/Users/bytedance/wk/amora/.plan/nvidia-probe-semantic-measurement-gap-plan.md`.
P1 probes have moderate semantic gaps. They are executable early, but their
results should usually be reported as fitted equivalents rather than direct
hardware facts.

P1 covers:

- L1, constant, texture, and instruction-cache behavior
- scheduler and issue behavior
- register file and operand collector behavior

## Common Methodology Rules

1. Run P0 first. P1 analysis depends on known clock, occupancy, arithmetic
   latency, shared-memory behavior, and basic throughput.
2. Use disassembly checks for every generated kernel.
3. Use NCU/CUPTI metrics to classify bottleneck source.
4. Use PC Sampling when stall attribution matters.
5. Use NVBit only in separate validation runs for dynamic instruction/register
   evidence.
6. Report simulator cache, scheduler, and operand-collector fields as
   behavioral/effective estimates unless direct metadata exists.

Risk scale:

- Low: repeated timing curve has a sharp feature and matches counters.
- Medium: curve is stable but several hardware mechanisms can explain it.
- High: estimate maps a hidden hardware policy onto a simulator-only structure.

## Probe: `l1_cache/pointer_chase.cu`

### Goal

Measure L1 path hit latency for global/read-only/constant/texture-style access
paths where supported.

### Parameters

- `l1d_cache_config::l1_latency`
- `m_L1D_config`
- `m_L1C_config`
- `m_L0C_config`
- `m_L1T_config`

### Methodology

1. Allocate a pointer-chase list that fits inside the target cache level.
2. Construct dependent loads so the next address depends on the previous load.
3. Generate variants:
   - ordinary global load
   - cache-hinted global load if available
   - read-only path
   - constant path
   - texture path if the backend supports it
4. Keep working set below the expected L1 capacity for hit-latency runs.
5. Sweep active warps to separate single-warp hit latency from scheduler hiding.
6. Collect NCU/CUPTI `l1tex__*` hit, request, sector, and throughput metrics.
7. Compare against shared-memory latency and L2 pointer-chase latency controls.

### Reasoning

Dependent pointer chasing removes memory-level parallelism and makes the latency
of the selected path visible. Multiple source variants are necessary because
NVIDIA cache paths do not map one-to-one to simulator cache objects.

### Risk Estimate

Risk: Medium.

Main risks:

- Loads may bypass or use different cache paths depending on compiler, address
  space, and cache operators.
- L1 behavior can be sectorized or unified with shared memory.
- Hit-rate counters may not isolate a single path.

Mitigation:

- Disassemble load opcodes and cache modifiers.
- Use working sets that clearly fit L1 and clearly exceed L1 as controls.
- Report path-specific estimates instead of collapsing them prematurely.

## Probe: `l1_cache/working_set.cu`

### Goal

Estimate cache capacity and line-size knees for L1-like paths.

### Parameters

- `cache_config::m_line_sz`
- `cache_config::m_nset`
- `m_L1D_config`
- `m_L1I_L1_half_C_cache_config`
- `m_L0I_config`
- `m_L1C_config`
- `m_L0C_config`
- `m_L1T_config`

### Methodology

1. Run pointer-chase or strided load loops across increasing working-set sizes.
2. Use randomized pointer permutations to defeat simple prefetch/stream effects.
3. Sweep access stride to detect line-size transitions.
4. Repeat for data, read-only, constant, texture, and instruction-footprint
   variants when practical.
5. Fit latency or miss-rate knees:
   - first knee: likely L1 capacity/effective capacity
   - stride knee: likely line/sector granularity
6. Validate with NCU/CUPTI hit-rate and sector metrics.

### Reasoning

Cache capacity and line size are usually visible as knees in latency or
transaction curves. Randomized dependent traversals reduce false bandwidth
signals and emphasize cache residency.

### Risk Estimate

Risk: Medium.

Main risks:

- Replacement policy and cache partitioning can blur capacity knees.
- Real hardware may use sector lines, making simulator `m_line_sz` ambiguous.
- Instruction-cache variants require careful code-size generation.

Mitigation:

- Preserve full curves and report effective capacity ranges.
- Distinguish line size from sector size when counters expose sectors.
- Use confidence ranges rather than a single number when knees are broad.

## Probe: `l1_cache/conflict_sets.cu`

### Goal

Estimate associativity-like behavior and cache-index conflicts.

### Parameters

- `cache_config::m_assoc`
- `cache_config::m_nset`
- `cache_config::m_mshr_entries`
- `cache_config::m_mshr_max_merge`
- `cache_config::m_miss_queue_size`
- `l1d_cache_config::l1_banks`

### Methodology

1. Build address sets with controlled spacing so candidate addresses map to the
   same index under simple indexing hypotheses.
2. Sweep number of conflicting lines.
3. Measure latency and miss metrics as conflict-set size grows.
4. Add parallel miss variants to stress MSHR and miss-queue behavior.
5. Compare conflict behavior across address-bit hypotheses.
6. Use NCU/CUPTI L1 sectors, misses, and replay metrics.

### Reasoning

Associativity-like behavior appears when adding one more line to a conflict set
causes a stable miss-rate or latency increase. Parallel miss streams can reveal
outstanding miss capacity, though that is more coupled than associativity.

### Risk Estimate

Risk: Medium to High.

Main risks:

- NVIDIA cache indexing may be hashed or undocumented.
- Replacement policy can obscure exact associativity.
- MSHR and miss-queue estimates are coupled with L2 latency and warp scheduling.

Mitigation:

- Treat associativity as effective associativity.
- Test multiple index hypotheses.
- Mark MSHR/miss-queue estimates as `coupled_inference`.

## Probe: `l1_cache/analyze.py`

### Goal

Convert cache latency, working-set, and conflict curves into fitted simulator
cache parameters.

### Methodology

1. Detect stable knees using segmented regression or derivative thresholds.
2. Detect line/sector granularity from stride-response changes.
3. Compare fitted values against NCU/CUPTI hit/miss and sector counters.
4. Emit separate estimates by cache path:
   - data
   - read-only
   - constant
   - texture
   - instruction, when measured
5. Assign confidence based on repeatability and counter agreement.

### Reasoning

Cache structure is not directly exposed. The analyzer must maintain the
difference between measured behavior and simulator configuration equivalents.

### Risk Estimate

Risk: Medium.

Main risks:

- Overfitting noisy knees.
- Collapsing sectorized behavior into an incorrect single line size.

Mitigation:

- Require repeatable knees across repetitions.
- Store raw curves in the result package.
- Emit ranges when one number is not defensible.

## Probe: `scheduler_policy/ready_warps.cu`

### Goal

Infer scheduler count, active-warp scaling, and broad scheduling behavior.

### Parameters

- `shader_core_config::num_subcores_in_SM`
- `shader_core_config::gpgpu_num_sched_per_core`
- `shader_core_config::gpgpu_scheduler_string`
- `shader_core_config::gpgpu_max_insn_issue_per_warp`

### Methodology

1. Create kernels with controlled numbers of ready warps per SM.
2. Use independent arithmetic instructions to minimize dependency stalls.
3. Sweep active warps from one warp upward.
4. Measure issue throughput and per-warp progress.
5. Add variants where only selected warps are ready at controlled intervals.
6. Collect NCU/CUPTI `smsp__*` issue, eligible-warps, active-warps, and stall
   metrics.
7. Use PC Sampling to identify not-selected or scoreboard-related stalls.

### Reasoning

Scheduler capacity and policy are visible in how issue rate scales with ready
warps and how progress is distributed across warps. The result maps behavior to
simulator policy labels, not vendor policy names.

### Risk Estimate

Risk: High.

Main risks:

- Scheduler policy is not directly exposed.
- Warp distribution across SMSPs/subcores is not fully controllable.
- Measured behavior is coupled with instruction pipe and scoreboard behavior.

Mitigation:

- Use several independent readiness patterns.
- Report behavioral classification with confidence, not a hard policy name.
- Cross-check with arithmetic throughput P0 results.

## Probe: `scheduler_policy/mixed_issue.cu`

### Goal

Detect dual-issue or mixed-pipeline issue behavior.

### Parameters

- `shader_core_config::gpgpu_dual_issue_diff_exec_units`
- `shader_core_config::pipeline_widths_string`
- `shader_core_config::pipe_widths`
- `shader_core_config::gpgpu_max_insn_issue_per_warp`

### Methodology

1. Generate instruction streams that alternate independent operations from two
   different classes:
   - FP32 + INT
   - FP32 + SFU
   - FP32 + memory
   - FP32 + tensor when tensor P2 support is available
2. Compare measured throughput against single-pipe baselines.
3. Sweep active warps and independent chain count.
4. Collect NCU/CUPTI pipe utilization and issue metrics.
5. Validate dynamic instruction mix with NVBit.

### Reasoning

If two instruction classes can issue in the same cycle or overlap more strongly
than single-pipe limits predict, mixed streams should exceed the slower
single-pipe expectation. This informs simulator dual-issue and pipeline width
fields.

### Risk Estimate

Risk: Medium to High.

Main risks:

- Apparent overlap may come from latency hiding rather than dual issue.
- Instruction pairing rules can be architecture-specific and opcode-specific.
- Counters may aggregate in ways that hide issue pairing.

Mitigation:

- Compare against both one-warp and many-warp cases.
- Use disassembly and NVBit opcode counts.
- Mark dual-issue classification as medium confidence unless multiple streams
  agree.

## Probe: `scheduler_policy/analyze.py`

### Goal

Fit scheduler and issue behavior from ready-warp and mixed-issue results.

### Methodology

1. Estimate throughput scaling versus ready warps.
2. Detect saturation point and per-SMSP issue capacity.
3. Compare mixed-stream throughput against additive, max-only, and partially
   overlapped models.
4. Classify scheduler behavior by similarity:
   - round-robin-like
   - greedy-like
   - two-level-like
   - unknown/coupled
5. Emit simulator mappings with confidence and `coupled_with` metadata.

### Reasoning

The analyzer avoids pretending NVIDIA exposes simulator scheduler names. It
should map observed behavior onto the closest simulator abstraction.

### Risk Estimate

Risk: High.

Main risks:

- Multiple simulator policies can fit the same throughput data.
- PC Sampling is statistical and may not resolve fine-grained policy.

Mitigation:

- Require multiple readiness patterns for policy classification.
- Report unknown when policy evidence is weak.

## Probe: `register_file/register_bank_sweep.sass`

### Goal

Infer register bank count and bank-conflict effects using controlled register
numbering.

### Parameters

- `shader_core_config::gpgpu_num_reg_banks`
- `shader_core_config::reg_file_port_throughput`
- `gpgpu_reg_bank_use_warp_id`
- `num_regular_register_file_read_ports_per_bank`
- `num_regular_register_file_write_ports_per_bank`

### Methodology

1. Generate SASS or inline-PTX variants with explicit source and destination
   register numbers.
2. Construct same-bank and distributed-bank operand patterns under candidate
   bank mappings.
3. Sweep:
   - register stride
   - number of source operands
   - destination register pattern
   - warp ID if bank mapping may include warp ID
4. Measure throughput degradation relative to conflict-free candidates.
5. Collect NCU/CUPTI scoreboard or issue-stall counters where available.
6. Use NVBit register instrumentation only for narrow validation windows.

### Reasoning

Register bank conflicts should appear as periodic throughput changes when
register numbers alias under the true bank mapping. SASS is required because
compiler register allocation would otherwise hide the experiment.

### Risk Estimate

Risk: High.

Main risks:

- Handwritten SASS tooling may be fragile.
- Register renaming or assembler constraints may alter intended numbering.
- Bank mapping may include undocumented hashing or warp ID.
- Throughput changes may be caused by operand collectors rather than banks.

Mitigation:

- Disassemble and verify register numbers.
- Test multiple mapping hypotheses.
- Mark port counts and collector-related outputs as `coupled_inference`.

## Probe: `register_file/register_latency.cu`

### Goal

Measure dependent register read-after-write latency and operand delivery cost.

### Parameters

- `max_latency_regular_register_file_latency`
- `shader_core_config::reg_file_port_throughput`
- `gpgpu_operand_collector_num_units_*`

### Methodology

1. Build dependent arithmetic chains that vary operand count and register reuse.
2. Compare:
   - same register reuse
   - rotating registers
   - candidate same-bank registers
   - candidate distributed-bank registers
3. Use P0 arithmetic latency as the baseline.
4. Attribute excess latency or throughput loss to register/operand delivery only
   when arithmetic opcode and schedule are unchanged.
5. Collect scoreboard and issue-stall metrics.

### Reasoning

Register-file latency is hard to isolate because arithmetic latency already
includes operand delivery. The useful signal is differential: same opcode,
different register pattern.

### Risk Estimate

Risk: High.

Main risks:

- Arithmetic latency, scoreboard behavior, and operand collectors are entangled.
- Compiler register allocation can invalidate CUDA-source variants.
- Counters may not expose register-bank conflicts directly.

Mitigation:

- Prefer SASS-controlled variants for high-confidence claims.
- Report differential penalties separately from absolute latency.
- Couple estimates with arithmetic and scheduler probe IDs.

## Probe: `register_file/analyze.py`

### Goal

Fit register-bank and operand-delivery behavior from SASS and CUDA variants.

### Methodology

1. Detect periodic throughput penalties across register-number strides.
2. Score candidate bank mappings.
3. Compare same-bank and distributed-bank confidence.
4. Estimate port pressure only when operand-count sweeps show stable plateaus.
5. Emit:
   - high/medium confidence for bank count when periodicity is clear
   - low/medium confidence for port counts
   - coupled estimates for operand collector parameters

### Reasoning

Register bank count may be observable through periodicity. Operand collector
structure is usually not directly observable and must remain explicitly coupled.

### Risk Estimate

Risk: High.

Main risks:

- False periodicity from scheduler or instruction-cache alignment.
- Multiple bank mappings fitting similar data.

Mitigation:

- Require repeatability across unroll factors and warp counts.
- Include negative controls with unrelated register permutations.

## P1 Output Requirements

Every P1 result should include:

- dependency on P0 baselines
- source and disassembly hashes
- raw latency/throughput curves
- counter evidence and metric availability
- PC Sampling evidence when used
- NVBit validation stream path when used
- simulator estimate
- confidence
- `coupled_with` references
- risk notes

## P1 Acceptance Criteria

P1 is complete when AMORA can report:

- L1 hit-latency estimate for at least one data path
- L1 line or sector granularity estimate
- effective L1 capacity range
- scheduler issue-scaling curve
- mixed-issue classification for at least FP32+INT and FP32+SFU
- register-bank periodicity result or an explicit unsupported/indeterminate
  outcome
- clear separation between direct measurements and simulator-equivalent fitted
  parameters
