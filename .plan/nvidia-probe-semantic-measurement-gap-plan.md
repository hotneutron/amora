# NVIDIA Probe Semantic and Measurement Gap Plan

## Summary

Implement the first executable AMORA probing suite for NVIDIA GPUs. The suite
will map the simulator-oriented parameters identified in
`/Users/bytedance/wk/amora/.plan/probing-suite-microarchitecture-plan.md` onto
real-hardware measurements using two coordinated tracks:

1. Tooling adapters for readily available evidence from NCU, CUPTI, and NVBit.
2. Custom CUDA/SASS microkernels for parameters that cannot be measured directly
   by available tools.

The implementation order should start with high-signal, low-gap probes that use
available CUDA runtime metadata, NCU/CUPTI metrics, and simple kernels. It should
then move toward semantic-gap-heavy probes where a simulator parameter is only an
indirect or coupled approximation of real hardware behavior.

## Current State Analysis

The repository has an empty implementation skeleton:

- `/Users/bytedance/wk/amora/amora/core`
- `/Users/bytedance/wk/amora/amora/backends`
- `/Users/bytedance/wk/amora/amora/probes`
- `/Users/bytedance/wk/amora/amora/reports`
- `/Users/bytedance/wk/amora/amora/schemas`
- `/Users/bytedance/wk/amora/tools/doc_fetch`
- `/Users/bytedance/wk/amora/tools/metric_scrape`
- `/Users/bytedance/wk/amora/tools/report_render`

The existing plan at
`/Users/bytedance/wk/amora/.plan/probing-suite-microarchitecture-plan.md`
defines the target parameter families:

- topology and occupancy
- arithmetic latency and initiation interval
- issue width, scheduler behavior, and functional-unit throughput
- register file and operand collector behavior
- shared memory
- L1, constant, texture, and instruction caches
- SM memory pipeline and coalescing
- L2, DRAM, memory partitions, and DRAM timing
- interconnect and address mapping
- tensor core and matrix engine behavior
- TMA, DMA, and async copy behavior
- synchronization and barrier behavior

The NVIDIA documentation tree now contains CUPTI and NVBit notes:

- `/Users/bytedance/wk/amora/docs/vendors/nvidia/cupti/docs.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/cupti/host-profiling.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/cupti/range-profiling.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/cupti/pm-sampling.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/cupti/pc-sampling.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/cupti/sass-metrics.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/cupti/api-notes.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/nvbit/docs.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/nvbit/instrumentation-model.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/nvbit/examples.md`
- `/Users/bytedance/wk/amora/docs/vendors/nvidia/nvbit/limitations.md`

Key constraints from those notes:

- CUPTI is the programmable metric backend.
- CUPTI Range Profiling is the preferred exact metric path.
- CUPTI Host Profiling is needed for metric enumeration, config images, and
  metric evaluation.
- CUPTI PM Sampling is useful for time-varying behavior but is not a replacement
  for exact per-kernel metrics.
- CUPTI PC Sampling and SASS Metrics provide attribution, not complete semantic
  reconstruction.
- NVBit is a dynamic SASS instrumentation layer and should be used for dynamic
  instruction and memory-reference streams, not as a hardware counter backend.
- NVBit and Nsight/NCU-style injection workflows should be run as separate
  passes and correlated offline.

## Semantic and Measurement Gap Matrix

| Parameter Family | Simulator Semantics | Readily Available Evidence | Remaining Gap | Implementation Priority |
|---|---|---|---|---|
| Topology and occupancy | Static limits such as SM count, warp size, max CTAs, max warps, registers, shared memory | CUDA device attributes, occupancy APIs, launch metadata, NCU `launch__*`, simple persistent-CTA kernels | Cluster topology and exact residency policies are partly architectural and partly scheduler/runtime policy | P0 |
| Arithmetic latency | Fixed simulator latency per opcode class | Dependent CUDA/SASS loops, CUPTI/NCU instruction metrics, NVBit opcode validation | Compiler scheduling and SASS selection can hide intended dependency chains unless disassembly is verified | P0 |
| Arithmetic throughput and functional-unit count | Unit counts and initiation intervals | Independent instruction streams, NCU pipe utilization, CUPTI Range Profiling, NVBit opcode histograms | Unit count is inferred from throughput and clocks, not directly exposed | P0 |
| Shared memory | Bank count, bank mapping, latency, broadcast/multicast behavior | Shared-memory stride kernels, NCU shared metrics, CUPTI Range Profiling | Bank mapping and broadcast policy are semantic approximations across generations | P0 |
| L1 and cache geometry | Explicit cache set/line/associativity/MSHR parameters | Pointer chase, working-set sweeps, NCU `l1tex__*`, CUPTI metrics | Real cache policies are adaptive and may not map cleanly to simulator replacement/MSHR fields | P1 |
| Scheduler and issue | Scheduler count, dual-issue policy, per-warp issue limits | Controlled ready-warp kernels, NCU `smsp__*` stall/issue metrics, PC Sampling | Scheduler policy names such as GTO or round-robin are simulator abstractions, not vendor-exposed facts | P1 |
| Register file and operand collector | Bank count, ports, collector units, RF latency | Register-numbered SASS loops, NVBit register-value tools, NCU stall metrics | Operand collector structure is mostly not observable; measurements are coupled with scheduler and pipe behavior | P1 |
| SM memory pipeline and coalescing | Coalescing cycles, queue depths, PRT limits, scalar units | Lane-address kernels, NVBit memory traces, NCU/CUPTI L1TEX metrics | Internal queue depths and coalescing windows are inferred from cliffs and can be confounded by L1/L2 behavior | P2 |
| L2 and DRAM bandwidth/latency | Cache config, partitions, burst length, DRAM timings, scheduler queues | Pointer chase, streaming kernels, NCU `lts__*`/`dram__*`, CUPTI Range Profiling/PM Sampling | DRAM timing fields are highly coupled; exact bank-group policy and scheduler internals are not directly exposed | P2 |
| Interconnect/address mapping | Flit size, routing delay, VC/switch allocation, address mask | Partition-camping sweeps, latency under load, CUPTI/NCU memory partition metrics where available | NoC/router fields are simulator-specific and only indirectly recoverable | P3 |
| Tensor core | Tensor latency, rate, unit count, supported shapes | WMMA/MMA microkernels, NCU tensor metrics, CUPTI Range Profiling, NVBit opcode validation | Shape-specific extra latency and issue sharing are coupled with scheduler and operand delivery | P2 |
| TMA/async copy | Copy-engine queues, requests per cycle, transfer entries, completion path | Hopper-style async/TMA kernels where supported, NCU/CUPTI memory metrics, PC/SASS attribution | TMA internal state names are simulator structures; real hardware exposes limited direct observability | P3 |
| Synchronization/barriers | Barrier counts, fence costs, mbarrier/grid/cluster semantics | Barrier/fence microkernels, PC Sampling stalls, NCU scheduler stalls | Semantic mapping from real barrier instructions to simulator barrier classes is architecture-specific | P2 |

## Measurement Evidence Tiers

Every estimate should be tagged with one of these evidence tiers:

- `direct_metadata`: CUDA runtime/device attribute or launch metadata.
- `direct_counter`: CUPTI/NCU metric with close semantic match.
- `timing_direct`: custom microkernel timing with a single dominant parameter.
- `instrumented_stream`: NVBit dynamic instruction, memory, or register stream.
- `coupled_inference`: fitted estimate involving multiple hidden parameters.
- `unsupported`: probe skipped with explicit reason.

Confidence should be derived from evidence tier, repeatability, disassembly
verification, metric availability, and degree of parameter coupling.

## Proposed Changes

### 1. Create Probe Suite Package Skeleton

Files:

- `/Users/bytedance/wk/amora/amora/schemas/hardware_profile.schema.json`
- `/Users/bytedance/wk/amora/amora/schemas/probe_result.schema.json`
- `/Users/bytedance/wk/amora/amora/schemas/simulator_parameter_map.yaml`
- `/Users/bytedance/wk/amora/amora/core/runner.py`
- `/Users/bytedance/wk/amora/amora/core/statistics.py`
- `/Users/bytedance/wk/amora/amora/core/parameter_model.py`
- `/Users/bytedance/wk/amora/amora/core/capabilities.py`
- `/Users/bytedance/wk/amora/amora/backends/base.py`
- `/Users/bytedance/wk/amora/amora/backends/nvidia_cuda.py`
- `/Users/bytedance/wk/amora/amora/backends/ncu.py`
- `/Users/bytedance/wk/amora/amora/backends/cupti.py`
- `/Users/bytedance/wk/amora/amora/backends/nvbit.py`
- `/Users/bytedance/wk/amora/amora/reports/markdown.py`
- `/Users/bytedance/wk/amora/amora/reports/json_report.py`

What:

- Define schemas, backend interfaces, probe result records, capability
  discovery, statistical fitting, and report generation.

Why:

- The existing repository has package directories but no executable probing
  framework.
- The CUPTI/NVBit tools have different runtime constraints and must be modeled
  as separate evidence sources.

How:

- `base.py` defines abstract methods for device discovery, build, run, collect,
  parse, and capability query.
- `nvidia_cuda.py` handles CUDA compilation, launch, buffers, timers, and device
  attributes.
- `ncu.py` wraps `ncu --query-metrics`, `--set`, `--metrics`, and CSV raw-page
  export.
- `cupti.py` initially wraps a small CLI/helper binary contract rather than
  embedding all CUPTI C++ code in Python.
- `nvbit.py` invokes NVBit tools in separate runs using `LD_PRELOAD` or
  `CUDA_INJECTION64_PATH`.

### 2. Define Stable Result Schemas

Files:

- `/Users/bytedance/wk/amora/amora/schemas/probe_result.schema.json`
- `/Users/bytedance/wk/amora/amora/schemas/hardware_profile.schema.json`

Probe result fields:

- `probe_id`
- `family`
- `target`
- `backend`
- `kernel`
- `launch_config`
- `source_kind`: `cuda|ptx|sass|external_binary|tool_only`
- `disassembly_hash`
- `measurement_evidence_tier`
- `raw_measurements`
- `counter_values`
- `nvbit_streams`
- `estimates`
- `unsupported_reason`

Estimate fields:

- `parameter`
- `value`
- `unit`
- `confidence`
- `evidence_tier`
- `probe_ids`
- `coupled_with`
- `notes`

Why:

- Semantic gaps must remain visible in the data model. A coupled estimate should
  not look like a direct measurement.

### 3. Implement Capability Discovery First

Files:

- `/Users/bytedance/wk/amora/amora/core/capabilities.py`
- `/Users/bytedance/wk/amora/amora/backends/nvidia_cuda.py`
- `/Users/bytedance/wk/amora/amora/backends/ncu.py`
- `/Users/bytedance/wk/amora/amora/backends/cupti.py`
- `/Users/bytedance/wk/amora/amora/backends/nvbit.py`

What:

- Discover CUDA device properties, available NCU metrics, CUPTI feature support,
  NVBit availability, `nvcc`, `nvdisasm`, driver version, and profiling
  permissions.

Why:

- Metric availability and tool compatibility vary by CUDA version, driver,
  GPU architecture, MIG/MPS/vGPU mode, and permissions.

Acceptance:

- Unsupported features are reported before probe execution.
- The runner can skip probes instead of producing misleading empty estimates.

### 4. Implement P0 Readily Available Probes

#### 4.1 Topology and Occupancy

Files:

- `/Users/bytedance/wk/amora/amora/probes/topology/device_attributes.py`
- `/Users/bytedance/wk/amora/amora/probes/topology/persistent_cta.cu`
- `/Users/bytedance/wk/amora/amora/probes/topology/occupancy.py`

Measurements:

- CUDA device attributes for SM count, warp size, registers, shared memory,
  max threads, max blocks, and clock data.
- Persistent CTA kernel to estimate concurrent CTA residency.
- Sweep block size, register use, and dynamic shared memory.

Maps:

- `shader_core_config::warp_size`
- `shader_core_config::max_warps_per_shader`
- `shader_core_config::max_cta_per_core`
- `shader_core_config::gpgpu_shader_registers`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`
- `gpgpu_sim_config::num_shader()`

Gap handling:

- Treat CUDA metadata as direct for limits, but persistent CTA results as
  runtime-policy evidence.
- `n_simt_clusters` and `n_simt_cores_per_cluster` remain coupled unless
  architecture-specific metadata is available.

#### 4.2 Arithmetic Latency and Throughput

Files:

- `/Users/bytedance/wk/amora/amora/probes/arithmetic_latency/dependent_chain.cu`
- `/Users/bytedance/wk/amora/amora/probes/arithmetic_latency/independent_chains.cu`
- `/Users/bytedance/wk/amora/amora/probes/arithmetic_latency/analyze.py`

Measurements:

- Device-side cycle timing for dependent chains.
- Independent chains for reciprocal throughput.
- NCU/CUPTI counters for instruction counts and pipe utilization.
- NVBit opcode histogram pass for dynamic instruction validation.

Maps:

- `shader_core_config::max_sp_latency`
- `shader_core_config::max_int_latency`
- `shader_core_config::max_sfu_latency`
- `shader_core_config::max_dp_latency`
- `gpgpu_num_sp_units`
- `gpgpu_num_int_units`
- `gpgpu_num_sfu_units`
- `gpgpu_num_dp_units`

Gap handling:

- Require disassembly verification so compiler scheduling does not invalidate
  dependency chains.
- Unit counts are inferred from throughput and should be marked
  `coupled_inference`.

#### 4.3 Shared Memory

Files:

- `/Users/bytedance/wk/amora/amora/probes/shared_memory/pointer_chase.cu`
- `/Users/bytedance/wk/amora/amora/probes/shared_memory/bank_stride.cu`
- `/Users/bytedance/wk/amora/amora/probes/shared_memory/analyze.py`

Measurements:

- Pointer chase latency.
- Stride sweep for bank count and conflict periodicity.
- Broadcast and multicast-like patterns.
- NCU/CUPTI shared-memory request, transaction, and bank-conflict metrics.

Maps:

- `shader_core_config::gpgpu_shmem_num_banks`
- `shader_core_config::gpgpu_smem_latency`
- `gpgpu_shmem_limited_broadcast`
- `gpgpu_shmem_warp_parts`
- `memory_shared_memory_minimum_latency`

Gap handling:

- Bank count can be high confidence when stride periodicity is clean.
- Broadcast/multicast policy is semantic and should be medium confidence unless
  validated by counters.

### 5. Implement P1 Probes with Moderate Semantic Gaps

#### 5.1 L1, Constant, Texture, and Instruction Cache

Files:

- `/Users/bytedance/wk/amora/amora/probes/l1_cache/pointer_chase.cu`
- `/Users/bytedance/wk/amora/amora/probes/l1_cache/working_set.cu`
- `/Users/bytedance/wk/amora/amora/probes/l1_cache/conflict_sets.cu`
- `/Users/bytedance/wk/amora/amora/probes/l1_cache/analyze.py`

Measurements:

- Hit latency, capacity knees, line size, associativity-like conflicts.
- NCU/CUPTI `l1tex__*` metrics.
- Optional source variants for global, read-only, texture, constant, and
  instruction paths.

Maps:

- `m_L1D_config`
- `m_L1I_L1_half_C_cache_config`
- `m_L0I_config`
- `m_L1C_config`
- `m_L0C_config`
- `m_L1T_config`
- `cache_config::m_nset`
- `cache_config::m_line_sz`
- `cache_config::m_assoc`
- `l1d_cache_config::l1_latency`
- `l1d_cache_config::l1_banks`

Gap handling:

- Real caches may be unified, sectorized, adaptive, or policy-dependent. Report
  simulator fields as fitted equivalents, not direct hardware facts.

#### 5.2 Scheduler and Issue

Files:

- `/Users/bytedance/wk/amora/amora/probes/scheduler_policy/ready_warps.cu`
- `/Users/bytedance/wk/amora/amora/probes/scheduler_policy/mixed_issue.cu`
- `/Users/bytedance/wk/amora/amora/probes/scheduler_policy/analyze.py`

Measurements:

- Active-warp sweep.
- Single-pipe and mixed-pipe issue streams.
- NCU/CUPTI `smsp__*` issue and stall metrics.
- PC Sampling for stall attribution.

Maps:

- `shader_core_config::num_subcores_in_SM`
- `shader_core_config::gpgpu_num_sched_per_core`
- `shader_core_config::gpgpu_scheduler_string`
- `shader_core_config::gpgpu_max_insn_issue_per_warp`
- `shader_core_config::gpgpu_dual_issue_diff_exec_units`
- `shader_core_config::pipeline_widths_string`
- `shader_core_config::pipe_widths`

Gap handling:

- Scheduler string is a simulator policy label. Map by behavioral similarity and
  keep confidence below direct metadata/counter estimates.

#### 5.3 Register File and Operand Collector

Files:

- `/Users/bytedance/wk/amora/amora/probes/register_file/register_bank_sweep.sass`
- `/Users/bytedance/wk/amora/amora/probes/register_file/register_latency.cu`
- `/Users/bytedance/wk/amora/amora/probes/register_file/analyze.py`

Measurements:

- Controlled register-numbering SASS loops.
- Same-bank versus distributed-bank throughput.
- Dependent read-after-write chains.
- NVBit register-value instrumentation for validation windows.
- NCU/CUPTI stall metrics for operand collector or scoreboard pressure where
  available.

Maps:

- `shader_core_config::gpgpu_num_reg_banks`
- `shader_core_config::reg_file_port_throughput`
- `gpgpu_reg_bank_use_warp_id`
- `num_regular_register_file_read_ports_per_bank`
- `num_regular_register_file_write_ports_per_bank`
- `max_latency_regular_register_file_latency`
- `gpgpu_operand_collector_num_units_*`

Gap handling:

- Operand collector fields are mostly hidden. Treat collector estimates as
  coupled with scheduler and functional-unit throughput.

### 6. Implement P2 Probes with Strong Coupling

#### 6.1 SM Memory Pipeline and Coalescing

Files:

- `/Users/bytedance/wk/amora/amora/probes/memory_pipeline/lane_patterns.cu`
- `/Users/bytedance/wk/amora/amora/probes/memory_pipeline/outstanding_requests.cu`
- `/Users/bytedance/wk/amora/amora/probes/memory_pipeline/analyze.py`

Measurements:

- Lane address pattern sweeps.
- Independent memory instruction depth sweep.
- Active warp sweep.
- NVBit memory-reference traces.
- NCU/CUPTI `l1tex__*`, memory dependency, and request metrics.

Maps:

- `memory_subcore_queue_size`
- `memory_intermidiate_stages_subcore_unit`
- `memory_sm_prt_size`
- `memory_l1d_max_lookups_per_cycle_per_bank`
- `memory_maximum_coalescing_cycles`
- `memory_num_scalar_units_per_subcore`
- `memory_subcore_link_to_sm_byte_size`
- `memmory_max_concurrent_requests_standard_per_sm`

Gap handling:

- Queue depths and coalescing windows are inferred from throughput cliffs. Mark
  as coupled with L1, L2, and scheduler behavior.

#### 6.2 L2 and DRAM

Files:

- `/Users/bytedance/wk/amora/amora/probes/l2_cache/pointer_chase.cu`
- `/Users/bytedance/wk/amora/amora/probes/global_memory/streaming.cu`
- `/Users/bytedance/wk/amora/amora/probes/global_memory/partition_sweep.cu`
- `/Users/bytedance/wk/amora/amora/probes/global_memory/row_policy_sweep.cu`
- `/Users/bytedance/wk/amora/amora/probes/global_memory/analyze.py`

Measurements:

- L2 hit latency with working sets exceeding L1.
- L2 capacity and conflict-like behavior.
- Streaming bandwidth plateaus.
- Address bit sweeps for partition camping.
- Row-hit and row-miss style access patterns.
- NCU/CUPTI `lts__*` and `dram__*` metrics.

Maps:

- `memory_config::m_L2_config`
- `gpgpu_cache:dl2`
- `gpgpu_l2_rop_latency`
- `memory_config::m_n_mem`
- `memory_config::m_n_sub_partition_per_memory_channel`
- `memory_config::scheduler_type`
- `memory_config::busW`
- `memory_config::BL`
- `memory_config::nbk`
- `memory_config::nbkgrp`
- `memory_config::tCCD`
- `memory_config::tRCD`
- `memory_config::tRAS`
- `memory_config::tRP`
- `memory_config::CL`
- `memory_config::WL`
- `dram_latency`
- `dram_data_command_freq_ratio`

Gap handling:

- Report DRAM timing parameters as fitted effective parameters, not literal
  vendor DRAM timings.
- Multiple simulator fields may be explained by one bandwidth or latency curve;
  record `coupled_with` aggressively.

#### 6.3 Tensor Core and Synchronization

Files:

- `/Users/bytedance/wk/amora/amora/probes/tensor_core/mma_latency.cu`
- `/Users/bytedance/wk/amora/amora/probes/tensor_core/mma_throughput.cu`
- `/Users/bytedance/wk/amora/amora/probes/synchronization/barrier_latency.cu`
- `/Users/bytedance/wk/amora/amora/probes/synchronization/fence_latency.cu`

Measurements:

- Dependent and independent MMA instruction streams.
- Shape and datatype sweeps.
- CTA barrier and fence timing under varying active warp counts.
- NCU/CUPTI tensor, stall, and barrier-related metrics.
- NVBit opcode validation for MMA instruction mix.

Maps:

- `gpgpu_tensor_core_avail`
- `gpgpu_num_tensor_core_units`
- `tensor_latency`
- `tensor_rate_per_cycle`
- `shader_core_config::max_tensor_core_latency`
- `tensor_extra_latency_16816_fp32_1688_fp32`
- `gpgpu_num_cta_barriers`
- `BARRIER_OP`
- `MEMORY_BARRIER_OP`
- `MBARRIER_OP`
- `CLUSTER_BARRIER_OP`

Gap handling:

- Tensor unit count is coupled with clock, scheduler, issue limits, and operand
  delivery.
- Barrier operation classes are ISA and simulator semantic mappings, not direct
  hardware counter names.

### 7. Implement P3 Biggest-Gap Probes Last

#### 7.1 TMA, DMA, and Async Copy

Files:

- `/Users/bytedance/wk/amora/amora/probes/tma_copy/async_copy_latency.cu`
- `/Users/bytedance/wk/amora/amora/probes/tma_copy/tma_transfer_sweep.cu`
- `/Users/bytedance/wk/amora/amora/probes/tma_copy/analyze.py`

Measurements:

- Issue-to-completion latency.
- Transfer bandwidth by size, stride, rank, and alignment.
- Wait/barrier completion semantics.
- NCU/CUPTI memory metrics and PC/SASS attribution where supported.

Maps:

- `tma_unit_sm::kMaxRequestsPerCycle`
- `TMACommand`
- `TMATransferEntry`
- `TMAOpcodeFamily`
- `TMADirection`
- `TMATransferType`
- `TMAOperandForm`
- `m_command_queue`
- `m_in_flight_transfers`
- `m_outstanding_requests`
- `m_outstanding_stores_per_warp`

Gap handling:

- Most TMA names are simulator internal states. Report only behavioral
  equivalents unless a direct ISA event or metric exists.

#### 7.2 Interconnect and Address Mapping

Files:

- `/Users/bytedance/wk/amora/amora/probes/interconnect/address_mapping.cu`
- `/Users/bytedance/wk/amora/amora/probes/interconnect/injection_rate.cu`
- `/Users/bytedance/wk/amora/amora/probes/interconnect/analyze.py`

Measurements:

- Address bit to partition mapping.
- Bandwidth and latency under controlled injection rates.
- Multi-CTA partition-camping tests.
- CUPTI/NCU partition or fabric metrics when available.

Maps:

- `icnt_flit_size`
- `gpgpu_mem_addr_mapping`
- `gpgpu_mem_address_mask`
- `routing_delay`
- `vc_alloc_delay`
- `sw_alloc_delay`
- `credit_delay`
- `input_speedup`
- `output_speedup`
- `internal_speedup`
- `output_buffer_size`
- `use_noc_latency`

Gap handling:

- Router and VC fields are simulator constructs. They should be estimated only
  as effective parameters after L2/DRAM and coalescing probes are stable.

### 8. Add Tool Correlation Workflow

Files:

- `/Users/bytedance/wk/amora/amora/core/correlation.py`
- `/Users/bytedance/wk/amora/amora/backends/ncu.py`
- `/Users/bytedance/wk/amora/amora/backends/cupti.py`
- `/Users/bytedance/wk/amora/amora/backends/nvbit.py`

What:

- Correlate separate runs by `probe_id`, kernel name, launch config, problem
  size, binary hash, and disassembly hash.

Why:

- NVBit and Nsight/NCU-style tooling cannot normally run together. CUPTI
  profiling may require replay. AMORA needs offline correlation across separate
  evidence passes.

Acceptance:

- A single probe can attach timing results, NCU/CUPTI metrics, PC/SASS
  attribution, and NVBit streams without assuming they came from one process.

### 9. Add Reports and Coverage Matrix

Files:

- `/Users/bytedance/wk/amora/amora/reports/markdown.py`
- `/Users/bytedance/wk/amora/amora/reports/json_report.py`

Report sections:

- target identity and tool versions
- capability discovery results
- semantic gap matrix for executed probe families
- measurement coverage matrix by simulator parameter
- high-confidence estimates
- coupled estimates
- unsupported parameters with reasons
- raw result summary
- suggested next probes to reduce uncertainty

## Ordered Roadmap

### P0: Readily Available and Low-Gap

1. Schemas and parameter map.
2. Capability discovery.
3. CUDA device metadata and occupancy probes.
4. Arithmetic latency and throughput microkernels.
5. Shared-memory latency and bank-conflict probes.
6. NCU/CUPTI metric query and CSV/result ingestion.
7. NVBit opcode histogram integration for validation.

Expected outcome:

- First JSON and Markdown reports with high-confidence estimates for warp size,
  SM count, occupancy limits, basic arithmetic latency/throughput, shared memory
  bank count, shared latency, and direct NCU/CUPTI metrics.

### P1: Moderate-Gap Structure

1. L1/cache geometry probes.
2. Scheduler and issue behavior probes.
3. Register bank and operand delivery probes.
4. PC Sampling and SASS Metrics attribution integration.

Expected outcome:

- Medium/high-confidence fitted cache parameters.
- Behavioral scheduler and dual-issue classifications.
- Initial register-bank and operand-delivery estimates with explicit coupling.

### P2: Strongly Coupled Subsystems

1. SM memory pipeline and coalescing probes.
2. L2 and DRAM probes.
3. Tensor core probes.
4. Synchronization and barrier probes.
5. PM Sampling support for long-running and phase-based probes.

Expected outcome:

- Effective L2/DRAM and memory-pipeline parameters with coupling metadata.
- Tensor throughput and latency estimates.
- Barrier/fence cost profiles.

### P3: Biggest Semantic Gaps

1. TMA/async-copy probe family.
2. Interconnect and address-mapping probe family.
3. Effective router/queue parameter fitting.
4. Architecture-specific refinements for Hopper/Blackwell-style features.

Expected outcome:

- Behavioral equivalents for simulator-specific TMA and interconnect fields.
- Explicit list of parameters that remain unobservable without vendor-private
  documentation or stronger hardware counter support.

## Assumptions and Decisions

- First executable target is NVIDIA CUDA hardware.
- Framework interfaces should remain portable, but implementation starts with
  NVIDIA-specific backends.
- Tooling adapters and custom microkernels are implemented in parallel tracks.
- CUPTI/NCU metrics are evidence, not the only source of truth.
- NVBit is used for dynamic instruction, memory-reference, and register-stream
  validation, not for hardware counter access.
- The first implementation should emit reports and profiles. It should not
  automatically rewrite simulator configuration files.
- Every parameter estimate must include confidence, evidence tier, probe IDs,
  and coupling notes.

## Verification Steps

### Static Verification

1. Validate JSON schemas.
2. Validate `simulator_parameter_map.yaml` contains only documented target
   parameter names.
3. Unit-test statistical helpers for median, MAD, slope fitting, plateau
   detection, knee detection, and periodicity detection.
4. Unit-test report rendering from synthetic probe results.

### Tool Verification

1. Verify `nvcc`, `nvdisasm`, CUDA runtime, NCU, CUPTI helper, and NVBit tool
   availability are detected independently.
2. Verify unavailable tools produce `unsupported` evidence records.
3. Verify NCU metric query parsing against a saved sample.
4. Verify NVBit output parsing against a small opcode histogram sample.

### Kernel Verification

1. Compile and disassemble every microkernel.
2. Reject or warn when disassembly does not match the expected dependency or
   instruction pattern.
3. Run a no-op kernel to validate launch plumbing.
4. Run timer calibration to validate monotonic device timing.
5. Run repeated arithmetic and shared-memory probes to confirm low variance.

### Acceptance Criteria

The implementation is complete for the first milestone when:

- AMORA can run P0 probes on one NVIDIA CUDA target.
- It emits a machine-readable JSON hardware profile.
- It emits a Markdown report with a semantic gap and coverage matrix.
- It provides estimates for at least:
  - `shader_core_config::warp_size`
  - `shader_core_config::max_warps_per_shader`
  - `shader_core_config::gpgpu_shmem_size`
  - `shader_core_config::gpgpu_shmem_num_banks`
  - `shader_core_config::gpgpu_smem_latency`
  - `shader_core_config::max_sp_latency`
  - `shader_core_config::max_int_latency`
  - `shader_core_config::max_sfu_latency`
  - `cache_config::m_line_sz`
  - `l1d_cache_config::l1_latency`
- Each estimate has value, unit, confidence, evidence tier, probe IDs, and
  coupling notes.
- Unsupported probes are listed with explicit reasons.

## Rollout Notes

Implement P0 first and keep P1-P3 behind explicit probe-family flags. This keeps
early reports useful while avoiding false precision for the biggest semantic
gaps. Promote a probe family from experimental to default only after its
disassembly checks, repeatability checks, and cross-tool validation are stable.
