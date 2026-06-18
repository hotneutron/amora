# Probing Suite Plan: Reverse-Engineer Microarchitectural Parameters

## Summary

Build a real-hardware-first probing suite that uses ISA-level microkernels to
reverse engineer performance-relevant microarchitectural parameters for GPUs and
AI accelerators. The suite should measure latencies, bandwidths, capacities,
issue/throughput limits, scheduling behavior, cache and memory policies, tensor
unit behavior, TMA-like copy engines, synchronization costs, and interconnect /
off-chip memory characteristics.

The output should be a structured parameter report that maps measurements onto
the simulator names used in this repository, especially the blocks shown in:

- `.plan/20260617-1617-nvidia-gpu-architecture-diagram.md`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-cache.h`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/dram.h`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/operation_type.h`
- `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/`

The first implementation target is real hardware with an available ISA. The
suite should assume it can emit, assemble, or run low-level ISA kernels. Portable
kernel APIs can be used only as launch and allocation scaffolding.

## Current State Analysis

The repo models a GPGPU-Sim / Accel-Sim style architecture with these major
performance blocks:

| Diagram Block | Repo Names / Paths |
|---|---|
| GPU top level | `gpgpu_sim`, `gpgpu_sim_config`, `gpu-sim.h`, `gpu-sim.cc` |
| SIMT clusters | `simt_core_cluster`, `shader.h`, `shader.cc` |
| SM / shader core | `SM`, `shader_core_config`, `remodeling/sm.h`, `remodeling/sm.cc` |
| Subcores | `Subcore`, `num_subcores_in_SM`, `remodeling/subcore.h`, `remodeling/subcore.cc` |
| Warp scheduling | `scheduler_unit`, `gpgpu_scheduler_string`, `gpgpu_num_sched_per_core`, `gpgpu_max_insn_issue_per_warp` |
| Pipeline widths | `-gpgpu_pipeline_widths`, `pipeline_widths_string`, `pipe_widths` |
| Functional units | `functional_unit`, `functional_unit_with_queue`, `m_sp_pipeline`, `m_tensor_pipeline`, `m_tma_pipeline` |
| Register file | `gpgpu_shader_registers`, `gpgpu_num_reg_banks`, `reg_file_port_throughput`, `Register_file` |
| Shared memory | `gpgpu_shmem_size`, `gpgpu_shmem_num_banks`, `gpgpu_smem_latency` |
| L1 / constant / texture caches | `m_L1D_config`, `m_L1I_L1_half_C_cache_config`, `m_L0I_config`, `m_L1C_config`, `m_L0C_config`, `m_L1T_config` |
| LD/ST path | `ldst_unit_sm`, `memory_subcore_queue_size`, `memory_l1d_minimum_latency`, `memory_maximum_coalescing_cycles` |
| Tensor units | `gpgpu_tensor_core_avail`, `gpgpu_num_tensor_core_units`, `tensor_latency`, `tensor_rate_per_cycle` |
| TMA / copy engine | `tma_unit_sm`, `TMACommand`, `TMATransferEntry`, `m_command_queue`, `m_in_flight_transfers` |
| Interconnect | `icnt_wrapper`, `local_interconnect`, `-icnt_flit_size`, Intersim router parameters |
| L2 | `m_L2_config`, `l2_cache`, `-gpgpu_cache:dl2`, `-gpgpu_l2_rop_latency` |
| DRAM | `memory_config`, `dram_t`, `-gpgpu_n_mem`, `-gpgpu_dram_buswidth`, `-gpgpu_dram_burst_length`, `-gpgpu_dram_timing_opt` |

The suite should not try to infer every simulator field from a single probe.
Instead, it should maintain a measurement-to-parameter confidence model because
several repo parameters are coupled. For example, observed global memory
bandwidth depends on `gpgpu_n_mem`, `gpgpu_dram_buswidth`,
`gpgpu_dram_burst_length`, `dram_data_command_freq_ratio`, L2 policy, memory
coalescing, and interconnect saturation.

## Proposed Changes

### 1. Add a Probe Suite Design Under `tools/probe_suite/`

Create a new suite with this conceptual layout:

```text
tools/probe_suite/
  README.md
  schema/
    hardware_profile.schema.json
    probe_result.schema.json
    simulator_parameter_map.yaml
  core/
    runner.py
    calibrator.py
    statistics.py
    parameter_model.py
  backends/
    isa_backend.py
    cuda_launch_backend.py
    external_assembler_backend.py
  probes/
    arithmetic_latency/
    issue_throughput/
    scheduler_policy/
    register_file/
    shared_memory/
    l1_cache/
    l2_cache/
    global_memory/
    interconnect/
    tensor_core/
    tma_copy/
    synchronization/
  reports/
    render_markdown.py
    render_json.py
```

Why:

- The suite needs to run outside the simulator on real hardware.
- The measured output still needs to map back to repo-defined simulator names.
- Separating probes, backends, schemas, and calibration keeps accelerator-specific
  ISA support isolated.

How:

- `backends/isa_backend.py` defines the common interface:

```python
class ISABackend:
    def assemble(self, source: str, target: str) -> bytes: ...
    def launch(self, binary: bytes, launch_config: dict) -> dict: ...
    def read_timer(self) -> str: ...
    def supports_counter(self, name: str) -> bool: ...
```

- `cuda_launch_backend.py` can provide allocation, launch, timing, and result
  buffer plumbing for Nvidia targets while keeping the hot loop in ISA.
- `external_assembler_backend.py` handles non-Nvidia accelerators where the
  vendor assembler/compiler is external.

### 2. Define a Hardware Profile Schema

Add `schema/hardware_profile.schema.json` with the normalized output format.

Required top-level fields:

```json
{
  "target": {
    "vendor": "nvidia|amd|intel|custom",
    "device_name": "string",
    "isa_name": "string",
    "isa_version": "string",
    "driver_version": "string"
  },
  "measurements": {},
  "repo_parameter_estimates": {},
  "confidence": {},
  "raw_results": []
}
```

`repo_parameter_estimates` should use repo names directly, for example:

```json
{
  "shader_core_config::n_simt_clusters": 132,
  "shader_core_config::n_simt_cores_per_cluster": 1,
  "shader_core_config::warp_size": 32,
  "shader_core_config::num_subcores_in_SM": 4,
  "shader_core_config::gpgpu_num_sched_per_core": 4,
  "shader_core_config::gpgpu_num_sp_units": 128,
  "shader_core_config::gpgpu_num_tensor_core_units": 4,
  "shader_core_config::gpgpu_shmem_size": 233472,
  "shader_core_config::gpgpu_shmem_num_banks": 32,
  "shader_core_config::gpgpu_smem_latency": 28,
  "shader_core_config::max_sp_latency": 4,
  "shader_core_config::max_tensor_core_latency": 32,
  "memory_config::m_n_mem": 12,
  "memory_config::busW": 64,
  "memory_config::BL": 16,
  "memory_config::dram_latency": 200
}
```

### 3. Define the Simulator Parameter Map

Add `schema/simulator_parameter_map.yaml`.

This file should map each probe family to the repo-defined parameters it can
estimate.

Initial map:

```yaml
topology:
  estimates:
    - shader_core_config::n_simt_clusters
    - shader_core_config::n_simt_cores_per_cluster
    - gpgpu_sim_config::num_shader()
    - shader_core_config::n_thread_per_shader
    - shader_core_config::warp_size
    - shader_core_config::max_warps_per_shader
    - shader_core_config::max_cta_per_core

scheduler:
  estimates:
    - shader_core_config::num_subcores_in_SM
    - shader_core_config::gpgpu_num_sched_per_core
    - shader_core_config::gpgpu_scheduler_string
    - shader_core_config::gpgpu_max_insn_issue_per_warp
    - shader_core_config::gpgpu_dual_issue_diff_exec_units

pipeline:
  estimates:
    - shader_core_config::pipeline_widths_string
    - shader_core_config::pipe_widths
    - shader_core_config::max_sp_latency
    - shader_core_config::max_int_latency
    - shader_core_config::max_sfu_latency
    - shader_core_config::max_dp_latency
    - shader_core_config::max_tensor_core_latency
    - gpgpu_num_sp_units
    - gpgpu_num_dp_units
    - gpgpu_num_int_units
    - gpgpu_num_sfu_units

register_file:
  estimates:
    - shader_core_config::gpgpu_shader_registers
    - shader_core_config::gpgpu_num_reg_banks
    - shader_core_config::reg_file_port_throughput
    - num_regular_register_file_read_ports_per_bank
    - num_regular_register_file_write_ports_per_bank
    - max_latency_regular_register_file_latency

shared_memory:
  estimates:
    - shader_core_config::gpgpu_shmem_size
    - shader_core_config::gpgpu_shmem_per_block
    - shader_core_config::gpgpu_shmem_num_banks
    - shader_core_config::gpgpu_smem_latency
    - gpgpu_shmem_limited_broadcast
    - gpgpu_shmem_warp_parts

caches:
  estimates:
    - m_L1D_config
    - m_L1I_L1_half_C_cache_config
    - m_L0I_config
    - m_L1C_config
    - m_L0C_config
    - m_L1T_config
    - cache_config::m_nset
    - cache_config::m_line_sz
    - cache_config::m_assoc
    - cache_config::m_mshr_entries
    - cache_config::m_mshr_max_merge
    - cache_config::m_miss_queue_size
    - l1d_cache_config::l1_latency
    - l1d_cache_config::l1_banks

memory_pipeline:
  estimates:
    - memory_subcore_queue_size
    - memory_intermidiate_stages_subcore_unit
    - memory_sm_prt_size
    - memory_shared_memory_minimum_latency
    - memory_l1d_minimum_latency
    - memory_l1d_max_lookups_per_cycle_per_bank
    - memory_maximum_coalescing_cycles
    - memory_num_scalar_units_per_subcore
    - memory_subcore_link_to_sm_byte_size
    - memmory_max_concurrent_requests_shmem_per_sm
    - memmory_max_concurrent_requests_standard_per_sm

l2_dram:
  estimates:
    - memory_config::m_L2_config
    - gpgpu_cache:dl2
    - gpgpu_l2_rop_latency
    - memory_config::m_n_mem
    - memory_config::m_n_sub_partition_per_memory_channel
    - memory_config::scheduler_type
    - memory_config::busW
    - memory_config::BL
    - memory_config::nbk
    - memory_config::nbkgrp
    - memory_config::tCCD
    - memory_config::tRCD
    - memory_config::tRAS
    - memory_config::tRP
    - memory_config::CL
    - memory_config::WL
    - dram_latency
    - dram_data_command_freq_ratio

interconnect:
  estimates:
    - icnt_flit_size
    - gpgpu_mem_addr_mapping
    - gpgpu_mem_address_mask
    - routing_delay
    - vc_alloc_delay
    - sw_alloc_delay
    - credit_delay
    - input_speedup
    - output_speedup
    - internal_speedup

tensor_tma_sync:
  estimates:
    - gpgpu_tensor_core_avail
    - gpgpu_num_tensor_core_units
    - tensor_latency
    - tensor_rate_per_cycle
    - tma_unit_sm::kMaxRequestsPerCycle
    - TMACommand
    - TMATransferEntry
    - sync_debug_enable
```

### 4. Implement Probe Families

Each probe family should produce:

- generated ISA source
- launch configuration
- expected dependency pattern
- raw timings
- counter readings when available
- inferred repo parameter estimates
- confidence score

#### 4.1 Topology and Occupancy Probes

Goal:

- Estimate `n_simt_clusters`, `n_simt_cores_per_cluster`, total SM count,
  `warp_size`, `max_warps_per_shader`, `max_cta_per_core`, register and shared
  memory occupancy limits.

Methods:

- Use persistent CTAs with atomic slot claiming to infer concurrent CTA count.
- Vary threads per CTA and registers per thread to infer warp size and resident
  warp limits.
- Vary dynamic shared memory allocation to infer `gpgpu_shmem_size` and
  `gpgpu_shmem_per_block`.

Mapped parameters:

- `shader_core_config::n_simt_clusters`
- `shader_core_config::n_simt_cores_per_cluster`
- `shader_core_config::warp_size`
- `shader_core_config::max_warps_per_shader`
- `shader_core_config::max_cta_per_core`
- `shader_core_config::gpgpu_shader_registers`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`

#### 4.2 Arithmetic Latency Probes

Goal:

- Estimate instruction latencies and initiation intervals for ALU, INT, SFU,
  DP, branch, predicate, uniform, half, tensor, and miscellaneous operations.

Methods:

- Build dependent instruction chains:
  - `dst_i = op(dst_{i-1})`
  - measure cycles per instruction via ISA timer.
- Build independent multi-chain kernels to separate latency from throughput.
- Sweep chain count to find initiation interval and pipeline occupancy.

Mapped parameters:

- `shader_core_config::max_sp_latency`
- `shader_core_config::max_int_latency`
- `shader_core_config::max_sfu_latency`
- `shader_core_config::max_dp_latency`
- `shader_core_config::max_tensor_core_latency`
- `sfu_latency`
- `tensor_latency`
- `branch_latency`
- `half_latency`
- `uniform_latency`
- `predicate_latency`
- `miscellaneous_queue_latency`
- `miscellaneous_no_queue_latency`
- `sfu_initiation`
- `tensor_initiation`
- `branch_initiation`
- `half_initiation`
- `uniform_initiation`
- `predicate_initiation`

#### 4.3 Issue Width and Scheduler Probes

Goal:

- Estimate scheduler count, issue width, dual-issue policy, and per-warp issue
  limits.

Methods:

- Construct kernels with independent instructions targeting one execution unit.
- Construct mixed-unit instruction streams, e.g. ALU + SFU, ALU + memory, ALU +
  tensor, to detect `gpgpu_dual_issue_diff_exec_units`.
- Vary active warps and subcore-local work distribution to infer
  `num_subcores_in_SM` and `gpgpu_num_sched_per_core`.
- Use controlled warp readiness patterns to distinguish round-robin,
  greedy-then-oldest, and two-level scheduling behavior.

Mapped parameters:

- `shader_core_config::num_subcores_in_SM`
- `shader_core_config::gpgpu_num_sched_per_core`
- `shader_core_config::gpgpu_scheduler_string`
- `shader_core_config::gpgpu_max_insn_issue_per_warp`
- `shader_core_config::gpgpu_dual_issue_diff_exec_units`
- `shader_core_config::pipeline_widths_string`
- `shader_core_config::pipe_widths`

#### 4.4 Functional Unit Count and Throughput Probes

Goal:

- Estimate the number and throughput of SP, INT, DP, SFU, tensor, and memory
  units.

Methods:

- Saturate each unit class with independent instruction streams.
- Sweep active warps and CTAs to find the saturation point.
- Normalize throughput per SM and per cycle.

Mapped parameters:

- `gpgpu_num_sp_units`
- `gpgpu_num_dp_units`
- `gpgpu_num_int_units`
- `gpgpu_num_sfu_units`
- `gpgpu_tensor_core_avail`
- `gpgpu_num_tensor_core_units`
- `gpgpu_num_mem_units`
- `tensor_rate_per_cycle`

#### 4.5 Register File and Operand Collector Probes

Goal:

- Estimate register file banks, read/write ports, bank conflict policy, operand
  collector throughput, and register file latency.

Methods:

- Generate ISA sequences with controlled source/destination register numbering.
- Sweep register-bank conflict patterns and operand counts.
- Measure throughput degradation under same-bank vs distributed-bank accesses.
- Use dependent read-after-write chains to estimate register file latency.

Mapped parameters:

- `shader_core_config::gpgpu_num_reg_banks`
- `shader_core_config::reg_file_port_throughput`
- `gpgpu_reg_bank_use_warp_id`
- `is_opc_improved`
- `cu_num_ports`
- `is_rf_cache_enabled`
- `num_regular_register_file_read_ports_per_bank`
- `num_regular_register_file_write_ports_per_bank`
- `max_latency_regular_register_file_latency`
- `max_operands_regular_register_file`
- `gpgpu_operand_collector_num_units_*`
- `gpgpu_operand_collector_num_in_ports_*`
- `gpgpu_operand_collector_num_out_ports_*`

#### 4.6 Shared Memory Probes

Goal:

- Estimate shared memory size, bank count, bank width / mapping, broadcast
  behavior, and latency.

Methods:

- Pointer-chase in shared memory for latency.
- Generate stride sweeps to identify bank count and bank mapping.
- Compare uniform broadcast, multicast-like, and conflict-heavy access patterns.
- Sweep shared memory allocation to find per-block and per-SM limits.

Mapped parameters:

- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`
- `shader_core_config::gpgpu_shmem_num_banks`
- `gpgpu_shmem_limited_broadcast`
- `gpgpu_shmem_warp_parts`
- `gpgpu_smem_latency`
- `memory_shared_memory_minimum_latency`
- `memmory_max_concurrent_requests_shmem_per_sm`

#### 4.7 L1 / Constant / Texture / Instruction Cache Probes

Goal:

- Estimate cache line size, capacity, associativity, hit latency, MSHR count,
  miss queue depth, bank count, and replacement behavior.

Methods:

- Pointer-chase for hit latency.
- Stride and working-set sweeps for capacity and line size.
- Conflict-set construction for associativity.
- Parallel miss streams for MSHR and miss queue pressure.
- Read-only, constant, texture, and instruction-specific variants when the ISA
  exposes those paths.

Mapped parameters:

- `m_L1D_config`
- `m_L1I_L1_half_C_cache_config`
- `m_L0I_config`
- `m_L1C_config`
- `m_L0C_config`
- `m_L1T_config`
- `cache_config::m_nset`
- `cache_config::m_line_sz`
- `cache_config::m_assoc`
- `cache_config::m_mshr_entries`
- `cache_config::m_mshr_max_merge`
- `cache_config::m_miss_queue_size`
- `l1d_cache_config::l1_latency`
- `l1d_cache_config::l1_banks`
- `gpgpu_l1_banks_byte_interleaving`
- `gpgpu_l1_banks_hashing_function`

#### 4.8 SM Memory Pipeline and Coalescing Probes

Goal:

- Estimate memory coalescing windows, scalar-unit count, subcore-to-SM link
  bandwidth, queue depths, and concurrent request limits.

Methods:

- Generate warp memory instructions with controlled lane address patterns.
- Sweep contiguous, strided, scattered, and scalar access patterns.
- Vary independent memory instructions per warp and active warps per SM.
- Detect throughput cliffs that correspond to queue and PRT saturation.

Mapped parameters:

- `memory_subcore_queue_size`
- `memory_intermidiate_stages_subcore_unit`
- `memory_sm_prt_size`
- `memory_l1d_minimum_latency`
- `memory_l1d_max_lookups_per_cycle_per_bank`
- `memory_maximum_coalescing_cycles`
- `memory_num_scalar_units_per_subcore`
- `memory_subcore_link_to_sm_byte_size`
- `memmory_max_concurrent_requests_standard_per_sm`

#### 4.9 L2 and DRAM Probes

Goal:

- Estimate L2 cache geometry, L2 hit latency, DRAM bandwidth, DRAM latency,
  memory partition count, bank group behavior, burst length, and scheduling
  policy.

Methods:

- Use pointer-chase working sets that exceed L1 but fit L2 for L2 latency.
- Use capacity and conflict sweeps for L2 geometry.
- Use streaming reads/writes to estimate peak bandwidth.
- Use bank/partition conflict address sweeps to infer memory address mapping.
- Use row-hit / row-miss style address patterns to infer DRAM timing effects.
- Use mixed read/write streams to infer scheduler behavior and return queue
  pressure.

Mapped parameters:

- `memory_config::m_L2_config`
- `gpgpu_cache:dl2`
- `gpgpu_l2_rop_latency`
- `memory_config::m_n_mem`
- `memory_config::m_n_sub_partition_per_memory_channel`
- `gpgpu_n_mem_per_ctrlr`
- `memory_config::scheduler_type`
- `gpgpu_dram_partition_queues`
- `gpgpu_frfcfs_dram_sched_queue_size`
- `gpgpu_dram_return_queue_size`
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
- `dram_bnk_indexing_policy`
- `dram_bnkgrp_indexing_policy`

#### 4.10 Interconnect and Address Mapping Probes

Goal:

- Estimate address-to-partition mapping, interconnect flit size, latency, and
  saturation behavior.

Methods:

- Sweep physical/virtual address bits and detect bandwidth/latency partitioning
  changes.
- Run many-CTA global-memory traffic patterns with controlled destination
  partitions.
- Measure latency under increasing injection rate to estimate routing and
  allocator delays.

Mapped parameters:

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

#### 4.11 Tensor Core / Matrix Engine Probes

Goal:

- Estimate tensor instruction latency, throughput, unit count, supported tile
  shapes, pipeline initiation interval, and operand-layout constraints.

Methods:

- Dependent tensor instruction chains for latency.
- Independent tensor instruction streams for throughput.
- Sweep matrix shapes, data types, accumulator types, and operand layouts.
- Mix tensor and non-tensor instructions to detect issue sharing and dual-issue
  restrictions.

Mapped parameters:

- `gpgpu_tensor_core_avail`
- `gpgpu_num_tensor_core_units`
- `tensor_latency`
- `tensor_rate_per_cycle`
- `shader_core_config::max_tensor_core_latency`
- `tensor_extra_latency_16816_fp32_1688_fp32`

#### 4.12 TMA / DMA / Async Copy Probes

Goal:

- Estimate copy engine queue depths, request rate, sectorization, outstanding
  transfer limits, completion behavior, and interaction with synchronization.

Methods:

- Generate ISA async-copy or TMA-like commands when available.
- Sweep transfer size, rank/dimension count, alignment, and stride.
- Measure issue-to-completion latency and steady-state bandwidth.
- Pair copy commands with barrier/wait instructions to infer completion semantics.

Mapped parameters:

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
- `Subcore::m_tma_pipeline`
- `SM::m_tma_unit_shared_of_sm`
- `m_EX_TMA_reception_latches_per_subcore`

#### 4.13 Synchronization and Barrier Probes

Goal:

- Estimate barrier latency, memory fence cost, mbarrier-like behavior, cluster
  barrier behavior, and wait/arrive semantics.

Methods:

- Measure CTA barrier latency as a function of active warps.
- Measure memory fence latency after different memory traffic patterns.
- For Hopper-like ISAs, generate `SYNCS`, `UCGABAR`, `FENCE`, and async-copy
  sequences to infer readiness and completion rules.

Mapped parameters:

- `gpgpu_num_cta_barriers`
- `BARRIER_OP`
- `MEMORY_BARRIER_OP`
- `GRID_BARRIER_OP`
- `MBARRIER_OP`
- `CLUSTER_BARRIER_OP`
- `sync_debug_enable`
- `sync_debug_print_budget`
- `sync_debug_skip_runtime_budget`

### 5. Add Calibration and Inference Logic

Add `core/calibrator.py` and `core/parameter_model.py`.

The calibrator should:

1. Normalize raw timings to cycles when a cycle counter exists.
2. Use robust statistics: median, MAD, confidence intervals, outlier rejection.
3. Fit simple models first:
   - dependent chain slope → latency
   - independent chain reciprocal throughput → initiation interval
   - working-set knee → capacity
   - stride periodicity → bank or partition count
   - bandwidth plateau → peak throughput
4. Emit direct estimates when confidence is high.
5. Emit coupled estimates when several parameters cannot be separated.

The parameter model should classify estimates:

```json
{
  "parameter": "shader_core_config::gpgpu_smem_latency",
  "value": 28,
  "unit": "cycles",
  "confidence": 0.87,
  "probe_ids": ["shared_memory.pointer_chase.latency"],
  "coupled_with": [],
  "notes": "Median dependent shared-memory load latency."
}
```

### 6. Add Report Generation

Add Markdown and JSON reports.

Markdown report sections:

1. Target identity
2. Probe coverage matrix
3. High-confidence repo parameter estimates
4. Medium/low-confidence estimates
5. Coupled parameters requiring additional probes
6. Raw benchmark summary
7. Suggested simulator config overrides

JSON report should be machine-readable and stable enough for future scripts to
generate simulator config fragments.

### 7. Add Accelerator Portability Layer

Because the goal includes arbitrary GPUs and AI accelerators with an ISA, keep
the suite backend-oriented.

Required backend capabilities:

| Capability | Required | Purpose |
|---|---:|---|
| Device allocation | Yes | Allocate buffers and scratch regions |
| Kernel launch | Yes | Run probes |
| ISA assembly or binary injection | Yes | Preserve exact instruction sequences |
| Device timer access | Yes | Measure latency without host noise |
| Global memory atomics | Preferred | Topology / occupancy probes |
| Performance counters | Optional | Improve confidence, not required |
| Cache controls | Optional | Force cache path where ISA supports it |

Backend interface should expose unsupported features explicitly so the runner can
skip incompatible probes rather than silently returning misleading data.

## Assumptions & Decisions

1. Primary target is real hardware, not simulator-only calibration.
2. Probe kernels are ISA-level microkernels; portable APIs are allowed only for
   launch, memory allocation, and result collection.
3. The first suite version should produce measured parameter reports, not modify
   simulator config files automatically.
4. Repo-defined parameter names are the canonical output vocabulary.
5. Some parameters are not directly observable and must be reported as coupled
   estimates with confidence scores.
6. The suite should be accelerator-neutral at the framework level, but Nvidia
   should be the first concrete backend because the repo architecture and naming
   are Nvidia/GPGPU-Sim oriented.
7. The suite should avoid relying on vendor performance counters for correctness;
   counters are supplemental evidence.

## Verification Steps

### Static Verification

1. Validate `hardware_profile.schema.json` and `probe_result.schema.json`.
2. Validate `simulator_parameter_map.yaml` contains only repo-defined parameter
   names or documented class/member names.
3. Run unit tests for model fitting:
   - latency slope extraction
   - throughput plateau detection
   - cache capacity knee detection
   - bank conflict periodicity detection
   - confidence score calculation

### Backend Verification

1. Run a no-op ISA kernel and verify launch/result plumbing.
2. Run a timer calibration kernel and verify monotonic device timing.
3. Run a dependent ALU chain with known instruction count and verify stable
   cycle-per-instruction output.
4. Run repeated probes and verify median/MAD stability across runs.

### Probe Verification

1. Arithmetic probes should recover internally consistent latency and throughput:
   dependent chains report higher cycles/instruction than independent saturated
   streams.
2. Shared-memory stride probes should show periodic conflict behavior.
3. Cache probes should show clear latency knees as working set crosses cache
   levels.
4. DRAM streaming probes should show a bandwidth plateau.
5. Tensor probes should report throughput scaling until tensor unit saturation.
6. TMA / async-copy probes should report distinct issue latency, transfer
   bandwidth, and completion/wait latency when the ISA supports such operations.

### Acceptance Criteria

The plan is implemented successfully when:

1. The suite can run at least one ISA backend on real hardware.
2. It emits a Markdown report and JSON hardware profile.
3. The report includes estimates for at least these repo names:
   - `shader_core_config::warp_size`
   - `shader_core_config::max_warps_per_shader`
   - `shader_core_config::gpgpu_shmem_size`
   - `shader_core_config::gpgpu_shmem_num_banks`
   - `shader_core_config::gpgpu_smem_latency`
   - `shader_core_config::max_sp_latency`
   - `shader_core_config::gpgpu_num_sched_per_core`
   - `shader_core_config::pipeline_widths_string`
   - `cache_config::m_line_sz`
   - `cache_config::m_assoc`
   - `l1d_cache_config::l1_latency`
   - `memory_config::m_n_mem`
   - `memory_config::busW`
   - `memory_config::BL`
   - `dram_latency`
4. Each estimate includes value, unit, confidence, probe IDs, and notes.
5. Unsupported probes are skipped with explicit reasons.

## Rollout Plan

1. Land schemas and parameter map first.
2. Implement the backend interface and one Nvidia ISA backend.
3. Implement arithmetic latency, shared memory, and cache probes first because
   they validate the runner and statistics model.
4. Add scheduler, register file, L2/DRAM, and interconnect probes.
5. Add tensor and TMA probes after the base timing harness is stable.
6. Add report generation once raw probe results and parameter estimates are
   stable.

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| ISA changes across vendors or generations | Keep ISA templates backend-local and feature-gated. |
| Compiler/assembler rewrites probe loops | Use explicit ISA assembly and disassemble binaries during verification. |
| Host timing noise | Prefer device-side cycle counters and repeat runs with robust statistics. |
| Coupled parameters misidentified as direct estimates | Emit coupled estimates and confidence metadata. |
| Cache and memory policies adapt dynamically | Use multiple access patterns and report policy-dependent results. |
| Performance counters unavailable | Treat counters as optional; rely on timing and controlled microkernels. |
| Arbitrary AI accelerators lack CUDA-like launch APIs | Keep launch/assembly/timer capabilities abstract in `ISABackend`. |
