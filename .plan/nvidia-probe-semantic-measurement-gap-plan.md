# NVIDIA Probe Semantic and Measurement Gap Plan

## Summary

This document is the NVIDIA-specific specialization of the generalized AMORA
probing plan in `.plan/probing-suite-microarchitecture-plan.md`. It defines how
AMORA should use CUDA metadata, NCU, CUPTI, NVBit, disassembly, microkernels,
published NVIDIA information, and simulator traces to produce trustworthy
GPGPU-Sim/AMORA-facing parameter estimates.

The core rule is the same as the generalized plan:

```text
published facts
-> backend capabilities
-> raw observations
-> normalized hardware-neutral measurements
-> NVIDIA backend interpretations
-> simulator mapping contracts
-> fitted simulator-equivalent estimates
```

NVIDIA tools can expose more direct execution facts than end-to-end timing for
many CUDA-visible behaviors. Therefore NCU/CUPTI counters are primary evidence
when a metric contract has a close semantic match. Microkernel timing remains
essential for controlled workload generation, counter cross-checking, fallback
coverage, and behaviors that are not directly exposed by profiler metrics.

## Revision History

### 2026-06-18: Alignment With Generalized AMORA Plan

Source inputs:

- Generalized plan:
  `.plan/probing-suite-microarchitecture-plan.md`
- Previous NVIDIA plan:
  `.plan/nvidia-probe-semantic-measurement-gap-plan.md`
- NVIDIA plan reaction document:
  `.plan/20260618_nvidia-probe-semantic-measurement-gap-plan_comments.md`
- Existing P0-P3 methodology files:
  `.plan/nvidia-p0-kernel-methodology.md`,
  `.plan/nvidia-p1-kernel-methodology.md`,
  `.plan/nvidia-p2-kernel-methodology.md`,
  `.plan/nvidia-p3-kernel-methodology.md`

Major changes:

- Rewrote the NVIDIA plan as a backend-specific instance of the generalized
  layered architecture.
- Added explicit NVIDIA backend capabilities, tool registry, ISA/SASS semantic
  records, and metric resolver policy.
- Preserved the design decision that NCU/CUPTI direct counters can be primary
  evidence when their semantic contract is direct.
- Preserved the design decision that microkernel timing is not universally
  primary because it is affected by runtime fog.
- Added layered result requirements matching the generalized plan:
  `raw_observation`, `normalized_measurement`, `backend_interpretation`, and
  `simulator_estimate`.
- Reframed each NVIDIA probe family as a mapping contract from a
  hardware-neutral concept through NVIDIA interpretation to simulator
  parameters.
- Added explicit use of published NVIDIA information as trust-and-verify
  anchors.
- Added explicit simulator trace usage for directly observable simulator states
  such as queue length, scheduler state, cache state, pipeline state, and memory
  partition state.
- Added capability-gated execution modes so probes can be skipped, downgraded,
  or reported as unsupported instead of emitting weak scalars.
- Added compact uncertainty categories with detailed variance and fitting
  records linked from reports.
- Added implementation phases that update the current P0 scaffolding before
  expanding into P1-P3.

### 2026-06-18: Measurement-Contract Revision

This earlier revision responded to
`.plan/20260618_nvidia-probe-semantic-measurement-gap-plan_comments.md`.

Key decisions retained:

- Use the evidence source with the closest semantic match.
- Treat NCU/CUPTI direct metrics as primary when the metric maps directly to the
  target behavior.
- Use microkernel timing as controlled workload evidence, validation, or
  fallback.
- Trust and verify published NVIDIA parameters and stable CUDA metadata.
- Treat simulator parameters and dynamic simulator states as directly observable
  inside the simulator.
- Require fitting metadata, uncertainty, clock-domain handling, metric mapping,
  and explicit rejection/downgrade behavior.

### Superseded Assumptions From Earlier Versions

- Superseded: "Microkernel timing is the primary evidence source."
  Replacement: Use semantic-match primary evidence. For NVIDIA, direct
  NCU/CUPTI counters can be stronger than end-to-end timing when the metric
  contract is direct.

- Superseded: "A probe family can directly emit simulator parameters."
  Replacement: A probe emits layered observations and measurements. Simulator
  parameters are emitted only through mapping contracts.

- Superseded: "Metric naming is mostly a local implementation detail."
  Replacement: NVIDIA metric names vary by architecture and tool version, so
  logical metrics require candidate names, normalization, validation, and
  fallback rules.

- Superseded: "Hidden hardware and simulator state have the same observability."
  Replacement: hardware internals may be indirect or unobservable, but simulator
  internals are directly traceable. The semantic gap is the mapping from
  hardware evidence to simulator-equivalent behavior.

## NVIDIA Backend Scope

This plan targets NVIDIA CUDA-capable GPUs and maps observations to
GPGPU-Sim/AMORA-style simulator parameters.

Primary NVIDIA tool stack:

- CUDA runtime and driver metadata
- Nsight Compute CLI (`ncu`)
- CUPTI Range Profiling
- CUPTI Host Profiling
- CUPTI PM Sampling
- CUPTI PC Sampling
- CUPTI SASS Metrics
- NVBit
- `nvdisasm` and `cuobjdump`
- Published NVIDIA documentation and public specifications
- AMORA/GPGPU-Sim internal simulator traces

The NVIDIA backend is a reference backend for AMORA, not the global abstraction.
Concept names should remain hardware-neutral until this plan maps them through
NVIDIA-specific semantics.

## Evidence Policy

Use the evidence source with the closest semantic match.

| Evidence Source | Primary When | Validation When | Main Risk |
|---|---|---|---|
| Published NVIDIA facts | NVIDIA exposes stable limits, units, or architecture facts | Checking inferred values | Specs may omit SKU modes or dynamic behavior |
| CUDA metadata | Runtime-visible limit maps directly to a target concept | Cross-checking profiler launch metadata | Runtime policy may differ from physical structure |
| NCU direct metrics | Metric directly matches the behavior | Checking timing or fitted curves | Replay, permissions, metric availability |
| CUPTI Range Profiling | Programmatic metric collection is needed | Automating NCU-equivalent checks | API complexity and pass constraints |
| CUPTI Host Profiling | Metric enumeration/config/evaluation is needed | Resolving supported metrics | Version and architecture variation |
| CUPTI PM Sampling | Time-varying PM state matters | Checking phase behavior | Sampling granularity |
| CUPTI PC/SASS Metrics | Source/SASS attribution is needed | Explaining stall and instruction anomalies | Sampling or patching overhead |
| NVBit | Executed instruction or memory stream is needed | Verifying SASS and dynamic mix | Instrumentation overhead and injection conflicts |
| Microkernel timing | Timing behavior is the target or no direct metric exists | Cross-checking direct counters | Runtime fog |
| Simulator trace | Simulator state/parameter behavior must be known | Mapping hardware observations to simulator knobs | Simulator may omit hardware mechanisms |

Direct counters are not automatically trusted. A metric must be selected through
a metric resolver and pass its validation rule before it can be primary.

## NVIDIA Backend Capabilities

The NVIDIA backend should discover and record these capabilities before running
probes:

```yaml
backend: nvidia_cuda
instruction_control:
  can_emit_low_level_isa: partial
  can_verify_disassembly: true
  can_control_register_assignment: partial
  can_validate_dynamic_instruction_stream: true
timing:
  has_device_timer: true
  timer_domains:
    - SM
    - globaltimer_or_ns
profiling:
  has_per_kernel_counters: true
  has_range_counters: true
  has_pm_sampling: true
  has_pc_sampling: true
  has_sass_metrics: true
  has_dynamic_binary_instrumentation: true
memory_control:
  can_select_address_space: true
  can_control_cache_policy: partial
  can_allocate_shared_memory: true
  can_control_alignment: true
simulator_mapping:
  target_simulator: gpgpu_sim_or_amora
  simulator_state_trace_available: true
```

Capability values are target-specific. The runner must record actual tool
availability, permission status, architecture support, and unsupported reasons.

## NVIDIA ISA And SASS Semantic Records

Every low-level probe must identify the SASS/PTX semantic class it intends to
exercise and the architectural block it is expected to stress.

Example:

```yaml
instruction: FFMA
backend: nvidia_sass
semantic_class: fp32_fma
architectural_block:
  primary: fp32_pipe
  secondary:
    - scheduler
    - register_file
    - operand_delivery
verification:
  disassembly_required: true
  dynamic_instruction_count_optional: true
mapping_confidence: high
known_mismatches:
  - compiler may replace source operations with alternate opcodes
  - operand reuse and register banking can alter observed throughput
```

Initial semantic records should cover:

- integer add/mul and predicate/control operations,
- FP32 add/mul/FMA,
- FP64 operations where supported,
- SFU operations,
- global, local, shared, constant, and texture load/store paths,
- barrier and fence operations,
- MMA/tensor instructions,
- async-copy/TMA-like instructions where visible,
- branch and control-flow instructions.

## Tool Registry And Metric Resolver

NVIDIA metric names are resolved through logical metric contracts.

Each logical metric must define:

- `logical_name`
- `tool_candidates`
- `candidate_metrics_by_arch`
- `preferred_metric`
- `normalization`
- `validation_rule`
- `fallback_source`
- `unsupported_reason`

Example:

```yaml
logical_name: sm_active_cycles
tool_candidates:
  - ncu
  - cupti_range
candidate_metrics_by_arch:
  default:
    - sm__cycles_active.avg
    - sm__cycles_elapsed.avg
normalization: use active cycles for instruction latency and issue fits
validation_rule: reject if active cycles are zero or launch metadata reports no executed kernel
fallback_source: device_clock64_timing
unsupported_reason: metric unavailable from NCU/CUPTI query
```

Metric resolver responsibilities:

- query available metrics per target GPU,
- select candidate metrics by architecture and CUDA/tool version,
- record exact metric names used,
- record units and aggregation suffixes,
- record profiler replay or sampling mode,
- emit unsupported status instead of silently substituting weak metrics.

## Layered Result Schema

NVIDIA probe output must preserve all layers.

```yaml
raw_observation:
  source: ncu
  raw_values:
    smsp__cycles_active.avg: 100000
    smsp__sass_thread_inst_executed_op_ffma_pred_on.sum: 25000
  probe_id: arithmetic_throughput.fp32_ffma.nvidia
  binary_hash: string
  disassembly_hash: string

normalized_measurement:
  concept: scalar_instruction_reciprocal_throughput
  value: 0.25
  unit: sm_cycles_per_instruction
  variance:
    count: 20
    median: 0.25
    mad: 0.01
    min: 0.24
    max: 0.27

backend_interpretation:
  backend: nvidia_cuda
  instruction_semantics: fp32_fma
  architectural_block: fp32_pipe
  clock_domain: SM
  primary_evidence: direct_counter
  validation_evidence:
    - disassembly_hash
    - nvbit_opcode_histogram
    - timing_cross_check

simulator_estimate:
  parameter: gpgpu_num_sp_units
  value: 128
  unit: effective_units_per_sm
  evidence_tier: coupled_inference
  fit_status: conditionally_identified
  uncertainty_category: conditional_scalar
  assumptions:
    - FFMA maps to the intended FP32 pipe class
    - active-cycle normalization is valid
  coupled_with:
    - scheduler_issue_width
    - operand_delivery
    - clock_domain
```

## Measurement Contract Fields

Every simulator estimate requires a contract with these fields:

- `parameter`
- `simulator_component`
- `hardware_behavior`
- `nvidia_backend_interpretation`
- `observability`
- `primary_evidence`
- `validation_evidence`
- `probe_workload`
- `metric_mapping`
- `formula_or_fit`
- `clock_domain`
- `fit_status_required`
- `uncertainty_category`
- `scalar_output_allowed`
- `known_mismatches`
- `rejection_rules`
- `downgrade_rules`
- `fallback`

Observability values:

- `published`
- `metadata`
- `direct_counter`
- `tool_derived_counter`
- `instrumented_stream`
- `timing`
- `simulator_trace`
- `coupled_inference`
- `behavioral_only`
- `unsupported`

## Fitting, Variance, And Uncertainty

Fit status values:

- `direct`
- `uniquely_identified`
- `bounded`
- `conditionally_identified`
- `underconstrained`
- `behavioral_only`
- `unsupported`

Compact uncertainty categories:

- `stable_scalar`: low variance and direct semantic match.
- `bounded_range`: lower and upper bounds are more defensible than one scalar.
- `conditional_scalar`: scalar valid under stated assumptions.
- `multi_fit`: multiple parameter sets explain the data.
- `behavioral_class`: emit class or curve, not hardware scalar.
- `indeterminate`: evidence is insufficient or contradictory.

Variance records must include sample count, median, MAD, min, max, coefficient
of variation, and per-pass/per-run counter variance where available.

## Clock-Domain Policy

Every timing-derived or rate-derived NVIDIA result must record:

- `clock_domain`: `SM|L1TEX|L2|DRAM|fabric|copy_engine|host|mixed|unknown`
- `clock_source`: `clock64|globaltimer|NCU metric|CUPTI metric|published|host_timer`
- `clock_locked`: `true|false|unknown`
- `observed_clock_range`
- `native_unit`
- `conversion_method`
- `clock_assumptions`

Rules:

- Keep native units unless a simulator mapping contract defines conversion.
- Convert to simulator cycles only with an explicit simulator clock ratio.
- Keep SM, L2, DRAM, fabric, and copy-engine domains separate.

## Probe Execution Modes

Each NVIDIA probe should support these modes where applicable:

1. `capability`: discover CUDA device, tool, permission, and metric support.
2. `counter`: collect NCU/CUPTI Range Profiling metrics.
3. `sampling`: collect CUPTI PM/PC/SASS sampling or attribution data.
4. `timing`: run low-overhead microkernel timing without profiler attachment.
5. `instrumented`: run NVBit validation separately.
6. `disassembly`: verify SASS/PTX and record hashes.
7. `sim_trace`: run simulator with internal state tracing.
8. `fit`: fuse evidence and emit layered results.

Small-kernel policy:

- Separate timing-only runs from profiler runs.
- Lengthen inner loops or batch kernels when profiler replay would dominate.
- Correlate runs by probe ID, binary hash, disassembly hash, launch config,
  metric set, device ID, and tool version.

## Parameter Mapping Contracts

### Topology And Occupancy

Hardware-neutral concept:

- topology limits and residency-like capacity.

Primary NVIDIA evidence:

- CUDA runtime/driver metadata,
- published NVIDIA device specifications,
- NCU launch metadata.

Validation:

- persistent CTA residency kernel,
- simulator occupancy-state trace.

Simulator parameters:

- `gpgpu_sim_config::num_shader()`
- `shader_core_config::n_simt_clusters`
- `shader_core_config::n_simt_cores_per_cluster`
- `shader_core_config::warp_size`
- `shader_core_config::max_warps_per_shader`
- `shader_core_config::max_cta_per_core`
- `shader_core_config::gpgpu_shader_registers`
- `shader_core_config::gpgpu_shmem_size`
- `shader_core_config::gpgpu_shmem_per_block`

Scalar policy:

- allow scalar for SM count, warp size, max resident blocks, max resident
  threads/warps, register limits, and shared-memory limits when metadata or
  published facts are direct;
- treat cluster decomposition as table-backed, fitted, or unsupported unless
  an explicit architecture mapping exists.

### Arithmetic Latency

Hardware-neutral concept:

- dependent operation latency for a specific instruction semantic class.

Primary NVIDIA evidence:

- direct NCU/CUPTI instruction and active-cycle metrics when available;
- dependent-chain timing when no direct counter contract exists.

Validation:

- SASS dependency-chain verification,
- NVBit opcode histogram or dynamic instruction stream,
- occupancy sweep to detect scheduler/latency hiding confounders.

Simulator parameters:

- `shader_core_config::max_sp_latency`
- `shader_core_config::max_int_latency`
- `shader_core_config::max_sfu_latency`
- `shader_core_config::max_dp_latency`
- `shader_core_config::max_tensor_core_latency`

Scalar policy:

- allow scalar only for stable opcode-class latency with matching counters,
  SASS, timing, and low variance;
- otherwise emit `conditional_scalar` or `bounded_range`.

### Arithmetic Throughput And Functional Units

Hardware-neutral concept:

- reciprocal throughput and issue saturation for an instruction semantic class.

Primary NVIDIA evidence:

- NCU/CUPTI pipe utilization, instruction counts, active cycles, issue metrics;
- independent-stream microkernels for saturation.

Validation:

- NVBit opcode mix,
- timing plateau,
- simulator pipeline trace.

Simulator parameters:

- `shader_core_config::pipeline_widths_string`
- `shader_core_config::pipe_widths`
- `gpgpu_num_sp_units`
- `gpgpu_num_dp_units`
- `gpgpu_num_int_units`
- `gpgpu_num_sfu_units`
- `gpgpu_num_tensor_core_units`

Scalar policy:

- throughput plateau can be `stable_scalar` when direct metrics are stable;
- unit counts are `published`, `conditionally_identified`, or
  `coupled_inference`, not raw facts unless documented.

### Scheduler And Issue

Hardware-neutral concept:

- ready-work scheduling behavior, issue eligibility, and stall attribution.

Primary NVIDIA evidence:

- `smsp__*` issue, eligible-warps, active-warps, and stall metrics;
- CUPTI PC Sampling and SASS Metrics.

Validation:

- ready-warp and mixed-issue kernels,
- arithmetic throughput coupling,
- simulator scheduler trace.

Simulator parameters:

- `shader_core_config::num_subcores_in_SM`
- `shader_core_config::gpgpu_num_sched_per_core`
- `shader_core_config::gpgpu_scheduler_string`
- `shader_core_config::gpgpu_max_insn_issue_per_warp`
- `shader_core_config::gpgpu_dual_issue_diff_exec_units`

Scalar policy:

- emit exact counts only when metadata, published facts, or robust counter
  evidence support them;
- emit scheduler policy as `behavioral_class`, not as a claimed NVIDIA policy
  name.

### Register File And Operand Delivery

Hardware-neutral concept:

- operand delivery conflict behavior, register-bank effects, and port-like
  saturation.

Primary NVIDIA evidence:

- SASS-controlled register-number sweeps,
- NCU/CUPTI stall and issue metrics.

Validation:

- NVBit register/instruction validation where available,
- throughput and latency probe coupling,
- simulator operand-collector trace.

Simulator parameters:

- `shader_core_config::gpgpu_num_reg_banks`
- `shader_core_config::reg_file_port_throughput`
- `num_regular_register_file_read_ports_per_bank`
- `num_regular_register_file_write_ports_per_bank`
- `max_latency_regular_register_file_latency`
- `gpgpu_operand_collector_num_units_*`
- `gpgpu_operand_collector_num_in_ports_*`
- `gpgpu_operand_collector_num_out_ports_*`

Scalar policy:

- register-bank count may be scalar if periodicity is stable across independent
  register-number hypotheses;
- ports and operand collectors usually require `multi_fit` or
  `behavioral_class` unless multiple probe families agree.

### Shared Memory

Hardware-neutral concept:

- explicitly managed local memory latency, bandwidth, and conflict behavior.

Primary NVIDIA evidence:

- NCU/CUPTI shared-memory transaction and conflict metrics;
- bank-stride microkernels.

Validation:

- SASS shared-memory instruction verification,
- timing impact,
- simulator shared-memory trace.

Simulator parameters:

- `shader_core_config::gpgpu_shmem_num_banks`
- `gpgpu_shmem_limited_broadcast`
- `gpgpu_shmem_warp_parts`
- `gpgpu_smem_latency`
- `memory_shared_memory_minimum_latency`
- `memmory_max_concurrent_requests_shmem_per_sm`

Scalar policy:

- bank count can be scalar when conflict periodicity is clean;
- broadcast/multicast and queue behavior should be behavioral or fitted unless
  directly supported by counters.

### L1, Constant, Texture, And Instruction Caches

Hardware-neutral concept:

- cache-like latency plateaus, capacity knees, transaction granularity, and
  path-specific behavior.

Primary NVIDIA evidence:

- NCU/CUPTI `l1tex__*`, constant, texture, and instruction-cache metrics;
- pointer-chase and working-set probes.

Validation:

- SASS load path and cache modifier verification,
- L2/DRAM traffic checks,
- simulator cache trace.

Simulator parameters:

- `m_L1D_config`
- `l1d_cache_config::l1_latency`
- `l1d_cache_config::l1_banks`
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

Scalar policy:

- latency and line/sector behavior can be scalar or bounded when direct metrics
  and access patterns agree;
- associativity, MSHR, and replacement policy are fitted simulator equivalents.

### SM Memory Pipeline And Coalescing

Hardware-neutral concept:

- request formation, coalescing, sectorization, and SM-to-memory-pipeline
  pressure.

Primary NVIDIA evidence:

- NCU/CUPTI L1TEX sector/request/replay metrics;
- lane-pattern kernels.

Validation:

- NVBit memory-reference stream,
- simulator memory-pipeline trace.

Simulator parameters:

- `memory_l1d_minimum_latency`
- `memory_l1d_max_lookups_per_cycle_per_bank`
- `memory_maximum_coalescing_cycles`
- `memory_subcore_link_to_sm_byte_size`
- `memmory_max_concurrent_requests_standard_per_sm`

Scalar policy:

- coalescing and sector behavior can be direct when counters match expected lane
  addresses;
- queue and outstanding-request limits are fitted or bounded.

### L2, DRAM, And Memory Partitions

Hardware-neutral concept:

- shared-cache behavior, memory partitioning, global memory latency, bandwidth,
  and saturation.

Primary NVIDIA evidence:

- NCU/CUPTI `lts__*`, `dram__*`, partition metrics where available;
- CUPTI PM Sampling for phase behavior;
- published memory specs as trust-and-verify anchors.

Validation:

- streaming, pointer-chase, partition-camping, and row-policy workloads;
- simulator L2/DRAM/partition traces.

Simulator parameters:

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

Scalar policy:

- bandwidth can be stable or bounded when direct metrics and published anchors
  agree;
- DRAM timing-like parameters are usually `underconstrained` or `multi_fit`;
- emit alternative fits and residuals for timing/scheduler combinations.

### Tensor Core

Hardware-neutral concept:

- matrix operation latency, throughput, supported shapes, and datatype/layout
  behavior.

Primary NVIDIA evidence:

- NCU/CUPTI tensor instruction, tensor-pipe, active-cycle, and utilization
  metrics;
- MMA microkernels.

Validation:

- SASS MMA opcode verification,
- NVBit dynamic opcode mix where practical,
- simulator tensor-pipeline trace.

Simulator parameters:

- `gpgpu_tensor_core_avail`
- `gpgpu_num_tensor_core_units`
- `tensor_latency`
- `tensor_rate_per_cycle`
- `shader_core_config::max_tensor_core_latency`
- `tensor_extra_latency_16816_fp32_1688_fp32`

Scalar policy:

- tensor throughput may be direct when counters match opcode shape and datatype;
- unit count and shape-specific extra latency are conditional or fitted unless
  published.

### Synchronization And Barriers

Hardware-neutral concept:

- barrier, fence, arrival, and completion behavior under controlled patterns.

Primary NVIDIA evidence:

- NCU/CUPTI scheduler/barrier stall metrics;
- CUPTI PC Sampling stall attribution.

Validation:

- barrier/fence microkernels,
- simulator synchronization trace.

Simulator parameters:

- `gpgpu_num_cta_barriers`
- `BARRIER_OP`
- `MEMORY_BARRIER_OP`
- `GRID_BARRIER_OP`
- `MBARRIER_OP`
- `CLUSTER_BARRIER_OP`

Scalar policy:

- barrier count/feature support can be direct when documented or metadata-backed;
- latency should be reported per scope and arrival pattern.

### TMA, DMA, And Async Copy

Hardware-neutral concept:

- async copy command issue, transfer bandwidth, completion, in-flight capacity,
  and overlap behavior.

Primary NVIDIA evidence:

- architecture feature checks,
- SASS verification,
- NCU/CUPTI async/copy metrics where available,
- PM Sampling for phase behavior.

Validation:

- async/TMA workloads,
- simulator TMA/copy-engine trace.

Simulator parameters:

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

Scalar policy:

- feature presence and instruction support can be direct;
- internal queue values are simulator-equivalent fitted parameters.

### Interconnect And Address Mapping

Hardware-neutral concept:

- fabric saturation, address-to-partition behavior, injection pressure, and
  routing-like latency.

Primary NVIDIA evidence:

- memory partition and L2 metrics where available,
- traffic imbalance and partition-camping workloads,
- PM Sampling for phase behavior.

Validation:

- NVBit memory address stream,
- simulator interconnect and partition traces.

Simulator parameters:

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

Scalar policy:

- address mapping can be inferred or bounded if partition counters and address
  sweeps agree;
- router microparameters should usually be `behavioral_class`,
  `bounded_range`, or `multi_fit`, not exact hardware facts.

## Implementation Plan

### Phase N0: Bring Current P0 Scaffolding Into Layered Output

Files:

- `amora/schemas/results.py`
- `amora/reports/json_report.py`
- `amora/backends/nvidia/cuda.py`
- `amora/probes/nvidia/p0/*.py`

Work:

- Add layered result objects matching this plan.
- Add fit status, uncertainty category, variance, assumptions, and coupled
  parameters to simulator estimates.
- Record binary/disassembly hashes when available.
- Keep current P0 dry-run behavior compatible with tests.

### Phase N1: Add NVIDIA Capability And Metric Resolution

Files:

- `amora/backends/nvidia/capabilities.py`
- `amora/backends/nvidia/metrics.py`
- `amora/backends/nvidia/ncu.py`
- `amora/backends/nvidia/cupti.py`

Work:

- Query CUDA, NCU, CUPTI, NVBit, and disassembly tool availability.
- Resolve logical metrics to target-specific candidate metric names.
- Emit unsupported reasons and fallback routes.

### Phase N2: Add ISA/SASS Semantic Records

Files:

- `amora/backends/nvidia/isa_semantics.yaml`
- `amora/backends/nvidia/disassembly.py`
- `amora/backends/nvidia/nvbit.py`

Work:

- Define SASS/PTX semantic classes for P0-P3 probes.
- Add verification hooks for required opcode and dependency patterns.
- Add NVBit validation records where practical.

### Phase N3: Convert P0-P3 Methodology Files

Files:

- `.plan/nvidia-p0-kernel-methodology.md`
- `.plan/nvidia-p1-kernel-methodology.md`
- `.plan/nvidia-p2-kernel-methodology.md`
- `.plan/nvidia-p3-kernel-methodology.md`

Work:

- Add per-probe measurement contracts.
- Mark primary evidence source and validation evidence.
- Add clock-domain policy, rejection rules, downgrade rules, and fallback.
- Add scalar-output policy and expected fit status.

### Phase N4: Add Simulator Trace Contracts

Files:

- `amora/backends/simulator_trace.py`
- `amora/probes/simulator_state/*.py`

Work:

- Define simulator trace points for scheduler state, queue lengths, pipeline
  occupancy, cache state, interconnect state, and memory partition state.
- Use traces to compare fitted simulator-equivalent parameters with dynamic
  simulator behavior.

### Phase N5: Expand Beyond P0

Work order:

1. topology and occupancy,
2. arithmetic latency and throughput,
3. shared/local memory,
4. coalescing and L1/L2 behavior,
5. scheduler and operand delivery,
6. DRAM and partition behavior,
7. tensor core,
8. synchronization,
9. TMA/async copy,
10. interconnect and address mapping.

Reasoning:

- Start with metadata-backed and direct-counter-backed probes.
- Move next to probes with stable microkernel contracts.
- Leave highly coupled hidden structures for later phases where simulator
  traces and fitting infrastructure exist.

## Acceptance Criteria

This NVIDIA plan is implemented when:

- every NVIDIA probe emits layered results,
- every simulator parameter estimate references a measurement contract,
- every logical metric is resolved through the metric resolver,
- direct NCU/CUPTI metrics are primary only when the metric contract is direct,
- microkernel timing is separated from profiler runs,
- disassembly and NVBit validation are recorded where required,
- published parameters and CUDA metadata are used as trust-and-verify anchors,
- simulator dynamic state trace contracts exist for fitted mappings,
- every non-direct estimate reports fit status, uncertainty category, variance,
  assumptions, and coupled parameters,
- clock domains and conversion methods are explicit,
- unsupported, downgraded, and rejected measurements appear in reports.

## Key Decision Record

- Decision: NCU/CUPTI direct counters can be primary evidence.
  Reason: when metric semantics match the target behavior, counters expose
  execution facts that end-to-end timing can hide behind runtime fog.

- Decision: Microkernels are controlled workload generators, validation tools,
  and fallback measurement paths, not universally primary truth.
  Reason: microkernel timing is also affected by clocks, scheduling, cache
  state, launch overhead, and interference.

- Decision: Published NVIDIA parameters are trust-and-verify anchors.
  Reason: documented facts should not be re-inferred unless the goal is
  validation or detecting mode-specific differences.

- Decision: Simulator internal states are directly observable.
  Reason: queue lengths, scheduler state, cache state, and pipeline state can be
  traced in the simulator. The challenge is fitting hardware observations to
  simulator-equivalent behavior.

- Decision: Hidden NVIDIA structures map to simulator-equivalent behavior only
  when contracts and fitting metadata justify it.
  Reason: many simulator knobs are not literal NVIDIA hardware facts.
