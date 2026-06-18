# Nvidia GPU Architectural Block Diagram — 2026-06-17

## Simulator-Oriented Architecture

This diagram reflects the Nvidia GPU architecture modeled by this repository.
It is based on the simulator's own block structure: `gpgpu_sim`, SIMT clusters,
SMs, subcores, functional-unit pipelines, memory hierarchy, interconnect, L2,
DRAM, tensor pipelines, and TMA units.

```mermaid
flowchart TB
    Host[Host Runtime / Trace Frontend]

    GPU[gpgpu_sim<br/>Top-Level GPU Simulator]

    Host --> GPU

    GPU --> CoreDomain[CORE Clock Domain]
    GPU --> ICNTDomain[ICNT Clock Domain]
    GPU --> MemDomain[L2 / DRAM Clock Domain]

    CoreDomain --> Cluster0[SIMT Core Cluster 0]
    CoreDomain --> ClusterN[SIMT Core Cluster N]

    subgraph Cluster[SIMT Core Cluster]
        SM0[SM / Shader Core]
        SMN[SM / Shader Core]
    end

    Cluster0 --> SM0
    ClusterN --> SMN

    subgraph SM[Streaming Multiprocessor]
        Fetch[Fetch + L0/L1 I-Cache]
        Decode[Decode]
        Schedulers[Warp Schedulers]
        Scoreboard[Scoreboards<br/>Read / Write Hazards]

        subgraph Subcores[Subcores]
            Subcore0[Subcore 0]
            Subcore1[Subcore 1]
            SubcoreN[Subcore N]
        end

        subgraph Pipelines[Execution Pipelines]
            INT[INT / ALU]
            SP[SP]
            DP[DP]
            SFU[SFU]
            Tensor[Tensor Core Pipe]
            Branch[Branch]
            Misc[Misc / No-Queue]
            MemPipe[Memory Pipe]
            TMA[TMA Pipe]
        end

        LDST[Shared LD/ST Unit]
        TMAUnit[TMA Unit<br/>Tensor Memory Accelerator]
        L1D[L1 Data / Shared Memory Path]
        Const[Constant Cache]
        Barriers[CTA Barriers / Sync State]
    end

    SM0 --> Fetch
    Fetch --> Decode --> Schedulers
    Schedulers --> Scoreboard
    Scoreboard --> Subcore0
    Scoreboard --> Subcore1
    Scoreboard --> SubcoreN

    Subcore0 --> INT
    Subcore0 --> SP
    Subcore0 --> DP
    Subcore0 --> SFU
    Subcore0 --> Tensor
    Subcore0 --> Branch
    Subcore0 --> Misc
    Subcore0 --> MemPipe
    Subcore0 --> TMA

    MemPipe --> LDST
    TMA --> TMAUnit
    LDST --> L1D
    TMAUnit --> L1D
    Misc --> Barriers

    L1D --> ShaderMemIF[Shader Memory Interface]
    Const --> ShaderMemIF
    TMAUnit --> ShaderMemIF

    ShaderMemIF --> ICNT[Interconnect / ICNT]

    ICNT --> MemPartitions[Memory Partitions]

    subgraph MemorySystem[Memory System]
        L2[L2 Cache / Subpartition]
        DRAMSched[Memory Partition / DRAM Scheduler]
        DRAM[DRAM Model]
    end

    MemPartitions --> L2
    L2 --> DRAMSched
    DRAMSched --> DRAM

    DRAM --> DRAMSched --> L2 --> ICNT
    ICNT --> ShaderMemIF
```

## Performance-Critical Parameter Taxonomy

The items below are ordered by typical impact on kernel runtime. The exact order
is workload-dependent: compute-bound, memory-bound, tensor-bound, and
synchronization-bound kernels will stress different parts of the diagram. Repo
names are used where the simulator exposes a concrete parameter, class, or
modeled structure.

| Priority | Big-Ticket Item | Diagram Blocks | Why It Matters | Repo-Defined Parameters |
|---:|---|---|---|---|
| 1 | **Occupancy and SM residency** | `gpgpu_sim`, SIMT clusters, SM, subcores | Determines how many warps and CTAs can be resident, which controls latency hiding and total parallel work in flight. | `shader_core_config::n_simt_clusters`<br>`shader_core_config::n_simt_cores_per_cluster`<br>`gpgpu_sim_config::num_shader()`<br>`shader_core_config::n_thread_per_shader`<br>`shader_core_config::warp_size`<br>`shader_core_config::max_warps_per_shader`<br>`shader_core_config::max_cta_per_core`<br>`shader_core_config::gpgpu_shader_registers`<br>`shader_core_config::gpgpu_shmem_size`<br>`shader_core_config::gpgpu_shmem_per_block` |
| 2 | **Issue, scheduling, and execution throughput** | Warp schedulers, scoreboards, subcores, execution pipelines | Defines the compute roofline: how many instructions can issue, which pipelines can dual-issue, and how fast ALU/SFU/DP pipelines accept dependent and independent work. | `shader_core_config::num_subcores_in_SM`<br>`shader_core_config::gpgpu_num_sched_per_core`<br>`shader_core_config::gpgpu_scheduler_string`<br>`shader_core_config::gpgpu_max_insn_issue_per_warp`<br>`shader_core_config::gpgpu_dual_issue_diff_exec_units`<br>`shader_core_config::pipeline_widths_string`<br>`shader_core_config::pipe_widths`<br>`gpgpu_num_sp_units`<br>`gpgpu_num_dp_units`<br>`gpgpu_num_int_units`<br>`gpgpu_num_sfu_units`<br>`shader_core_config::max_sp_latency`<br>`shader_core_config::max_int_latency`<br>`shader_core_config::max_sfu_latency`<br>`shader_core_config::max_dp_latency` |
| 3 | **Register file and operand delivery** | Schedulers, scoreboards, execution pipelines | A kernel can be pipeline-rich but operand-starved. Register banks, ports, collector units, and register latency affect bank conflicts, dependency stalls, and achieved issue rate. | `shader_core_config::gpgpu_num_reg_banks`<br>`shader_core_config::reg_file_port_throughput`<br>`gpgpu_reg_bank_use_warp_id`<br>`num_regular_register_file_read_ports_per_bank`<br>`num_regular_register_file_write_ports_per_bank`<br>`max_latency_regular_register_file_latency`<br>`gpgpu_operand_collector_num_units_*`<br>`gpgpu_operand_collector_num_in_ports_*`<br>`gpgpu_operand_collector_num_out_ports_*` |
| 4 | **Shared memory, L1, and SM memory pipeline** | LD/ST unit, L1 data / shared memory path, shader memory interface | Controls latency and bandwidth for on-chip data movement. Bank conflicts, coalescing, queue limits, and L1 lookup bandwidth often decide whether a kernel reaches its compute roofline. | `shader_core_config::gpgpu_shmem_num_banks`<br>`gpgpu_shmem_limited_broadcast`<br>`gpgpu_shmem_warp_parts`<br>`gpgpu_smem_latency`<br>`memory_shared_memory_minimum_latency`<br>`memory_l1d_minimum_latency`<br>`memory_l1d_max_lookups_per_cycle_per_bank`<br>`memory_maximum_coalescing_cycles`<br>`memory_subcore_link_to_sm_byte_size`<br>`memmory_max_concurrent_requests_shmem_per_sm`<br>`memmory_max_concurrent_requests_standard_per_sm`<br>`m_L1D_config`<br>`l1d_cache_config::l1_latency`<br>`l1d_cache_config::l1_banks` |
| 5 | **L2, DRAM, and global memory service** | Interconnect, memory partitions, L2, DRAM scheduler, DRAM model | Dominates memory-bound kernels. Cache geometry, memory partition count, bus width, burst length, DRAM timing, and scheduler policy determine global-memory latency and sustained bandwidth. | `memory_config::m_L2_config`<br>`gpgpu_cache:dl2`<br>`gpgpu_l2_rop_latency`<br>`memory_config::m_n_mem`<br>`memory_config::m_n_sub_partition_per_memory_channel`<br>`gpgpu_n_mem_per_ctrlr`<br>`memory_config::scheduler_type`<br>`gpgpu_dram_partition_queues`<br>`gpgpu_frfcfs_dram_sched_queue_size`<br>`gpgpu_dram_return_queue_size`<br>`memory_config::busW`<br>`memory_config::BL`<br>`memory_config::nbk`<br>`memory_config::nbkgrp`<br>`memory_config::tCCD`<br>`memory_config::tRCD`<br>`memory_config::tRAS`<br>`memory_config::tRP`<br>`memory_config::CL`<br>`memory_config::WL`<br>`dram_latency`<br>`dram_data_command_freq_ratio` |
| 6 | **Tensor and matrix engine throughput** | Tensor core pipe, subcores, schedulers | Dominates GEMM, convolution, attention, and other matrix-heavy kernels. Tensor latency, unit count, issue rate, and shape-specific extra latency set the tensor roofline. | `gpgpu_tensor_core_avail`<br>`gpgpu_num_tensor_core_units`<br>`tensor_latency`<br>`tensor_rate_per_cycle`<br>`shader_core_config::max_tensor_core_latency`<br>`tensor_extra_latency_16816_fp32_1688_fp32` |
| 7 | **TMA / async copy / DMA behavior** | TMA pipe, TMA unit, L1/shared path, shader memory interface | Controls producer/consumer kernels that overlap global-to-shared movement with compute. Queue depth, request rate, transfer state, and outstanding limits decide whether copy engines hide memory latency. | `tma_unit_sm::kMaxRequestsPerCycle`<br>`TMACommand`<br>`TMATransferEntry`<br>`TMAOpcodeFamily`<br>`TMADirection`<br>`TMATransferType`<br>`TMAOperandForm`<br>`m_command_queue`<br>`m_in_flight_transfers`<br>`m_outstanding_requests`<br>`m_outstanding_stores_per_warp`<br>`Subcore::m_tma_pipeline`<br>`SM::m_tma_unit_shared_of_sm`<br>`m_EX_TMA_reception_latches_per_subcore` |
| 8 | **Synchronization, fences, and barrier progress** | Barriers / sync state, misc pipeline, TMA completion path | Barrier and fence costs directly affect tiled kernels, reductions, cluster-scope communication, and async-copy completion. Incorrect readiness modeling can create large performance or progress errors. | `gpgpu_num_cta_barriers`<br>`BARRIER_OP`<br>`MEMORY_BARRIER_OP`<br>`GRID_BARRIER_OP`<br>`MBARRIER_OP`<br>`CLUSTER_BARRIER_OP`<br>`sync_debug_enable`<br>`sync_debug_print_budget`<br>`sync_debug_skip_runtime_budget` |
| 9 | **Interconnect and address mapping** | Shader memory interface, ICNT, memory partitions | Determines how SM requests reach L2/DRAM partitions. Address mapping and ICNT latency/bandwidth can create partition camping, congestion, and backpressure. | `icnt_flit_size`<br>`gpgpu_mem_addr_mapping`<br>`gpgpu_mem_address_mask`<br>`routing_delay`<br>`vc_alloc_delay`<br>`sw_alloc_delay`<br>`credit_delay`<br>`input_speedup`<br>`output_speedup`<br>`internal_speedup` |
| 10 | **Instruction, constant, texture, and cache policy details** | Fetch, L0/L1 instruction cache, constant cache, texture cache | Usually secondary for dense compute kernels, but important for large instruction footprints, lookup-heavy kernels, texture paths, and constant-cache broadcast behavior. | `m_L1I_L1_half_C_cache_config`<br>`m_L0I_config`<br>`m_L1C_config`<br>`m_L0C_config`<br>`m_L1T_config`<br>`cache_config::m_nset`<br>`cache_config::m_line_sz`<br>`cache_config::m_assoc`<br>`cache_config::m_mshr_entries`<br>`cache_config::m_mshr_max_merge`<br>`cache_config::m_miss_queue_size` |

## Parameter Impact Notes

Occupancy parameters define how much latency the machine can hide. Scheduler and
pipeline parameters define the compute roofline once enough warps are resident.
Register, shared-memory, and cache parameters often decide whether a kernel can
actually reach that roofline.

For memory-bound kernels, L2, DRAM, interconnect, coalescing, and address-mapping
parameters dominate. For modern GEMM, attention, and producer/consumer kernels,
tensor throughput, TMA / async-copy behavior, and synchronization progress become
first-order performance terms.

## Repo Mapping

| Architectural Block | Repo Location |
|---|---|
| GPU top-level simulator | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/gpu-sim.h`, `gpu-sim.cc` |
| SIMT clusters | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/shader.h`, `shader.cc` |
| Remodeled SM | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/sm.h`, `sm.cc` |
| Subcores and issue pipeline | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/subcore.h`, `subcore.cc` |
| Functional units | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/functional_unit.cc` |
| Operation types and warp instructions | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/operation_type.h`, `abstract_hardware_model.h` |
| LD/ST unit | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/ldst_unit_sm.h`, `ldst_unit_sm.cc` |
| TMA unit | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/remodeling/tma_unit_sm.h`, `tma_unit_sm.cc` |
| Interconnect | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/icnt_wrapper.h`, `icnt_wrapper.cc`, `local_interconnect.h`, `intersim2/` |
| L2 and cache model | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/l2cache.h`, `l2cache.cc`, `gpu-cache.h`, `gpu-cache.cc` |
| DRAM model | `simulator-remodeled/gpu-simulator/gpgpu-sim/src/gpgpu-sim/dram.h`, `dram.cc` |

## Interpretation Notes

This is a simulator-oriented architectural diagram, not a transistor-accurate
Nvidia product floorplan. The diagram emphasizes the components represented in
the repo and the paths exercised by the timing model.
