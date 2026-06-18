# CUDA Performance Counter Ecosystem Survey — 2026-06-17

## Summary

The CUDA performance-monitoring ecosystem has three distinct layers:

1. **Nsight Compute / NCU**: the highest-level kernel profiler and practical
   entry point for counter collection.
2. **CUPTI / PerfWorks**: the API layer used to build profilers, enumerate
   metrics, configure passes, collect counter data, sample PM state, and map
   metrics to source/SASS.
3. **NVBit**: a dynamic SASS instrumentation framework. It is useful for
   instruction traces, opcode histograms, register inspection, and memory traces,
   but it is not the primary API for hardware PM counter access.

For reverse engineering microarchitectural parameters, use NCU for broad
counter discovery and first-pass profiling, CUPTI Range Profiling / PM Sampling
for programmable collection, and NVBit for ground-truth dynamic instruction and
memory-reference streams.

## Main Tools and What They Expose

| Tool / API | Role | Best For | Notes |
|---|---|---|---|
| Nsight Compute CLI (`ncu`) | Kernel profiler and metric frontend | Fast exploration of available metrics, sections, Speed-of-Light summaries, scheduler stalls, memory metrics, source/SASS attribution | Uses section sets and metrics. May replay kernels multiple times to collect all requested metrics. |
| CUPTI Range Profiling API | Programmable metric collection | Building custom profilers that collect metrics for kernels or ranges | Current recommended path over older Profiling/Event/Metric APIs. |
| CUPTI Host Profiling API | Metric enumeration/config/evaluation | Querying base metrics, submetrics, passes, metric properties | Wraps newer PerfWorks-style host concepts. |
| CUPTI PM Sampling API | Periodic hardware PM sampling | Timeline-style performance monitor samples | Samples hardware performance monitors periodically. Useful when exact replay profiling is too intrusive. |
| CUPTI PC Sampling API | Warp PC and stall-reason sampling | Finding instruction PCs responsible for stalls | Sampling-based, not exact counter collection. |
| CUPTI SASS Metrics API | Source/SASS-level metrics | Instruction-level metric attribution | Uses SASS patching; important for source/SASS correlation. |
| NVBit | Dynamic binary instrumentation | Instruction count, opcode histogram, memory trace, register value capture | Cannot be run together with Nsight/nvprof-style injection tools in the common workflow. |

## Counter / Metric Taxonomy

### Nsight Compute Metric Prefixes

Common NCU metric names encode the hardware unit and aggregation suffix:

- `gpu__*`: device-level / whole GPU metrics, e.g. DRAM Speed-of-Light
  throughput.
- `gpc__*`: GPC-level metrics and clocks.
- `sm__*`: SM-level metrics, e.g. SM throughput and active cycles.
- `smsp__*`: SM subpartition / scheduler-level metrics.
- `l1tex__*`: L1/TEX path metrics, including local/shared/global memory paths.
- `lts__*`: L2 cache slice metrics.
- `dram__*`: DRAM / framebuffer memory metrics.
- `launch__*`: static launch metadata such as registers per thread.
- `sass__*` or source/SASS views: instruction-level attribution when available
  through NCU sections or CUPTI SASS metrics.

Important suffix patterns:

- `.sum`: summed raw-like count.
- `.avg`: average over instances.
- `.min` / `.max`: per-instance extrema.
- `.per_second`: rate.
- `.pct_of_peak_sustained_elapsed`: percent of sustained peak over elapsed time.
- `.pct_of_peak_sustained_active`: percent of sustained peak over active cycles.

### Performance Categories

| Category | Representative Metrics |
|---|---|
| SM / compute throughput | `sm__throughput.avg.pct_of_peak_sustained_elapsed`, pipe-specific utilization metrics |
| Scheduler / latency stalls | `smsp__warps_*`, `smsp__average_warps_issue_stalled_*` |
| Occupancy | achieved occupancy metrics, active warps, launch limits such as registers/shared memory |
| Instruction mix | SASS instruction statistics, opcode histograms via NCU or NVBit |
| Shared memory | bank conflicts, transactions, shared load/store throughput |
| L1/TEX | `l1tex__*` requests, sectors, hit rates, bandwidth |
| L2 | `lts__*` requests, bytes, sectors, hit rates, throughput |
| DRAM | `dram__bytes*`, DRAM throughput, memory-controller metrics where supported |
| Interconnect / external links | NVLINK (`nvl*`), C2C, PCIe metrics via newer CUPTI profiling APIs |
| Source/SASS attribution | PC sampling, SASS metrics, source counters |

## Key Repos and Docs

### Nsight Compute / NCU

- Nsight Compute Profiling Guide:
  `https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html`
  - Best source for metric structure, metric collection, replay, sections, rules,
    sampling, reproducibility, roofline charts, and memory charts.
  - Documents that NCU uses section sets, sections, and metrics; `--set full`
    collects a broad profile while the default `basic` set is lighter.

- Nsight Compute CLI:
  `https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html`
  - Best source for automation.
  - Useful commands:

```bash
ncu --list-sets
ncu --list-sections
ncu --query-metrics
ncu --query-metrics-mode suffix --metrics sm__throughput
ncu --set full ./app
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__bytes.sum.per_second ./app
ncu --page raw --csv ./app
```

- Nsight Compute Customization Guide:
  `https://docs.nvidia.com/nsight-compute/CustomizationGuide/index.html`
  - Best source for section files, rule files, source counters, derived metrics,
    PM sampling timelines, and counter domains.
  - Important for understanding how NCU groups and requests metrics.

### CUPTI / PerfWorks

- CUPTI documentation index:
  `https://docs.nvidia.com/cupti/index.html`
  - Best overview of Activity, Callback, Host Profiling, Range Profiling,
    PC Sampling, SASS Metric, PM Sampling, Profiling, and Checkpoint APIs.

- CUPTI usage guide:
  `https://docs.nvidia.com/cupti/main/main.html`
  - Best source for practical API flow and samples.
  - Notes that `cuptiSubscribe()` should be called before profiling sessions and
    that multiple CUPTI subscribers can interfere.

- CUPTI API modules:
  `https://docs.nvidia.com/cupti/api/modules.html`
  - Best source for exact API structs and entry points.
  - Relevant modules:
    - `CUPTI_RANGE_PROFILER_API`
    - `CUPTI_PROFILER_HOST_API`
    - `CUPTI_PM_SAMPLING_API`
    - `CUPTI_PCSAMPLING_API`
    - `CUPTI_SASS_METRICS_API`

- CUPTI release notes:
  `https://docs.nvidia.com/cupti/release-notes/release-notes.html`
  - Important because APIs and architecture support move quickly.
  - CUDA 12.8 added Blackwell support and newer hardware event tracing.
  - Newer releases deprecate older Event/Metric APIs in favor of Range Profiling.

### NVBit

- Official NVBit repo:
  `https://github.com/NVlabs/NVBit`
  - Research prototype from NVIDIA Architecture Research Group.
  - Dynamic SASS instrumentation without recompilation.
  - Can inspect SASS instructions when functions are loaded and inject calls
    before/after instructions.
  - Good examples: dynamic instruction counters, instruction tracers, memory
    reference tracers, profiling tools.

- NVBit MICRO 2019 paper:
  `https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus`
  - Best architectural description of how NVBit works internally:
    dynamic recompilation at SASS level, ABI-compliant injected code, register
    requirement analysis, basic-block instrumentation, and source correlation.

- NVBit tutorial repo:
  `https://github.com/eunomia-bpf/nvbit-tutorial`
  - Practical examples for:
    - `instr_count`
    - `instr_count_bb`
    - `opcode_hist`
    - memory trace tools
    - register value recording
  - Useful for building instrumentation-side measurements that complement PMCs.

- NVBit tutorial docs:
  `https://eunomia.dev/others/nvbit-tutorial/`
  - More readable walkthrough of the same examples.

### Related Survey / Integration Repos

- MemSysExplorer NVBit profiler documentation:
  `https://msx.ece.tufts.edu/docs/html/profilers/nvbit.html`
  - Example of wrapping NVBit instrumentation into a higher-level profiling
    workflow for memory access frequency and working-set statistics.

## Practical Architecture for a Full-Picture PMC Survey

```text
CUDA application / kernel
        |
        | launch interception / profiling injection
        v
Nsight Compute CLI
        |
        | section sets, metrics, replay, source/SASS views
        v
NCU metric report (.ncu-rep / CSV / raw page)

CUDA application / custom profiler
        |
        | CUPTI Activity / Callback / Range Profiling / PM Sampling
        v
CUPTI + PerfWorks metric configuration and evaluation
        |
        v
Counter data image + evaluated metrics

CUDA binary
        |
        | LD_PRELOAD / CUDA_INJECTION64_PATH
        v
NVBit SASS instrumentation
        |
        v
Dynamic instruction traces / opcode histograms / memory traces
```

## What Each Layer Can and Cannot Tell You

### NCU

Strengths:

- Easiest way to list and collect metrics.
- Good report structure: Speed-of-Light, Memory Workload Analysis, Scheduler
  Stats, Instruction Stats, Source/SASS views.
- Handles replay, sections, derived metrics, and peak-normalized throughput.

Limitations:

- Not a raw event interface in the old `nvprof --events` sense.
- Many metric sets require replay, which can perturb concurrent or non-replayable
  workloads.
- Some metrics are architecture-specific and not stable across GPU generations.

### CUPTI

Strengths:

- Programmatic profiler construction.
- Lets tools define ranges, configure metrics, evaluate data, collect PM samples,
  and collect PC/stall samples.
- Best path for automation in a probing suite.

Limitations:

- More complex than NCU.
- API support varies by CUDA version, driver, architecture, MIG/vGPU/MPS mode,
  and profiling permissions.
- Older Event/Metric APIs are deprecated; prefer Range Profiling and Host
  Profiling APIs.

### NVBit

Strengths:

- Observes dynamic SASS execution directly.
- Works without application source code.
- Useful for validating instruction mix, memory reference stream, register use,
  and control flow.

Limitations:

- Instrumentation overhead can be high.
- It is not the primary interface to hardware PM counters.
- It conflicts with Nsight/nvprof-style injection workflows in common usage.

## Recommended Workflow

1. Use `ncu --list-sets`, `ncu --list-sections`, and `ncu --query-metrics` to
   discover metric names on the target GPU.
2. Run `ncu --set full` on representative kernels to identify the limiting
   subsystem: SM, tensor, shared/L1, L2, DRAM, scheduler stalls, or occupancy.
3. Export `--page raw --csv` for machine-readable metric harvesting.
4. Inspect NCU section files and the Customization Guide to understand which
   metrics are grouped into each report section.
5. Use CUPTI Range Profiling for programmable collection of the selected metrics.
6. Use CUPTI PM Sampling when time-varying behavior matters more than exact
   replayed kernel counters.
7. Use CUPTI PC Sampling / SASS Metrics for source/SASS-level attribution.
8. Use NVBit to collect dynamic instruction/memory traces that NCU/CUPTI counters
   cannot express directly.
9. Cross-check:
   - NCU instruction stats vs NVBit opcode histogram.
   - NCU memory transactions/bytes vs NVBit memory-reference stream.
   - NCU stall reasons vs CUPTI PC sampling.
   - NCU throughput counters vs microbenchmark-derived bandwidth/latency.

## Reading Order

1. Nsight Compute Profiling Guide, especially metric collection, metric
   structure, replay, sampling, reproducibility, memory charts, and roofline.
2. Nsight Compute CLI docs for automation and metric query commands.
3. Nsight Compute Customization Guide for section files, derived metrics, PM
   sampling timelines, and counter domains.
4. CUPTI overview and Range Profiling / Host Profiling API docs.
5. CUPTI PM Sampling, PC Sampling, and SASS Metrics API docs.
6. NVBit paper for instrumentation architecture.
7. NVBit repo and tutorial examples for practical dynamic instrumentation.

## Implications for the GPU Probing Suite

For a probing suite that reverse engineers microarchitectural parameters:

- Treat NCU metrics as the reference vocabulary for CUDA-visible counters.
- Treat CUPTI as the programmable collection backend.
- Treat NVBit as the dynamic-instruction oracle, not as the hardware-counter
  oracle.
- Build adapters that map NCU/CUPTI metric names onto simulator parameters:
  `shader_core_config::*`, `cache_config::*`, `memory_config::*`, and TMA/tensor
  model names.
- Do not assume metric availability across architectures. Query first, then
  select probes and counters.
