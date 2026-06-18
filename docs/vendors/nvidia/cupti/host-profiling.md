# CUPTI Host Profiling API

## Source

- CUPTI Host Profiling usage: https://docs.nvidia.com/cupti/main/main.html#cupti-profiler-host-api
- CUPTI Profiler Host API module: https://docs.nvidia.com/cupti/api/group__CUPTI__PROFILER__HOST__API.html

## Purpose

The Host Profiling API performs host-side operations needed by Range Profiling
and PM Sampling:

1. Enumerate supported metrics.
2. Inspect metric properties and submetrics.
3. Build config images that schedule raw counters.
4. Evaluate counter-data images into readable metric values.

## Metric Naming

CUPTI uses the same PerfWorks-style metric vocabulary exposed through Nsight
Compute. A metric usually starts with a unit prefix such as `sm__`, `smsp__`,
`l1tex__`, `lts__`, or `dram__`, then adds rollups and submetrics.

Common rollups:

- `.sum`: sum across instances.
- `.avg`: average across instances.
- `.min`: minimum instance value.
- `.max`: maximum instance value.

Common counter submetrics:

- `.per_second`
- `.per_cycle_active`
- `.per_cycle_elapsed`
- `.pct_of_peak_sustained_active`
- `.pct_of_peak_sustained_elapsed`

## Key APIs

- `cuptiProfilerHostInitialize`
- `cuptiProfilerHostGetBaseMetrics`
- `cuptiProfilerHostGetSubMetrics`
- `cuptiProfilerHostGetMetricProperties`
- `cuptiProfilerHostConfigAddMetrics`
- `cuptiProfilerHostGetConfigImageSize`
- `cuptiProfilerHostGetConfigImage`
- `cuptiProfilerHostGetNumOfPasses`
- `cuptiProfilerHostEvaluateToGpuValues`
- `cuptiProfilerHostGetSinglePassSets`
- `cuptiProfilerHostGetMetricsInSinglePassSet`

## Config Images

After metric selection, CUPTI creates a config image that records scheduling
information for the required raw metrics. A metric set may require multiple
passes for Range Profiling. PM Sampling requires a single-pass configuration.

For AMORA, cache config images only when these inputs match:

- chip name
- CUDA/CUPTI version
- selected metrics
- profiler type
- single-pass metric set name, when used
- device partition information, when profiling green contexts

## Evaluation

Range Profiling and PM Sampling decode hardware data into counter-data images.
Host Profiling then evaluates selected metrics for a range index or sample
index.

Store both the evaluated value and the original metric name. The suffix carries
semantic information that is needed for simulator-parameter mapping.

## AMORA Mapping

The Host Profiling API is the enumeration and normalization layer for:

- `shader_core_config::*` throughput and stall proxies from `sm__` and `smsp__`
  metrics.
- `cache_config::*` and `l1d_cache_config::*` from `l1tex__` and cache hit/miss
  metrics.
- `memory_config::*` from `lts__`, `dram__`, NVLINK, C2C, and PCIe metrics.
- Tensor/TMA probes from tensor-pipe, async-copy, and memory-pipeline metrics
  where available.
