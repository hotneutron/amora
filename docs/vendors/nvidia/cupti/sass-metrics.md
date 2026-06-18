# CUPTI SASS Metrics API

## Source

- CUPTI SASS Metrics usage: https://docs.nvidia.com/cupti/main/main.html#cupti-sass-metric-api
- CUPTI SASS Metrics API module: https://docs.nvidia.com/cupti/api/group__CUPTI__SASS__METRICS__API.html

## Purpose

The SASS Metrics API collects metrics at SASS assembly instruction level. It
uses SASS patching and supports a larger set of SASS instruction-level metrics
than the old Activity API path.

It is supported on Turing and newer GPUs, compute capability 7.5 and above.

## Collection Granularity

CUPTI can collect SASS metrics at:

- GPU level: `CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_GPU`
- SM level: `CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_SM`
- SMSP level: `CUPTI_SASS_METRICS_OUTPUT_GRANULARITY_SMSP`

SMSP-level output is especially relevant for AMORA because the simulator has
subcore and scheduler-level structures.

## Flow

1. Query chip name with `cuptiDeviceGetChipName`.
2. Query metric count with `cuptiSassMetricsGetNumOfMetrics`.
3. Enumerate metrics with `cuptiSassMetricsGetMetrics`.
4. Query metric properties with `cuptiSassMetricsGetProperties`.
5. Build `CUpti_SassMetrics_Config` entries.
6. Set device config with `cuptiSassMetricsSetConfig`.
7. Enable SASS patching with `cuptiSassMetricsEnable`.
8. Launch workload.
9. Query data properties with `cuptiSassMetricsGetDataProperties`.
10. Flush metric data with `cuptiSassMetricsFlushData`.
11. Disable with `cuptiSassMetricsDisable`.
12. Unset config with `cuptiSassMetricsUnsetConfig`.

## Lazy Patching

`enableLazyPatching` patches kernels at first launch instead of eagerly patching
all kernels in the module. This is preferable for AMORA probe binaries that may
contain many kernels but only launch a selected subset.

## AMORA Use

Use SASS Metrics when the probe needs instruction-level attribution that plain
range metrics cannot provide:

- per-instruction execution or activity counters
- source/SASS correlation for hot instructions
- SMSP-level imbalance
- validating NVBit instruction counts against CUPTI-attributed metrics

SASS Metrics are complementary to NVBit. CUPTI provides hardware metric
attribution; NVBit provides explicit dynamic instrumentation streams.

## Caveats

- Metrics must be queried per chip.
- SASS patching changes execution and can add overhead.
- Flush data before disabling; data collected after the last flush can be
  discarded when `cuptiSassMetricsDisable` resets patched kernels.
- Set and unset configuration per CUDA device.
