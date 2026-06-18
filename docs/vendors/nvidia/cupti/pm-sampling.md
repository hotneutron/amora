# CUPTI PM Sampling API

## Source

- CUPTI PM Sampling usage: https://docs.nvidia.com/cupti/main/main.html#cupti-pm-sampling-api
- CUPTI PM Sampling API module: https://docs.nvidia.com/cupti/api/group__CUPTI__PM__SAMPLING__API.html

## Purpose

PM Sampling periodically samples GPU performance monitors. It produces a
timeline of metric samples rather than replayed exact per-kernel counter values.

It is useful when replay profiling is too intrusive or when AMORA needs to see
phase changes inside a long-running workload.

PM Sampling is supported on Turing and newer GPUs, compute capability 7.5 and
above.

## Host-Side Requirements

PM Sampling uses the Host Profiling API for:

- metric enumeration
- metric-property queries
- single-pass metric-set discovery
- config-image creation
- counter-data evaluation

The config image must be single-pass. CUDA 13.2 adds
`cuptiProfilerHostGetSinglePassSets` and
`cuptiProfilerHostGetMetricsInSinglePassSet` to discover compatible sets.

## Target-Side Flow

1. `cuptiPmSamplingEnable` for a device.
2. `cuptiPmSamplingSetConfig` with config image, hardware buffer size, sampling
   interval, and trigger mode.
3. `cuptiPmSamplingGetCounterDataSize`.
4. `cuptiPmSamplingCounterDataImageInitialize`.
5. `cuptiPmSamplingStart`.
6. Periodically call `cuptiPmSamplingDecodeData`, especially for long workloads.
7. `cuptiPmSamplingStop`.
8. Decode remaining data.
9. Evaluate samples with Host Profiling APIs.
10. `cuptiPmSamplingDisable`.

## Trigger Modes

- `GPU_SYSCLK_INTERVAL`: sampling interval is expressed in GPU system clock
  cycles.
- `GPU_TIME_INTERVAL`: sampling interval is expressed in nanoseconds. This mode
  is not supported on Turing and GA100 chips.

## AMORA Use

Use PM Sampling for:

- long-running probes where replay would distort behavior
- producer/consumer pipelines
- async copy or TMA-like phase behavior
- DVFS, throttling, and clock-correlation checks
- broad triage before choosing a narrower Range Profiling metric set

PM Sampling should complement Range Profiling. It should not be treated as an
exact replacement for per-kernel counter collection.

## Caveats

- Sampling frequency is limited by GPU size, load, and system pressure.
- Hardware buffer overflow must be checked during decode.
- The metric set must fit in one pass.
- Samples need workload timeline context to be useful; correlate with CUPTI
  Activity records or NVTX ranges.
