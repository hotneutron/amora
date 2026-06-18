# CUPTI Documentation Notes

## Sources

- CUPTI documentation index: https://docs.nvidia.com/cupti/index.html
- CUPTI usage guide: https://docs.nvidia.com/cupti/main/main.html
- CUPTI API modules: https://docs.nvidia.com/cupti/api/modules.html
- CUPTI release notes: https://docs.nvidia.com/cupti/release-notes/release-notes.html

## Role in AMORA

CUPTI is the programmable profiling and tracing layer for CUDA applications. It
is the preferred backend for AMORA when a probe needs machine-readable metrics,
kernel/range attribution, PC sampling, or source/SASS-level measurements.

CUPTI exposes these major API families:

| API | Header | AMORA Use |
|---|---|---|
| Activity | `cupti_activity.h` | CUDA API, kernel, memcpy, graph, NVTX, memory, and topology traces. |
| Callback | `cupti_callbacks.h` | Runtime/driver interception, launch correlation, module-load capture. |
| Host Profiling | `cupti_profiler_host.h` | Metric enumeration, config-image creation, metric evaluation. |
| Range Profiling | `cupti_range_profiler.h` | Exact metric collection over kernels or user ranges. |
| PM Sampling | `cupti_pmsampling.h` | Periodic hardware performance-monitor samples. |
| PC Sampling | `cupti_pcsampling.h` | Warp PC and scheduler stall-reason samples. |
| SASS Metrics | `cupti_sass_metrics.h` | SASS instruction-level metrics through SASS patching. |
| Checkpoint | `cupti_checkpoint.h` | Save/restore state for user replay workflows. |

## Profiling Model

CUPTI distinguishes tracing from profiling:

- Tracing records when CUDA activities happen and how they correlate.
- Profiling collects GPU performance metrics for kernels or ranges, often using
  replay when a metric set cannot be captured in one pass.

For AMORA, use Activity and Callback APIs to understand what ran, then use Host
Profiling plus Range Profiling or PM Sampling to collect hardware metrics.

## Version and Compatibility Notes

- CUPTI follows CUDA toolkit and driver compatibility rules. A driver/toolkit
  mismatch can surface as `CUPTI_ERROR_NOT_INITIALIZED`.
- CUDA 13.0 drops the legacy Event and Metric APIs. Use Range Profiling and Host
  Profiling instead.
- CUDA 13.0 deprecates the older Profiling API from `cupti_profiler_target.h`
  and old PerfWorks host APIs. Use Range Profiling for new work.
- CUDA 13.0 drops PC Sampling Activity APIs and source/SASS metrics from the
  Activity API. Use PC Sampling API and SASS Metrics API instead.
- CUDA 13.2 adds user-defined activity records and single-pass PM sampling metric
  set queries.
- CUDA 13.3 makes multiple subscribers and user-defined activity records
  production features.

## Initialization Notes

For tracing on CUDA 13.3+ and driver r610+, prefer `cuptiSubscribe_v2` and V2
Activity APIs. The V2 model allows multiple activity subscribers when the first
subscriber enables `allowMultipleSubscribers`.

For older CUDA versions, `cuptiSubscribe` supports only one subscriber. A second
tool may receive `CUPTI_ERROR_MULTIPLE_SUBSCRIBERS_NOT_SUPPORTED`. AMORA should
detect and report this clearly instead of continuing with partial data.

## AMORA Implementation Guidance

- Query support first: CUPTI feature and metric availability vary by GPU, driver,
  CUDA version, MIG/vGPU/MPS mode, and profiling permissions.
- Keep metric selection dynamic. Do not assume an NCU metric name is available on
  every architecture.
- Store raw counter-data images or decoded metric values with enough metadata to
  identify CUDA version, driver version, chip name, metric names, replay mode,
  range names, and collection scope.
- Treat device-level metrics such as NVLINK, C2C, and PCIe separately from
  context-level metrics unless the current CUPTI release explicitly supports the
  mixed configuration.
