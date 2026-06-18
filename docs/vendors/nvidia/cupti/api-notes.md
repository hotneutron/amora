# CUPTI API Notes for AMORA

## Source

- CUPTI usage guide: https://docs.nvidia.com/cupti/main/main.html
- CUPTI API modules: https://docs.nvidia.com/cupti/api/modules.html
- CUPTI release notes: https://docs.nvidia.com/cupti/release-notes/release-notes.html

## Preferred APIs

Use these APIs for new AMORA work:

- Range Profiling API for exact metric collection.
- Host Profiling API for metric enumeration, configuration, and evaluation.
- PM Sampling API for time-series hardware PM samples.
- PC Sampling API for PC/stall attribution.
- SASS Metrics API for SASS instruction-level metrics.
- Activity and Callback APIs for traces, correlation, and module-load events.

Avoid these for new work:

- legacy CUPTI Event API
- legacy CUPTI Metric API
- deprecated CUPTI Profiling API as the primary collection path
- old PC Sampling and source/SASS metrics via Activity records

## Subscriber Rules

CUDA 13.3+ supports multiple activity subscribers with V2 APIs. Earlier versions
effectively require a single subscriber for Activity and Callback APIs.

AMORA should:

- call subscription APIs before enabling tracing features
- detect existing CUPTI sessions
- use V2 APIs when available
- fail clearly if another profiler blocks subscription
- avoid running with Nsight Compute, Nsight Systems, Compute Sanitizer, or NVBit
  injection unless the specific combination is known to work

## Correlation Rules

For a useful probe record, capture:

- CUDA API correlation IDs
- kernel name and demangled name
- context ID and device ID
- stream ID
- graph ID and graph node ID when present
- NVTX range or AMORA probe ID
- module/cubin information for source/SASS correlation

Activity records may be delivered out of order. Use timestamps and correlation
IDs rather than buffer order.

## Buffering Rules

CUPTI clients provide activity buffers through callbacks. Choose buffer sizes
large enough to prevent drops but small enough to avoid long delivery latency.
The CUPTI guide suggests typical activity buffers between 1 MB and 10 MB.

Flush explicitly at profiling boundaries:

- before process exit
- before changing user-defined activity field selections
- before detaching CUPTI
- after PC sampling ranges
- periodically during long PM sampling runs

## Architecture and Mode Limits

Feature support varies with:

- GPU architecture
- CUDA toolkit version
- driver version
- MIG, MPS, vGPU, WSL, MCDM, and confidential-compute mode
- profiling permissions
- device partitioning and green contexts

AMORA should include a capability-probe step before selecting probes and metrics.

## Result Metadata

Every CUPTI-backed AMORA result should record:

- CUDA toolkit version
- CUPTI API version
- driver version
- chip name
- device name
- profiling API used
- metric names and suffixes
- range mode and replay mode, if applicable
- sampling period and dropped-sample counts, if applicable
- counter-data decode status
- limitations or unsupported metrics reported by CUPTI
