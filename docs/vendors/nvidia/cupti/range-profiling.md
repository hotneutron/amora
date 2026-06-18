# CUPTI Range Profiling API

## Source

- CUPTI Range Profiling usage: https://docs.nvidia.com/cupti/main/main.html#range-profiling-api
- CUPTI Range Profiling API module: https://docs.nvidia.com/cupti/api/group__CUPTI__RANGE__PROFILER__API.html

## Purpose

Range Profiling is the main CUPTI path for exact metric collection over kernels
or user-defined ranges. It replaces the older profiling/event/metric workflow
for new AMORA work.

It is supported on Turing and newer GPUs, compute capability 7.5 and above.

## Concepts

Range modes:

- Auto range: each kernel launch is one range. CUPTI synchronizes at kernel
  boundaries and can use kernel replay.
- User range: the tool explicitly uses push/pop APIs. A range can include
  multiple kernels and can be nested.

Replay modes:

- Kernel replay: CUPTI replays kernels. Supported with auto range.
- User replay: AMORA saves/restores workload state and replays the range.
- Application replay: AMORA relaunches the application until all passes are
  collected.

Replay requires deterministic workload behavior. Non-deterministic kernels,
host-device races, random data generation, and external I/O can invalidate
multi-pass metric collection.

## Target-Side Flow

1. Use Host Profiling to enumerate metrics and build a config image.
2. `cuptiRangeProfilerEnable` for the CUDA context.
3. `cuptiRangeProfilerGetCounterDataSize`.
4. `cuptiRangeProfilerCounterDataImageInitialize`.
5. `cuptiRangeProfilerSetConfig`.
6. `cuptiRangeProfilerStart`.
7. For user ranges, call `cuptiRangeProfilerPushRange` and
   `cuptiRangeProfilerPopRange`.
8. Launch workload.
9. `cuptiRangeProfilerStop`.
10. Repeat if not all passes were submitted.
11. `cuptiRangeProfilerDecodeData`.
12. Use Host Profiling evaluation APIs.
13. `cuptiRangeProfilerDisable`.

## AMORA Use

Use Range Profiling when a probe needs stable, per-kernel or per-range metric
values:

- arithmetic and tensor throughput probes
- scheduler and stall probes
- shared/L1/L2/DRAM traffic probes
- occupancy and achieved active-warp probes
- synchronization and barrier microbenchmarks

For the probing suite, prefer small deterministic kernels and narrow metric
sets. This reduces replay count and makes the inferred simulator parameters
easier to attribute.

## Caveats

- Auto range may serialize kernel launches and introduce synchronization.
- Multi-pass collection can perturb caches, residency, and concurrent workloads.
- User ranges require balanced push/pop calls.
- Nested ranges increase replay cost.
- Device-level metrics and context-level metrics may have collection-scope
  restrictions depending on CUPTI version.
