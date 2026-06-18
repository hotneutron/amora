# CUPTI PC Sampling API

## Source

- CUPTI PC Sampling usage: https://docs.nvidia.com/cupti/main/main.html#cupti-pc-sampling-api
- CUPTI PC Sampling API module: https://docs.nvidia.com/cupti/api/group__CUPTI__PCSAMPLING__API.html
- CUPTI PC Sampling Utility API module: https://docs.nvidia.com/cupti/api/group__CUPTI__PCSAMPLING__UTILITY.html

## Purpose

PC Sampling periodically samples warp program counters and warp scheduler state.
It is sampling-based, not exact counter collection. It is useful for identifying
which instructions are associated with stall reasons.

PC Sampling is supported on Turing and newer GPUs, compute capability 7.5 and
above.

## What It Records

At a fixed cycle interval, the sampler in each SM selects an active warp and
records:

- program counter
- function and cubin correlation information
- selected stall-reason counters
- total and dropped sample counts

The sampler chooses a random active warp. The sampled warp may differ from the
warp selected by the scheduler in the same cycle.

## Configuration

Important attributes:

- collection mode: continuous or kernel serialized
- sampling period: `2^N` cycles for `N` between 5 and 31
- selected stall reasons
- scratch buffer size
- hardware buffer size
- start/stop control

Lower sampling periods increase sample frequency but can cause dropped samples.
Very high periods can produce too few samples for stable attribution.

## Source and SASS Correlation

PC records include fields such as cubin CRC, PC offset, and function name. A
tool can:

1. Extract cubins with `cuobjdump -xelf all`.
2. Disassemble cubins with `nvdisasm`.
3. Match cubins with `cuptiGetCubinCrc`.
4. Map SASS to CUDA source with `cuptiGetSassToSourceCorrelation`.

For JIT-compiled cubins, use module resource callbacks such as
`CUPTI_CBID_RESOURCE_MODULE_LOADED` or `CUPTI_CBID_RESOURCE_MODULE_PROFILED` to
capture the binary.

## AMORA Use

Use PC Sampling to attach observed stalls to instruction PCs in:

- arithmetic dependency-chain probes
- memory pointer-chase probes
- shared-memory bank-conflict probes
- tensor and async-copy overlap probes
- synchronization and barrier probes

PC Sampling is strongest as an explanation layer. It tells AMORA where sampled
stalls concentrate, while Range Profiling or PM Sampling provides the metric
values.

## Caveats

- PC Sampling does not support simultaneous sampling of multiple CUDA contexts
  on one GPU. Disable sampling on one context before configuring another.
- Sampling lacks exact time resolution.
- A low sampling period can overflow buffers and drop samples.
- AMORA should store dropped-sample counters with every result.
