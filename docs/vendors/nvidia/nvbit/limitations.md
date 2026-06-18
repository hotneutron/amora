# NVBit Limitations

## Source

- Official NVBit repository: https://github.com/NVlabs/NVBit
- NVBit tutorial docs: https://eunomia.dev/others/nvbit-tutorial/
- NVBit MICRO 2019 paper page: https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus

## Not a Hardware Counter API

NVBit observes and modifies dynamic SASS execution. It is not the primary API
for hardware performance counters. For PM counters, use Nsight Compute or CUPTI
Range Profiling, PM Sampling, PC Sampling, and SASS Metrics.

## Injection Conflicts

NVBit uses the same injection mechanism as common NVIDIA profiling tools. The
tutorial notes that it cannot be used together with `nvprof`, Nsight Systems, or
Nsight Compute in the usual workflow.

AMORA should run NVBit and CUPTI/NCU passes separately, then correlate results
offline.

## Platform Limits

Current public documentation lists Linux-only support, specific host
architectures, CUDA/driver requirements, and a supported compute-capability
range. These limits change by release.

AMORA should record:

- NVBit version
- CUDA toolkit version
- driver version
- GPU compute capability
- `nvdisasm` path and version when available
- injection method used

## Overhead

Instruction-level instrumentation can be very expensive. Memory traces and
register-value recording can slow a workload by hundreds or thousands of times.

Mitigations:

- instrument selected kernels only
- instrument selected instruction ranges only
- prefer basic-block instrumentation for broad counts
- cap trace size and record dropped entries
- keep injected device functions small

## Correctness Risk

Removing or replacing SASS instructions can break the target program. Even pure
injection can perturb timing, scheduling, memory pressure, and cache behavior.

For AMORA, use NVBit measurements to validate dynamic instruction facts. Avoid
using NVBit-instrumented timings as direct microarchitectural latency or
bandwidth estimates unless the probe is explicitly calibrated for instrumentation
overhead.

## Source Correlation Limits

Source/SASS correlation depends on available cubins, debug/source line
information, and toolchain compatibility. JIT-compiled code and stripped binaries
need extra handling.

For source/SASS attribution with lower instrumentation overhead, also evaluate
CUPTI PC Sampling and CUPTI SASS Metrics.
