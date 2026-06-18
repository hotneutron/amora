# NVBit Instrumentation Model

## Source

- Official NVBit repository: https://github.com/NVlabs/NVBit
- NVBit MICRO 2019 paper page: https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus

## Model

NVBit instruments GPU binaries dynamically at SASS level. When a GPU function is
loaded, an NVBit tool can inspect its SASS instructions and insert calls to tool
device functions before or after selected instructions.

The MICRO 2019 abstract describes the core implementation ideas:

- dynamic recompilation at SASS level
- register-requirement analysis
- ABI-compliant instrumented code
- tool code written in CUDA/C/C++
- selective application to precompiled binaries and libraries
- basic-block instrumentation
- multiple function injections at the same location
- inspection of ISA-visible state
- dynamic selection of instrumented or uninstrumented code
- source-code correlation

## Host and Device Split

An NVBit tool usually has:

- host-side instrumentation logic that receives callbacks and modifies loaded
  functions
- device-side injected functions that run from instrumented GPU code
- communication buffers or channels for returning data to the CPU

For AMORA, keep the host side responsible for selecting kernels and SASS
locations, and keep injected device functions minimal. Heavy formatting or
aggregation should happen off the hot path.

## What to Instrument

Useful AMORA instrumentation points:

- every instruction for exact dynamic instruction count
- one point per basic block for lower-overhead control-flow counts
- memory instructions for address, width, predicate, and active-lane traces
- selected arithmetic instructions for dependency-chain verification
- register reads or writes for bank-mapping experiments
- branch instructions for divergence and control-flow probes

## Output Semantics

NVBit output is instrumentation-defined. AMORA should make every NVBit tool emit
structured records with:

- probe ID
- kernel name
- launch index
- function name
- static SASS PC or instruction index
- opcode
- predicate state or active mask when relevant
- thread/warp/block identity when relevant
- memory address and width when relevant
- collection limits and dropped-record counts

## Overhead Strategy

Prefer progressive instrumentation:

1. Basic-block counts to identify hot code.
2. Opcode histogram to classify instruction mix.
3. Instruction-level counting for selected kernels.
4. Memory/register traces only for narrow windows.

This keeps AMORA probes usable on real workloads while still allowing precise
microbenchmark instrumentation when needed.
