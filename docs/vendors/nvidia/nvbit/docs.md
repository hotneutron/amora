# NVBit Documentation Notes

## Sources

- Official NVBit repository: https://github.com/NVlabs/NVBit
- NVBit MICRO 2019 paper page: https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus
- NVBit tutorial repository: https://github.com/eunomia-bpf/nvbit-tutorial
- NVBit tutorial docs: https://eunomia.dev/others/nvbit-tutorial/

## Role in AMORA

NVBit is a dynamic SASS instrumentation framework for NVIDIA GPUs. It lets tools
inspect and instrument compiled GPU binaries without recompiling the target
application.

For AMORA, NVBit should be treated as a dynamic-instruction and memory-reference
oracle, not as the primary hardware performance counter interface. Use CUPTI and
Nsight Compute for hardware metrics; use NVBit to validate instruction streams,
opcode mixes, memory-reference streams, register values, and control-flow facts.

## What NVBit Provides

NVBit tools can:

- inspect SASS instructions when a function is loaded
- inject device-function calls before or after selected SASS instructions
- count dynamic instructions
- build opcode histograms
- trace memory references
- record register values
- instrument basic blocks
- selectively instrument kernels or instruction ranges
- remove SASS instructions, although correctness is not guaranteed after removal

## Requirements from Current Public Docs

The current official README lists:

- SM compute capability `>= 3.5 && <= 12.1`
- host CPU: `x86_64` or `aarch64`
- OS: Linux
- GCC `>= 8.5.0`
- CUDA `>= 12.0`
- CUDA driver `<= 575.xx`
- `nvdisasm` available in `PATH`

These bounds move over time. AMORA should record the exact NVBit release and
target driver/toolkit versions in every NVBit-backed result.

## Launch and Injection

NVBit tools are shared libraries loaded into the target process. Common launch
forms:

```bash
LD_PRELOAD=./tools/instr_count/instr_count.so ./app
CUDA_INJECTION64_PATH=./tools/instr_count/instr_count.so ./app
```

NVBit uses the same general injection mechanism as NVIDIA profiling tools, so it
cannot normally be run together with `nvprof`, Nsight Systems, or Nsight Compute.

## AMORA Integration Pattern

1. Compile or select a microbenchmark kernel.
2. Run an NVBit tool to collect dynamic SASS facts.
3. Run a separate CUPTI/NCU pass for hardware metrics.
4. Join results by kernel name, launch configuration, problem size, and generated
   probe ID.
5. Cross-check:
   - NVBit instruction count vs CUPTI/NCU instruction metrics.
   - NVBit opcode histogram vs SASS/source metrics.
   - NVBit memory-reference stream vs L1/L2/DRAM transaction metrics.
   - NVBit register traces vs register-bank and dependency probe hypotheses.

## Caution

NVBit adds overhead by saving/restoring application state around injected calls.
Instruction-level tools can slow workloads by orders of magnitude. AMORA should
support selective instrumentation by kernel and instruction range.
