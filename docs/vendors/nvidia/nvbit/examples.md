# NVBit Example Tools

## Source

- NVBit tutorial docs: https://eunomia.dev/others/nvbit-tutorial/
- NVBit tutorial repository: https://github.com/eunomia-bpf/nvbit-tutorial
- Official NVBit repository: https://github.com/NVlabs/NVBit

## Tutorial Tool Pattern

The tutorial repository organizes examples under `tools/` and test applications
under `test-apps/`. A common quick-start flow is:

```bash
export PATH=/usr/local/cuda/bin:$PATH
cd tools && make && cd ..
cd test-apps && make && cd ..
LD_PRELOAD=./tools/instr_count/instr_count.so ./test-apps/vectoradd/vectoradd
```

The tutorial also documents using `CUDA_INJECTION64_PATH` as an alternative to
`LD_PRELOAD`.

## Useful Examples for AMORA

| Tool Pattern | AMORA Use |
|---|---|
| `instr_count` | Dynamic instruction count per kernel. |
| `instr_count_bb` | Lower-overhead basic-block execution counts. |
| `opcode_hist` | Dynamic opcode mix for pipeline and issue probes. |
| memory trace tools | Memory-reference stream, coalescing, stride, and cache-probe validation. |
| register value recording | Register-bank and operand-delivery probe support. |
| instruction replacement examples | Experimental perturbation or controlled SASS edits; use cautiously. |

## Environment Controls

The tutorial documents selective instrumentation controls such as:

```bash
KERNEL_BEGIN=0 KERNEL_END=1 LD_PRELOAD=./tools/instr_count/instr_count.so ./app
INSTR_END=100 LD_PRELOAD=./tools/instr_count/instr_count.so ./app
TOOL_VERBOSE=1 LD_PRELOAD=./tools/instr_count/instr_count.so ./app
```

AMORA should standardize similar controls for every NVBit-backed probe:

- kernel begin/end
- instruction begin/end
- maximum records
- output path
- binary or JSONL output mode
- verbose diagnostics

## Build Notes

The tutorial notes that recent CUDA 12.x toolchains can require compiling device
code with `nvcc -dc` and linking the final shared library with `g++` rather than
`nvcc`.

AMORA should keep NVBit tools in a separate build target from regular CUDA probe
binaries because NVBit tool shared libraries have different linking and runtime
injection requirements.

## Expected Overhead

The tutorial gives approximate overhead bands:

- basic-block instruction count: low single-digit multiples
- CUDA graph instruction count: moderate overhead
- instruction count and opcode histogram: tens to hundreds of times slower
- memory trace and register recording: hundreds to thousands of times slower

These are workload-dependent. AMORA result schemas should record whether data
came from a basic-block, instruction-level, memory-trace, or register-trace mode.
