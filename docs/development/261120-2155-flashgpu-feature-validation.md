# FlashGPU-Sim feature validation

This adapter executes exact GoogleTest and standalone binaries through native CUDA, PTX, strict
SASS functional execution, or SASS detailed timing. It also builds independent
NVIDIA semantic probes. Source, binary, configuration, command, exit status,
assertions and skipped cases remain visible in caller-selected evidence.

## Layout and ownership

The layout follows [baseline probe development](nvidia-baseline-probes.md) and
Amora's existing backend/probe split:

- `amora/probes/nvidia/feature_validation/` owns CUDA kernels, the host TMEM
  capacity probe, declared semantic contracts and NVIDIA reference extraction.
- `amora/backends/flashgpu_sim/` owns build caching, runtime selection, isolated
  configuration directories, bounded subprocess execution and result accounting.
- `tests/backends/test_flashgpu_sim.py` checks receipt immutability, errors,
  process-group timeout cleanup, environment isolation and result accounting.
- The invoking project owns plans, trackers, reference snapshots and results.
  Always supply `--evidence-root`; there is no per-project database in Amora.

The semantic tests use GoogleTest assertions and XML. They live outside the
36-probe baseline parameter-fitting registry: their result is a semantic verdict,
not an identified latency or throughput parameter. Existing NVIDIA capability,
profiling and four-layer measurement contracts retain their scope.

## Commands

Run from the Amora checkout (or install it first). Set `CUDA_INSTALL_PATH` to
the desired CUDA toolkit.

```bash
python -m amora flashgpu_sim list
python -m amora flashgpu_sim build \
  --probe dsm --sim-root ../FlashGPU-Sim \
  --cuda-root "$CUDA_INSTALL_PATH" \
  --build-root ../gpu-perf/build/flashgpu \
  --evidence-root ../gpu-perf/local_artifacts/flashgpu-validation
```

The build command prints the cached binary's path and hash. Substitute that path
for `<binary>` below. The same binary can run on a compatible GPU or on the
simulator.

```bash
python -m amora flashgpu_sim run \
  --binary <binary> --sim-root ../FlashGPU-Sim \
  --cuda-root "$CUDA_INSTALL_PATH" \
  --evidence-root ../gpu-perf/local_artifacts/flashgpu-validation \
  --label dsm-native --preset SM90_H100 --mode native

python -m amora flashgpu_sim run \
  --binary <binary> --sim-root ../FlashGPU-Sim \
  --cuda-root "$CUDA_INSTALL_PATH" \
  --evidence-root ../gpu-perf/local_artifacts/flashgpu-validation \
  --label dsm-ptx --preset SM90_H200 --mode ptx
```

`--mode sass` and `--mode sass-timing` decode the exact executable using the
specified toolkit. Select a compatible preset. Unsupported instructions retain
the simulator's explicit failure; there is no PTX fallback.

Use `--gtest-filter` to isolate a case, `--timeout` to bound execution,
`--gpu` and `--omp-threads` for device/CPU selection, and repeated
`--env NAME=VALUE` for benchmark controls. Use distinct run labels: existing
logs, XML and receipts are never overwritten.

```bash
python -m amora flashgpu_sim run \
  --binary <binary> --sim-root ../FlashGPU-Sim \
  --cuda-root "$CUDA_INSTALL_PATH" \
  --evidence-root ../gpu-perf/local_artifacts/flashgpu-validation \
  --label tcgen05-raw --preset SM100_B200 --mode ptx \
  --config-override gpgpu_ptx_reorder_sass_guided=0 \
  --config-override gpgpu_ptx_reorder=0
```

Overrides are recorded in the receipt with the effective configuration hash.
They describe a different execution contract and must be reported alongside
default-preset failures.

For standalone stock calibration programs, repeat `--arg=VALUE` to pass their
arguments without GoogleTest flags:

```bash
python -m amora flashgpu_sim run \
  --binary <stock-tma-completion-binary> --sim-root ../FlashGPU-Sim \
  --cuda-root "$CUDA_INSTALL_PATH" \
  --evidence-root ../gpu-perf/local_artifacts/stock-flashgpu-h100 \
  --label tma-a16-new --preset SM90_H100 --mode ptx \
  --arg=--blocks=1 --arg=--iters=32 --arg=--warmup=8 \
  --arg=--tiles=8 --arg=--repeat=1 --arg=--case=a16 \
  --arg=--issue=serial --arg=--hot
```

Standalone receipts retain the exact program arguments and configuration.
Their successful execution does not substitute for numerical assertions.

Build `tmem-capacity` to observe the production TMEM manager. This host program
prints allocation and readback observations; exit zero means execution completed,
not that the observed capacity conforms to NVIDIA's ISA.

```bash
python -m amora flashgpu_sim command \
  --label tmem-observation --cwd . \
  --evidence-root ../gpu-perf/local_artifacts/flashgpu-validation -- <binary>

python -m amora flashgpu_sim summarize \
  --evidence-root ../gpu-perf/local_artifacts/flashgpu-validation \
  --output ../gpu-perf/local_artifacts/flashgpu-validation/results.json
```

The summary retains failed, skipped, crashed, timed-out and missing-result runs.
It can also read the initial investigation's legacy `commands.jsonl` receipts.

## Prerequisites and evidence limits

CUDA builds require the toolkit and FlashGPU-Sim's GoogleTest checkout/library.
Build the simulator and its selected test suites first using its documented
`setup_environment`. The host TMEM probe requires G++ and the subject source.
The build adapter keys artifacts by source/support hashes and compiler identity.

NVIDIA reference extraction is optional:

```bash
python -m pip install -e '.[flashgpu-validation]'
python -m amora flashgpu_sim extract-spec --evidence-root <new-evidence-directory>
```

A semantic pass covers asserted shapes, types, memory effects and synchronization.
A timing pass confirms timing execution; silicon accuracy needs matching hardware
measurements, stable conditions and a specified metric. The tcgen05 uniform-input
smoke does not qualify stride/swizzle decoding, exceptional floating values,
two-CTA operation or narrow-precision variants.

## Revision History

- 2026-11-20 21:55 -0800: Added guide for the backend/probe layout, explicit evidence ownership, CLI and validation boundaries.
- 2026-11-20 23:11 -0800: Updated invoking-project evidence examples to its Git-ignored local_artifacts directory.
- 2026-11-20 23:15 -0800: Added standalone stock timing-program execution for the local-H100 comparison; backend/CLI checks now pass 16 tests.
