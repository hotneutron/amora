# FlashGPU-Sim validation code integration

## User requests

```text
investigate ~/wk/FLashGPU-Sim, validate the claims on
GPU features: TMA, TMEM, DSM, tcgen05, mbarrier, mma, wgmma, ldmatrix/stmatrix, etc.
persist the validation plan in .plan/ and execute it with a tracker.
```

```text
naming convention for the markdowns: {YYMMDD}-{HHMM}-*.md
```

```text
re-org the code layout. or better, add the code to ~/wk/amora/
```

```text
read amora's readme and docs for the best code layout
```

## Layout decision and source inputs

Read [Amora README](../README.md), [repository rules](../RULES.md),
[probe development](../docs/development/nvidia-baseline-probes.md), and the
existing NVIDIA/GCoM runners and measurement contracts.

Use `amora/probes/nvidia/feature_validation/` for independent NVIDIA kernels,
semantic contracts and reference extraction. Use
`amora/backends/flashgpu_sim/` for simulator-specific build/run mechanics.
Wire `amora flashgpu_sim` into the existing CLI. Keep parameter-fitting baseline
probes separate because these GoogleTest checks validate semantics and preserve
assertions rather than estimate simulator parameters.

## Execution and evidence ownership

The investigation plan, tracker, reference excerpts and numerical results remain
in [gpu-perf's validation area](../../gpu-perf/.plan/).
Amora owns reusable code and packaging. FlashGPU-Sim tracked source is unchanged.
Existing Amora deletions and untracked traces/plans are preserved.

Kernel sources moved from gpu-perf's planning directory. The three initial
standalone scripts were consolidated into the backend and reference modules,
with explicit source/toolkit/evidence roots, bounded subprocess groups,
immutable run files, exact hashes, source/target metadata and skip/failure
accounting. CUDA sources are included in installed wheels.

## Validation

H100-specific follow-up request, preserved verbatim:

```text
I mean the report for "stock flashgpu-sim code " against the H100 we have here
```

Added standalone-program arguments to the existing run adapter so stock TMA and
cp.async calibration programs can execute through the same receipt/runtime
contract. No copied simulator implementation or parameter changes are needed.
The H100 comparison's raw evidence belongs to the invoking project's ignored
`local_artifacts/stock-flashgpu-h100/`.

- Targeted backend and existing CLI tests passed before final packaging checks.
- Built and executed the relocated TMEM probe; observations match the original.
- Built both relocated CUDA probes through the new adapter.
- Reproduced the DSM atomic mismatch through the adapter with SASS guidance disabled.
- Wheel contents and final test counts are recorded in the gpu-perf tracker.

## Revision History

- 2026-11-20 21:55 -0800: Recorded documentation inputs, ownership decision, migration and verification.
- 2026-11-20 23:11 -0800: Updated invoking-project evidence examples to its Git-ignored local_artifacts directory.
- 2026-11-20 23:15 -0800: Extended the adapter for stock standalone timing programs and verified the new behavior together with existing CLI checks (16 passing tests).
