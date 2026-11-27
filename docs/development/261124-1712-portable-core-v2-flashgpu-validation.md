# Portable_core v2 validation with CUDA and FlashGPU-Sim

## Purpose and ownership

This adapter materializes the source-pinned portable_core v2 mathematical cases
as standalone CUDA programs for native H100 and execution-driven simulator
comparison. It follows the existing workload/backend split:

- `benchmark_generators/portable_core/expanded/v2/` owns unchanged cases,
  schemas, deterministic input recipes and independent oracles.
- `amora/benchmarking/portable_v2/` owns corpus sealing, scheduling and
  coverage/timing accounting.
- `amora/backends/nvidia/portable_core_v2.py` and `.cu` own the separately
  named `cuda-explicit-node-boundaries-v1` realization.
- The existing `amora/backends/flashgpu_sim/` owns runtime/profile selection
  and bounded command receipts.
- The invoking project owns snapshots, inputs, binaries, results and reports.

This standalone realization is separate from the proposed eager Torch CUDA
connector. CPU release checks and V1 fingerprints retain their meaning.

## Corpus and semantics

The complete source-available suite/profile union contains 1,155 unique cases,
18 shapes and 30 operators. Equal case IDs/hashes retain every suite/profile
alias instead of being counted repeatedly. Exact H3 reference profiles stay
source blocked; development controls do not qualify unavailable reference data.

The corpus seals canonical program/case/upstream identities, physical and
logical input hashes, strides, offsets, port order and independent per-node
oracle bytes. Input staging preserves declared backing storage and views.

Each already-unrolled node executes once in topological order on one CUDA
stream. Node outputs are materialized with declared FP16/FP32/int32 storage.
FP32 working arithmetic uses disabled FMA contraction; FP16 temporary/store
points explicitly round to nearest even. Scatter is deterministic, and top-k
uses the lowest source index to break ties. The current CUDA top-k supports
k<=8; source cases use k=4. The identity-matrix half-scale fixture lowers to
multiply-by-one-half and does not establish dense GEMM or tensor-core coverage.

## Commands

Use a Python environment with NumPy; CPU release qualification additionally
requires Torch. From the Amora checkout:

```bash
V2_RUN_ROOT=../gpu-perf/local_artifacts/flashgpu-portable-v2-new
python -m amora flashgpu_sim portable-v2 prepare --root "$V2_RUN_ROOT"
python -m amora flashgpu_sim portable-v2 build --root "$V2_RUN_ROOT" \
  --cuda-root "$CUDA_INSTALL_PATH"

python -m amora flashgpu_sim portable-v2 run --root "$V2_RUN_ROOT" \
  --mode native --scope pilot --gpu 7 --samples 0
python -m amora flashgpu_sim portable-v2 run --root "$V2_RUN_ROOT" \
  --mode native --scope full --gpu 7 --repeats 3 --samples 30 --warmup 3

python -m amora flashgpu_sim portable-v2 run --root "$V2_RUN_ROOT" \
  --mode ptx --scope full --gpu 7 --samples 1 --warmup 0 \
  --workers 16 --omp-threads 8 --chunk-size 16 --timeout 1800
python -m amora flashgpu_sim portable-v2 report --root "$V2_RUN_ROOT"
```

Physical GPU 7 becomes logical CUDA device 0 through `CUDA_VISIBLE_DEVICES=7`.
Native placement is checked against the device UUID. Simulation uses the stock
H100 preset and the same binary and input/oracle banks; it does not execute
on the physical GPU.

The scheduler requires accepted native correctness for each simulator case.
Crashed batches never silently hide later rows: uncompleted cases get singleton
retries. `--selection-indices <JSON-list>` and `--run-tag <label>` allow explicit
checkpoint continuation. Retained completed observations must keep their
original source, executable, input and runtime provenance.

## Measurement and limitations

Inputs, allocation and output comparison stay outside the measured program body.
Native correctness checks cover every node boundary and unchanged input storage
before and after timing. Failed correctness does not produce timing rows.

Timing uses a CUDA stream program span, including launch-submission gaps.
It is not a single-kernel duration or an SM-cycle counter. The simulator's
event span comes from its modeled clock and execution; keep it labeled as
simulation. Hardware clocks remain free-running while stock simulator clocks
remain unchanged.

The comparison admits native timing only with three independent process runs,
at least 30 samples per run, median span >=10 microseconds and relative MAD
<=5% within and across repeat medians. Short or unstable cases remain
inconclusive. A passed numerical row does not imply qualified performance,
complete ISA support, fusion, or dense Tensor Core/TMA/WGMMA coverage.

## Comparison output

The report includes native raw samples, per-repeat medians, all simulator
observations and the selected source validation paths. Completed observations
outrank interrupted retries; date and path order make selection deterministic.
Functional-mode samples never become timing pairs. Shape, operator, family,
node-count, state-edge and unroll summaries retain the declared native timing
quality gate. Operator groups count whole programs containing that operator
and overlap; their program spans do not isolate an individual operator's cost.

## Integrity checks

The unit suite checks corpus/file/bank tampering, physical stride/offset
preservation, wrong binaries, missing node-port checks, duplicated or out-of-scope
results, timing leakage after failed correctness, completed-versus-interrupted
observation precedence and CLI compatibility.

```bash
python -m pytest tests/benchmarks/test_portable_cuda_v2.py \
  tests/benchmarks/test_portable_core_v2.py \
  tests/backends/test_flashgpu_sim.py tests/test_cli.py
```

## Revision History

- 2026-11-24 17:12 -0800: Documented the standalone v2 CUDA realization, GPU 7 selection, sealed corpus, runtime/measurement contracts and public CLI.

- 2026-11-24 19:35 -0800: Documented deterministic completed-observation precedence, raw observation links and grouped span summaries; 51 regression tests passed against the completed source-available validation campaign.

- 2026-11-27T15:58:55-08:00: Integrated the upstream Torch CUDA qualification adapter alongside the standalone program realization. `amora/backends/nvidia/portable_cuda_v2.py` owns program builds/runs; `portable_core_v2.py` retains the upstream device Target and public program exports. Existing CUDA workload source bytes are preserved.
