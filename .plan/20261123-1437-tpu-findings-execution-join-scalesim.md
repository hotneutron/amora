# TPU7x execution attribution, module spans, and SCALE-Sim pilot

2026-11-23. Follow-up to [timestamp normalization](20261123-1319-tpu-findings-timestamp-normalization.md). One TPU7x instance, one logical TensorCore per call, JAX/jaxlib 0.12.0.1 and libtpu 0.0.50.1. Both native replays used the same verified runtime lock; the second used the separately recreated environment.

## 1. Execution ID join: passed for the tested stack

1. Use host-local containment of the user marker and the PJRT launch linkage.
   Match `_p` to `_c` on the `PJRT_LoadedExecutable_Execute` event family and
   require exactly one nested `DoEnqueueProgram`. Do not use host/device
   temporal containment or equate the high-level PJRT `run_id` with libtpu's ID.
2. Match `DoEnqueueProgram`, `CompleteCallbacks`, and the TPU:0 `XLA Modules`
   row by `(run_id, queue_id, device_ordinal, core_type)` within one capture.
   On the device side, the ordinal is obtained from the selected TPU track.
3. Reject missing, duplicate, ambiguous and out-of-scope rows. These private
   trace fields are an observed contract for this runtime/capture mode, not
   a public cross-version ABI.

All **36/36** saved A/B launches joined. The new independent replays joined
**272/272** marked calls, including 32 repeated launches of the identical
executable. The checker rejected 21 idle markers across all captures, four
multi-enqueue markers, and six mutated-key cases per capture; 12 unmarked or
ambiguous module executions were excluded. Both native replays passed the
existing numerical oracle before and after capture, then released the TPU.

## 2. Device-local span: admitted as an experiment metric only

The device `XLA Modules` duration and producer `device_duration_ps` agreed
within **0.301 ns** across 272 joined modules. In 217 calls, an independent
pair of event types on the same device clock (`sequencer_overlay` and
`continuation_queue_completed_run_id_low`) also bracketed the module; that
sequencer envelope was **157.3–163.4 ns** wider than the module. The maximum
per-shape difference in module median across replays was **2.85%**.

The artifact `profiled-module-spans.jsonl` exposes
`profiled_xla_module_span_ns` on those **217** corroborated rows only, with
trace/run keys and target/runtime provenance. Other joined rows remain raw
observations. These are profiler-reported single-TensorCore spans, not an
independent measurement of physical clock frequency. The join does not repair
cross-plane timestamps or justify host/device subtraction, multi-core
elapsed time, or generic `device_duration_ns`. The production adapter is unchanged.

## 3. Held-out SCALE-Sim pilot: operation-level empirical fit

- Pinned SCALE-Sim `87fd03d4ab3ac800119fc88904440c6da420226b`. Ran all 12
  frozen BF16-to-FP32 GEMMs in both one-array weight-stationary proxies
  (256×256 primary; 128×128 sensitivity). Used `Total Cycles` excluding initial
  prefetch. Neither proxy uses a legacy TPU cycles-to-time coefficient.
  These runs inherit the example's abstract buffer and bandwidth settings;
  they do not model the actual dual-MXU TensorCore or its physical memory system.
- Used the unique **GEMM fusion HLO event**, not host time or the enclosing
  module. All 272 calls contained the expected output fusion plus separate
  `copy-start` / `copy-done` HLO events. The StableHLO contains one dot and
  the optimized fusion lowers it to a convolution; native operand movement
  and implementation-specific work *inside* the fusion are not isolated.
  This is an empirical predictor of the observed fusion span, **not** a match
  against isolated MXU cycles, an HBM timing calibration, or an ISA model.
- Fit `fusion_us = a × predictor + b` on eight frozen aligned/tail shapes in
  replay 0. Scored four disjoint shapes from replay 1, using medians of ten
  calls per shape; the repeated-executable control block was excluded. Also
  fit the same two-parameter model using logical MACs instead of cycles.
  Maximum fusion median change across replays was 2.27%; relative MAD never
  exceeded 1.66% within the ten-call groups.

| Predictor | Fit MAPE (8) | Held-out MAPE (4) | Held-out MAE | Worst held-out relative error |
|---|---:|---:|---:|---:|
| SCALE-Sim 256×256 proxy | 8.31% | 5.58% | 0.183 µs | 13.12% |
| SCALE-Sim 128×128 sensitivity | 4.02% | 4.66% | 0.162 µs | 7.79% |
| Logical MACs baseline | 6.38% | 5.96% | 0.247 µs | 11.62% |

| Held-out M×N×K | Observed fusion | 256×256 prediction | 128×128 prediction |
|---|---:|---:|---:|
| 768×512×512 | 2.897 µs | 2.517 µs | 2.700 µs |
| 768×1024×768 | 4.485 µs | 4.433 µs | 4.491 µs |
| 769×509×517 | 3.172 µs | 2.997 µs | 2.925 µs |
| 773×1031×767 | 5.037 µs | 5.163 µs | 4.840 µs |

The primary proxy barely improves MAPE over MACs and has a worse worst-case
error. The smaller proxy scoring better does **not** identify hardware geometry
or warrant selecting it on this holdout. Keep this as a pilot; do not publish a
TPU7x simulator backend or transfer the coefficients to another generation.
Next validation should freeze one proxy and prospectively expand the tail,
aspect-ratio and K regimes on an independent target/runtime row.

## Evidence and reproducibility

Local artifacts: `out/tpu_profiler_experiments/20261123-1437-execution-simulation/`.
All three downloaded archives and 94 inventoried campaign files were verified;
local parsing of both native traces reproduced the on-host join receipt
byte-for-byte. `capture.py`, `join.py`, `simulate.py`, and `compare.py` retain
campaign, matching, simulation and fit logic. The pinned simulator required
its documented default layout section and NumPy 2.3.5 (NumPy 2.5.5 rejected a
one-element array conversion); the simulator source was not patched. Its
local environment did not change the TPU runtime lock.

| Artifact | SHA-256 |
|---|---|
| Frozen `cases.json` | `8b9709205ae704506538d59a3c0c977f4546abc8c106afa5f4a119b2cfdb6fba` |
| SCALE-Sim source archive | `ec41fe971b74d4fbbd1f91b9bf0185f0b30e32fd90f9451641828f6e2f6c1136` |
| `local-replay-joins.json` | `b73e9f5bb603ef778ac0e71225d31ecae8d9761320a3697e8f3619e26cec9194` |
| `profiled-module-spans.jsonl` | `ba242d180d0471263f326a91cc1b5df3b359464ba85999ee6445db82cad19ad4` |
| `simulation/summary.json` | `302299b5fed8c169c6b9118374357f46cdd9d74f7c552c301a6ff182bb51920c` |
| `comparison.json` | `59da5fafe189dcc6cb63446ccc882f9387386d75c47376c5b2931f45cf57549b` |

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-23 | Validated trace-scoped execution IDs and the experimental device-local duration; completed the pinned GEMM proxy and held-out comparison without changing backend admission. |
