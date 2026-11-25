# TPU7x non-GEMM timing pilot

2026-11-23. Follow-up to the [execution join and GEMM pilot](20261123-1437-tpu-findings-execution-join-scalesim.md). This study used standalone JAX kernels, not Portable-core V2 accelerator connectors.

**Result:** Add, ReLU, row sum, softmax, transpose, and indexed gather all
produced attributable TPU7x module spans. The simpler logical-bytes predictor
beat the padded-bytes proxy on five of six families. Gather remained poorly
predicted by either proxy. No production timing model or simulator backend is
admitted.

## Measurement contract

- Same single TensorCore and pinned JAX/jaxlib 0.12.0.1 / libtpu 0.0.50.1 stack
  as the GEMM pilot. Two locked environments ran eight frozen shapes per
  family, eight calls per shape. All 96 compiled cases passed the NumPy
  oracle before and after capture, and the TPU was released after both runs.
- The capture used explicit compute/sync tracing with counters disabled.
  The enqueue/completion execution-key join matched **768/768** marked calls.
  Sixty idle and 24 two-launch markers yielded no timing rows, and 72
  out-of-scope modules were excluded. Device-local sequencer markers
  corroborated 696 of the 768 joined spans. Per-shape medians using only those
  rows differed by at most 1.55% from the corresponding all-joined medians.
- The label is the device-local **whole XLA module span**. It is not isolated
  operator duration, physical HBM traffic, or host-to-device synchronized time.
  Transpose realized as a copy on this stack. Gather and softmax included
  additional operations in the executable. The highest per-shape replay
  median difference was 4.28%. One reduction group reached 5.33% relative MAD;
  this is a diagnostic pilot, not a production timing qualification.

## Held-out prediction

Fit a separate affine predictor per family on five replay-0 shapes and score
three disjoint replay-1 shapes. The predictors have the same parameter count.
Logical requested bytes count declared operand reads and outputs; gather counts
selected rows and indexes, not the full backing table. The second predictor
rounds matrix extents to an abstract 8×128 tile. This describes neither native
layout nor measured traffic. Softmax's temporary/intermediate work is not
explicitly priced by either proxy.

| Family | Logical bytes MAPE | Padded bytes MAPE |
|---|---:|---:|
| Add | 1.75% | 5.24% |
| ReLU | 0.99% | 3.55% |
| Row sum | 2.53% | 4.83% |
| Softmax | 5.58% | 10.71% |
| Transpose | 4.45% | 5.94% |
| Gather | 17.39% | 14.53% |
| Macro average | 5.45% | 7.47% |

Gather's worst held-out errors were 39.68% and 33.96%, respectively. Do not
select the padded model just because it improved gather on this small holdout.
A prospective gather study should vary locality, row reuse and table working
set at fixed logical work before fitting another model. Treat fused softmax
and layout-only copies as separate realization families.

## Simulator boundary and evidence

The upstream [StableHLO parser](https://github.com/scalesim-project/SCALE-Sim/tree/8050a0f964f6a02c48fba1b6a42bba2eca0b31e3/scalesim/stablehlo_parser)
parsed all 96 StableHLO files. The pinned branch includes older-generation
predictors for several constituent ops, lacks a v6e gather predictor file,
and has no TPU7x model inventory. Parser coverage does not establish lowering,
fusion, or latency coverage. No upstream predictor was executed, no pickles
were loaded, and no v4/v6e coefficients were reused. The table above reports
only the two separately fitted TPU7x proxies.

Verified metadata and score receipts are local-only in
`out/tpu_profiler_experiments/20261123-1515-non-gemm/`. The metadata archive
verified 326 local files; its full inventory also records 12 raw captures
retained on the remote host. The derived `non-gemm-comparison.json` SHA-256 is
`9c51473e35a1abaeb711b2304013a1a4e4e590a652b0d0e8e600cb43b364b6b1`.
No additional runs were needed to score the captured experiment.

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-23 | Recorded the existing six-family native campaign, replay and join controls, held-out proxy errors, and the upstream parser/model boundary. |
