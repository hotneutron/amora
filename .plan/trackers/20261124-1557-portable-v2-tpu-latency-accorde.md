# Portable-core V2 TPU latency and Accorde training tracker

Evidence root: `out/portable_v2_tpu_models/20261124-1557/`. Source corpus:
[verified V2 correctness and IR](20261123-1650-portable-core-v2-cuda-tpu-backends.md).
Original Accorde execution baseline: `bf465099` in the sibling repository.
Source and review documentation are maintained separately from local run
artifacts. No paid allocation or device reset was required.

## Request

```text
complete the V2 corpus with latency labels.
also work on the TPU-specific PPP path to accept v5*/v6*/v7* and use the completed V2 corpus to train.
then work on the traning of all the viable paths to different TPU models.
```

Access update, verbatim:

```text
tpu machine connection is restored.
```

## Gate tracker

| Gate | Pre — freeze | Control | Results |
|---|---|---|---|
| L0 — case admission | 1,141 frozen TPU7x V2 cases; 1,130 native correctness passes and eleven unsupported cases. Reuse the admitted inputs, output-only realization and runtime. | Verify the archived correctness chain, CPU input/StableHLO hashes, target and ownership receipts. Do not invent labels for denied rows. | Prepared all 1,130 features and recorded eleven explicit `not_admissible` states. |
| L1 — latency label | Synchronized host dispatch to output readiness on one TPU7x TensorCore. 20 warmups, three repeats × 30 singleton samples, 5% within/across-repeat relative MAD and 1 µs minimum median. No compile, transfer or profiler in the measured interval. | Exact StableHLO; exact optimized HLO body after removing only file/function/location/stack-frame tables and normalizing numeric `stack_frame_id`. Retain raw and normalized hashes. Recheck correctness before/after; independently recompute statistics from raw samples. | Initial SSH authentication failed then was restored. The raw HLO hash check failed the first two-case pilot; a native diff confirmed only the stated stack metadata changed. The r2 pilot independently admitted two labels (207,895.5 and 212,843.5 ns). First pass complete and independently verified: 1,106 labelled, 24 unstable and 11 not admissible. |
| L2 — recovery | At most one additional identical attempt for an unstable admitted case, after the first pass is verified. | Keep accepted labels and their hashes unchanged; continue to reject unstable or non-identical executions. | **Complete.** All 24 retries passed; the 1,106 accepted rows were preserved exactly. Final TPU7x ledger: **1,130 measured labels and 11 inadmissible cases**, no missing eligible labels. |
| P0 — TPU PPP features | Recognize v5e/v5p, v6e, TPU7x by exact known aliases. Vendor-published chip peaks are typed as envelope features, not per-core rates or clocks. | Reject ambiguous family strings and uncalibrated cycle costing. Even with a matching calibration-shaped control, the existing v5p FP16 example with deferred work remains rejected. | Generation resolver, static workload join, public chip envelope and strict calibrated PPP admission implemented in Accorde. No full native XLA→PPP cycle lowering or measured v5/v6 resource calibration is claimed. |
| M0 — pilot training | Separate earlier measured GEMM fusion and non-GEMM whole-module domains; use pre-existing fit/holdout shapes, not V2 rows. | Fit only train rows and test only disjoint groups; no checkpoint or feature reuse across label domains. | Trained ridge, Extra Trees, Accorde MLP and chip-envelope residual on each pilot. See scores below. |
| M1 — full V2 training | Require a fully labelled, compatible output-only cohort. Separate target/runtime digests; one generation and label domain per fit. | Freeze canonical program groups independent of labels: 904 train / 226 test TPU7x rows. Reject incomplete labels; exclude identity, target labels, debug/HLO hashes and observed counters from features. | **Complete for TPU7x.** All 1,130 labels used; 904 train / 226 test cases across 305 / 76 disjoint canonical program groups. Six routes evaluated; model reload reproduced every held-out prediction. |
| M2 — other TPU generations | Verified target/runtime, separate correctness placement receipts, measurements and split on each generation. | No reuse of TPU7x labels or calibrated coefficients as measured v5/v6 ground truth. | v5e, v5p and v6e hardware and labels not supplied. No measured model trained for those targets. |

## Held-out pilot results (not V2 model qualification)

| Label domain | Fit / held out | Train median | Static ridge | Extra Trees | Accorde MLP | Chip-envelope residual |
|---|---:|---:|---:|---:|---:|---:|
| TPU7x BF16 GEMM fusion | 8 / 4 | 20.55% | 4.79% | 6.83% | 16.83% | 13.97% |
| TPU7x non-GEMM whole module | 30 / 18 | 29.01% | 6.78% | 8.71% | 8.01% | 15.85% |

Values are held-out MAPE. The simplest learned route wins these tiny pilots;
this is not enough to select a production model. The residual features use
published chip-rate envelopes, not calibrated per-TensorCore PPP cycles.

## Full V2 TPU7x held-out results

The label is synchronized host dispatch through output readiness on the pinned
TPU7x runtime, not an isolated kernel, TPU-only cycle count, or throughput.

| Route | MAPE | MAE (µs) | p95 APE | Worst APE |
|---|---:|---:|---:|---:|
| Train median | 16.24% | 26.28 | 36.39% | 41.55% |
| Static ridge | 6.86% | 11.46 | 14.10% | 25.35% |
| Static Extra Trees | 4.16% | 6.67 | 15.26% | 26.07% |
| StableHLO ridge | 6.42% | 10.69 | 15.01% | 21.46% |
| Accorde MLP | 7.57% | 11.96 | 28.35% | 62.30% |
| Chip-envelope residual | 45.87% | 77.46 | 93.77% | 111.10% |

Static Extra Trees leads mean error, but the held-out dual-residual-weave
family averages 25.99% APE across four cases. Static ridge has the lowest p95
APE and StableHLO ridge the lowest worst APE. These are single-target grouped
holdout results, not evidence of transfer or approval for deployment. The
chip-envelope residual is not competitive on this endpoint.

## Handoff and receipts

- `tools/portable_core_v2/collect_tpu_latency.py`: plan, timed shard, assembly
  and missing-only recovery. The remote campaign reuses the previously pinned
  output-only executable source; no accepted label may be replaced by retries.
- Accorde paths: `src/backend/common/ppp/tpu_generations.py`,
  `tools/ppp_profiling/build_tpu_latency_dataset.py`, and
  `tools/ppp_profiling/train_tpu_latency.py` in the Accorde checkout.
- `latency-plan-r2.json` SHA-256:
  `0e438a60cb8c8407425c88117a33336d21ebb80667b30e242bd4d1f162dd4d56`.
  `r2-pilot-shard.json` SHA-256:
  `d60b00d24efb02cb3885179ab62f0cf500f4460b3c50850ad1a4fcf80a0bbd13`.
- `gemm-fusion-fit/` and `non-gemm-module-fit/` retain the separate pilot fits.
  `v2-features.json` remains the pre-measurement feature snapshot.
- `recovered-v2-latency.json` (all accepted labels, including 24 retries)
  SHA-256: `808d51ff6e8f9bf15fafff7d420afa9eecd470db8eefa0328e831c7151324296`.
  `v2-tpu7x-labelled-dataset.json` (features and frozen split) SHA-256:
  `99e65d971d9c4033c6fbb975f431b685fe2e0676ec712124de5b166a16ebaa2b`.
- `v2-tpu7x-fit/evaluation.json` SHA-256:
  `aa32d5ae174fdae6ec813b0f2430b8d6f334f7f6183826562a5dbc8cbc9e8f96`.
  `v2-fit-reload-verification.json` confirms all 226 held-out predictions
  per route reproduced after reload on CPU. The fitted artifacts and raw
  native sample archives remain local.

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-24 | Froze the latency protocol, generation binding and held-out model paths. Recorded access recovery, HLO identity pilot, and exploratory measured model comparisons. |
| r2 | 2026-11-24 | Verified the sealed first-pass archive and all 1,106 accepted labels against raw samples. Started the bounded retry for just the 24 unstable rows; held-out groups and accepted labels are unchanged. |
| r3 | 2026-11-24 | Verified all 24 recovery labels and reproduced all six full-corpus held-out routes from saved artifacts. Only the TPU7x host-latency endpoint is measured; older-generation and calibrated cycle models remain gated. |
| r4 | 2026-11-25 | Prepared the Amora source and documentation for review; run artifacts and fitted models remain local. |
