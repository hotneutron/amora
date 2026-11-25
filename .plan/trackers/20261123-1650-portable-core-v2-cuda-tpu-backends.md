# Portable-core V2 CUDA and TPU execution tracker

Plan: [CUDA + TPU backends](../20261123-1650-plan-portable-core-v2-cuda-tpu-backends.md), r3. Baseline: `56749f4ed228ce1daece388195a882caf101083f`. Local evidence directory: `out/portable_v2_backends/`.

## Request

```text
execute the backend plan with a tracker with gating details (pre -> control -> results).
```

## Gate ledger

| Gate | Pre — prerequisites and frozen scope | Control — test that can fail the gate | Results — actual outcome and next gate |
|---|---|---|---|
| B0 — contract freeze | Committed V2 catalogue on `tpu`. Four smoke snapshots plus selected H3 development cases; 29 target-independent witness expectations frozen before native startup. | Verify manifest digests, independently hash the CPU input bytes, inspect without importing Torch/JAX, and rerun legacy snapshot/fingerprint checks. | **Pass.** All five frozen snapshots verify unchanged from the new wheel. The six v1 fingerprints and existing V2 CPU tests pass; no hashed CPU source changed. |
| B1 — CPU plumbing | No accelerator initialization in the controller. New `amora benchmarks inspect-v2` and `qualify-v2` only. | Foreign device, wrong manifest hash, duplicate or missing selection, invalid layout/index, misordered output ports and worker timeout are non-passes. Exercises run on CPU; native evidence requires the device worker. | **Pass.** 12 focused device-protocol and lowering controls passed, plus 24 inherited recorded cases. Separate CPU JAX replay matched all 29 expected witness decisions. The on-host campaign verifier rejected missing/mismatched rows by comparing every result against the complete frozen suite selection. |
| B2 — CUDA correctness | Eager Torch single-device adapter, explicit CUDA stream and placement, FP32 working arithmetic, no autocast/TF32. The initial local probe was blocked; the user subsequently supplied `h100sjc0`. | Require an NVIDIA fingerprint and device-local packed storage and output, independent oracle parity, and a storage-boundary witness. | **Pass on the eight H100s for the frozen cases below.** The initial local `device_unavailable` receipts remain diagnostic only. No CPU fallback. |
| B3 — TPU correctness | TPU7x with verified Python 3.12.3, JAX/jaxlib 0.12.0.1 and libtpu 0.0.50.1 lock. No active holder detected before any shard; only explicit `tpu:0`. No second-generation host was approved. | Verify every frozen case hash and all local-vs-remote positive input digests. Assert output/node placement, CPU oracle parity, distinct FP16/FP32 storage controls and expected unsupported rows. Recheck ownership during and after each worker. | **Pass for this TPU7x witness set.** 26 native passes; two physical layouts and one duplicate scatter correctly denied. All five pairs of FP16/FP32 controls passed. Verified 104 case IR files and 175 evidence files in total; all post-exit ownership scans showed no holders. Other TPU generations remain unqualified. |
| B4 — timing | Requires separate approval and accepted output-only correctness receipts. | No timing command, published timing rows, profiler collection, or simulator fit in this delivery. | **Not authorized; not run.** |
| B5 — packaging | Keep accelerator dependencies optional and authorize hardware CI separately. | Clean wheel with framework imports blocked, full CPU regression, unchanged snapshots, and per-case receipt ledger. | **Local pass.** Wheel imported both adapters and verified all five freezes without frameworks. Full regression: 252 passed, 17 skipped, plus the separate pinned JAX CPU test pass. Ledger has 58 rows: 29 TPU7x decisions and 29 CUDA `not_evaluated` rows. No hardware CI was enabled. |

## Frozen native matrix and evidence

| Snapshot | Pre — expected positive / negative | Results — observed |
|---|---:|---|
| `operation-gate-v2` smoke | 10 / 0 | 10 passed, including admitted/refused historical operation types. No H100 status is inherited. |
| `h3-semantics` smoke | 6 / 1 | Six passed; duplicate scatter rejected. |
| `h3-programs` smoke | 6 / 0 | Six passed, including fork/join and matched chain. |
| `mechanism-controls` smoke | 3 / 0 | Three diagnostics passed, including top-k tie/port checks. |
| `h3-access-controls` (selected development rows) | 1 / 2 | Unique scatter passed; physical stride and offset rows rejected. |

- Frozen `witnesses.json` SHA-256:
  `193339089b0c694a64d9d9ea3de3c1de441cb5d44467d7db30a7a40230f806d5`.
  `cpu-input-pins.json` holds the independent CPU input digests. Both files
  were compared byte-for-byte with the remote copies after retrieval.
- `native-verification.json` binds the on-host results to the staged source,
  expected rows and raw artifact hashes. TPU evidence archive SHA-256:
  `d857632b1ed24e9fd49bebb969556d6d2bd2c139da87c5cfd2dffbf133b5d9ae`.
  Target identity SHA-256:
  `d2b21928876a763a1c6a02d0305cb8d1e7b9d23cccb5a426531c8019aca3c172`.
- `coverage-ledger.json` records case, backend, target/runtime and realization
  identities; unsupported and not-evaluated cells stay explicit. Its SHA-256 is
  `460e683fadc53a3ec7dda4b9cf412b826ae67d966d63bed0d42ddf0abb7e01e2`.
  Installed wheel SHA-256:
  `2a3d2089e4fee37ea10628a7af44c1a8a94c438089e9fde85a20ecdc0161f014`.

## Full V2 artifact collection — TPU before CUDA

| Phase | Pre — inputs and prerequisites | Control — what must match | Results |
|---|---|---|---|
| TPU7x full correctness/IR | 1,141 distinct cases across the five source-available V2 snapshots. Frozen `full-pins.json` SHA-256 `f8816d695c1975e8933faf2558e442817a2672f0ed2b2294c33663842e6d61eb`: 1,130 expected passes, seven duplicate-index denials, four layout denials. Same adapter source bytes and verified runtime lock as B3. | Run at most 32 cases per worker. Verify case IDs, CPU input digests, TPU placement, boundary controls, and post-exit ownership. Preserve unqualified rows and validate archive, receipt and IR hashes after collection. | **Complete and verified.** All 38 shards finished; 1,130 passes and 11 expected denials, zero mismatches. All six archives and 5,057 inventory files passed hash checks, including 4,520 StableHLO/optimized HLO artifacts. Final ownership scan was empty. No timing or profiling. |
| CUDA full correctness | TPU campaign and artifact verification completed first. Authorized `h100sjc0` (eight H100 80GB HBM3, SM 90, MIG disabled, driver 595.71.05); existing Python 3.11.2, NumPy 2.3.5 and Torch 2.11.0+cu130. Same 1,141-case pins and payload hash as TPU. | Start with one shard on one GPU, then use idle GPUs in parallel with one visible UUID per worker. Check device identity and placement, input digests, FP16/FP32 controls, and ownership before and after each shard. Preserve accepted rows and rerun only unexecuted cases. | **Complete and verified.** 38 shards, 1,130 passes and 11 expected denials, zero mismatches. Two shards initially saw 1 MiB in use and were not run; all 29 cases in those shards passed after a fresh idle check, with no accepted cases rerun. All eight GPUs were idle at the end. No performance timing, compiler IR or native disassembly from the eager route. |

Campaign directory: `out/portable_v2_backends/20261123-1816-full/`. The six TPU
archives are under `archives/`; their extracts are under `evidence/`. Verified
summary `tpu-verification.json` SHA-256:
`60f5778d85e1ffce7967d90a721d6c299c5b2a0ed121c0dafce71a5d4ba680c0`.
Initial local CUDA admission receipt `cuda-admission/result.json` SHA-256:
`4903ff196dc419ff4756a9c6a99d1a3b2796b0a4d92e7b2cbfaa8a12cf6fc5fc`.

The H100 run's receipts, worker logs and final verification are in
`out/portable_v2_backends/20261123-2038-cuda-full/`. All 236 files across the
original and supplemental CUDA archives were verified. Final CUDA summary
`cuda-final-verification.json` SHA-256:
`416ec3bb5b326e640f2f11388eb7cee27a02576b7095aeb2b4a3263f6ddc5862`.
CUDA archive SHA-256:
`714a5b0dba47e62f3e496a766af5759ec82b99e5c912bcce9c06c17cac7b71d1`.
Supplemental archive SHA-256:
`8dc247e45955510363547dc4569b741fd046c63cf1dfe65371d159826e468509`.

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-23 | Opened the pre/control/results ledger for correctness-first execution. Recorded target and timing boundaries before any native V2 run. |
| r2 | 2026-11-23 | Recorded the isolated adapters, verified TPU7x gate, expected layout/index denials, CPU/wheel controls and the outstanding CUDA and other-generation hardware gates. No timing work or historical H3 source promotion. |
| r3 | 2026-11-23 | Started full source-available V2 correctness and artifact collection with frozen inputs, TPU first and CUDA deferred until TPU evidence verification. |
| r4 | 2026-11-23 | Completed and verified all TPU artifacts, then attempted CUDA admission. CUDA was blocked before native execution because no authorized CUDA target was available. |
| r5 | 2026-11-23 | Completed the full frozen CUDA matrix on the authorized eight-H100 host. Recovered only two previously unexecuted shards and verified both archives and all device releases. No timing or compiler/disassembly capture. |
