# TPU7x native qualification

Accepted on 2026-11-17. Instance: `gke-tpu-f20b10b7-2pbv`.
Runtime: Python 3.12.3, JAX/jaxlib 0.12.0.1, libtpu 0.0.50.1, NumPy 2.5.5.
An independent environment was recreated from the same 19 hash-locked wheels
and passed native compute, transfer, Pallas, collective and trace replay.

Runtime discovery reported eight addressable TensorCores on four chip
coordinates, matching the GKE node's four-chip `2x2x1` topology. This evidence
qualifies the recorded instance/runtime/recipes; other generations remain
catalogued without native qualification.

## Accepted evidence

| Evidence | Coverage | Accepted run |
|---|---|---|
| Portable-core correctness | All 184 unique core28 smoke cases; complete core20/core24 subsets; six semantic fingerprints verified | `native-restore-002-core28-000` through `005` |
| Compute and H2D throughput controls | 12 cases: topology, dispatch, five GEMM shapes through 2048³, two vector shapes, two H2D shapes and a bounded dense four-step matrix chain | `native-throughput-fixed-002` |
| D2H completion | Fresh device buffer, 1024² FP32 payload; includes device-local copy and host transfer | `native-d2h-fixed-002` |
| Pallas memory/DMA | Two HBM/VMEM streaming sizes and explicit DMA/semaphore tiling | `native-fixed-pallas-001` |
| Collective sweep | `psum`, `all_gather`, `ppermute`; 2/4/8 participants × 128/65536 elements: 18 cases | `native-restore-002-collectives` |
| Trace availability | Verified XPlane and JSON/Perfetto artifacts; TPU device track present | `native-restore-002-profile` |
| Independent native replay | Compute, H2D/D2H, Pallas streaming/DMA, all three collective primitives and trace capture | `native-replay-smoke-001`, `native-feature-replay-001`, `native-profile-replay-001` |

The [run index](README.md) links the 15 accepted immutable reports. All 18
source manifests, including three retained diagnostic campaigns, passed local
artifact verification after download. Qualified reports link to those manifests;
raw sources remain under the local execution artifact directory.
Resolve compiler files through launch-descriptor and manifest paths; historical
raw compiler metadata may contain program-local basenames.

## Timing interpretation and quality

All normalized timings use a synchronized **host dispatch-to-completion**
interval. Throughput controls use K=32 with 20 warmup batches, 30 measured
batches per repeat and three repeats. Every warmup/measured batch retains all
outputs until they are drained. The reported per-invocation value amortizes K;
it is not isolated device latency.

The unchanged relative-MAD gate is 5% within each repeat and across repeat
medians. Accepted throughput controls have a maximum within-repeat relative
MAD of 1.91%; accepted D2H is 0.82%; Pallas is 4.96%. Collective timing also
passed its frozen gate. D2H uses K=1, while Pallas uses K=1 and the collective
sweep uses K=16; compare their time domains and work units explicitly.

Requested bytes and logical FLOPs divided by these intervals are effective
program rates. Physical HBM traffic, MXU instruction counts/utilization,
isolated copy latency and device clock cycles are not derived from them.
AMORA did not set or observe device clocks for this baseline. Trace capture is
available, but device-only interval semantics remain unqualified.

## Corrected implementation and retained diagnostics

- TPU lowering rejected tiled HBM block specs for explicit DMA. The fix passes
  whole-array HBM refs with a trivial map and selects DMA tile regions inside
  the kernel; all three Pallas cases then passed unchanged numerical/noise gates.
- Reusing one JAX array for D2H could reuse its host cache. Each D2H invocation
  now requests a fresh device buffer with `may_alias=False`; the additional
  device-local copy is part of the declared completion interval.
- Batched timing now warms the same batch/retention/drain unit that it measures.
  D2H completion has a separate singleton protocol to avoid conflating it with
  throughput batching.
- The unconditioned random four-step chain failed the frozen numerical oracle.
  Its diagnostic is retained. The accepted chain uses dense dyadic Hadamard
  weights with operator norm at most one and dyadic inputs; it passed the
  original tolerances. This qualifies that fixture, not arbitrary random chains.

No failed/inconclusive run was overwritten or promoted. The original noisy
single-device sweep, failed DMA lowering and failed random-chain campaign
remain in the local evidence store.

## Validation and provenance

Local regression: 227 passed, seven hardware/optional skips. Explicit native
TPU smoke tests: two passed. The wheel contains all seven architecture profiles
and Apache provenance/license assets; its CLI can list probes with accelerator
framework imports blocked.

Runtime digest:
`f2160dbc42ad690e562353729f9538ea05e5deac477c2da445827c29dbc8a71c`.
Target configuration digest:
`3153678c9c78e536c9572ee8937c6820ef992b9c3256ba1d0ed3e5f4d8b81957`.
Lock SHA-256:
`7cd241d382bb7be16d125bb837f1eacf6e19d3d80173f370581353652b737fd6`.

The [execution tracker](../../../.plan/trackers/20261116-1659-tpu-backend.md)
records source baselines, exact artifact paths, corrections and acceptance gates.

## Revision History

| Revision | UTC timestamp | Change |
|---|---|---|
| r1 | 2026-11-17T19:46:22Z | Accepted the first TPU7x instance after native correctness, timing/DMA controls, collective sweep, independent replay and local artifact verification; retained diagnostic campaigns and explicit timing/observability limits. |
