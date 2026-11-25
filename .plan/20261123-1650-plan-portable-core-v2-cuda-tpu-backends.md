# Portable-core V2: CUDA and TPU backend plan

**Status:** Proposal for review, 2026-11-23. This document authorizes no code
changes, hardware runs, resource allocation, or timing collection.

## Request

```text
persist a CUDA + TPU backend plan for v2 for me to review
```

## Decision

Add **PyTorch CUDA** (`cuda_torch`) and **JAX/PJRT TPU** (`tpu_jax`) as opt-in
realizations of the same frozen V2 cases. Deliver device correctness first,
then qualify separately named timing modes. Do not route V2 through legacy
case dictionaries, change the CPU release matrix, or infer a hardware pass
from portable correctness. The first CUDA lane is eager Torch on one device;
the first TPU lane is a single-device JAX executable. CUDA Graph, Torch
compilation, Pallas and distributed execution are separate realizations, not
fallbacks or requirements for the initial connectors.

The [V2 tracker](trackers/20261123-1522-portable-core-h3-operation-coverage.md)
records 1,505 required CPU checks; those checks qualify neither accelerator.
The [V2 executor](../benchmark_generators/portable_core/expanded/v2/execution.py)
is CPU-only. The [TPU legacy bridge](../amora/backends/tpu/portable_core.py)
and [measurement entrypoint](../amora/backends/tpu/measurement.py) still
consume the legacy tier manifests. Exact historical H3 stays
`blocked_missing_source` until Accorde supplies approved per-case sources.
Independent H3 controls are usable immediately under their own IDs. Gate-v2
references use the current portable fixtures, not the historical CUDA RNG
sequence, labels, or admission decisions.

## 1. Integration and compatibility

- **One loader:** consume the output of `amora-portable-core freeze --suite`
  via [V2 snapshot verification](../benchmark_generators/portable_core/expanded/v2/runner.py).
  Check the separately pinned manifest digest for imported handoffs. Select
  case IDs from that snapshot or a frozen selection manifest, not a parallel
  CUDA/TPU catalogue. Preserve the program, case, input recipe, and upstream
  identities without embedding backend options in them.
- **Isolated adapters:** add shared adapter/receipt code under
  `amora/benchmarking/portable_v2/`, with backend-specific code in
  `amora/backends/nvidia/portable_core_v2.py` and
  `amora/backends/tpu/portable_core_v2.py`. Keep the hashed V2 CPU source files
  unchanged where possible. Existing V2 snapshots pin the source bytes of
  `expanded/v2/*.py`; a necessary semantic or schema change must introduce a
  new revision and explicit migration, not silently invalidate or rewrite
  frozen snapshots. V1 tiers, fingerprints and measurement entrypoints stay
  unchanged. `amora-portable-core check` stays the CPU release checker.
- **Prepare → execute → validate:** the shared path binds typed ports,
  dimensions, source layout and bounded inputs against the V2 spec. Adapter
  preparation owns transfers, device storage construction, compilation and
  executable-cache identity. Prepared execution accepts only those bound
  values, without host reads, implicit repacking, device fallback or hidden
  recompilation. Correctness can return all node ports or explicit boundary
  witnesses; measure only the output-only realization. If a debug build
  changes compilation/fusion, retain its identity separately and verify the
  exact timing executable's outputs against the oracle as well.
- **Versioned receipts:** record the source manifest and case hashes, input
  hashes (including physical layout when claimed), oracle and tolerance
  revisions, adapter/code/config hashes, target/runtime fingerprint, lowering
  mode, launch/executable identity and capability status. Compute a separate
  `realization_sha256` for the device implementation and protocol; do not hash
  process IDs, pointers or wall timestamps into it. Distinguish device query
  facts, compile observations, and measured evidence. Absent artifacts stay
  `not_available`, not inferred from names or optimizer logs.
- **Fail closed and keep dependencies optional:** `inspect-v2` reports static
  capability without importing Torch/JAX or opening a device. A run names its
  backend, visible device and bounded case selection explicitly. Require
  positive proof of the selected accelerator and, for profiling/timing,
  approved ownership of the relevant resource scope. Missing library, runtime,
  dtype/operation/layout support, out-of-bounds inputs, failed correctness and
  unavailable exclusivity are separate outcomes. A denied case must never
  yield a timing row or inherit a target's neighbouring case status.

**Proposed CLI** (review the names; not implemented):

```bash
amora-portable-core freeze --suite operation-gate-v2 --profile smoke \
  --out out/portable_v2/gate-smoke
amora benchmarks qualify-v2 --frozen out/portable_v2/gate-smoke \
  --backend cuda_torch --device cuda:0 --out out/portable_v2/cuda-check
amora benchmarks qualify-v2 --frozen out/portable_v2/gate-smoke \
  --backend tpu_jax --device tpu:0 --out out/portable_v2/tpu-check
```

Add a separate `amora benchmarks measure-v2` only after the two correctness
lanes are admitted. It must consume a passed qualification receipt and reject
changes to the case/input, target, runtime, adapter, or output-only realization.

## 2. Per-backend design

| Concern | CUDA | TPU |
|---|---|---|
| Execution | Torch tensors on explicit `cuda:N`; disable autocast and TF32 for the documented FP32-working lane in an isolated worker. Torch CUDA functionality is not Gluon/NCU/SASS admission. | `jax.jit` on explicitly selected addressable TPU device; verify PJRT placement and no CPU fallback. Record JAX, jaxlib, libtpu and instance/topology, including actual device/core mapping. Reuse discovery/ownership checks, not the legacy DAG evaluator. |
| Layout | Packed input first. For a strided/offset case, stage the backing storage and reconstruct a view; verify element stride, storage offset, allocation size and alias contract on-device. A packed copy does not satisfy the strided case. | Packed input first. JAX staging does not preserve NumPy byte strides by contract. Mark physical stride/offset cases unsupported unless a separately qualified lowering realizes the layout. A declared packed/logical equivalent is a different realization and cannot pass the original physical-layout case. |
| FP16 and state boundaries | Use explicit FP32 working operations and FP16 storage/round-trip points as declared, including reduction before broadcast. Retain the out-of-place/alias policy and ordered recurrence ports. | Use the same casts and state/port order in JAX. Retain StableHLO and optimized HLO when available, and demonstrate that fusion or precision rewrites do not erase observable rounding. An HLO hash is not a native executable/disassembly hash. |
| Indirect and discrete cases | Bind and validate indices before launch. Duplicate destinations require an explicitly deterministic contract or an unsupported result; forcing float atomics or a nondeterministic path under the same ID is not parity. | Same bound index and duplicate rules; do not treat a compiled scatter or sort as inherently deterministic. Check top-k tie policy and ordered output ports exactly. |
| Capability scope | Fingerprint device/SM, driver, Torch/CUDA runtime, stream policy and MIG mode when relevant. No family-wide inference from one device/revision. | Scope by generation profile, instance, SKU, dtype, local TensorCore/core mode and runtime; never gate on numeric generation order. Start on the qualified TPU7x instance, then validate a second generation (prefer v6e, or v5e if that's the available separately approved target). v4 and v5p are separate follow-up rows. |

These are contract gates, not item-level performance goals. ReLU, row
logsumexp, reduction→broadcast and a fork/join graph make the first end-to-end
witness set. Add an affine scan, grouped reads, threshold/top-k and duplicate
scatter controls before expanding. Track each required `(case, backend,
target)` row, including blocked and unsupported rows. The half-scale
contraction fixture can lower to scale-only; it does not validate a hardware
GEMM path. A dense, non-identity performance control needs its own case ID.

## 3. Timing and artifacts — separate admission

- **Common:** retain per-repeat samples; compile, transfer, first execution
  and warm timing separately. Warm the same invocation and synchronization
  unit to be measured. Freeze batch size, output retention, allocation and
  drain policies. Verify output identity before/after each timing campaign.
  Record whether output capture, trace/counters, or graph replay changes the
  realization. Report host completion and device-specific intervals as
  separate fields with their unit, resource scope and uncertainty. A batched
  per-invocation value is amortized throughput, not isolated latency.
- **CUDA:** for device-side timing, record paired CUDA events on the owned
  Torch stream, around the same documented launch/drain unit. Validate stream
  dependencies; extra streams must be joined explicitly or rejected. Reuse
  [existing CUDA-event evidence machinery](../amora/backends/nvidia/cuda_event_run.py)
  only when its payload, clock and scope match the new worker. Keep a
  synchronized host lane and a separate trace lane for launch attribution.
  Do not identify a Torch call or CUDA-event span with one kernel or SM cycles.
- **TPU:** use [synchronized runner semantics](../amora/backends/tpu/runner.py)
  (dispatch through readiness of all output leaves) for the first timing lane.
  Keep `device_duration_ns` unset. The [timestamp study](20261123-1319-tpu-findings-timestamp-normalization.md)
  verified host arithmetic but not host/device alignment. The [execution-ID
  pilot](20261123-1437-tpu-findings-execution-join-scalesim.md) supports a
  sequencer-corroborated module-span diagnostic on one pinned stack only. Admit
  it under its own name only after repeating the identity, clock and device
  scope controls for the new runtime and workload. Never sum overlapping
  streams, TPU cores or parent/child intervals, recover clock cycles from
  inferred frequencies, or equate estimated work rates with hardware counters.

No new simulator calibration, PPP/P³ lifting, H100 gate promotion or
performance-model work in this delivery. The two timing lanes are not
comparable just because both report microseconds.

## 4. Milestones and exit gates

| Gate | Deliverable | Acceptance before proceeding |
|---|---|---|
| **B0 — Contract freeze** | Confirm CLI placement, immutable snapshot boundary, backend/target capability schema and initial witness IDs. Pin CPU oracle, source/input recipes and target-specific install/runtime contracts. | No-framework import/discovery and all six v1 fingerprints pass. Existing V2 CPU snapshots/checks verify unchanged. Unsupported devices and features have explicit statuses and do not initialize hardware or fall back. |
| **B1 — CPU-only plumbing** | Implement loading, validation, adapter protocol, prepared execution hooks, shared output/boundary checks, receipts and shard accounting. | Deterministic source/manifest and receipt verification; tampered hashes/ports/layout, unsupported ops, missing backends, duplicate/missing shards and mislabelled devices fail closed. Fake devices do not count as native evidence. |
| **B2 — CUDA correctness** | Start with packed one-device FP16/FP32/int32 cases. Record CUDA placement and checks for the frozen witness set; expand the coverage table per operation after passing the slice. | Every required CUDA witness passes the independently pinned CPU oracle, including explicit store points, output arity/dtypes and exact indices/ties. Strides, duplicates, recurrence and alias policies are either proved by their own controls or marked unsupported. No timing required. |
| **B3 — TPU correctness and second generation** | Same frozen portable inputs on TPU7x; add a separately discovered generation/runtime row after the first passes. | Every required supported witness passes its parity/placement check; capture compiler artifacts and explicit unsupported rows. Repeat startup/ownership and parity per generation. If no second host is approved, deliver the first qualification without claiming transfer. |
| **B4 — Native timing pilot** *(separate approval)* | Freeze workload and estimands prospectively; compare a small matched subset only. Use three independent repeats of at least 30 samples after matched warmup, with singleton and batch results separate. | All runs start from accepted correctness receipts; preserve clock/domain and ownership evidence. Predeclare minimum sample duration and quality bounds (start with the existing 5% relative-MAD policy within repeats and across repeat medians, or justify an alternative before collection). For sub-resolution measurements, use a declared batch or report inconclusive. Never promote noisy/missing cases or replace accepted measurements. |
| **B5 — Packaging and rollout** | Add optional dependency documentation and opt-in CI recipes. Render a coverage ledger keyed by case/realization/target/runtime/protocol; retain raw traces locally. | Isolated wheel and CPU-only CI pass; hardware CI is separately enabled. Required witnesses are all accounted for; unanswered/blocked cells stay visible. Accorde can consume the source-only crosswalk without importing live research scripts or inheriting labels. |

## Review decisions

1. **Initial targets:** approve one CUDA device and the existing TPU7x as the
   first correctness pair; approve v6e as the preferred independent second TPU
   generation, subject to access and runtime qualification.
2. **Default realizations:** keep Torch eager and JAX JIT as the baseline;
   require separate evidence for optimized variants or physical strides on TPU.
3. **Release boundary:** approve correctness-first delivery (B0–B3 and local
   B5) without making timing, complete historical H3 source closure or another
   generation's availability a blocker for the first connector. B4 still needs
   explicit access and experiment approval; each blocked row remains visible.

## Revision History

| Revision | Date | Change |
|---|---|---|
| r1 | 2026-11-23 | Proposed isolated CUDA and TPU adapters using frozen V2 cases, separate device-correctness and timing gates, per-generation admission, and fail-closed layout/rounding policies. No implementation or runs. |
| r2 | 2026-11-23 | Marked the local CPU implementation and experiment references in the plan-only commit. Technical decisions are unchanged. |
| r3 | 2026-11-23 | Restored the links when the existing implementation and experiment findings were added to the same branch. |
