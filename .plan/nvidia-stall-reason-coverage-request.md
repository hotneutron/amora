# Plan: make the warp-issue stall vector dimensionally sound

To: AMORA NVIDIA backend maintainer
Status: Implemented in this working tree
Blocks: per-phase schedule extraction in the accorde project (see **Consumer** below)

## Actionable Plan

1. Move the canonical stall reason list and metric-family resolution into a shared helper.
   - Add `amora/backends/nvidia/stall_metrics.py`.
   - Keep the 20 logical reasons used by NVIDIA and GCoM comparisons.
   - Resolve one coherent metric family per record: choose the family with the
     most supported reasons, preferring
     `smsp__warp_issue_stalled_<reason>_per_warp_active.pct` on coverage ties
     over `smsp__average_warps_issue_stalled_<reason>_per_issue_active.ratio`.
   - Treat `no_instruction` and `no_instructions` as aliases while preserving
     AMORA's logical reason name `no_instructions`.

2. Stop building aggregate stall vectors from per-metric cross-launch maxima.
   - Update `collect_stall_attribution` in
     `amora/probes/nvidia/baseline/_sources.py`.
   - Run NCU with only the selected family.
   - Choose one whole launch row using `median_total_stall`.
   - Record `metric_family`, `unit`, `missing_reasons`, `complete`, and
     `launch_selection`.

3. Apply the same family-selection contract to benchmark detailed collection.
   - Update `amora/backends/nvidia/benchmark_detail.py`.
   - Emit `stall_histogram.unit`, `stall_histogram.metric_family`,
     `stall_histogram.missing_reasons`, and whole-row launch provenance.
   - Preserve logical metrics for the selected row only.

4. Keep downstream reports dimensionally honest.
   - Update `amora/benchmarking/detailed.py` so detailed Markdown includes a
     hardware stall `unit` column instead of assuming every NVIDIA value is a
     percentage.

5. Record target-host metric inventory.
   - Command used:
     `ncu --query-metrics --chips gh100 --query-metrics-mode base | rg 'smsp__(warp_issue_stalled|average_warps_issue_stalled).*_(per_warp_active|per_issue_active)'`
   - Host: NVIDIA H100 80GB HBM3, driver `595.71.05`.
   - `/usr/bin/ncu --version`: Nsight Compute `2022.3.0.0`.
   - GH100 catalog exposes `smsp__warp_issue_stalled_*_per_warp_active` and
     `smsp__average_warps_issue_stalled_*_per_issue_active` for 18 reasons in
     this installed NCU version.
   - `warpgroup_arrive` and `mma` were absent from both queried families here.
   - `no_instruction` is the catalog spelling; AMORA maps it to logical
     `no_instructions`.

## Execution Notes

Implemented files:

- `amora/backends/nvidia/stall_metrics.py`
- `amora/backends/nvidia/metrics.py`
- `amora/probes/nvidia/baseline/_sources.py`
- `amora/backends/nvidia/benchmark_detail.py`
- `amora/benchmarking/detailed.py`
- `tests/backends/test_nvidia_cuda.py`
- `tests/benchmarks/test_detailed.py`

Verification:

- `PYTHONPATH=. pytest -q tests/backends/test_nvidia_cuda.py tests/backends/test_ncu_run.py tests/backends/test_sass.py tests/benchmarks/test_detailed.py tests/benchmarks/test_classification.py tests/benchmarks/test_materialize.py`
  passed with `45 passed, 2 skipped`.
- Full `PYTHONPATH=. pytest -q` passed with `94 passed, 3 skipped`.

## Original Request Context

## Revision History

| Revision | Change |
|---|---|
| r2 | Rewritten against `20fd9b2` ("Add GCoM stall comparison for H100 probes"), which landed after r1 was drafted. **Superseded:** r1's central ask — expand `_STALL_LOGICALS` — is already done, and more completely than r1 proposed (20 reasons, including several r1 did not know about). r1's proposed name `no_instruction` was wrong; the metric is `no_instructions`. **New:** the expansion introduced a unit mismatch that makes the emitted vector dimensionally incoherent, verified against committed H100 output. That is now the primary request. **Retained:** the cross-launch `max` fold, and the request for a recorded `ncu --query-metrics` inventory. |
| r1 | Created. Requested an `ncu --query-metrics` inventory, expansion of `_STALL_LOGICALS` from 8 reasons, and a fix to the `dominant_stall` argmax. |

Source inputs for r2: `20fd9b2`, `amora/backends/nvidia/metrics.py` and
`amora/probes/nvidia/baseline/_sources.py` at that commit, and the committed
`out/nvidia/hopper/h100-80g.json`.

## What already landed — no longer requested

`20fd9b2` expanded `_STALL_LOGICALS` to 20 reasons, adding `selected`,
`dispatch_stall`, `warpgroup_arrive`, `mma`, `no_instructions`, `imc_miss`,
`sleeping`, `branch_resolving`, `membar`, `drain`, `tex_throttle`, and `misc`, each
with a `MetricResolver.CANDIDATES` entry. This covers r1's request and includes
`warpgroup_arrive` and `mma`, which r1 missed entirely.

## 1. The vector mixes two metric families with different units

This is the primary request.

`CANDIDATES` resolves in order and takes the first supported candidate. The eight
original reasons list `smsp__warp_issue_stalled_<r>_per_warp_active.pct` **first**, with
the `.ratio` form as fallback. The twelve reasons added in `20fd9b2` have **only**
`smsp__average_warps_issue_stalled_<r>_per_issue_active.ratio`.

On any host where both families exist, the emitted `stalls` dict therefore carries eight
values on a 0–100 percentage of warp-active cycles alongside twelve on a 0–1 average
warp count per issue-active cycle. Different numerator, different denominator, different
scale.

Verified in `out/nvidia/hopper/h100-80g.json`, first `stall_attribution` record:

```
wait               68.8600   [pct]
selected            1.0000   [ratio]
short_scoreboard    0.6000   [pct]
branch_resolving    0.0900   [ratio]
imc_miss            0.0500   [ratio]
...                          sum = 70.610 over 17 reasons
```

Three consequences:

- **`dominant_stall` is decided by units, not by behaviour.** A `.pct` reason will beat
  a `.ratio` reason at nearly any real value. In the second record of the same file,
  `wait` is reported dominant at 4.09 (pct) over `selected` at 1.0000 (ratio) — but a
  ratio of 1.0 means every issue-active cycle had a warp in that state, which is the
  larger quantity by any reading. Every `stall_attribution` record in that file reports a
  `.pct` reason as dominant.
- **The sum is meaningless**, so no coverage or residual can be computed from it.
- **Coverage varies silently between records** — 17 reasons in the first, 8 in the third,
  with nothing in the record distinguishing "this reason was ~0" from "this reason did
  not resolve on this run."

Requested: emit one family per record. Preferably add the `_per_warp_active.pct`
candidate for the twelve new reasons where it exists on the target architecture — that
fixes the mismatch at the root and keeps the established unit. Where it does not exist,
either convert or carry a per-reason `unit` field and refuse to argmax across units.
Whichever way, please record the chosen family in the record so a consumer can tell.

## 2. The vector is assembled from different launches

`collect_stall_attribution` still calls `collect_ncu_metrics(..., aggregate="max")` with
`launch_count=4`, and `_fold` takes the max **per metric independently** across
`ncu.raw_rows` — across launches. Each reason's value can come from a different launch.

The result is not a stall vector from any single execution: the reasons are not mutually
consistent, they cannot sum to a meaningful total, and the argmax is over numbers that
never co-occurred. `max` is a defensible aggregation for a latency or a bandwidth; for a
distribution over mutually exclusive warp states it is not.

Requested: for `stall_attribution` specifically, select one launch and report its vector
whole — median by total issue-stall, modal, or simply the last — and record which. If
cross-launch aggregation is wanted, aggregate whole vectors, not components.

## 3. Record an `ncu --query-metrics` inventory

Still outstanding. `20fd9b2` added twelve `.ratio`-only entries, which raises a
portability question the inventory would settle: whether those reasons also exist in the
`.pct` family on the target architectures (bearing directly on request 1), and which
reasons exist at all per architecture.

The plumbing is already there — `_discover_ncu_metrics` in
`amora/backends/nvidia/cuda.py` calls `ncu --query-metrics` at capability discovery and
populates `capabilities.ncu_metrics`; `list_metrics()` in `amora/backends/nvidia/ncu.py`
exposes it directly. So this is a query and a record, not new integration:

```
ncu --query-metrics | grep warp_issue_stalled
```

Record, under `.plan/` or `docs/`: the architecture and `ncu` version, every
`smsp__warp_issue_stalled_*_per_warp_active` and
`smsp__average_warps_issue_stalled_*_per_issue_active` base name, and any reason present
on one architecture and absent on another. `CANDIDATES` is a static class attribute, so
an architecture-specific reason needs to resolve `available=False` cleanly rather than
silently vanish from the vector.

## Consumer

The accorde project is replacing an analytical scheduling model whose interval time is
`max` over functional units. The check on that model is: what fraction of blocked warp
cycles is attributable to functional-unit occupancy — the only thing the analytical
model represents?

That needs the throttle group (`math_pipe_throttle`, `mio_throttle`, `lg_throttle`,
`tex_throttle`, and now `mma`) as a fraction of **all** blocked cycles. With mixed units
the denominator does not exist, which is why request 1 outranks everything else here.

`no_instructions` and `imc_miss` are the highest-value additions from `20fd9b2` for this
purpose: instruction-fetch and constant-cache pressure are structures the analytical
model has no term for at all, and they are the hardware anchors for validating GCoM's
per-subcore L0I and L0C against silicon. `warpgroup_arrive` and `mma` are directly the
Hopper async-pipeline structures the model tries to approximate with a `pipeline_stages`
divisor.

## Not requested

- No new probe. This extends an existing, working collector.
- No PC-sampling work — see `.plan/nvidia-pc-sampling-stall-request.md`, a separate and
  larger request. This one is per-kernel aggregates only.
- No change to `MetricResolver.resolve` semantics.

## Related

- `amora/probes/nvidia/baseline/_sources.py` — `_STALL_LOGICALS`,
  `collect_stall_attribution`, `_fold`
- `amora/backends/nvidia/metrics.py` — `MetricResolver.CANDIDATES`, `_is_supported`
- `amora/backends/nvidia/cuda.py` — `_discover_ncu_metrics`
- `out/nvidia/hopper/h100-80g.json` — the evidence for request 1
- `.plan/gcom-cuda-stall-reason-histogram-per-probe.md` — the per-probe histogram work
- accorde repository,
  `.plan/20260803-2229-plan-extract-scheduling-from-hardware-stall-traces.md`

## Direction, verbatim

Reproduced exactly as given. The home-directory path below is quoted text, not a path
reference by this document; every path this document cites is repo-relative per
`RULES.md`.

> write a request for fix in ~/wk/amora to expand the _STALL_LOGICALS to cover all the reasons. ask amora to run ncu --query-metrics.
