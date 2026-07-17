# Comments: GCoM H100 Probe Sim-vs-HW Comparison Plan

## Position

The original `gcom_cuda` backend plan is a good integration direction, but it should not be approved
as-is. GCoM is actively changing, simulator accuracy is not guaranteed, and the initial metric mapping
is too narrow for serious NVIDIA/Hopper analysis.

The plan should be amended so that versioning, GCoM stat-schema drift, richer grouped metric mapping,
and simulator accuracy confidence are first-class requirements before implementation.

## 1. Add Version Management as a Hard Requirement

Every `gcom_cuda` run should record all version-relevant inputs. Without this, results cannot be
reproduced or interpreted once GCoM changes.

Required metadata:

- AMORA git commit hash.
- GCoM git commit hash.
- GCoM dirty/clean state.
- GCoM branch name.
- Simulator binary path.
- Simulator binary build timestamp.
- NVBit tracer version or build hash, if available.
- CUDA toolkit version.
- NVIDIA driver version.
- GPU model used for trace generation.
- H100 config file path and hash.
- `gpgpusim.config` hash.
- `trace.config` hash.
- Relevant environment variables affecting tracing or simulation.

Every generated report should include this metadata.

## 2. Treat GCoM Stats and Metric Mapping as a Versioned Contract

AMORA should treat GCoM emitted stats as a versioned external API. The mapping table should not assume
GCoM stat names are stable.

`gcom_metrics_map.py` should carry explicit mapping metadata, conceptually:

```python
MAPPING_VERSION = "2026-06-h100-v1"

SUPPORTED_GCOM_STAT_SCHEMA = {
    "required": [...],
    "optional": [...],
    "known_unmapped": [...],
}
```

For each run:

- Save raw GCoM stats.
- Save parsed stat keys.
- Compare parsed keys against AMORA's known mapping contract.
- Report mapped stats.
- Report missing required stats.
- Report missing optional stats.
- Report newly observed unmapped stats.
- Report deprecated stats that AMORA still knows about but GCoM no longer emits.
- Avoid silently ignoring schema drift.

Required behavior:

- Missing required stats should mark affected metrics unavailable.
- Missing optional stats should warn but continue.
- New stats should be listed in coverage output.
- Core stats such as `gpu_sim_cycle` should be treated as required.
- The whole run should not fail unless a required core execution stat is absent.

Suggested report fields:

- `mapping_version`
- `gcom_commit`
- `mapped_stats`
- `missing_required_stats`
- `missing_optional_stats`
- `new_unmapped_stats`
- `deprecated_or_unused_stats`

## 3. Archive Raw Simulator Outputs

Every `gcom_cuda` run should archive enough raw material for future re-parsing. This matters because
AMORA's mapping may improve after expensive simulations have already been run.

Archive:

- GCoM stdout/stderr.
- Full simulator log.
- Parsed stat JSON.
- Raw trace metadata.
- Probe binary build command.
- Probe launch parameters.
- Config hashes or copied configs.
- Environment metadata.

## 4. Add Simulator Accuracy Model

The comparison should not report only percentage error. It should also explain how much confidence
AMORA has in each comparison and what type of relationship exists between the hardware metric and the
simulator value.

Each probe or counter row should include:

- `fidelity`: `direct`, `proportional`, `proxy`, or `unsupported`.
- `model_confidence`: `high`, `medium`, or `low`.
- `known_limitations`.
- `expected_error_band`, if known.
- `calibration_status`: `uncalibrated`, `validated`, or `tuned`.
- `architecture_scope`: `generic_cuda`, `nvidia_generic`, `nvidia_hopper`, or `h100_specific`.

This prevents overclaiming simulator equivalence. A direct cycle-derived metric and a TMA proxy should
not be reported with the same confidence.

## 5. Add Accuracy Validation Anchors

Before interpreting broad comparisons, use a small canonical probe set as validation anchors.

Suggested anchors:

- FP32 dependent latency.
- FP32 throughput.
- Shared memory pointer chase.
- L1 pointer chase.
- L2 pointer chase.
- DRAM streaming bandwidth.
- Barrier latency.
- Tensor core latency.
- Tensor core throughput.

The report should include a baseline accuracy summary:

- Passed anchors.
- Failed anchors.
- Whether broader comparison should be considered reliable.

## 6. Expand Metric Mapping by Metric Group

The current mapping table is only a seed. It should be expanded into a grouped taxonomy. Each metric
should identify:

- AMORA logical metric name.
- GCoM stat key or derivation.
- NCU metric name, if available.
- Fidelity level.
- Architecture scope.
- Notes and limitations.

Proposed metric groups:

| Group | Metrics to Add or Investigate |
|---|---|
| Core execution | Active cycles, elapsed cycles, executed instructions, warp/thread instruction counts, IPC, issue rate. |
| Warp scheduling | Active warps, eligible warps, issued warps, selected warps, no-eligible-warp stalls, not-selected stalls. |
| Instruction frontend / L1I | Instruction fetches, L1I accesses, L1I misses, L1I hit rate, decode/issue frontend stalls. |
| L1D / LSU | L1D load accesses, L1D store accesses, global load/store split, local memory traffic, sector counts, hit/miss rates. |
| Shared memory | Shared load/store counts, shared bank conflicts, shared replay/conflict stalls, shared latency proxies. |
| Register file / operand collection | Register bank conflicts, register read/write counts, operand collector stalls, reuse/collector pressure if modeled. |
| L2 cache | L2 read/write sectors, L2 hits/misses, L2 hit rate, L2 bandwidth, L2 queueing/backpressure. |
| DRAM | DRAM read/write bytes, read/write commands, row hit/miss/conflict, bank activity, bandwidth utilization. |
| Memory partitions | Partition utilization, partition camping indicators, queue occupancy, partition imbalance. |
| Interconnect / NoC | Injection/ejection traffic, SM-to-L2 latency, L2-to-DRAM latency, NoC queueing, interconnect bandwidth. |
| Compute pipelines | FP32 pipe activity, FP64 pipe activity, INT pipe activity, SFU pipe activity, CUDA-core utilization proxy. |
| Tensor pipelines | Tensor pipe activity, HMMA counts, WGMMA counts if supported, tensor issue rate, tensor latency, tensor utilization proxy. |
| Async copy / TMA | `cp.async` / LDGSTS counts, async copy latency, async copy bandwidth, native TMA metrics if exposed, TMA proxy metrics, mbarrier/wait behavior. |
| Stall taxonomy | Scoreboard, long scoreboard, barrier, dispatch, pipe busy, memory dependency, MIO/LG throttle, wait, dependency stalls. |
| Occupancy / residency | Theoretical occupancy, achieved occupancy, resident CTAs, resident warps, persistent CTA behavior. |
| Simulator-only diagnostics | Internal queue latencies, model-specific bottleneck stats, config-derived limits, unsupported-but-observed simulator counters. |

Hopper-specific features such as TMA, WGMMA, tensor pipeline behavior, and split L1I/L1D metrics
require investigation. They should not be assumed comparable unless GCoM exposes meaningful stats.

## 7. Separate Probe Comparison from Counter Comparison

The output should clearly separate two comparison layers:

- Probe-level comparison: AMORA scalar probe values.
- Counter-level comparison: NCU counters versus GCoM internal stats.

Example:

```text
memory_pipeline.lane_patterns
probe value: unsupported
counter comparison: available via sectors/request proxy
```

This avoids treating counter proxies as equivalent to probe values.

## 8. Add Metric Coverage Report

Every compare run should include a coverage summary:

- Total AMORA logical metrics.
- Metrics with NCU data.
- Metrics with GCoM direct mapping.
- Metrics with GCoM proportional mapping.
- Metrics with GCoM proxy mapping.
- Metrics unsupported by GCoM.
- Metrics missing due to GCoM version or stat drift.
- New GCoM stats not mapped by AMORA.
- Hopper-specific metrics covered.
- Hopper-specific metrics missing.

## 9. Add Architecture Profiles

Do not hardcode everything as H100-only. Use architecture scopes:

- `generic_cuda`
- `nvidia_generic`
- `nvidia_hopper`
- `h100_specific`
- `gcom_simulator_only`

This keeps the mapping usable for future A100, H100, B100/B200, or other GPU comparisons.

## 10. Clarify Unsupported vs Unknown vs Missing

The plan should avoid collapsing all unavailable data into `unsupported`. Use separate states:

- `unsupported`: simulator fundamentally does not model it.
- `missing_stat`: simulator may model it, but this GCoM version did not emit the stat.
- `unmapped`: simulator emitted a stat, but AMORA does not yet map it.
- `proxy_only`: not equivalent, but useful diagnostically.
- `not_applicable`: metric does not apply to this architecture or probe.

## 11. Adjust Implementation Order

Recommended phased implementation:

Phase 1: Backend foundation and version metadata.

- Add `gcom_cuda` backend skeleton.
- Add capability discovery.
- Add version/config hash reporting.
- Add raw log archival.
- Add minimal simulation execution.
- Add one smoke probe.

Phase 2: Probe-level comparison.

- Implement cycle-derived probe comparison.
- Use existing 36 NVIDIA probe IDs as source of truth.
- Add `compare`.
- Preserve unsupported/proxy distinctions.

Phase 3: Stat schema and coverage reporting.

- Parse full GCoM stats.
- Add schema snapshot.
- Add missing/new/unmapped stat reporting.
- Add metric coverage report.

Phase 4: Expanded metric mapping.

- Add richer grouped mapping from section 6.
- Start with generic metrics, then add NVIDIA/Hopper-specific metrics.
- Explicitly mark uncertain mappings as `proxy` or `missing_stat`.

Phase 5: Accuracy model.

- Add validation anchors.
- Add model confidence annotations.
- Add expected error-band notes where known.
- Add accuracy summary in reports.

## Recommendation

The original `gcom_cuda` plan should be revised before implementation. The backend structure is sound,
but approval should depend on adding:

- Mandatory version metadata.
- GCoM stat-schema and mapping drift detection.
- Raw simulator output archival.
- Simulator accuracy confidence annotations.
- Richer grouped metric mapping, especially for NVIDIA/Hopper-specific metrics.
- Separate probe-level and counter-level comparison.
- Metric coverage reporting.
