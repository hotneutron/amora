# Plan: `gcom_cuda` Probe Sim-vs-HW Comparison

## Summary

Model **GCoM CUDA** as a first-class AMORA backend/vendor named `gcom_cuda`.
It is a simulator backend that consumes NVIDIA CUDA traces and emits the same
AMORA `ProbeResult` shape as the real `nvidia` backend.

This plan is intentionally generic to `gcom_cuda`. H100 is one SKU under the
Hopper family, not the backend identity. Future simulator SKUs such as
`gcom_b200` and `gcom_v100` should fit the same backend, report hierarchy,
version contract, metric mapping, and comparison flow.

The comparison model has two layers:

- **Probe-level comparison:** AMORA scalar probe values from real NVIDIA
  hardware versus simulator-derived values.
- **Counter-level comparison:** real NCU/CUPTI counters versus GCoM internal
  stats mapped into AMORA logical metrics.

These layers must remain separate. A counter proxy can explain simulator
behavior without pretending to be the same thing as a probe scalar.

## Revision History

### 2026-06-26: Generic `gcom_cuda` Revision

Source inputs:

- `.plan/gcom-h100-probe-sim-vs-hw-comparison-comments.md`
- Previous H100-specific GCoM comparison plan
- Existing NVIDIA report hierarchy and grouped probe summary:
  `reports/nvidia/SUMMARY.md`
- Current NVIDIA baseline probe registry and methodology documents.

Major changes:

- Renamed the plan from H100-specific wording to the generic `gcom_cuda`
  backend/vendor model.
- Treated `gcom_h100` as a simulator SKU in the Hopper family, parallel to
  future `gcom_b200`, `gcom_v100`, and other simulator SKUs.
- Accepted versioned GCoM stat contracts, raw simulator output archival,
  unsupported/missing/unmapped/proxy distinctions, and phased implementation.
- Clarified that accuracy validation anchors are a selected subset of the
  canonical 36 NVIDIA probe IDs, not a separate probe inventory.
- Clarified that grouped metric mapping should reuse AMORA's existing grouped
  probe taxonomy and report layout, rather than inventing a parallel grouping.
- Updated report outputs to follow the same vendor/family/SKU hierarchy used by
  `reports/nvidia/`.
- Replaced machine-local absolute paths with repo-relative paths or environment
  variables.

Superseded assumptions:

- Superseded: `gcom_h100` is the backend identity.
  Replacement: `gcom_cuda` is the backend/vendor; `gcom_h100` is one SKU.

- Superseded: validation anchors are a new list of probes.
  Replacement: anchors are a flagged subset of the existing canonical 36 probe
  IDs and should be generated from the same probe registry and report tables.

- Superseded: metric groups are an independent table unrelated to reports.
  Replacement: metric mapping is grouped by the same report taxonomy already
  used in `reports/nvidia/SUMMARY.md`.

## Backend, Family, And SKU Model

Use this naming model:

- Backend/vendor: `gcom_cuda`
- Family examples: `hopper`, `blackwell`, `volta`
- SKU examples: `gcom_h100`, `gcom_b200`, `gcom_v100`
- Hardware comparison target examples: `h100-80g`, `b200`, `v100-32g`

The backend is generic. SKU-specific config files and expectations are selected
through an architecture profile.

Example profile keys:

```yaml
backend: gcom_cuda
family: hopper
sku: gcom_h100
hardware_reference:
  backend: nvidia
  family: hopper
  sku: h100-80g
architecture_scope: nvidia_hopper
config:
  gpgpusim_config: ${GCOM_ROOT}/gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/gpgpusim.config
  trace_config: ${GCOM_ROOT}/gpu-simulator/configs/tested-cfgs/SM90_H100_L2_50MB_80GB/trace.config
```

Do not hardcode H100 into backend APIs. H100-specific constants belong in the
`gcom_h100` SKU profile.

## Report Hierarchy

`gcom_cuda` should mirror the existing NVIDIA hierarchy:

```text
reports/
  nvidia/
    SUMMARY.md
    hopper/
      probes-h100-80g.md
  gcom_cuda/
    SUMMARY.md
    hopper/
      probes-gcom_h100.md
      sim-vs-hw-gcom_h100.md
      sim-vs-hw-gcom_h100.json
```

Raw run artifacts should live under ignored output directories:

```text
out/
  gcom_cuda/
    hopper/
      gcom_h100/
        <run_id>/
          version.json
          raw_stats.txt
          parsed_stats.json
          simulator.log
          trace_metadata.json
          config_hashes.json
```

The committed report files summarize results. The ignored output directory keeps
raw material for re-parsing and debugging.

## Versioned Contract

Version management and stat-schema handling are hard requirements.

Every `gcom_cuda` run must record:

- AMORA git commit hash.
- GCoM git commit hash.
- GCoM dirty/clean state.
- GCoM branch name.
- Simulator binary path.
- Simulator binary build timestamp.
- NVBit tracer version or build hash when available.
- CUDA toolkit version.
- NVIDIA driver version.
- GPU model used for trace generation.
- Family and SKU profile.
- GCoM config file paths and hashes.
- Trace config paths and hashes.
- Relevant environment variables affecting tracing or simulation.

Treat emitted GCoM stats as a versioned external API. `gcom_metrics_map.py`
should carry mapping metadata:

```python
MAPPING_VERSION = "2026-06-gcom-cuda-v1"

SUPPORTED_GCOM_STAT_SCHEMA = {
    "required": [...],
    "optional": [...],
    "known_unmapped": [...],
}
```

For each run, save:

- raw GCoM stats,
- parsed stat keys,
- mapped stats,
- missing required stats,
- missing optional stats,
- newly observed unmapped stats,
- deprecated stats AMORA still knows about but GCoM no longer emits.

Missing required stats mark affected metrics unavailable. Missing optional stats
warn but continue. New stats are listed in coverage output. A whole run should
fail only when a required core execution stat, such as `gpu_sim_cycle`, is
absent.

## Raw Output Archival

Archive enough raw material to re-parse old simulations when AMORA's mapping
improves:

- GCoM stdout and stderr,
- full simulator log,
- parsed stat JSON,
- raw trace metadata,
- probe binary build command,
- probe launch parameters,
- config hashes or copied configs,
- version metadata,
- environment metadata.

The archive is part of the evidence record. Report generation should never
depend only on a lossy parsed table.

## Accuracy Model

Every probe-level or counter-level comparison row should include:

- `fidelity`: `direct`, `proportional`, `proxy`, or `unsupported`.
- `model_confidence`: `high`, `medium`, or `low`.
- `known_limitations`.
- `expected_error_band`, when known.
- `calibration_status`: `uncalibrated`, `validated`, or `tuned`.
- `architecture_scope`: `generic_cuda`, `nvidia_generic`,
  `nvidia_hopper`, `h100_specific`, or `gcom_simulator_only`.

Percentage error alone is not enough. A direct cycle-derived metric and a TMA
proxy should not be reported with the same confidence.

## Accuracy Validation Anchors

The validation anchors are not a new probe list. They are a selected subset of
the canonical 36 NVIDIA probe IDs already represented in the grouped reports.

Anchor set for `gcom_h100` should be drawn from existing IDs:

- `arithmetic_latency.dependent_chain`
- `arithmetic_throughput.independent_chains`
- `shared_memory.pointer_chase`
- `l1_cache.pointer_chase`
- `l2_cache.pointer_chase`
- `global_memory.streaming`
- `synchronization.barrier_latency`
- `tensor_core.mma_latency`
- `tensor_core.mma_throughput`

These are useful anchors because they cover compute, memory, synchronization,
and tensor behavior while staying tied to existing probes. Reports should flag
which rows are anchors and include:

- passed anchors,
- failed anchors,
- unavailable anchors,
- whether broad comparison should be considered reliable.

Anchor status should be computed from the same comparison rows as the regular
report. Do not create a separate anchor execution path unless implementation
needs a fast smoke subset.

## Grouped Metric Mapping

AMORA already has grouped probe results, as shown by `reports/nvidia/SUMMARY.md`.
`gcom_cuda` should reuse that taxonomy for metric coverage and comparison
instead of introducing a competing grouping.

Use these report groups as the primary presentation units:

- Compute & Scheduling
- Register, Tensor & Sync
- On-chip Memory
- Global Memory & DRAM
- Transfer & Interconnect

Inside each group, each metric mapping row should identify:

- AMORA logical metric name,
- GCoM stat key or derivation,
- NCU metric name when available,
- fidelity level,
- architecture scope,
- notes and limitations,
- source probe IDs affected by the mapping.

The richer metric groups from the comment document are accepted as a mapping
checklist, but they should be folded into the existing report groups. For
example, "Warp scheduling" and "Instruction frontend / L1I" belong under
Compute & Scheduling, while "Async copy / TMA" belongs under Transfer &
Interconnect.

## Coverage Report

Every compare run should include coverage at both levels:

- total AMORA probe IDs,
- probe IDs with real hardware values,
- probe IDs with GCoM direct values,
- probe IDs with GCoM proportional values,
- probe IDs with GCoM proxy-only values,
- probe IDs unsupported by GCoM,
- metrics missing due to GCoM stat-schema drift,
- new GCoM stats not mapped by AMORA,
- Hopper-specific metrics covered,
- Hopper-specific metrics missing.

Coverage should be summarized by the same grouped report taxonomy used in
`reports/nvidia/SUMMARY.md`.

## Unsupported, Unknown, Missing, And Proxy States

Do not collapse all unavailable data into `unsupported`.

Use these states:

- `unsupported`: simulator fundamentally does not model the behavior.
- `missing_stat`: simulator may model it, but this GCoM version did not emit the
  required stat.
- `unmapped`: simulator emitted a stat, but AMORA does not yet map it.
- `proxy_only`: not equivalent, but useful diagnostically.
- `not_applicable`: metric does not apply to this architecture, SKU, or probe.

These states are accepted and should be represented in both JSON and Markdown
reports.

## Probe Inventory And Single Source Of Truth

The canonical probe IDs come from the NVIDIA baseline registry. `gcom_cuda`
must derive from that inventory instead of copying a second list.

Implementation rule:

```python
PLANNED_PROBES = tuple(amora.probes.nvidia.baseline.PROBES)
```

The `gcom_cuda` mapping table declares only simulator-specific policy:

- category,
- derivation kind,
- required GCoM stat keys,
- required hardware denominator fields,
- fidelity,
- architecture scope,
- known limitations.

It must not duplicate canonical probe metadata such as concept names, units, or
raw denominator values when those are available from real NVIDIA `ProbeResult`
records.

## Probe-Level Comparison

Cycle-derived probes use `gpu_sim_cycle` and hardware-matched denominators from
the real NVIDIA report. Examples:

- dependent arithmetic latency: `gpu_sim_cycle / chain_length`
- throughput plateaus: simulated curve points reduced by the same reducer as
  the hardware analyzer
- bandwidth: simulated bytes divided by simulated time
- multi-kernel differentials: separate simulated runs, then subtraction

If a required denominator is absent from the real hardware report, the simulated
probe row becomes `missing_stat` or `unsupported` with a reason. Do not hardcode
denominators in `gcom_cuda`.

## Counter-Level Comparison

GCoM emits internal stats that can be mapped into AMORA logical metrics. Add a
versioned `gcom_metrics_map.py` that records:

- logical metric name,
- required GCoM stat keys,
- derivation function,
- equivalent or nearest NCU metric name,
- fidelity,
- architecture scope,
- limitations.

Counter comparison rows should be reported separately from probe rows:

```text
memory_pipeline.lane_patterns
probe value: unsupported
counter comparison: available via sectors/request proxy
```

Proxy rows are diagnostic. They should not upgrade the probe-level result to a
direct comparable scalar.

## Backend Implementation Plan

New backend package:

```text
amora/
  backends/
    gcom_cuda/
      __init__.py
      config.py
      gcom.py
      build.py
      trace.py
      runner.py
      compare.py
  probes/
    gcom_cuda/
      baseline/
        __init__.py
        metrics_map.py
```

Expected CLI:

```bash
amora gcom_cuda inspect-capabilities
amora gcom_cuda list
amora nvidia run --all --output out/nvidia/hopper/h100-80g.json
amora gcom_cuda run --sku gcom_h100 --hw-baseline out/nvidia/hopper/h100-80g.json \
  --all --output out/gcom_cuda/hopper/gcom_h100/probes.json
amora gcom_cuda compare \
  --real out/nvidia/hopper/h100-80g.json \
  --sim out/gcom_cuda/hopper/gcom_h100/probes.json \
  --out-dir reports/gcom_cuda/hopper
```

`gcom_cuda run` accepts `--hw-baseline` because simulator derivations need
hardware-side denominators. Without it, comparable probes emit an unavailable
status instead of hardcoded counts.

## Implementation Order

### Phase 1: Backend Foundation And Version Metadata

- Add `gcom_cuda` backend skeleton.
- Add capability discovery.
- Add version/config hash reporting.
- Add raw log archival.
- Add minimal simulation execution.
- Add one smoke probe.

### Phase 2: Probe-Level Comparison

- Implement cycle-derived probe comparison.
- Use existing NVIDIA probe IDs as source of truth.
- Add `compare`.
- Preserve unsupported/proxy/missing distinctions.

### Phase 3: Stat Schema And Coverage Reporting

- Parse full GCoM stats.
- Add schema snapshot.
- Add missing/new/unmapped stat reporting.
- Add grouped metric coverage report.

### Phase 4: Expanded Metric Mapping

- Add richer grouped mapping using the existing report group taxonomy.
- Start with generic metrics, then add NVIDIA/Hopper-specific metrics.
- Explicitly mark uncertain mappings as `proxy_only` or `missing_stat`.

### Phase 5: Accuracy Model

- Add validation anchor flags.
- Add model confidence annotations.
- Add expected error-band notes where known.
- Add accuracy summary in reports.

## Initial `gcom_h100` Policy

The initial `gcom_h100` SKU should support three categories:

- `comparable`: direct cycle/stat derivation with known denominators.
- `approximate`: sweep, differential, or proxy behavior that may need multiple
  simulated launches.
- `unavailable`: one of `unsupported`, `missing_stat`, `unmapped`,
  `proxy_only`, or `not_applicable`.

The earlier 36-probe mapping table remains useful as a grounding worksheet, but
the implementation should store policy in `metrics_map.py` and derive the probe
inventory from the NVIDIA registry.

## Verification

1. `amora gcom_cuda inspect-capabilities` reports tracer, simulator, compiler,
   GPU, config, and SKU-profile status.
2. Version metadata and config hashes are present in every JSON report.
3. Raw simulator outputs are archived under `out/gcom_cuda/<family>/<sku>/`.
4. `amora gcom_cuda run --sku gcom_h100 --probe arithmetic_latency.dependent_chain`
   yields a finite simulated value when the simulator and trace are available.
5. `amora gcom_cuda run --sku gcom_h100 --all` returns the canonical NVIDIA
   probe inventory with comparable, approximate, and unavailable rows.
6. `amora gcom_cuda compare` writes Markdown and JSON under
   `reports/gcom_cuda/hopper/`.
7. The compare report includes probe-level comparison, counter-level
   comparison, grouped metric coverage, validation anchor summary, version
   metadata, and unavailable-state breakdown.
8. Non-CUDA tests still pass; simulator-dependent tests are gated.

## Open Decisions

- Exact SKU ID spelling for H100: default to `gcom_h100`, but allow profiles to
  include display names and config hashes.
- Whether `reports/gcom_cuda/hopper/sim-vs-hw-gcom_h100.md` should link to the
  hardware reference report or embed a small hardware summary table.
- Whether anchor pass/fail thresholds are global defaults or per-probe
  metadata.
- How much of the GCoM stat schema should be required before the first smoke
  comparison.
