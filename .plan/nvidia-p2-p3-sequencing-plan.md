# NVIDIA P2 / P3 Sequencing Plan

## Scope

This is a sequencing and dependency plan, not a fresh methodology. The full P2
and P3 probe specifications already exist in
`.plan/nvidia-p2-kernel-methodology.md` and
`.plan/nvidia-p3-kernel-methodology.md`. This document maps those 13 probes to a
concrete implementation order, records which depend on SASS validation and NCU
integration, and folds them into the existing `baseline/` tree and report
grouping.

## Status of prerequisites

P2/P3 lean far harder on direct counters than P0/P1 did. Two infrastructure
items gate most of the work:

- **NCU integration** (`.plan/nvidia-ncu-integration-plan.md`) — required for
  memory request/sector metrics, L2 hit rates, DRAM bytes, tensor-pipe
  utilization, partition/slice metrics. **Hard dependency** for the majority of
  P2/P3.
- **SASS validation** (`.plan/nvidia-sass-validation-plan.md`) — required to
  confirm exact MMA opcodes, async-copy/TMA instructions, fence scope, and
  memory access widths. **Hard dependency** for tensor-core, TMA, and fence
  probes.

Probes that can produce a useful timing-only first cut (then upgrade once NCU
lands) are marked "timing-first OK" below.

## Probe inventory and dependencies

### P2 (`.plan/nvidia-p2-kernel-methodology.md`)

| probe_id | primary evidence | NCU | SASS | timing-first OK |
| --- | --- | :-: | :-: | :-: |
| `memory_pipeline.lane_patterns` | request/sector counters | required | recommended | partial |
| `memory_pipeline.outstanding_requests` | stall/throughput + saturation | helpful | recommended | yes |
| `memory_pipeline.analyze` | merges the two | — | — | — |
| `l2_cache.pointer_chase` | L2 hit-rate + latency curve | helpful | recommended | yes |
| `global_memory.streaming` | DRAM bytes/throughput | helpful | optional | yes |
| `global_memory.partition_sweep` | partition/slice metrics + timing | required | optional | partial |
| `global_memory.row_policy_sweep` | DRAM metrics + timing | required | optional | partial |
| `global_memory.analyze` | merges streaming/partition/row | — | — | — |
| `tensor_core.mma_latency` | dependent MMA timing | helpful | required | yes |
| `tensor_core.mma_throughput` | tensor-pipe util + timing | required | required | partial |
| `synchronization.barrier_latency` | repeated-barrier timing | helpful | recommended | yes |
| `synchronization.fence_latency` | fence-sequence timing | helpful | required | partial |

### P3 (`.plan/nvidia-p3-kernel-methodology.md`)

| probe_id | primary evidence | NCU | SASS | timing-first OK |
| --- | --- | :-: | :-: | :-: |
| `tma_copy.async_copy_latency` | issue/wait/use timing | helpful | required | partial |
| `tma_copy.tma_transfer_sweep` | TMA/async-copy bytes + timing | required | required | partial |
| `tma_copy.analyze` | merges the two | — | — | — |
| `interconnect.address_mapping` | throughput/latency vs address | required | optional | partial |
| `interconnect.injection_rate` | multi-SM throughput saturation | required | optional | yes |
| `interconnect.analyze` | merges the two | — | — | — |

## Recommended implementation order

The order maximizes early value and respects the two infrastructure gates.

**Phase A — timing-first P2, before NCU (parallel with infra work)**

1. `synchronization.barrier_latency` — pure repeated-barrier timing; no counter
   needed for a first cut; lowest risk P2 probe.
2. `global_memory.streaming` — sustained DRAM bandwidth from timing + bytes
   moved; upgrade to direct DRAM counters when NCU lands.
3. `l2_cache.pointer_chase` — working-set latency curve (reuses the L1
   pointer-chase machinery at larger footprints).
4. `memory_pipeline.outstanding_requests` — independent-memory saturation curve.

**Phase B — after NCU integration**

5. `memory_pipeline.lane_patterns` + `memory_pipeline.analyze` (request/sector
   counters are the whole point).
6. `global_memory.partition_sweep`, `global_memory.row_policy_sweep`,
   `global_memory.analyze`.
7. Upgrade Phase-A probes to counter-primary where their contract is direct.

**Phase C — after SASS validation (tensor / sync exactness)**

8. `tensor_core.mma_latency` then `tensor_core.mma_throughput` (need verified
   MMA opcodes; throughput also needs tensor-pipe counters from Phase B).
9. `synchronization.fence_latency` (needs verified fence scope/instructions).

**Phase D — P3, after Phases B+C**

10. `tma_copy.async_copy_latency`, `tma_copy.tma_transfer_sweep`,
    `tma_copy.analyze` (need verified async-copy/TMA SASS + memory counters).
11. `interconnect.injection_rate` (timing-first OK, but only meaningful atop the
    P2 memory baselines it must rule out).
12. `interconnect.address_mapping`, `interconnect.analyze` (candidate-set
    generators; depend on P2 partition baselines).

## Codebase integration

- **Folding**: follow the P1 pattern — new groups under
  `amora/probes/nvidia/baseline/` (`memory_pipeline/`, `l2_cache/`,
  `global_memory/`, `tensor_core/`, `synchronization/`, `tma_copy/`,
  `interconnect/`), each with `.cu` + `.py` + `analyze.py` + `__init__.py`,
  registered in `baseline/__init__.py`.
- **Report grouping**: `amora/reports/probe_groups.py` already pre-registers all
  13 P2/P3 probe IDs into the 5 thematic groups, so no report change is needed
  as they land — the SUMMARY tables populate automatically.
- **Capability gating**: counter/SASS-dependent probes return structured
  `unsupported` (with registered source) when NCU/SASS are unavailable, exactly
  like the current kernel-bound probes without a GPU.
- **Fit-status discipline**: P2/P3 default to `bounded` / `behavioral_only` /
  `underconstrained` per the methodologies; exact scalars only when counters
  (and SASS where relevant) uniquely identify them. The em-dash rendering for
  null scalars is already in the report generator.

## Per-phase acceptance

- Phase A: four P2 probes runnable on H100 with timing evidence and registered
  sources; SUMMARY shows them in the Global Memory / On-chip groups.
- Phase B: memory-pipeline and global-memory probes emit `direct_counter`
  evidence with metric-resolver records; analyzers merge cleanly.
- Phase C: tensor-core probes report per-shape latency/throughput with SASS
  confirmation; fence probe reports scope-specific costs.
- Phase D: TMA probes report transfer descriptors with verified SASS;
  interconnect probes emit candidate sets / bounds, never exact hashes.

## Risks

- Architecture coverage: TMA/async-copy and some tensor shapes exist only on
  Hopper+; probes must capability-gate by compute capability and downgrade to
  `unsupported` on older arches (the V100 baseline in `reports/nvidia/volta/`
  already exercises the "older arch, fewer features" path).
- Counter availability varies by arch/driver; the metric resolver's
  candidate-list approach absorbs most of this.
- Over-attribution: partition/row-policy/interconnect probes must preserve
  candidate sets and avoid claiming proprietary mappings — enforced by the
  methodology's scalar policies.
