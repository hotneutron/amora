"""gcom_cuda baseline probe registry and table-driven runner factory.

The probe inventory is the single source of truth from the NVIDIA registry
(`PLANNED_PROBES = tuple(nvidia.baseline.PROBES)`); this module never re-lists
probe IDs. Each probe's simulated value is derived generically from GCoM stats
using the policy in :mod:`metrics_map` and a hardware denominator taken from the
real NVIDIA ``ProbeResult`` (supplied via ``--hw-baseline``).

When the simulator/trace is unavailable, or a probe's policy is `unavailable`,
or the required HW denominator is missing, the probe emits a structured
``ProbeResult.unsupported`` carrying the specific state — never a fabricated
number.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Callable

from amora.backends.gcom_cuda import config as cfg
from amora.backends.gcom_cuda.gcom import GcomCapabilities, discover_capabilities
from amora.probes.gcom_cuda.baseline import metrics_map as mm
from amora.probes.nvidia.baseline import PROBES as NVIDIA_PROBES
from amora.schemas.evidence import EvidenceTier, FitStatus, UncertaintyCategory
from amora.schemas.results import (
    BackendInterpretation,
    NormalizedMeasurement,
    ProbeIdentity,
    ProbeResult,
    RawObservation,
    SimulatorEstimate,
    ToolContext,
)

# Single source of truth: inventory comes from the NVIDIA registry.
PLANNED_PROBES: tuple[str, ...] = tuple(NVIDIA_PROBES)

# probe_id -> source .cu (probe_id "group.stem" -> baseline/group/stem.cu).
# This file: amora/probes/gcom_cuda/baseline/__init__.py
#   parents[0]=baseline parents[1]=gcom_cuda parents[2]=probes
_NVIDIA_BASELINE = Path(__file__).resolve().parents[2] / "nvidia" / "baseline"


def _probe_source(probe_id: str) -> Path | None:
    if "." not in probe_id:
        return None
    group, stem = probe_id.split(".", 1)
    src = _NVIDIA_BASELINE / group / f"{stem}.cu"
    return src if src.exists() else None


@dataclass(frozen=True)
class RunContext:
    """Inputs a gcom_cuda run needs beyond capabilities."""

    sku: str = cfg.DEFAULT_SKU
    hw_baseline: dict[str, Any] | None = None  # {probe_id: hw ProbeResult dict}
    sim_timeout: int = 1200  # per-probe sim wall-clock cap (s); latency-bound
    # probes that exceed it degrade to missing_stat instead of blocking the run.
    trace_timeout: int = 1800  # per-probe trace (instrumented kernel) cap (s).
    max_workers: int = 8  # probes run concurrently (GCoM is CPU sim; each sim is
    # OMP-threaded, so keep workers x OMP under the core count).
    omp_threads: int | None = None  # OMP threads per sim; None -> auto (cores //
    # max_workers, clamped 1..8) to avoid oversubscription when parallel.


def _tool_context(caps: GcomCapabilities) -> ToolContext:
    return ToolContext(tools=caps.to_dict())


def _resolve_omp_threads(ctx: RunContext) -> int:
    """Per-sim OMP threads: explicit, else cores//max_workers clamped to 1..8."""

    if ctx.omp_threads:
        return max(1, ctx.omp_threads)
    import os as _os

    cores = _os.cpu_count() or 8
    workers = max(1, ctx.max_workers)
    return max(1, min(8, cores // workers))


def _unsupported(probe_id: str, reason: str, caps: GcomCapabilities, *, state: str,
                 extra: dict[str, Any] | None = None) -> ProbeResult:
    raw = {"gcom_state": state}
    if extra:
        raw.update(extra)
    return ProbeResult.unsupported(
        probe_id, reason, backend="gcom_cuda", family="baseline",
        tool_context=_tool_context(caps), raw_values=raw,
    )


def _hw_denominator(probe_id: str, field_name: str, ctx: RunContext) -> float | None:
    """Pull a denominator from the real-HW ProbeResult raw values/metrics."""

    if ctx.hw_baseline is None:
        return None
    hw = ctx.hw_baseline.get(probe_id)
    if not hw:
        return None
    raw = hw.get("raw_observation", {}) or {}
    for bucket in (raw.get("values", {}), raw.get("metrics", {})):
        if isinstance(bucket, dict) and field_name in bucket:
            try:
                return float(bucket[field_name])
            except (TypeError, ValueError):
                return None
    return None


def _derive_value(policy: mm.ProbePolicy, stats: dict[str, float], denom: float | None,
                  profile: cfg.SkuProfile) -> tuple[float | None, str]:
    """Derive the simulated scalar from GCoM stats per the policy derivation kind.

    Returns (value, unit). Returns (None, "") when the derivation cannot run.
    """

    cycles = stats.get("gpu_sim_cycle")
    if cycles is None:
        return None, ""
    if policy.derivation == mm.PER_OP and denom:
        return cycles / denom, "cycles"
    if policy.derivation == mm.THROUGHPUT and denom:
        return denom / cycles, "ops/cycle"
    if policy.derivation == mm.BANDWIDTH:
        # Numerator is simulated DRAM bytes (reads+writes x atom), not a HW field.
        reads = stats.get("gpgpu_n_dram_reads", 0.0)
        writes = stats.get("gpgpu_n_dram_writes", 0.0)
        sim_bytes = (reads + writes) * cfg.DRAM_ATOM_BYTES
        seconds = cycles / profile.core_clock_hz
        if sim_bytes > 0 and seconds > 0:
            return (sim_bytes / seconds) / 1e9, "GB/s"
    return None, ""


def _result(probe_id: str, value: float, unit: str, policy: mm.ProbePolicy,
            caps: GcomCapabilities, stats: dict[str, float], hw: dict | None,
            log_path: Path | None) -> ProbeResult:
    concept = ((hw or {}).get("backend_interpretation") or {}).get("concept") or probe_id
    derived_metrics = {
        "gpu_sim_cycle": stats.get("gpu_sim_cycle"),
        "gpu_tot_sim_insn": stats.get("gpu_tot_sim_insn"),
        "gpu_ipc": stats.get("gpu_ipc"),
    }
    # GCoM-derived logical counters (for the counter-comparison layer).
    from amora.backends.gcom_cuda.runner import (
        derive_logical_metrics,
        extract_stall_reason_histogram,
    )

    logical = derive_logical_metrics(stats)
    stall_hist = extract_stall_reason_histogram(stats)
    if stall_hist is not None:
        derived_metrics["gcom_stall_reason_schema"] = stall_hist["schema"]
        derived_metrics["gcom_stall_reason_denominator"] = stall_hist.get("denominator")
        derived_metrics["gcom_stall_reason_denominator_key"] = stall_hist.get("denominator_key")
        derived_metrics["gcom_stall_reason_histogram"] = stall_hist["reasons"]
        derived_metrics["gcom_stall_reason_complete"] = stall_hist["complete"]
        if not stall_hist["complete"]:
            derived_metrics["gcom_stall_reason_missing"] = stall_hist["missing_reasons"]
    return ProbeResult(
        identity=ProbeIdentity(probe_id=probe_id, backend="gcom_cuda", family="baseline"),
        tool_context=_tool_context(caps),
        raw_observation=RawObservation(
            evidence_tier=EvidenceTier.SIMULATOR_TRACE,
            values={"gcom_state": policy.category, "sim_log": str(log_path) if log_path else None,
                    "stat_count": len(stats)},
            metrics=derived_metrics,
            units={"gpu_sim_cycle": "cycles"},
            source="gcom_cuda:accel-sim.out",
        ),
        normalized_measurement=NormalizedMeasurement(
            name=concept, value=value, unit=unit,
            fit_status=FitStatus.DIRECT if policy.fidelity == "direct" else FitStatus.BOUNDED,
            uncertainty=UncertaintyCategory.STABLE_SCALAR,
            assumptions=[policy.limitations] if policy.limitations else [],
        ),
        backend_interpretation=BackendInterpretation(
            concept=concept,
            interpretation={"derivation": policy.derivation, "architecture_scope": policy.architecture_scope},
            metric_resolver=logical,
        ),
        simulator_estimate=SimulatorEstimate(
            parameter=concept, value=value, unit=unit,
            evidence_tier=EvidenceTier.SIMULATOR_TRACE,
            fit_status=FitStatus.DIRECT if policy.fidelity == "direct" else FitStatus.BOUNDED,
            uncertainty=UncertaintyCategory.STABLE_SCALAR,
            mapping_contract=f"GCoM {policy.derivation} from gpu_sim_cycle (fidelity={policy.fidelity})",
        ),
    )


def _make_runner(probe_id: str) -> Callable[[GcomCapabilities, RunContext], list[ProbeResult]]:
    policy = mm.METRICS_MAP[probe_id]

    def run(caps: GcomCapabilities, ctx: RunContext) -> list[ProbeResult]:
        # 1. Policy-level unavailable probes short-circuit with their state.
        if policy.category == mm.UNAVAILABLE:
            return [_unsupported(
                probe_id, policy.limitations or "not comparable in gcom_cuda",
                caps, state=policy.state or mm.UNSUPPORTED,
            )]

        # 2. Need a real simulator to derive any value.
        if not caps.simulator_built:
            return [_unsupported(
                probe_id, "simulator not built; run build first", caps,
                state=mm.MISSING_STAT,
            )]

        # 3. Comparable/approximate need a HW denominator (single source of truth).
        denom = None
        if policy.hw_denominator is not None:
            denom = _hw_denominator(probe_id, policy.hw_denominator, ctx)
            if denom is None:
                return [_unsupported(
                    probe_id,
                    f"HW baseline denominator '{policy.hw_denominator}' required "
                    "(pass --hw-baseline)", caps, state=mm.MISSING_STAT,
                )]

        src = _probe_source(probe_id)
        if src is None:
            return [_unsupported(
                probe_id, "no kernel source for this probe", caps,
                state=mm.NOT_APPLICABLE,
            )]

        # 4. Single-trace derivations (per_op/throughput/bandwidth) run now.
        #    Sweep/differential need multi-trace reduction (later phase) -> honest
        #    missing_stat rather than a single-point approximation.
        if policy.derivation not in (mm.PER_OP, mm.THROUGHPUT, mm.BANDWIDTH):
            return [_unsupported(
                probe_id,
                f"derivation '{policy.derivation}' needs multi-trace reduction "
                "(not yet wired)", caps, state=mm.MISSING_STAT,
                extra={"category": policy.category},
            )]

        # Lazy import: trace/runner pull in GPU/sim-only machinery.
        from amora.backends.gcom_cuda import runner as gcom_runner
        from amora.backends.gcom_cuda import trace as gcom_trace
        from amora.backends.gcom_cuda.trace import TraceError
        from amora.backends.gcom_cuda.runner import SimulateError

        profile = cfg.get_sku_profile(ctx.sku)
        run_id = f"{probe_id.replace('.', '_')}_{int(time.time())}"
        out_dir = cfg.run_output_dir(profile, run_id)
        try:
            trace_dir = gcom_trace.trace_probe(probe_id, src, out_dir,
                                               timeout=ctx.trace_timeout)
            sim = gcom_runner.simulate(profile, trace_dir, log_path=out_dir / "gcom_sim.log",
                                       timeout=ctx.sim_timeout,
                                       omp_threads=_resolve_omp_threads(ctx))
        except (TraceError, SimulateError) as exc:
            return [_unsupported(probe_id, f"trace/sim failed: {exc}", caps,
                                 state=mm.MISSING_STAT)]
        if not sim.core_present():
            return [_unsupported(
                probe_id, f"simulator emitted no {gcom_runner.REQUIRED_CORE_STAT}",
                caps, state=mm.MISSING_STAT)]

        value, unit = _derive_value(policy, sim.stats, denom, profile)
        if value is None:
            return [_unsupported(probe_id, "derivation produced no value", caps,
                                 state=mm.MISSING_STAT)]
        hw = (ctx.hw_baseline or {}).get(probe_id)
        return [_result(probe_id, value, unit, policy, caps, sim.stats, hw,
                        out_dir / "gcom_sim.log")]

    return run


PROBES: dict[str, Callable[[GcomCapabilities, RunContext], list[ProbeResult]]] = {
    probe_id: _make_runner(probe_id) for probe_id in PLANNED_PROBES
}


def list_probes() -> list[dict[str, object]]:
    out = []
    for probe_id in PLANNED_PROBES:
        policy = mm.METRICS_MAP[probe_id]
        out.append({
            "probe_id": probe_id,
            "category": policy.category,
            "state": policy.state,
            "derivation": policy.derivation,
            "fidelity": policy.fidelity,
            "architecture_scope": policy.architecture_scope,
        })
    return out


def run_probe(probe_id: str, caps: GcomCapabilities | None = None,
              ctx: RunContext | None = None) -> list[ProbeResult]:
    capabilities = caps or discover_capabilities()
    context = ctx or RunContext()
    if probe_id in PROBES:
        return PROBES[probe_id](capabilities, context)
    return [_unsupported(probe_id, "probe not in gcom_cuda registry", capabilities,
                         state=mm.NOT_APPLICABLE)]


def run_all(caps: GcomCapabilities | None = None,
            ctx: RunContext | None = None) -> list[ProbeResult]:
    capabilities = caps or discover_capabilities()
    context = ctx or RunContext()

    # GCoM is a CPU simulator with no shared state between probes, so run them
    # concurrently. Results are reassembled in canonical probe order.
    workers = max(1, min(context.max_workers, len(PLANNED_PROBES)))
    if workers == 1:
        results: list[ProbeResult] = []
        for probe_id in PLANNED_PROBES:
            results.extend(run_probe(probe_id, capabilities, context))
        return results

    by_id: dict[str, list[ProbeResult]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(run_probe, probe_id, capabilities, context): probe_id
            for probe_id in PLANNED_PROBES
        }
        for future, probe_id in ((f, futures[f]) for f in futures):
            by_id[probe_id] = future.result()
    ordered: list[ProbeResult] = []
    for probe_id in PLANNED_PROBES:
        ordered.extend(by_id.get(probe_id, []))
    return ordered
