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

from dataclasses import dataclass
from pathlib import Path
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
_NVIDIA_BASELINE = Path(__file__).resolve().parents[3] / "nvidia" / "baseline"


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


def _tool_context(caps: GcomCapabilities) -> ToolContext:
    return ToolContext(tools=caps.to_dict())


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
        if policy.hw_denominator is not None:
            denom = _hw_denominator(probe_id, policy.hw_denominator, ctx)
            if denom is None:
                return [_unsupported(
                    probe_id,
                    f"HW baseline denominator '{policy.hw_denominator}' required "
                    "(pass --hw-baseline)", caps, state=mm.MISSING_STAT,
                )]

        # 4. Actual trace+simulate is GPU/sim-gated and lands with the execution
        #    path; until then, report a structured pending state rather than fake
        #    a value. (Phase 2 wires the derivation; Phase 1/2 scaffold here.)
        src = _probe_source(probe_id)
        if src is None:
            return [_unsupported(
                probe_id, "no kernel source for this probe", caps,
                state=mm.NOT_APPLICABLE,
            )]
        return [_unsupported(
            probe_id,
            "trace+simulate execution not available in this environment",
            caps, state=mm.MISSING_STAT,
            extra={"category": policy.category, "derivation": policy.derivation},
        )]

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
    results: list[ProbeResult] = []
    for probe_id in PLANNED_PROBES:
        results.extend(run_probe(probe_id, capabilities, context))
    return results
