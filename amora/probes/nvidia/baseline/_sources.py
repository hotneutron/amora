"""Helpers for registering CUDA kernel sources alongside probe results."""

from __future__ import annotations

import hashlib
from pathlib import Path

from amora.schemas.evidence import FitStatus, UncertaintyCategory


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a CUDA source on disk."""

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def source_descriptor(path: Path) -> dict[str, object]:
    """Describe a CUDA source for the probe report."""

    return {
        "kind": "cuda_source",
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


# Ordered weakest -> strongest, used to downgrade a fit one notch.
_FIT_ORDER = [
    FitStatus.UNSUPPORTED,
    FitStatus.UNDERCONSTRAINED,
    FitStatus.BEHAVIORAL_ONLY,
    FitStatus.BOUNDED,
    FitStatus.CONDITIONALLY_IDENTIFIED,
    FitStatus.UNIQUELY_IDENTIFIED,
    FitStatus.DIRECT,
]


def downgrade_fit(fit: FitStatus, *, notches: int = 1) -> FitStatus:
    """Lower a fit status by ``notches`` (never below UNDERCONSTRAINED)."""

    try:
        idx = _FIT_ORDER.index(fit)
    except ValueError:
        return fit
    new_idx = max(1, idx - notches)  # 1 == UNDERCONSTRAINED floor
    return _FIT_ORDER[new_idx]


def soften_uncertainty(uncertainty: UncertaintyCategory) -> UncertaintyCategory:
    """Relax a stable scalar to a bounded range after a SASS downgrade."""

    if uncertainty == UncertaintyCategory.STABLE_SCALAR:
        return UncertaintyCategory.BOUNDED_RANGE
    return uncertainty


def apply_sass_gating(sass, expectation, fit: FitStatus, uncertainty: UncertaintyCategory):
    """Apply SASS reject/downgrade/pass to a (fit, uncertainty) pair.

    Returns ``(decision, fit, uncertainty, downgrade_reason)``. ``decision`` is
    one of 'pass' / 'downgrade' / 'reject'. When ``sass`` is None (no
    disassembler) the decision is 'pass' unchanged. Probes handle 'reject' by
    returning a structured unsupported result.
    """

    # Imported lazily to avoid a hard dependency when SASS tooling is absent.
    from amora.backends.nvidia.sass import gate_decision

    if sass is None:
        return "pass", fit, uncertainty, None
    decision = gate_decision(sass, expectation)
    if decision == "downgrade":
        return (
            "downgrade",
            downgrade_fit(fit),
            soften_uncertainty(uncertainty),
            f"SASS validation downgrade: {sass.reason}",
        )
    return decision, fit, uncertainty, None


def collect_ncu_metrics(capabilities, source, logical_names, *, kernel_name, role="validation",
                        launch_count=8, aggregate="max", args=()):
    """Collect one or more logical NCU metrics for ``source`` (best effort).

    Resolves each logical name against the host's supported metric set, runs the
    driver once under NCU collecting the resolved counters, and returns a record
    dict ``{"role", "values": {logical: number}, "resolved": {logical: metric}}``
    or ``None`` when NCU / no metric is available (timing-only fallback).

    ``aggregate`` chooses how to fold multiple profiled launch rows per metric:
    "max" (default) or "sum".
    """

    # Lazy imports keep probes importable when NCU tooling is absent.
    from amora.backends.nvidia.metrics import MetricResolver
    from amora.backends.nvidia.ncu_run import NcuUnavailable, run_kernel_profiled

    resolver = MetricResolver(supported_metrics=getattr(capabilities, "ncu_metrics", frozenset()))
    resolved: dict[str, str] = {}
    for logical in logical_names:
        res = resolver.resolve(logical)
        if res.available and res.selected_name:
            resolved[logical] = res.selected_name
    if not resolved:
        return None
    try:
        ncu = run_kernel_profiled(
            source,
            capabilities=capabilities,
            metrics=tuple(resolved.values()),
            kernel_name=kernel_name,
            launch_count=launch_count,
            args=args,
        )
    except NcuUnavailable:
        return None

    def _fold(metric: str):
        vals = []
        for row in ncu.raw_rows:
            raw = (row.get(metric) or "").strip().replace(",", "")
            try:
                vals.append(float(raw))
            except ValueError:
                continue
        if not vals:
            v = ncu.metrics.get(metric)
            return v
        return sum(vals) if aggregate == "sum" else max(vals)

    values = {logical: _fold(metric) for logical, metric in resolved.items()}
    return {"role": role, "values": values, "resolved": resolved, "launches_profiled": len(ncu.raw_rows)}


_STALL_LOGICALS = (
    "stall_long_scoreboard",
    "stall_short_scoreboard",
    "stall_wait",
    "stall_barrier",
    "stall_lg_throttle",
    "stall_mio_throttle",
    "stall_math_pipe_throttle",
    "stall_not_selected",
)


def collect_stall_attribution(capabilities, source, *, kernel_name, launch_count=4, args=()):
    """Collect NCU warp-issue stall metrics and attribute the dominant reason.

    These ``smsp__warp_issue_stalled_*_per_warp_active`` metrics are the
    CUPTI/PC-sampling-derived stall reasons NCU surfaces. Returns a record
    ``{"role":"stall_attribution","dominant_stall","stalls":{reason:value}}`` or
    ``None`` when NCU/the metrics are unavailable. Best effort; never raises.
    """

    record = collect_ncu_metrics(
        capabilities, source, _STALL_LOGICALS,
        kernel_name=kernel_name, role="stall_attribution",
        launch_count=launch_count, aggregate="max", args=args,
    )
    if record is None:
        return None
    stalls = {k.replace("stall_", ""): v for k, v in record["values"].items() if v is not None}
    if not stalls:
        return None
    dominant = max(stalls, key=lambda k: stalls[k])
    return {
        "role": "stall_attribution",
        "dominant_stall": dominant,
        "stalls": stalls,
        "resolved": record["resolved"],
    }


def feature_gate(capabilities, probe_id, feature, *, tool_context):
    """Return an unsupported ProbeResult list when ``feature`` is absent, else None.

    Uses the curated published-facts table. When the device is in the table and
    lacks the feature, the probe is gated out cleanly (e.g. TMA on pre-Hopper).
    When the device is unknown the gate *allows* the probe (returns None) so an
    unseen GPU relies on the probe's own evidence rather than being mis-gated.
    """

    from amora.backends.nvidia.archinfo import facts_for_capabilities, supports_feature
    from amora.schemas.results import ProbeResult

    supported = supports_feature(capabilities, feature)
    if supported is False:
        facts = facts_for_capabilities(capabilities)
        arch = f"{facts.family}/{facts.model}" if facts else "this device"
        return [
            ProbeResult.unsupported(
                probe_id,
                f"{feature} is not available on {arch} (compute capability "
                f"{facts.compute_capability[0]}.{facts.compute_capability[1]})"
                if facts else f"{feature} is not available on this device",
                tool_context=tool_context,
                raw_values={"required_feature": feature,
                            "arch_facts": facts.to_dict() if facts else None},
            )
        ]
    return None
