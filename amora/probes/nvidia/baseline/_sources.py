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
