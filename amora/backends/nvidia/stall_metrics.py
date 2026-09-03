"""Shared helpers for Nsight Compute warp-issue stall metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


STALL_REASONS = (
    "selected",
    "not_selected",
    "dispatch_stall",
    "warpgroup_arrive",
    "long_scoreboard",
    "short_scoreboard",
    "barrier",
    "wait",
    "mio_throttle",
    "math_pipe_throttle",
    "mma",
    "no_instructions",
    "imc_miss",
    "sleeping",
    "branch_resolving",
    "membar",
    "drain",
    "lg_throttle",
    "tex_throttle",
    "misc",
)

STALL_LOGICALS = tuple(f"stall_{reason}" for reason in STALL_REASONS)

_REASON_ALIASES = {
    "no_instructions": ("no_instructions", "no_instruction"),
    # Hopper NCU catalogs expose this reason as ``gmma`` while the metric
    # description and AMORA's cross-backend taxonomy call it warpgroup_arrive.
    "warpgroup_arrive": ("warpgroup_arrive", "gmma"),
}


@dataclass(frozen=True)
class StallMetricFamily:
    """One coherent Nsight Compute stall-metric family."""

    key: str
    unit: str
    template: str

    def candidates_for(self, reason: str) -> tuple[str, ...]:
        aliases = _REASON_ALIASES.get(reason, (reason,))
        return tuple(self.template.format(reason=alias) for alias in aliases)


STALL_METRIC_FAMILIES = (
    StallMetricFamily(
        key="warp_issue_stalled_per_warp_active",
        unit="pct",
        template="smsp__warp_issue_stalled_{reason}_per_warp_active.pct",
    ),
    StallMetricFamily(
        key="average_warps_issue_stalled_per_issue_active",
        unit="ratio",
        template="smsp__average_warps_issue_stalled_{reason}_per_issue_active.ratio",
    ),
)


STALL_CYCLE_COUNTER_CANDIDATES = {
    "active_warp_cycles": ("smsp__warps_active.sum",),
    "active_smsp_cycles": ("smsp__cycles_active.sum",),
    "eligible_warp_cycles": ("smsp__warps_eligible.sum",),
}
STALL_CYCLE_REPLAY_METRIC = "profiler__replayer_passes"


@dataclass(frozen=True)
class StallCycleMetricSelection:
    """One per-warp-active issue-state family plus cycle denominators."""

    family: str | None
    input_unit: str | None
    reason_to_metric: dict[str, str]
    cycle_metrics: dict[str, str]
    replay_metric: str | None
    missing_reasons: tuple[str, ...]
    missing_cycle_metrics: tuple[str, ...]

    @property
    def available(self) -> bool:
        return bool(self.reason_to_metric) and not self.missing_cycle_metrics

    @property
    def metrics(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.cycle_metrics.values(),
                    *self.reason_to_metric.values(),
                    *((self.replay_metric,) if self.replay_metric else ()),
                )
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "input_unit": self.input_unit,
            "reason_to_metric": dict(self.reason_to_metric),
            "cycle_metrics": dict(self.cycle_metrics),
            "replay_metric": self.replay_metric,
            "missing_reasons": list(self.missing_reasons),
            "missing_cycle_metrics": list(self.missing_cycle_metrics),
            "available": self.available,
        }


def resolve_stall_cycle_metrics(
    supported_metrics: frozenset[str],
) -> StallCycleMetricSelection:
    """Resolve metrics suitable for absolute replay warp-cycle derivation.

    Only ``warp_issue_stalled_*_per_warp_active`` is dimensionally eligible.
    The ``.ratio`` suffix is preferred; ``.pct`` is a coherent fallback. The
    differently normalized ``average_warps_*_per_issue_active`` family is never
    selected by this contract.
    """

    cycle_metrics = {}
    missing_cycle_metrics = []
    for logical, candidates in STALL_CYCLE_COUNTER_CANDIDATES.items():
        selected = next(
            (
                candidate
                for candidate in candidates
                if metric_supported(candidate, supported_metrics)
            ),
            None,
        )
        if selected is None:
            missing_cycle_metrics.append(logical)
        else:
            cycle_metrics[logical] = selected

    reason_to_metric: dict[str, str] = {}
    selected_unit = None
    for unit in ("ratio", "pct"):
        resolved = {}
        for reason in STALL_REASONS:
            aliases = _REASON_ALIASES.get(reason, (reason,))
            for alias in aliases:
                candidate = (
                    f"smsp__warp_issue_stalled_{alias}_per_warp_active.{unit}"
                )
                if metric_supported(candidate, supported_metrics):
                    resolved[reason] = candidate
                    break
        if resolved:
            reason_to_metric = resolved
            selected_unit = unit
            break

    missing_reasons = tuple(
        reason for reason in STALL_REASONS if reason not in reason_to_metric
    )
    return StallCycleMetricSelection(
        family=(
            "warp_issue_stalled_per_warp_active"
            if reason_to_metric
            else None
        ),
        input_unit=selected_unit,
        reason_to_metric=reason_to_metric,
        cycle_metrics=cycle_metrics,
        replay_metric=(
            STALL_CYCLE_REPLAY_METRIC
            if metric_supported(STALL_CYCLE_REPLAY_METRIC, supported_metrics)
            else None
        ),
        missing_reasons=missing_reasons,
        missing_cycle_metrics=tuple(missing_cycle_metrics),
    )


STALL_LOGICAL_CANDIDATES = {
    f"stall_{reason}": tuple(
        candidate
        for family in STALL_METRIC_FAMILIES
        for candidate in family.candidates_for(reason)
    )
    for reason in STALL_REASONS
}


@dataclass(frozen=True)
class StallMetricSelection:
    """Resolved stall metrics constrained to a single metric family."""

    family: str | None
    unit: str | None
    logical_to_metric: dict[str, str]
    reason_to_metric: dict[str, str]
    missing_reasons: tuple[str, ...]
    candidates_by_reason: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "unit": self.unit,
            "logical_to_metric": dict(self.logical_to_metric),
            "reason_to_metric": dict(self.reason_to_metric),
            "missing_reasons": list(self.missing_reasons),
            "candidates_by_reason": {
                reason: list(candidates)
                for reason, candidates in self.candidates_by_reason.items()
            },
        }


def metric_supported(candidate: str, supported_metrics: frozenset[str]) -> bool:
    """Return whether ``candidate`` is present in an NCU metric catalog."""

    if candidate in supported_metrics:
        return True
    base = candidate.rsplit(".", 1)[0]
    return base in supported_metrics


def stall_reason_for_metric(metric: str) -> str | None:
    """Return AMORA's logical reason for an aggregate NCU metric name."""

    metric_base = metric.rsplit(".", 1)[0]
    for family in STALL_METRIC_FAMILIES:
        for reason in STALL_REASONS:
            for candidate in family.candidates_for(reason):
                if metric == candidate or metric_base == candidate.rsplit(".", 1)[0]:
                    return reason
    return None


def resolve_stall_metric_family(
    supported_metrics: frozenset[str],
    *,
    preferred_family: str = "warp_issue_stalled_per_warp_active",
) -> StallMetricSelection:
    """Resolve stall logicals without mixing NCU metric families.

    The per-warp-active family is preferred when coverage ties because it keeps
    the historical percentage unit. If it is unavailable, the ratio family is
    used as a coherent fallback instead of mixing units reason-by-reason.
    """

    candidates_by_reason = {
        reason: STALL_LOGICAL_CANDIDATES[f"stall_{reason}"]
        for reason in STALL_REASONS
    }
    best_family: StallMetricFamily | None = None
    best_reason_to_metric: dict[str, str] = {}
    best_score = -1
    for index, family in enumerate(STALL_METRIC_FAMILIES):
        resolved: dict[str, str] = {}
        for reason in STALL_REASONS:
            for candidate in family.candidates_for(reason):
                if metric_supported(candidate, supported_metrics):
                    resolved[reason] = candidate
                    break
        score = len(resolved)
        tie_break = (
            family.key == preferred_family,
            -index,
        )
        best_tie_break = (
            best_family is not None and best_family.key == preferred_family,
            -STALL_METRIC_FAMILIES.index(best_family) if best_family else 0,
        )
        if score > best_score or (score == best_score and tie_break > best_tie_break):
            best_family = family
            best_reason_to_metric = resolved
            best_score = score

    if best_family is None or best_score <= 0:
        return StallMetricSelection(
            family=None,
            unit=None,
            logical_to_metric={},
            reason_to_metric={},
            missing_reasons=STALL_REASONS,
            candidates_by_reason=candidates_by_reason,
        )

    logical_to_metric = {
        f"stall_{reason}": metric
        for reason, metric in best_reason_to_metric.items()
    }
    missing = tuple(reason for reason in STALL_REASONS if reason not in best_reason_to_metric)
    return StallMetricSelection(
        family=best_family.key,
        unit=best_family.unit,
        logical_to_metric=logical_to_metric,
        reason_to_metric=best_reason_to_metric,
        missing_reasons=missing,
        candidates_by_reason=candidates_by_reason,
    )


def parse_numeric_cell(value: object) -> float | None:
    """Parse an NCU CSV numeric cell."""

    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text.replace(",", ""))
    except ValueError:
        return None


def select_stall_launch_row(
    raw_rows: list[Mapping[str, object]],
    reason_to_metric: Mapping[str, str],
    *,
    strategy: str = "median_total_stall",
    excluded_total_reasons: frozenset[str] = frozenset(),
) -> tuple[int | None, dict[str, float], float | None]:
    """Select one whole launch row and return its stall vector.

    ``median_total_stall`` keeps all reasons from the same profiled launch while
    avoiding the component-wise max fold that can assemble a vector no launch
    actually produced.
    """

    candidates: list[tuple[float, int, dict[str, float]]] = []
    for index, row in enumerate(raw_rows):
        stalls: dict[str, float] = {}
        for reason, metric in reason_to_metric.items():
            value = parse_numeric_cell(row.get(metric))
            if value is not None:
                stalls[reason] = value
        if not stalls:
            continue
        candidates.append(
            (
                sum(
                    value
                    for reason, value in stalls.items()
                    if reason not in excluded_total_reasons
                ),
                index,
                stalls,
            )
        )
    if not candidates:
        return None, {}, None
    if strategy != "median_total_stall":
        raise ValueError(f"unknown stall launch selection strategy: {strategy}")
    candidates.sort(key=lambda item: (item[0], item[1]))
    total, index, stalls = candidates[len(candidates) // 2]
    return index, stalls, total
