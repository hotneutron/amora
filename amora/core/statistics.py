"""Small deterministic statistics helpers for probe analysis."""

from __future__ import annotations

from statistics import median
from typing import Iterable


def summarize_samples(samples: Iterable[float]) -> dict[str, float | int]:
    """Return robust summary statistics for a sequence of numeric samples."""

    values = [float(sample) for sample in samples]
    if not values:
        raise ValueError("cannot summarize an empty sample set")

    med = median(values)
    deviations = [abs(value - med) for value in values]
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "median": med,
        "mad": median(deviations),
    }


def slope_per_operation(
    total_cycles: float,
    baseline_cycles: float,
    operation_count: int,
) -> float:
    """Compute cycles per operation after subtracting a measured baseline."""

    if operation_count <= 0:
        raise ValueError("operation_count must be positive")
    adjusted = float(total_cycles) - float(baseline_cycles)
    if adjusted < 0:
        raise ValueError("baseline_cycles cannot exceed total_cycles")
    return adjusted / operation_count


def detect_periodic_peaks(values: Iterable[float], *, min_ratio: float = 1.2) -> int | None:
    """Return the smallest repeat period for stable peaks in a stride curve.

    The helper is intentionally conservative. It only returns a period when at
    least two peaks exist and their spacing is stable.
    """

    series = [float(value) for value in values]
    if len(series) < 4:
        return None

    baseline = median(series)
    threshold = baseline * min_ratio
    peaks = [index for index, value in enumerate(series) if value >= threshold]
    if len(peaks) < 2:
        return None

    gaps = [right - left for left, right in zip(peaks, peaks[1:])]
    if not gaps:
        return None
    first_gap = gaps[0]
    if first_gap <= 0:
        return None
    if all(gap == first_gap for gap in gaps):
        return first_gap
    return None
