import pytest

from amora.core.statistics import detect_periodic_peaks, slope_per_operation, summarize_samples


def test_summarize_samples_reports_median_and_mad():
    summary = summarize_samples([10, 12, 14, 100])

    assert summary["count"] == 4
    assert summary["min"] == 10.0
    assert summary["max"] == 100.0
    assert summary["median"] == 13.0
    assert summary["mad"] == 2.0


def test_summarize_samples_rejects_empty_input():
    with pytest.raises(ValueError, match="empty sample set"):
        summarize_samples([])


def test_slope_per_operation_subtracts_baseline():
    assert slope_per_operation(120, 20, 10) == 10.0


def test_slope_per_operation_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="positive"):
        slope_per_operation(10, 0, 0)
    with pytest.raises(ValueError, match="baseline"):
        slope_per_operation(10, 20, 1)


def test_detect_periodic_peaks_returns_stable_spacing():
    period = detect_periodic_peaks([10, 30, 10, 30, 10, 30], min_ratio=1.5)

    assert period == 2


def test_detect_periodic_peaks_returns_none_for_noisy_spacing():
    period = detect_periodic_peaks([10, 20, 10, 10, 20, 10, 20], min_ratio=1.5)

    assert period is None
