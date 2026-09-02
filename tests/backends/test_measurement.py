"""Tests for independent timing and NCU measurement-lane orchestration."""

from __future__ import annotations

from amora.backends.nvidia import measurement
from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.cuda_event_run import (
    CudaEventProcessSample,
    CudaEventTimingResult,
    DeviceInterval,
    DeviceIntervalResult,
)
from amora.backends.nvidia.measurement import collect_command_measurement_bundle
from amora.backends.nvidia.ncu_run import NcuPcSamplingResult, NcuResult, PcStallSample


IDENTITY = {
    "ttgir_sha256": "ttgir-a",
    "cubin_sha256": "cubin-a",
}
DEVICE = {"uuid": "GPU-test", "name": "H100"}


def _timing():
    process = CudaEventProcessSample(
        process_index=0,
        samples_us=(10.0, 11.0),
        warmup_launches=2,
        launch_batch_size=4,
        cache_protocol="warm_reuse",
        clock_policy="application_clocks",
        device=DEVICE,
        subject_identity=IDENTITY,
        subject_metadata={},
        measurement_axes={},
        tool_versions={"target": "test"},
        command=("timing",),
        cwd=None,
        environment_overrides={},
        stdout="",
        stderr="",
        returncode=0,
        started_at="start",
        completed_at="end",
    )
    return CudaEventTimingResult(
        target_command=("timing",),
        process_samples=(process,),
        median_us=10.5,
        p05_us=10.05,
        p95_us=10.95,
        process_cv=0.0,
        launch_batch_size=4,
        cache_protocol="warm_reuse",
        clock_policy="application_clocks",
        device=DEVICE,
        subject_identity=IDENTITY,
        subject_metadata={},
        measurement_axes={},
        tool_versions={"target": "test"},
        provenance={},
    )


def _target_evidence(identity=IDENTITY, axes=None):
    return {
        "kind": "cuda_event_timing",
        "schema_version": 1,
        "subject_identity": dict(identity),
        "device": dict(DEVICE),
        "subject_metadata": {},
        "measurement_axes": dict(axes or {}),
        "tool_versions": {"target": "test"},
    }


def test_bundle_separates_cuda_latency_from_profiler_duration(monkeypatch):
    calls = []
    monkeypatch.setattr(measurement, "run_command_cuda_events", lambda *a, **k: _timing())

    def aggregate(*args, **kwargs):
        calls.append(("aggregate", args, kwargs))
        return NcuResult(
            metrics={
                "gpu__time_duration.sum": 50_000.0,
                "dram__bytes_read.sum": 4096.0,
                "smsp__warp_issue_stalled_selected_per_warp_active.pct": 10.0,
                "smsp__warp_issue_stalled_not_selected_per_warp_active.pct": 20.0,
                "smsp__warp_issue_stalled_wait_per_warp_active.pct": 30.0,
                "smsp__warp_issue_stalled_barrier_per_warp_active.pct": 10.0,
            },
            raw_rows=[
                {
                    "gpu__time_duration.sum": "50000",
                    "dram__bytes_read.sum": "4096",
                    "smsp__warp_issue_stalled_selected_per_warp_active.pct": "10",
                    "smsp__warp_issue_stalled_not_selected_per_warp_active.pct": "20",
                    "smsp__warp_issue_stalled_wait_per_warp_active.pct": "30",
                    "smsp__warp_issue_stalled_barrier_per_warp_active.pct": "10",
                }
            ],
            target_evidence=_target_evidence(),
            cache_control=kwargs["cache_control"],
        )

    def pc(*args, **kwargs):
        calls.append(("pc", args, kwargs))
        return NcuPcSamplingResult(
            samples=[
                PcStallSample(
                    0,
                    "k",
                    "NOP",
                    3.0,
                    {"wait": 3.0},
                    sass_joined=True,
                    sass_opcode="NOP",
                    sass_instruction="/*0000*/ NOP ;",
                )
            ],
            target_evidence=_target_evidence(),
            cache_control=kwargs["cache_control"],
        )

    monkeypatch.setattr(measurement, "run_command_profiled", aggregate)
    monkeypatch.setattr(measurement, "run_command_pc_sampling", pc)
    bundle = collect_command_measurement_bundle(
        timing_target=("timing",),
        ncu_target=("ncu-target",),
        capabilities=NvidiaCapabilities(),
        aggregate_metrics=("gpu__time_duration.sum", "dram__bytes_read.sum"),
        kernel_name="k",
        cache_control="none",
        repeats=1,
        expected_cache_protocol="warm_reuse",
    )
    payload = bundle.to_dict()

    assert payload["latency_oracle"] == "cuda_events"
    assert payload["timing"]["median_us"] == 10.5
    assert "gpu__time_duration.sum" not in payload["aggregate_counters"]["metrics"]
    assert payload["profiler_duration"]["metrics"] == {
        "gpu__time_duration.sum": 50_000.0
    }
    assert payload["aggregate_counters"]["missing_metrics"] == []
    histogram = payload["aggregate_counters"]["structural_stall_histogram"]
    assert histogram["denominator"] == 40.0
    assert histogram["reason_fractions"] == {"wait": 0.75, "barrier": 0.25}
    assert histogram["excluded_reasons"] == ["selected", "not_selected"]
    assert payload["ncu_duration_role"] == "diagnostic_profiler_perturbation_only"
    assert payload["identity_check"]["status"] == "pass"
    assert all(call[2]["cache_control"] == "none" for call in calls)
    assert calls[0][2]["environment_overrides"]["AMORA_MEASUREMENT_LANE"] == "ncu_aggregate"
    assert calls[0][2]["environment_overrides"]["AMORA_CACHE_PROTOCOL"] == "warm_reuse"
    assert calls[1][2]["environment_overrides"]["AMORA_MEASUREMENT_LANE"] == "ncu_pc_sampling"
    assert calls[1][2]["environment_overrides"]["AMORA_CACHE_PROTOCOL"] == "warm_reuse"


def test_bundle_marks_identity_drift_invalid(monkeypatch):
    monkeypatch.setattr(measurement, "run_command_cuda_events", lambda *a, **k: _timing())
    monkeypatch.setattr(
        measurement,
        "run_command_profiled",
        lambda *a, **k: NcuResult(
            metrics={"dram__bytes_read.sum": 1.0},
            raw_rows=[{"dram__bytes_read.sum": "1"}],
            target_evidence=_target_evidence(
                {"ttgir_sha256": "ttgir-a", "cubin_sha256": "cubin-other"}
            ),
        ),
    )

    bundle = collect_command_measurement_bundle(
        timing_target=("timing",),
        ncu_target=("ncu-target",),
        capabilities=NvidiaCapabilities(),
        aggregate_metrics=("dram__bytes_read.sum",),
        kernel_name="k",
        cache_control="all",
        repeats=1,
        collect_pc_sampling=False,
    )

    assert bundle.valid_for_comparison is False
    assert "aggregate_counters_identity_mismatch:cubin_sha256" in bundle.identity_check["reasons"]


def test_bundle_repeats_pc_sampling_and_pools_exact_offsets(monkeypatch, tmp_path):
    monkeypatch.setattr(measurement, "run_command_cuda_events", lambda *a, **k: _timing())
    monkeypatch.setattr(
        measurement,
        "run_command_profiled",
        lambda *a, **k: NcuResult(
            metrics={"dram__bytes_read.sum": 1.0},
            raw_rows=[{"dram__bytes_read.sum": "1"}],
            target_evidence=_target_evidence(),
        ),
    )
    report_paths = []

    def pc(*args, **kwargs):
        report_paths.append(kwargs["report_path"])
        repeat_index = int(
            kwargs["environment_overrides"]["AMORA_PC_SAMPLING_REPEAT_INDEX"]
        )
        return NcuPcSamplingResult(
            samples=[
                PcStallSample(
                    0x60,
                    "k",
                    "BRA",
                    10.0 + repeat_index,
                    {"barrier": 10.0 + repeat_index},
                    sass_joined=True,
                    sass_opcode="BRA",
                    sass_instruction="/*0060*/ BRA ;",
                )
            ],
            report_path=kwargs["report_path"],
            report_sha256=f"hash-{repeat_index}",
            target_evidence=_target_evidence(),
        )

    monkeypatch.setattr(measurement, "run_command_pc_sampling", pc)
    bundle = collect_command_measurement_bundle(
        timing_target=("timing",),
        ncu_target=("ncu-target",),
        capabilities=NvidiaCapabilities(),
        aggregate_metrics=("dram__bytes_read.sum",),
        kernel_name="k",
        cache_control="none",
        repeats=1,
        pc_sampling_repeats=3,
        pc_report_path=tmp_path / "pc_sampling.ncu-rep",
    )

    assert [path.name for path in report_paths] == [
        "pc_sampling.repeat-01.ncu-rep",
        "pc_sampling.repeat-02.ncu-rep",
        "pc_sampling.repeat-03.ncu-rep",
    ]
    assert bundle.pc_sampling["repeat_count"] == 3
    assert bundle.pc_sampling["support_count"] == 33.0
    assert bundle.pc_sampling["by_offset"] == [
        {
            "function": "k",
            "pc_offset": 0x60,
            "sass_opcode": "BRA",
            "sass_instruction": "/*0060*/ BRA ;",
            "support_count": 33.0,
            "stall_counts": {"barrier": 33.0},
            "repeat_support_counts": [10.0, 11.0, 12.0],
            "sass_joined_in_all_repeats": True,
        }
    ]
    assert bundle.identity_check["status"] == "pass"


def test_bundle_marks_runtime_axis_drift_invalid(monkeypatch):
    timing = _timing()
    timing = CudaEventTimingResult(
        **{**timing.__dict__, "measurement_axes": {"cta_load": 1}}
    )
    monkeypatch.setattr(
        measurement, "run_command_cuda_events", lambda *a, **k: timing
    )
    monkeypatch.setattr(
        measurement,
        "run_command_profiled",
        lambda *a, **k: NcuResult(
            metrics={"dram__bytes_read.sum": 1.0},
            raw_rows=[{"dram__bytes_read.sum": "1"}],
            target_evidence=_target_evidence(axes={"cta_load": 2}),
        ),
    )

    bundle = collect_command_measurement_bundle(
        timing_target=("timing",),
        ncu_target=("ncu-target",),
        capabilities=NvidiaCapabilities(),
        aggregate_metrics=("dram__bytes_read.sum",),
        kernel_name="k",
        cache_control="none",
        repeats=1,
        collect_pc_sampling=False,
        expected_measurement_axes={"cta_load": 1},
    )

    assert bundle.valid_for_comparison is False
    assert "aggregate_counters_measurement_axes_mismatch" in bundle.identity_check["reasons"]


def test_bundle_rejects_mixed_stall_metric_families():
    try:
        collect_command_measurement_bundle(
            timing_target=("timing",),
            ncu_target=("ncu",),
            capabilities=NvidiaCapabilities(),
            aggregate_metrics=(
                "smsp__warp_issue_stalled_wait_per_warp_active.pct",
                "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
            ),
            kernel_name="k",
            cache_control="none",
        )
    except ValueError as exc:
        assert "mixes NCU stall metric families" in str(exc)
    else:
        raise AssertionError("mixed stall families were accepted")


def test_bundle_rejects_multiple_aggregate_launch_rows(monkeypatch):
    monkeypatch.setattr(measurement, "run_command_cuda_events", lambda *a, **k: _timing())
    monkeypatch.setattr(
        measurement,
        "run_command_profiled",
        lambda *a, **k: NcuResult(
            metrics={"dram__bytes_read.sum": 2.0},
            raw_rows=[
                {"dram__bytes_read.sum": "1"},
                {"dram__bytes_read.sum": "2"},
            ],
            target_evidence=_target_evidence(),
        ),
    )

    try:
        collect_command_measurement_bundle(
            timing_target=("timing",),
            ncu_target=("ncu-target",),
            capabilities=NvidiaCapabilities(),
            aggregate_metrics=("dram__bytes_read.sum",),
            kernel_name="k",
            cache_control="none",
            repeats=1,
            collect_pc_sampling=False,
        )
    except ValueError as exc:
        assert "exactly one filtered launch row" in str(exc)
    else:
        raise AssertionError("multiple aggregate launch rows were accepted")


def test_bundle_records_missing_aggregate_metrics(monkeypatch):
    monkeypatch.setattr(measurement, "run_command_cuda_events", lambda *a, **k: _timing())
    monkeypatch.setattr(
        measurement,
        "run_command_profiled",
        lambda *a, **k: NcuResult(
            metrics={"dram__bytes_read.sum": 1.0},
            raw_rows=[{"dram__bytes_read.sum": "1"}],
            target_evidence=_target_evidence(),
        ),
    )

    bundle = collect_command_measurement_bundle(
        timing_target=("timing",),
        ncu_target=("ncu-target",),
        capabilities=NvidiaCapabilities(),
        aggregate_metrics=(
            "dram__bytes_read.sum",
            "lts__t_sectors_op_read.sum",
        ),
        kernel_name="k",
        cache_control="none",
        repeats=1,
        collect_pc_sampling=False,
    )

    assert bundle.aggregate_counters["missing_metrics"] == [
        "lts__t_sectors_op_read.sum"
    ]
    assert "lts__t_sectors_op_read.sum" not in bundle.aggregate_counters["metrics"]
