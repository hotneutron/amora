"""Tests for command-target CUDA-event and device-interval protocols."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from amora.backends.nvidia import cuda_event_run
from amora.backends.nvidia.cuda_event_run import (
    MeasurementProtocolError,
    parse_cuda_event_payload,
    parse_device_interval_payload,
    run_command_cuda_events,
    run_command_device_intervals,
)


def _timing_payload(
    *,
    samples=(21.31, 21.29, 21.35),
    unit="us_per_launch",
    uuid="GPU-test",
    cubin="cubin-a",
    cache_protocol="warm_reuse",
):
    return {
        "schema_version": 1,
        "kind": "cuda_event_timing",
        "timing": {
            "unit": unit,
            "samples": list(samples),
            "warmup_launches": 20,
            "launches_per_sample": 100,
            "cache_protocol": cache_protocol,
            "clock_policy": "application_clocks",
        },
        "device": {
            "uuid": uuid,
            "name": "NVIDIA H100 80GB HBM3",
            "sm_clock_mhz": "1980",
        },
        "subject_identity": {
            "kernel_name": "jit_kernel",
            "ttgir_sha256": "ttgir-a",
            "cubin_sha256": cubin,
        },
        "measurement_axes": {},
        "tool_versions": {"cuda_runtime": "13.1"},
    }


def _interval_payload(**instrumentation):
    values = {
        "unprofiled_event_overhead_percent": 1.4,
        "overhead_threshold_percent": 2.0,
        "sass_bracketing_verified": True,
        "instruction_sequence_unchanged": True,
    }
    values.update(instrumentation)
    return {
        "schema_version": 1,
        "kind": "cuda_device_intervals",
        "clock": "globaltimer_ns",
        "intervals": [
            {
                "name": "wgmma_exposed_wait",
                "start_ns": 1000,
                "end_ns": 1120,
                "duration_ns": 120,
            }
        ],
        "instrumentation": values,
        "device": {"uuid": "GPU-test", "name": "H100"},
        "subject_identity": {
            "ttgir_sha256": "ttgir-a",
            "cubin_sha256": "cubin-a",
        },
        "measurement_axes": {},
        "tool_versions": {"cuda_runtime": "13.1"},
    }


def test_parse_valid_cuda_event_payload():
    result = parse_cuda_event_payload(_timing_payload())

    assert result.samples_us == (21.31, 21.29, 21.35)
    assert result.median_us == 21.31
    assert result.launch_batch_size == 100
    assert result.cache_protocol == "warm_reuse"


@pytest.mark.parametrize(
    "payload, match",
    [
        (_timing_payload(samples=()), "non-empty"),
        (_timing_payload(samples=(math.nan,)), "finite"),
        (_timing_payload(samples=(0.0,)), "positive"),
        (_timing_payload(unit="ns"), "unit"),
        (_timing_payload(cache_protocol="mystery"), "cache_protocol"),
    ],
)
def test_parse_cuda_event_payload_rejects_invalid_timing(payload, match):
    with pytest.raises(MeasurementProtocolError, match=match):
        parse_cuda_event_payload(payload)


def test_parse_cuda_event_payload_rejects_missing_identity():
    payload = _timing_payload()
    del payload["subject_identity"]["cubin_sha256"]

    with pytest.raises(MeasurementProtocolError, match="cubin_sha256"):
        parse_cuda_event_payload(payload)


def test_parse_cuda_event_payload_requires_clock_policy():
    payload = _timing_payload()
    del payload["timing"]["clock_policy"]

    with pytest.raises(MeasurementProtocolError, match="clock_policy"):
        parse_cuda_event_payload(payload)


def test_parse_cuda_event_payload_rejects_measurement_axis_mismatch():
    payload = _timing_payload()
    payload["measurement_axes"] = {"cta_load": 1}

    with pytest.raises(MeasurementProtocolError, match="measurement axes"):
        parse_cuda_event_payload(
            payload, expected_measurement_axes={"cta_load": 132}
        )


def test_parse_cuda_event_payload_accepts_operation_scope():
    payload = _timing_payload()
    payload["measurement_context"] = {
        "scope": "application_operation",
        "launch_ordinal": -1,
        "ordered_launches": [
            {
                "launch_ordinal": 0,
                "kernel_name": "kernel_a",
                "subject_identity": {"cubin_sha256": "launch-cubin"},
            }
        ],
    }

    result = parse_cuda_event_payload(payload)

    assert result.measurement_context["launch_ordinal"] == -1


def test_run_cuda_events_retains_process_samples_and_summary(monkeypatch):
    outputs = [_timing_payload(samples=(10.0, 12.0)), _timing_payload(samples=(14.0, 16.0))]

    def fake_run(args, **kwargs):
        payload = outputs.pop(0)
        return subprocess.CompletedProcess(
            args, 0, "target log\n" + json.dumps(payload) + "\n", ""
        )

    monkeypatch.setattr(cuda_event_run.subprocess, "run", fake_run)
    result = run_command_cuda_events(
        ("python", "fixture.py"),
        repeats=2,
        expected_launch_batch_size=100,
        expected_cache_protocol="warm_reuse",
        environment_overrides={"CUDA_VISIBLE_DEVICES": "0"},
    )

    assert result.median_us == 13.0
    assert result.p05_us == pytest.approx(10.3)
    assert result.p95_us == pytest.approx(15.7)
    assert len(result.process_samples) == 2
    assert result.to_dict()["latency_oracle"] == "cuda_events"
    assert result.provenance["environment_overrides"] == {
        "CUDA_VISIBLE_DEVICES": "0"
    }


@pytest.mark.parametrize(
    "second, match",
    [
        (_timing_payload(cubin="cubin-b"), "identity"),
        (_timing_payload(uuid="GPU-other"), "GPU UUID"),
    ],
)
def test_run_cuda_events_rejects_cross_process_drift(monkeypatch, second, match):
    outputs = [_timing_payload(), second]

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, json.dumps(outputs.pop(0)), "")

    monkeypatch.setattr(cuda_event_run.subprocess, "run", fake_run)

    with pytest.raises(MeasurementProtocolError, match=match):
        run_command_cuda_events(("python", "fixture.py"), repeats=2)


def test_run_cuda_events_rejects_nonzero_process(monkeypatch):
    monkeypatch.setattr(
        cuda_event_run.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 2, "", "boom"),
    )

    with pytest.raises(MeasurementProtocolError, match="returned 2"):
        run_command_cuda_events(("python", "fixture.py"), repeats=1)


def test_run_cuda_events_uses_real_process_isolation(tmp_path):
    target = (
        sys.executable,
        "-c",
        (
            "import json, os; "
            "print(json.dumps({"
            "'schema_version': 1, 'kind': 'cuda_event_timing', "
            "'timing': {'unit': 'us_per_launch', 'samples': [2.0], "
            "'warmup_launches': 1, 'launches_per_sample': 1, "
            "'cache_protocol': 'warm_reuse', 'clock_policy': 'uncontrolled'}, "
            "'device': {'uuid': 'GPU-test'}, "
            "'subject_identity': {'ttgir_sha256': 't', 'cubin_sha256': 'c'}, "
            "'tool_versions': {'python': 'test'}}))"
        ),
    )

    result = run_command_cuda_events(target, repeats=2, cwd=tmp_path)

    assert len(result.process_samples) == 2
    assert [
        sample.environment_overrides["AMORA_PROCESS_REPEAT"]
        for sample in result.process_samples
    ] == ["0", "1"]


def test_cuda_event_fixture_preserves_environment_measurement_context(tmp_path):
    context = {
        "operation_id": "operation",
        "launch_ordinal": 0,
        "ordered_launches": [],
    }
    completed = subprocess.run(
        [sys.executable, "tests/fixtures/cuda_event_target.py"],
        check=True,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "AMORA_SMOKE_MEASUREMENT_CONTEXT": json.dumps(context),
        },
    )
    payload = json.loads(completed.stdout.splitlines()[-1])

    assert payload["measurement_context"] == context


def test_device_interval_downgrades_slope_change():
    result = parse_device_interval_payload(
        _interval_payload(
            unprofiled_event_slope_change_percent=3.0,
            slope_threshold_percent=2.0,
        )
    )

    assert result.evidence_status == "diagnostic"
    assert "instrumentation_slope_change_exceeds_threshold" in result.diagnostic_reasons


def test_device_interval_downgrades_measurement_axis_mismatch():
    payload = _interval_payload()
    payload["measurement_axes"] = {"cta_load": 2}

    result = parse_device_interval_payload(
        payload, expected_measurement_axes={"cta_load": 1}
    )

    assert result.evidence_status == "diagnostic"
    assert "measurement_axes_do_not_match_expected_case" in result.diagnostic_reasons


def test_device_intervals_are_qualifying_when_all_guards_pass():
    result = parse_device_interval_payload(_interval_payload())

    assert result.evidence_status == "qualifying"
    assert result.diagnostic_reasons == ()
    assert result.intervals[0].duration_ns == 120.0


def test_device_interval_downgrades_all_requested_diagnostic_conditions():
    payload = _interval_payload(
        sass_bracketing_verified=False,
        instruction_sequence_unchanged=False,
        unprofiled_event_overhead_percent=4.0,
    )
    payload["clock"] = ""
    payload["intervals"][0].update(end_ns=900, duration_ns=50)

    result = parse_device_interval_payload(payload)

    assert result.evidence_status == "diagnostic"
    assert set(result.diagnostic_reasons) == {
        "non_monotonic_interval:wgmma_exposed_wait",
        "duration_mismatch:wgmma_exposed_wait",
        "sass_bracketing_not_verified",
        "instruction_sequence_not_verified_unchanged",
        "instrumentation_overhead_exceeds_threshold",
        "clock_source_not_documented",
    }


def test_barrier_topology_synthetic_target_protocols(tmp_path):
    target = str(
        Path(__file__).parents[1] / "fixtures" / "barrier_topology_target.py"
    )
    command = (
        sys.executable,
        target,
        "--panel",
        "BT-CAUSAL",
        "--topology",
        "pairwise",
    )
    axes = {
        "panel": "BT-CAUSAL",
        "topology": "pairwise",
        "M": 128,
        "N": 128,
        "K": 256,
        "tile_shape": "128x128x64",
        "grid": "132x1x1",
        "pipeline_depth": 3,
        "producer_asymmetry_level": "transition",
        "cache_protocol": "disjoint_rotation",
        "cta_concurrency": 132,
        "address_partition_mapping": "round_robin",
        "launch_batch_size": 8,
        "clock_policy": "base",
    }
    identity_fields = (
        "kernel_name",
        "source_sha256",
        "ttgir_sha256",
        "ptx_sha256",
        "sass_sha256",
        "cubin_sha256",
    )

    timing = run_command_cuda_events(
        command,
        repeats=2,
        cwd=tmp_path,
        expected_launch_batch_size=8,
        expected_cache_protocol="disjoint_rotation",
        required_identity_fields=identity_fields,
        expected_measurement_axes=axes,
    )
    intervals = run_command_device_intervals(
        command,
        cwd=tmp_path,
        required_identity_fields=identity_fields,
        expected_max_overhead_percent=5.0,
        expected_measurement_axes=axes,
    )

    assert timing.process_cv == 0.0
    assert timing.subject_metadata["numerical_result_valid"] is True
    assert intervals.evidence_status == "qualifying"
    assert {interval.name for interval in intervals.intervals} >= {
        "fast_tma_issue_to_barrier_release",
        "slow_tma_issue_to_barrier_release",
        "fast_barrier_release_to_consumer_issue",
    }
