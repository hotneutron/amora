"""Process-isolated CUDA-event timing and device-interval protocols.

CUDA events must be recorded by the process that owns the CUDA stream. AMORA
therefore defines the target's JSON protocol, runs the target in fresh processes,
validates every payload, and owns the resulting evidence record.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


CUDA_EVENT_SCHEMA_VERSION = 1
CUDA_EVENT_UNIT = "us_per_launch"
CACHE_PROTOCOLS = frozenset(
    {
        "warm_reuse",
        "disjoint_rotation",
        "cold_flush",
    }
)
DEFAULT_IDENTITY_FIELDS = ("ttgir_sha256", "cubin_sha256")


class MeasurementProtocolError(ValueError):
    """Raised when a target fails or emits invalid measurement evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _string_mapping(value: object, *, field_name: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise MeasurementProtocolError(f"{field_name} must be an object")
    result: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str) or not item.strip():
            raise MeasurementProtocolError(
                f"{field_name} must contain non-empty string keys and values"
            )
        result[key] = item
    return result


def _finite_number(value: object, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MeasurementProtocolError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise MeasurementProtocolError(f"{field_name} must be finite")
    return result


def _axis_mapping(value: object) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MeasurementProtocolError("measurement_axes must be an object")
    axes = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise MeasurementProtocolError(
                "measurement_axes keys must be non-empty strings"
            )
        if isinstance(item, float) and not math.isfinite(item):
            raise MeasurementProtocolError(
                f"measurement_axes.{key} must be finite"
            )
        if not isinstance(item, (str, int, float, bool)):
            raise MeasurementProtocolError(
                f"measurement_axes.{key} must be a JSON scalar"
            )
        axes[key] = item
    return axes


def _json_object(value: object, *, field_name: str) -> dict[str, Any]:
    """Validate that ``value`` is a JSON-compatible object."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise MeasurementProtocolError(f"{field_name} must be an object")
    try:
        encoded = json.dumps(value, allow_nan=False, sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise MeasurementProtocolError(
            f"{field_name} must contain finite JSON values"
        ) from exc
    if not isinstance(decoded, dict):
        raise MeasurementProtocolError(f"{field_name} must be an object")
    return decoded


def _positive_int(value: object, *, field_name: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MeasurementProtocolError(f"{field_name} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        comparator = "non-negative" if allow_zero else "positive"
        raise MeasurementProtocolError(f"{field_name} must be {comparator}")
    return value


def extract_json_payload(
    stdout: str,
    *,
    expected_kind: str | None = None,
    require_final_line: bool = True,
) -> dict[str, Any]:
    """Extract a JSON object emitted by a target.

    Unprofiled target commands must place the payload on their final non-empty
    stdout line. NCU appends its own output, so profiler callers may scan backward
    for a matching protocol object with ``require_final_line=False``.
    """

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    candidates = lines[-1:] if require_final_line else reversed(lines)
    for line in candidates:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            if require_final_line:
                break
            continue
        if not isinstance(value, dict):
            if require_final_line:
                break
            continue
        if expected_kind is None or value.get("kind") == expected_kind:
            return value
        if require_final_line:
            break
    location = "final stdout line" if require_final_line else "stdout"
    kind = f" {expected_kind!r}" if expected_kind else ""
    raise MeasurementProtocolError(f"no{kind} JSON object found in target {location}")


def extract_measurement_identity(stdout: str) -> dict[str, Any] | None:
    """Best-effort extraction of target identity from profiler-mixed stdout."""

    try:
        payload = extract_json_payload(stdout, require_final_line=False)
    except MeasurementProtocolError:
        return None
    identity = payload.get("subject_identity")
    device = payload.get("device")
    if not isinstance(identity, Mapping) or not isinstance(device, Mapping):
        return None
    return {
        "kind": payload.get("kind"),
        "schema_version": payload.get("schema_version"),
        "subject_identity": dict(identity),
        "device": dict(device),
        "subject_metadata": (
            dict(payload["subject_metadata"])
            if isinstance(payload.get("subject_metadata"), Mapping)
            else {}
        ),
        "subject_artifacts": (
            dict(payload["subject_artifacts"])
            if isinstance(payload.get("subject_artifacts"), Mapping)
            else {}
        ),
        "measurement_axes": (
            dict(payload["measurement_axes"])
            if isinstance(payload.get("measurement_axes"), Mapping)
            else {}
        ),
        "tool_versions": (
            dict(payload["tool_versions"])
            if isinstance(payload.get("tool_versions"), Mapping)
            else {}
        ),
        "measurement_context": (
            dict(payload["measurement_context"])
            if isinstance(payload.get("measurement_context"), Mapping)
            else {}
        ),
    }


@dataclass(frozen=True)
class CudaEventProcessSample:
    process_index: int
    samples_us: tuple[float, ...]
    warmup_launches: int
    launch_batch_size: int
    cache_protocol: str
    clock_policy: str | None
    device: dict[str, str]
    subject_identity: dict[str, str]
    subject_metadata: dict[str, Any]
    measurement_axes: dict[str, Any]
    tool_versions: dict[str, str]
    command: tuple[str, ...]
    cwd: str | None
    environment_overrides: dict[str, str]
    stdout: str
    stderr: str
    returncode: int
    started_at: str
    completed_at: str
    measurement_context: dict[str, Any] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def median_us(self) -> float:
        return float(statistics.median(self.samples_us))

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_index": self.process_index,
            "samples_us": list(self.samples_us),
            "median_us": self.median_us,
            "warmup_launches": self.warmup_launches,
            "launch_batch_size": self.launch_batch_size,
            "cache_protocol": self.cache_protocol,
            "clock_policy": self.clock_policy,
            "device": dict(self.device),
            "subject_identity": dict(self.subject_identity),
            "subject_metadata": dict(self.subject_metadata),
            "measurement_axes": dict(self.measurement_axes),
            "tool_versions": dict(self.tool_versions),
            "measurement_context": dict(self.measurement_context),
            "command": list(self.command),
            "cwd": self.cwd,
            "environment_overrides": dict(self.environment_overrides),
            "stdout": self.stdout,
            "stderr": self.stderr,
            "returncode": self.returncode,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "target_payload": dict(self.payload),
        }


@dataclass(frozen=True)
class CudaEventTimingResult:
    target_command: tuple[str, ...]
    process_samples: tuple[CudaEventProcessSample, ...]
    median_us: float
    p05_us: float
    p95_us: float
    process_cv: float
    launch_batch_size: int
    cache_protocol: str
    clock_policy: str | None
    device: dict[str, str]
    subject_identity: dict[str, str]
    subject_metadata: dict[str, Any]
    measurement_axes: dict[str, Any]
    tool_versions: dict[str, str]
    provenance: dict[str, Any]
    measurement_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CUDA_EVENT_SCHEMA_VERSION,
            "kind": "cuda_event_timing_result",
            "latency_oracle": "cuda_events",
            "unit": CUDA_EVENT_UNIT,
            "target_command": list(self.target_command),
            "process_samples": [sample.to_dict() for sample in self.process_samples],
            "median_us": self.median_us,
            "p05_us": self.p05_us,
            "p95_us": self.p95_us,
            "process_cv": self.process_cv,
            "launch_batch_size": self.launch_batch_size,
            "cache_protocol": self.cache_protocol,
            "clock_policy": self.clock_policy,
            "device": dict(self.device),
            "subject_identity": dict(self.subject_identity),
            "subject_metadata": dict(self.subject_metadata),
            "measurement_axes": dict(self.measurement_axes),
            "tool_versions": dict(self.tool_versions),
            "measurement_context": dict(self.measurement_context),
            "provenance": dict(self.provenance),
        }


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_cuda_event_payload(
    payload: Mapping[str, Any],
    *,
    process_index: int = 0,
    command: tuple[str, ...] = (),
    cwd: str | None = None,
    environment_overrides: Mapping[str, str] | None = None,
    stdout: str = "",
    stderr: str = "",
    returncode: int = 0,
    started_at: str = "",
    completed_at: str = "",
    expected_launch_batch_size: int | None = None,
    expected_cache_protocol: str | None = None,
    required_identity_fields: tuple[str, ...] = DEFAULT_IDENTITY_FIELDS,
    expected_measurement_axes: Mapping[str, Any] | None = None,
    expected_measurement_context: Mapping[str, Any] | None = None,
) -> CudaEventProcessSample:
    """Validate one target-emitted CUDA-event payload."""

    if payload.get("schema_version") != CUDA_EVENT_SCHEMA_VERSION:
        raise MeasurementProtocolError("unsupported CUDA-event schema_version")
    if payload.get("kind") != "cuda_event_timing":
        raise MeasurementProtocolError("payload kind must be cuda_event_timing")
    timing = payload.get("timing")
    if not isinstance(timing, Mapping):
        raise MeasurementProtocolError("timing must be an object")
    if timing.get("unit") != CUDA_EVENT_UNIT:
        raise MeasurementProtocolError(
            f"timing unit must be {CUDA_EVENT_UNIT!r}, got {timing.get('unit')!r}"
        )
    raw_samples = timing.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise MeasurementProtocolError("timing.samples must be a non-empty list")
    samples = tuple(
        _finite_number(value, field_name=f"timing.samples[{index}]")
        for index, value in enumerate(raw_samples)
    )
    if any(value <= 0.0 for value in samples):
        raise MeasurementProtocolError("timing samples must be positive")
    warmup = _positive_int(
        timing.get("warmup_launches"),
        field_name="timing.warmup_launches",
        allow_zero=True,
    )
    batch_size = _positive_int(
        timing.get("launches_per_sample"),
        field_name="timing.launches_per_sample",
    )
    cache_protocol = timing.get("cache_protocol")
    if cache_protocol not in CACHE_PROTOCOLS:
        raise MeasurementProtocolError(
            f"timing.cache_protocol must be one of {sorted(CACHE_PROTOCOLS)}"
        )
    if expected_launch_batch_size is not None and batch_size != expected_launch_batch_size:
        raise MeasurementProtocolError(
            "unexpected launches_per_sample: "
            f"{batch_size} != {expected_launch_batch_size}"
        )
    if expected_cache_protocol is not None and cache_protocol != expected_cache_protocol:
        raise MeasurementProtocolError(
            f"unexpected cache protocol: {cache_protocol!r} != "
            f"{expected_cache_protocol!r}"
        )
    raw_clock_policy = timing.get("clock_policy")
    if not isinstance(raw_clock_policy, str) or not raw_clock_policy.strip():
        raise MeasurementProtocolError(
            "timing.clock_policy must be a non-empty string"
        )
    clock_policy = raw_clock_policy
    device = _string_mapping(payload.get("device"), field_name="device")
    if not device.get("uuid"):
        raise MeasurementProtocolError("device.uuid is required")
    identity = _string_mapping(
        payload.get("subject_identity"), field_name="subject_identity"
    )
    missing = [name for name in required_identity_fields if not identity.get(name)]
    if missing:
        raise MeasurementProtocolError(
            f"subject_identity is missing required fields: {', '.join(missing)}"
        )
    raw_metadata = payload.get("subject_metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise MeasurementProtocolError("subject_metadata must be an object")
    measurement_axes = _axis_mapping(payload.get("measurement_axes"))
    if (
        expected_measurement_axes is not None
        and measurement_axes != dict(expected_measurement_axes)
    ):
        raise MeasurementProtocolError("measurement axes do not match the expected case")
    tool_versions = _string_mapping(
        payload.get("tool_versions"), field_name="tool_versions"
    )
    if not tool_versions:
        raise MeasurementProtocolError("tool_versions must not be empty")
    measurement_context = _json_object(
        payload.get("measurement_context"), field_name="measurement_context"
    )
    if (
        expected_measurement_context is not None
        and measurement_context != dict(expected_measurement_context)
    ):
        raise MeasurementProtocolError(
            "measurement context does not match the expected launch"
        )
    ordered_launches = measurement_context.get("ordered_launches")
    launch_ordinal = measurement_context.get("launch_ordinal")
    if isinstance(ordered_launches, list) and isinstance(launch_ordinal, int):
        operation_scope = measurement_context.get("scope") == "application_operation"
        if launch_ordinal == -1 and operation_scope:
            pass
        elif launch_ordinal < 0 or launch_ordinal >= len(ordered_launches):
            raise MeasurementProtocolError("measurement launch ordinal is invalid")
        else:
            descriptor = ordered_launches[launch_ordinal]
            if not isinstance(descriptor, Mapping):
                raise MeasurementProtocolError(
                    "selected launch descriptor must be an object"
                )
            descriptor_identity = descriptor.get("subject_identity")
            if not isinstance(descriptor_identity, Mapping):
                raise MeasurementProtocolError(
                    "selected launch descriptor must include subject_identity"
                )
            for name, expected in descriptor_identity.items():
                if identity.get(name) != expected:
                    raise MeasurementProtocolError(
                        "selected launch identity does not match descriptor: "
                        f"{name}"
                    )
            if descriptor.get("kernel_name") != identity.get("kernel_name"):
                raise MeasurementProtocolError(
                    "selected launch kernel does not match subject identity"
                )
    return CudaEventProcessSample(
        process_index=process_index,
        samples_us=samples,
        warmup_launches=warmup,
        launch_batch_size=batch_size,
        cache_protocol=cache_protocol,
        clock_policy=clock_policy,
        device=device,
        subject_identity=identity,
        subject_metadata=dict(raw_metadata),
        measurement_axes=measurement_axes,
        tool_versions=tool_versions,
        command=tuple(command),
        cwd=cwd,
        environment_overrides=dict(environment_overrides or {}),
        stdout=stdout,
        stderr=stderr,
        returncode=returncode,
        started_at=started_at,
        completed_at=completed_at,
        measurement_context=measurement_context,
        payload=dict(payload),
    )


def run_command_cuda_events(
    target: tuple[str, ...],
    *,
    repeats: int = 7,
    timeout: int = 180,
    expected_launch_batch_size: int | None = None,
    expected_cache_protocol: str | None = None,
    required_identity_fields: tuple[str, ...] = DEFAULT_IDENTITY_FIELDS,
    expected_measurement_axes: Mapping[str, Any] | None = None,
    expected_measurement_context: Mapping[str, Any] | None = None,
    cwd: str | Path | None = None,
    environment_overrides: Mapping[str, str] | None = None,
) -> CudaEventTimingResult:
    """Execute a CUDA-event target in fresh processes and validate all repeats."""

    if not target or not target[0]:
        raise MeasurementProtocolError("target command must not be empty")
    if repeats <= 0:
        raise MeasurementProtocolError("repeats must be positive")
    overrides = dict(environment_overrides or {})
    if any(
        not isinstance(key, str) or not isinstance(value, str)
        for key, value in overrides.items()
    ):
        raise MeasurementProtocolError("environment overrides must be strings")
    cwd_text = str(cwd) if cwd is not None else None
    process_samples = []
    for process_index in range(repeats):
        process_overrides = {
            **overrides,
            "AMORA_MEASUREMENT_LANE": "cuda_event_timing",
            "AMORA_PROCESS_REPEAT": str(process_index),
        }
        if expected_cache_protocol is not None:
            process_overrides["AMORA_CACHE_PROTOCOL"] = expected_cache_protocol
        run_environment = dict(os.environ)
        run_environment.update(process_overrides)
        started_at = _utc_now()
        try:
            completed = subprocess.run(
                target,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd_text,
                env=run_environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise MeasurementProtocolError(
                f"CUDA-event target process {process_index} failed: {exc}"
            ) from exc
        completed_at = _utc_now()
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout).strip()
            raise MeasurementProtocolError(
                f"CUDA-event target process {process_index} returned "
                f"{completed.returncode}: {diagnostic[-500:] or 'no output'}"
            )
        payload = extract_json_payload(
            completed.stdout, expected_kind="cuda_event_timing"
        )
        process_samples.append(
            parse_cuda_event_payload(
                payload,
                process_index=process_index,
                command=target,
                cwd=cwd_text,
                environment_overrides=process_overrides,
                stdout=completed.stdout,
                stderr=completed.stderr,
                returncode=completed.returncode,
                started_at=started_at,
                completed_at=completed_at,
                expected_launch_batch_size=expected_launch_batch_size,
                expected_cache_protocol=expected_cache_protocol,
                required_identity_fields=required_identity_fields,
                expected_measurement_axes=expected_measurement_axes,
                expected_measurement_context=expected_measurement_context,
            )
        )

    first = process_samples[0]
    for sample in process_samples[1:]:
        if sample.subject_identity != first.subject_identity:
            raise MeasurementProtocolError("subject identity changed across process repeats")
        if sample.subject_metadata != first.subject_metadata:
            raise MeasurementProtocolError("subject metadata changed across process repeats")
        if sample.measurement_axes != first.measurement_axes:
            raise MeasurementProtocolError("measurement axes changed across process repeats")
        if sample.tool_versions != first.tool_versions:
            raise MeasurementProtocolError("tool versions changed across process repeats")
        if sample.measurement_context != first.measurement_context:
            raise MeasurementProtocolError(
                "measurement context changed across process repeats"
            )
        if sample.device.get("uuid") != first.device.get("uuid"):
            raise MeasurementProtocolError("GPU UUID changed across process repeats")
        if sample.launch_batch_size != first.launch_batch_size:
            raise MeasurementProtocolError("launch batching changed across process repeats")
        if sample.cache_protocol != first.cache_protocol:
            raise MeasurementProtocolError("cache protocol changed across process repeats")
        if sample.clock_policy != first.clock_policy:
            raise MeasurementProtocolError("clock policy changed across process repeats")

    pooled = [value for process in process_samples for value in process.samples_us]
    process_medians = [process.median_us for process in process_samples]
    mean_process_median = statistics.fmean(process_medians)
    process_cv = (
        statistics.pstdev(process_medians) / mean_process_median
        if len(process_medians) > 1 and mean_process_median > 0.0
        else 0.0
    )
    return CudaEventTimingResult(
        target_command=tuple(target),
        process_samples=tuple(process_samples),
        median_us=float(statistics.median(pooled)),
        p05_us=_percentile(pooled, 0.05),
        p95_us=_percentile(pooled, 0.95),
        process_cv=process_cv,
        launch_batch_size=first.launch_batch_size,
        cache_protocol=first.cache_protocol,
        clock_policy=first.clock_policy,
        device=dict(first.device),
        subject_identity=dict(first.subject_identity),
        subject_metadata=dict(first.subject_metadata),
        measurement_axes=dict(first.measurement_axes),
        tool_versions=dict(first.tool_versions),
        provenance={
            "process_isolation": "one_target_process_per_repeat",
            "repeat_count": repeats,
            "summary_population": "all_target_emitted_cuda_event_samples",
            "process_cv_population": "per_process_medians",
            "cwd": cwd_text,
            "environment_overrides": overrides,
            "generated_environment": {
                "AMORA_MEASUREMENT_LANE": "cuda_event_timing",
                "AMORA_PROCESS_REPEAT": "<zero-based repeat index>",
                "AMORA_CACHE_PROTOCOL": expected_cache_protocol,
            },
        },
        measurement_context=dict(first.measurement_context),
    )


@dataclass(frozen=True)
class DeviceInterval:
    name: str
    start_ns: float
    end_ns: float
    duration_ns: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "duration_ns": self.duration_ns,
        }


@dataclass(frozen=True)
class DeviceIntervalResult:
    clock: str | None
    intervals: tuple[DeviceInterval, ...]
    instrumentation: dict[str, Any]
    device: dict[str, str]
    subject_identity: dict[str, str]
    subject_metadata: dict[str, Any]
    measurement_axes: dict[str, Any]
    tool_versions: dict[str, str]
    evidence_status: str
    diagnostic_reasons: tuple[str, ...]
    target_command: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    measurement_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CUDA_EVENT_SCHEMA_VERSION,
            "kind": "cuda_device_interval_result",
            "clock": self.clock,
            "intervals": [interval.to_dict() for interval in self.intervals],
            "instrumentation": dict(self.instrumentation),
            "device": dict(self.device),
            "subject_identity": dict(self.subject_identity),
            "subject_metadata": dict(self.subject_metadata),
            "measurement_axes": dict(self.measurement_axes),
            "tool_versions": dict(self.tool_versions),
            "evidence_status": self.evidence_status,
            "diagnostic_reasons": list(self.diagnostic_reasons),
            "target_command": list(self.target_command),
            "provenance": dict(self.provenance),
            "measurement_context": dict(self.measurement_context),
        }


def parse_device_interval_payload(
    payload: Mapping[str, Any],
    *,
    required_identity_fields: tuple[str, ...] = DEFAULT_IDENTITY_FIELDS,
    expected_max_overhead_percent: float | None = None,
    expected_measurement_axes: Mapping[str, Any] | None = None,
    expected_measurement_context: Mapping[str, Any] | None = None,
    target_command: tuple[str, ...] = (),
    provenance: Mapping[str, Any] | None = None,
) -> DeviceIntervalResult:
    """Validate intervals while downgrading questionable mechanism evidence."""

    if payload.get("schema_version") != CUDA_EVENT_SCHEMA_VERSION:
        raise MeasurementProtocolError("unsupported device-interval schema_version")
    if payload.get("kind") != "cuda_device_intervals":
        raise MeasurementProtocolError("payload kind must be cuda_device_intervals")
    raw_intervals = payload.get("intervals")
    if not isinstance(raw_intervals, list) or not raw_intervals:
        raise MeasurementProtocolError("intervals must be a non-empty list")
    diagnostic_reasons: list[str] = []
    intervals = []
    previous_start: float | None = None
    for index, raw in enumerate(raw_intervals):
        if not isinstance(raw, Mapping):
            raise MeasurementProtocolError(f"intervals[{index}] must be an object")
        name = raw.get("name")
        if not isinstance(name, str) or not name.strip():
            raise MeasurementProtocolError(f"intervals[{index}].name is required")
        start = _finite_number(raw.get("start_ns"), field_name=f"intervals[{index}].start_ns")
        end = _finite_number(raw.get("end_ns"), field_name=f"intervals[{index}].end_ns")
        duration = _finite_number(
            raw.get("duration_ns"), field_name=f"intervals[{index}].duration_ns"
        )
        if end < start or duration < 0.0:
            diagnostic_reasons.append(f"non_monotonic_interval:{name}")
        if previous_start is not None and start < previous_start:
            diagnostic_reasons.append(f"non_monotonic_interval_order:{name}")
        previous_start = start
        if not math.isclose(duration, end - start, rel_tol=1e-9, abs_tol=1e-6):
            diagnostic_reasons.append(f"duration_mismatch:{name}")
        intervals.append(DeviceInterval(name, start, end, duration))

    raw_instrumentation = payload.get("instrumentation")
    if not isinstance(raw_instrumentation, Mapping):
        raise MeasurementProtocolError("instrumentation must be an object")
    instrumentation = dict(raw_instrumentation)
    if instrumentation.get("sass_bracketing_verified") is not True:
        diagnostic_reasons.append("sass_bracketing_not_verified")
    if instrumentation.get("instruction_sequence_unchanged") is not True:
        diagnostic_reasons.append("instruction_sequence_not_verified_unchanged")
    overhead = _finite_number(
        instrumentation.get("unprofiled_event_overhead_percent"),
        field_name="instrumentation.unprofiled_event_overhead_percent",
    )
    if overhead < 0.0:
        raise MeasurementProtocolError("instrumentation overhead must be non-negative")
    threshold_value = (
        expected_max_overhead_percent
        if expected_max_overhead_percent is not None
        else instrumentation.get("overhead_threshold_percent")
    )
    if threshold_value is None:
        diagnostic_reasons.append("instrumentation_overhead_threshold_not_declared")
    else:
        threshold = _finite_number(
            threshold_value, field_name="instrumentation.overhead_threshold_percent"
        )
        if threshold < 0.0:
            raise MeasurementProtocolError("instrumentation threshold must be non-negative")
        if overhead > threshold:
            diagnostic_reasons.append("instrumentation_overhead_exceeds_threshold")
    slope_change = instrumentation.get("unprofiled_event_slope_change_percent")
    if slope_change is not None:
        slope_change_value = _finite_number(
            slope_change,
            field_name="instrumentation.unprofiled_event_slope_change_percent",
        )
        slope_threshold_value = instrumentation.get("slope_threshold_percent")
        if slope_threshold_value is None:
            diagnostic_reasons.append("instrumentation_slope_threshold_not_declared")
        else:
            slope_threshold = _finite_number(
                slope_threshold_value,
                field_name="instrumentation.slope_threshold_percent",
            )
            if abs(slope_change_value) > slope_threshold:
                diagnostic_reasons.append("instrumentation_slope_change_exceeds_threshold")

    clock = payload.get("clock")
    if not isinstance(clock, str) or not clock.strip():
        clock = None
        diagnostic_reasons.append("clock_source_not_documented")
    device = _string_mapping(payload.get("device"), field_name="device")
    if not device.get("uuid"):
        raise MeasurementProtocolError("device.uuid is required")
    identity = _string_mapping(
        payload.get("subject_identity"), field_name="subject_identity"
    )
    missing = [name for name in required_identity_fields if not identity.get(name)]
    if missing:
        raise MeasurementProtocolError(
            f"subject_identity is missing required fields: {', '.join(missing)}"
        )
    raw_metadata = payload.get("subject_metadata", {})
    if not isinstance(raw_metadata, Mapping):
        raise MeasurementProtocolError("subject_metadata must be an object")
    measurement_axes = _axis_mapping(payload.get("measurement_axes"))
    if (
        expected_measurement_axes is not None
        and measurement_axes != dict(expected_measurement_axes)
    ):
        diagnostic_reasons.append("measurement_axes_do_not_match_expected_case")
    tool_versions = _string_mapping(
        payload.get("tool_versions"), field_name="tool_versions"
    )
    if not tool_versions:
        raise MeasurementProtocolError("tool_versions must not be empty")
    measurement_context = _json_object(
        payload.get("measurement_context"), field_name="measurement_context"
    )
    if (
        expected_measurement_context is not None
        and measurement_context != dict(expected_measurement_context)
    ):
        diagnostic_reasons.append("measurement_context_does_not_match_expected_launch")
    reasons = tuple(dict.fromkeys(diagnostic_reasons))
    return DeviceIntervalResult(
        clock=clock,
        intervals=tuple(intervals),
        instrumentation=instrumentation,
        device=device,
        subject_identity=identity,
        subject_metadata=dict(raw_metadata),
        measurement_axes=measurement_axes,
        tool_versions=tool_versions,
        evidence_status="diagnostic" if reasons else "qualifying",
        diagnostic_reasons=reasons,
        target_command=tuple(target_command),
        provenance=dict(provenance or {}),
        measurement_context=measurement_context,
    )


def run_command_device_intervals(
    target: tuple[str, ...],
    *,
    timeout: int = 180,
    required_identity_fields: tuple[str, ...] = DEFAULT_IDENTITY_FIELDS,
    expected_max_overhead_percent: float | None = None,
    expected_measurement_axes: Mapping[str, Any] | None = None,
    expected_measurement_context: Mapping[str, Any] | None = None,
    cwd: str | Path | None = None,
    environment_overrides: Mapping[str, str] | None = None,
) -> DeviceIntervalResult:
    """Run one instrumented target and validate its device intervals."""

    if not target or not target[0]:
        raise MeasurementProtocolError("target command must not be empty")
    overrides = {
        **dict(environment_overrides or {}),
        "AMORA_MEASUREMENT_LANE": "device_intervals",
    }
    run_environment = dict(os.environ)
    run_environment.update(overrides)
    cwd_text = str(cwd) if cwd is not None else None
    started_at = _utc_now()
    try:
        completed = subprocess.run(
            target,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd_text,
            env=run_environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MeasurementProtocolError(f"device-interval target failed: {exc}") from exc
    completed_at = _utc_now()
    if completed.returncode != 0:
        diagnostic = (completed.stderr or completed.stdout).strip()
        raise MeasurementProtocolError(
            f"device-interval target returned {completed.returncode}: "
            f"{diagnostic[-500:] or 'no output'}"
        )
    payload = extract_json_payload(
        completed.stdout, expected_kind="cuda_device_intervals"
    )
    return parse_device_interval_payload(
        payload,
        required_identity_fields=required_identity_fields,
        expected_max_overhead_percent=expected_max_overhead_percent,
        expected_measurement_axes=expected_measurement_axes,
        expected_measurement_context=expected_measurement_context,
        target_command=target,
        provenance={
            "cwd": cwd_text,
            "environment_overrides": overrides,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "returncode": completed.returncode,
            "started_at": started_at,
            "completed_at": completed_at,
        },
    )
