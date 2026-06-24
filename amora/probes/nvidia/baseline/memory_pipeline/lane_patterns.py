"""Memory-pipeline lane-address-pattern coalescing probe (P2, Phase B).

One warp issues global loads under controlled per-lane address patterns. The
primary evidence is the NCU request/sector counters: sectors-per-request reveals
how lane addresses coalesce into memory sectors. When NCU is available the probe
reports a direct sectors/request scalar (DIRECT_COUNTER); when NCU is absent it
degrades to a behavioral-only timing fallback that records the counters are
required. SASS validation confirms the timed kernel is a global-load stream.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.backends.nvidia.metrics import MetricResolver
from amora.backends.nvidia.ncu_run import NcuUnavailable, run_kernel_profiled
from amora.backends.nvidia.runner import CudaUnavailable, run_kernel
from amora.backends.nvidia.sass import SassExpectation
from amora.probes.nvidia.baseline._sources import apply_sass_gating, source_descriptor
from amora.schemas.evidence import EvidenceTier, FitStatus, UncertaintyCategory
from amora.schemas.results import (
    BackendInterpretation,
    LaunchDescriptor,
    NormalizedMeasurement,
    ProbeIdentity,
    ProbeResult,
    RawObservation,
    SimulatorEstimate,
    ToolContext,
)


PROBE_ID = "memory_pipeline.lane_patterns"
SOURCE = Path(__file__).with_name("lane_patterns.cu")

# The timed loop must hit global memory (LDG) without shared or local spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_mem_lane_patterns",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDS", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def _collect_counter(capabilities: NvidiaCapabilities, logical: str) -> dict | None:
    """Collect a global-load counter via NCU (primary role).

    The driver runs every lane pattern in turn, so we profile all launches and
    take the maximum value across the profiled rows (the strided patterns issue
    the most sectors). Returns a record dict with the resolved metric and value,
    or None when NCU/the counter is unavailable.
    """

    resolver = MetricResolver(supported_metrics=capabilities.ncu_metrics)
    resolution = resolver.resolve(logical)
    if not resolution.available or not resolution.selected_name:
        return None
    try:
        ncu = run_kernel_profiled(
            SOURCE,
            capabilities=capabilities,
            metrics=(resolution.selected_name,),
            kernel_name="amora_mem_lane_patterns",
            launch_count=8,  # cover all four lane patterns plus slack
        )
    except NcuUnavailable:
        return None
    max_value = None
    if resolution.selected_name in ncu.metrics:
        max_value = float(ncu.metrics[resolution.selected_name])
    for row in ncu.raw_rows:
        raw = (row.get(resolution.selected_name) or "").strip().replace(",", "")
        try:
            v = float(raw)
        except ValueError:
            continue
        max_value = v if max_value is None else max(max_value, v)
    return {
        "metric": resolution.selected_name,
        "logical": logical,
        "role": "primary",
        "value": max_value,
        "launches_profiled": len(ncu.raw_rows),
    }


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=60, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"memory-pipeline lane-patterns probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload

    sass = result.sass_validation
    decision, _, _, sass_downgrade = apply_sass_gating(
        sass, EXPECTATION, FitStatus.DIRECT, UncertaintyCategory.STABLE_SCALAR
    )
    if decision == "reject":
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"SASS validation rejected the measurement: {sass.reason}",
                tool_context=_tool_context(capabilities),
                raw_values={
                    "registered_source": src_descriptor,
                    "sass": sass.to_dict(),
                },
            )
        ]

    # NCU is the PRIMARY evidence: requests + sectors -> sectors/request.
    requests = _collect_counter(capabilities, "global_load_requests")
    sectors = _collect_counter(capabilities, "global_load_sectors")
    req_value = requests.get("value") if requests else None
    sec_value = sectors.get("value") if sectors else None
    sectors_per_request = None
    if req_value and sec_value is not None and req_value > 0:
        sectors_per_request = float(sec_value) / float(req_value)

    ncu_record: dict[str, object] = {}
    if requests is not None:
        ncu_record["global_load_requests"] = requests
    if sectors is not None:
        ncu_record["global_load_sectors"] = sectors
    if sectors_per_request is not None:
        ncu_record["sectors_per_request"] = sectors_per_request

    downgrade_reason = sass_downgrade
    if sectors_per_request is not None:
        # Direct counter evidence.
        evidence_tier = EvidenceTier.DIRECT_COUNTER
        fit = FitStatus.DIRECT
        uncertainty = UncertaintyCategory.STABLE_SCALAR
        value = sectors_per_request
        if decision == "downgrade":
            fit = FitStatus.BOUNDED
            uncertainty = UncertaintyCategory.BOUNDED_RANGE
    else:
        # NCU unavailable: timing-only behavioral fallback. The kernel still ran,
        # but no coalescing scalar can be derived without the counters.
        evidence_tier = EvidenceTier.TIMING_DIRECT
        fit = FitStatus.BEHAVIORAL_ONLY
        uncertainty = UncertaintyCategory.BEHAVIORAL_CLASS
        value = None
        note = "NCU global-load request/sector counters required to derive sectors/request"
        downgrade_reason = (
            f"{downgrade_reason}; {note}" if downgrade_reason else note
        )

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    if ncu_record:
        values["ncu"] = ncu_record

    assumptions = [
        "one warp issues many global loads under named lane address patterns",
        "NCU sectors/request is the primary coalescing signal; timing only confirms LDG activity",
        "max counter value across profiled lane patterns characterizes the worst-case coalescing",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(grid=(1, 1, 1), block=(32, 1, 1), mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=evidence_tier,
                values=values,
                metrics={
                    "global_load_requests": req_value,
                    "global_load_sectors": sec_value,
                    "sectors_per_request": sectors_per_request,
                },
                units={"sectors_per_request": "sectors/request"},
                source="amora.probes.nvidia.baseline.memory_pipeline.lane_patterns",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="coalescing_sectors_per_request",
                value=value,
                unit="sectors/request",
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="memory_coalescing",
                interpretation={
                    "nvidia_backend": "global-load sectors per request from controlled lane address patterns",
                    "sectors_per_request": sectors_per_request,
                },
                metric_resolver=ncu_record,
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="memory_coalescing_rule",
                value=value,
                unit="sectors/request",
                evidence_tier=evidence_tier,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="NCU sectors/request under lane patterns -> simulator memory coalescing rule",
                assumptions=assumptions,
            ),
        )
    ]
