"""Interconnect address-mapping probe (P3, Phase D).

Streams a fixed amount of global memory while sweeping the base-address byte
stride across large power-of-two strides. Throughput variation across strides
reveals partition/slice periodicity. The measurement is a behavioral class:
"uniform" when the max/min bandwidth ratio is small, otherwise
"periodic_camping". SASS validation confirms the kernel is a global-load stream.
"""

from __future__ import annotations

from pathlib import Path

from amora.backends.nvidia.cuda import NvidiaCapabilities
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


PROBE_ID = "interconnect.address_mapping"
SOURCE = Path(__file__).with_name("address_mapping.cu")

_PERIODIC_RATIO = 1.2

# The timed loop must hit global memory (LDG) without spills.
EXPECTATION = SassExpectation(
    kernel_symbol="amora_icn_address_mapping",
    required_opcodes={"LDG": 1},
    forbidden_opcodes=("LDL", "STL"),
)


def _tool_context(capabilities: NvidiaCapabilities) -> ToolContext:
    return ToolContext(tools=capabilities.to_dict())


def run(capabilities: NvidiaCapabilities) -> list[ProbeResult]:
    src_descriptor = source_descriptor(SOURCE)
    try:
        result = run_kernel(SOURCE, capabilities=capabilities, timeout=120, expectation=EXPECTATION)
    except CudaUnavailable as exc:
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"address-mapping probe could not execute: {exc}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor},
            )
        ]
    payload = result.payload
    sweep = list(payload["sweep"])
    gbps = [float(p["gbps"]) for p in sweep if float(p["gbps"]) > 0]
    max_gbps = max(gbps) if gbps else 0.0
    min_gbps = min(gbps) if gbps else 0.0
    ratio = (max_gbps / min_gbps) if min_gbps > 0 else None
    mapping_class = (
        "uniform" if (ratio is not None and ratio < _PERIODIC_RATIO) else "periodic_camping"
    )

    sass = result.sass_validation
    decision, fit, uncertainty, downgrade_reason = apply_sass_gating(
        sass, EXPECTATION, FitStatus.BEHAVIORAL_ONLY, UncertaintyCategory.BEHAVIORAL_CLASS
    )
    if decision == "reject":
        return [
            ProbeResult.unsupported(
                PROBE_ID,
                f"SASS validation rejected the measurement: {sass.reason}",
                tool_context=_tool_context(capabilities),
                raw_values={"registered_source": src_descriptor, "sass": sass.to_dict()},
            )
        ]

    values = {
        "registered_source": src_descriptor,
        "binary_sha256": result.binary_sha256,
        **payload,
    }
    if sass is not None:
        values["sass"] = sass.to_dict()
    assumptions = [
        "grid-stride reads with a per-step base displacement swept across power-of-two strides",
        "best-of-N CUDA-event timing per stride; bandwidth varies with partition interleave",
        f"max/min bandwidth ratio < {_PERIODIC_RATIO} classifies as uniform else periodic_camping",
    ]
    return [
        ProbeResult(
            identity=ProbeIdentity(
                probe_id=PROBE_ID,
                binary_hash=result.binary_sha256,
                disassembly_hash=sass.disassembly_hash if sass else None,
            ),
            tool_context=_tool_context(capabilities),
            launch=LaunchDescriptor(mode="kernel"),
            raw_observation=RawObservation(
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                values=values,
                metrics={
                    "max_gbps": max_gbps,
                    "min_gbps": min_gbps,
                    "bandwidth_ratio": ratio,
                    "address_mapping_class": mapping_class,
                },
                units={"max_gbps": "GB/s", "min_gbps": "GB/s"},
                source="amora.probes.nvidia.baseline.interconnect.address_mapping",
            ),
            normalized_measurement=NormalizedMeasurement(
                name="address_mapping_class",
                value=mapping_class,
                fit_status=fit,
                uncertainty=uncertainty,
                assumptions=assumptions,
            ),
            backend_interpretation=BackendInterpretation(
                concept="address_partition_mapping",
                interpretation={
                    "nvidia_backend": "partition/slice periodicity from base-stride bandwidth variation",
                    "bandwidth_ratio": ratio,
                    "address_mapping_class": mapping_class,
                },
                sass_validation=sass.to_dict() if sass else {},
                downgrade_reason=downgrade_reason,
            ),
            simulator_estimate=SimulatorEstimate(
                parameter="address_mapping_class",
                value=mapping_class,
                evidence_tier=EvidenceTier.TIMING_DIRECT,
                fit_status=fit,
                uncertainty=uncertainty,
                mapping_contract="base-stride bandwidth variation -> simulator address-partition mapping class (candidate/behavioral)",
                assumptions=assumptions,
            ),
        )
    ]
