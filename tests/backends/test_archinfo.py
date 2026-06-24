"""Unit tests for the published-facts table and capability gating (no GPU)."""

from amora.backends.nvidia.archinfo import (
    ASYNC_COPY,
    TMA,
    facts_for_device,
    supports_feature,
)
from amora.backends.nvidia.cuda import NvidiaCapabilities, NvidiaDevice
from amora.probes.nvidia.baseline._sources import feature_gate


def _caps(name: str | None) -> NvidiaCapabilities:
    devices = [NvidiaDevice(index=0, name=name)] if name else []
    return NvidiaCapabilities(gpu_available=bool(devices), devices=devices)


def test_facts_for_known_devices():
    h100 = facts_for_device("NVIDIA H100 80GB HBM3")
    assert h100 is not None
    assert h100.family == "hopper" and h100.model == "h100"
    assert h100.compute_capability == (9, 0)
    assert TMA in h100.features and ASYNC_COPY in h100.features

    v100 = facts_for_device("Tesla V100-SXM2-32GB")
    assert v100.family == "volta"
    assert TMA not in v100.features
    assert ASYNC_COPY not in v100.features  # cp.async is Ampere+


def test_facts_unknown_device_is_none():
    assert facts_for_device("Some Unknown GPU") is None
    assert facts_for_device(None) is None


def test_supports_feature_tristate():
    assert supports_feature(_caps("NVIDIA H100 80GB HBM3"), TMA) is True
    assert supports_feature(_caps("Tesla V100-SXM2-32GB"), TMA) is False
    # Unknown device -> None (callers should allow, not gate).
    assert supports_feature(_caps("Mystery GPU"), TMA) is None


def test_feature_gate_blocks_missing_feature():
    caps = _caps("Tesla V100-SXM2-32GB")
    gated = feature_gate(caps, "tma_copy.async_copy_latency", ASYNC_COPY, tool_context=None)
    assert gated is not None
    result = gated[0]
    assert result.raw_observation.evidence_tier.value == "unsupported"
    assert result.raw_observation.values["required_feature"] == ASYNC_COPY


def test_feature_gate_allows_supported_and_unknown():
    # Supported device -> no gate.
    assert feature_gate(_caps("NVIDIA H100 80GB HBM3"), "p", TMA, tool_context=None) is None
    # Unknown device -> no gate (rely on the probe's own evidence).
    assert feature_gate(_caps("Mystery GPU"), "p", TMA, tool_context=None) is None
