"""gcom_cuda capability discovery.

Peer of ``amora/backends/nvidia/cuda.py``: reports the status of the simulator
binary, NVBit tracer, CUDA compiler, real GPU (tracing needs one), and the SKU
config profile. Reuses nvidia's tool/device discovery so the two backends agree
on device identity.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from amora.backends.gcom_cuda import config as cfg
from amora.backends.nvidia.cuda import (
    NvidiaDevice,
    ToolStatus,
    _discover_devices,
    discover_tool,
)


@dataclass(frozen=True)
class GcomCapabilities:
    backend: str = "gcom_cuda"
    sku: str = cfg.DEFAULT_SKU
    family: str = "hopper"
    simulator_built: bool = False
    tracer_built: bool = False
    nvcc_available: bool = False
    gpu_available: bool = False
    config_present: bool = False
    tools: dict[str, ToolStatus] = field(default_factory=dict)
    devices: list[NvidiaDevice] = field(default_factory=list)
    unsupported_reasons: list[str] = field(default_factory=list)
    sku_profile: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "sku": self.sku,
            "family": self.family,
            "simulator_built": self.simulator_built,
            "tracer_built": self.tracer_built,
            "nvcc_available": self.nvcc_available,
            "gpu_available": self.gpu_available,
            "config_present": self.config_present,
            "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
            "devices": [device.to_dict() for device in self.devices],
            "unsupported_reasons": list(self.unsupported_reasons),
            "sku_profile": self.sku_profile,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def _tracer_built() -> bool:
    """Tracer is considered built if its NVBit injection .so is present."""

    if not cfg.TRACER_DIR.exists():
        return False
    return any(cfg.TRACER_DIR.rglob("*.so"))


def discover_capabilities(sku: str | None = None) -> GcomCapabilities:
    profile = cfg.get_sku_profile(sku)
    tools = {
        "nvcc": discover_tool("nvcc", ["--version"]),
        "nvidia-smi": discover_tool("nvidia-smi", ["--version"]),
    }
    devices, device_error = _discover_devices(tools["nvidia-smi"])

    simulator_built = cfg.SIM_BIN.exists()
    tracer_built = _tracer_built()
    config_present = profile.config_exists()

    reasons: list[str] = []
    if not simulator_built:
        reasons.append(f"simulator binary not built ({cfg.SIM_BIN})")
    if not tracer_built:
        reasons.append(f"NVBit tracer not built ({cfg.TRACER_DIR})")
    if not tools["nvcc"].available:
        reasons.append("nvcc not available")
    if device_error:
        reasons.append(f"trace generation needs a GPU: {device_error}")
    if not config_present:
        reasons.append(f"SKU config missing for {profile.sku}")

    return GcomCapabilities(
        sku=profile.sku,
        family=profile.family,
        simulator_built=simulator_built,
        tracer_built=tracer_built,
        nvcc_available=tools["nvcc"].available,
        gpu_available=bool(devices),
        config_present=config_present,
        tools=tools,
        devices=devices,
        unsupported_reasons=reasons,
        sku_profile=profile.to_dict(),
    )
