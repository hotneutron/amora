"""NVIDIA CUDA capability discovery."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from typing import Any


def _run(args: list[str], *, timeout: int = 10) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


@dataclass(frozen=True)
class ToolStatus:
    name: str
    path: str | None
    available: bool
    version: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "path": self.path,
            "available": self.available,
            "version": self.version,
            "error": self.error,
        }


@dataclass(frozen=True)
class NvidiaDevice:
    index: int
    name: str
    uuid: str | None = None
    driver_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "driver_version": self.driver_version,
        }


@dataclass(frozen=True)
class NvidiaCapabilities:
    backend: str = "nvidia_cuda"
    cuda_available: bool = False
    gpu_available: bool = False
    tools: dict[str, ToolStatus] = field(default_factory=dict)
    devices: list[NvidiaDevice] = field(default_factory=list)
    unsupported_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cuda_available": self.cuda_available,
            "gpu_available": self.gpu_available,
            "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
            "devices": [device.to_dict() for device in self.devices],
            "unsupported_reasons": list(self.unsupported_reasons),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def discover_tool(name: str, version_args: list[str] | None = None) -> ToolStatus:
    path = shutil.which(name)
    if not path:
        return ToolStatus(name=name, path=None, available=False, error=f"{name} not found")
    if not version_args:
        return ToolStatus(name=name, path=path, available=True)
    try:
        completed = _run([path, *version_args])
    except Exception as exc:  # pragma: no cover - defensive subprocess path
        return ToolStatus(name=name, path=path, available=True, error=str(exc))
    output = _first_line(completed.stdout) or _first_line(completed.stderr)
    return ToolStatus(
        name=name,
        path=path,
        available=True,
        version=output,
        error=None if completed.returncode == 0 else output,
    )


def _discover_devices(nvidia_smi: ToolStatus) -> tuple[list[NvidiaDevice], str | None]:
    if not nvidia_smi.available or not nvidia_smi.path:
        return [], "nvidia-smi not available"
    query = [
        nvidia_smi.path,
        "--query-gpu=index,name,uuid,driver_version",
        "--format=csv,noheader,nounits",
    ]
    completed = _run(query)
    if completed.returncode != 0:
        return [], _first_line(completed.stderr) or "nvidia-smi query failed"
    devices: list[NvidiaDevice] = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            index = int(parts[0])
        except ValueError:
            continue
        devices.append(
            NvidiaDevice(
                index=index,
                name=parts[1],
                uuid=parts[2] if len(parts) > 2 and parts[2] else None,
                driver_version=parts[3] if len(parts) > 3 and parts[3] else None,
            )
        )
    if not devices:
        return [], "no NVIDIA GPUs reported by nvidia-smi"
    return devices, None


def discover_capabilities() -> NvidiaCapabilities:
    tools = {
        "nvcc": discover_tool("nvcc", ["--version"]),
        "nvidia-smi": discover_tool("nvidia-smi", ["--version"]),
        "ncu": discover_tool("ncu", ["--version"]),
        "nvdisasm": discover_tool("nvdisasm", ["--version"]),
        "cuobjdump": discover_tool("cuobjdump", ["--version"]),
    }
    devices, device_error = _discover_devices(tools["nvidia-smi"])
    reasons = []
    if not tools["nvcc"].available:
        reasons.append("nvcc not available")
    if device_error:
        reasons.append(device_error)
    cuda_available = tools["nvcc"].available
    gpu_available = bool(devices)
    return NvidiaCapabilities(
        cuda_available=cuda_available,
        gpu_available=gpu_available,
        tools=tools,
        devices=devices,
        unsupported_reasons=reasons,
    )
