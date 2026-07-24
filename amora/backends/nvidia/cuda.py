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
    ncu_metrics: frozenset[str] = field(default_factory=frozenset)
    ncu_metrics_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "cuda_available": self.cuda_available,
            "gpu_available": self.gpu_available,
            "tools": {name: tool.to_dict() for name, tool in self.tools.items()},
            "devices": [device.to_dict() for device in self.devices],
            "unsupported_reasons": list(self.unsupported_reasons),
            "ncu_counters_available": bool(self.ncu_metrics),
            "ncu_metric_count": len(self.ncu_metrics),
            "ncu_metrics_error": self.ncu_metrics_error,
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


def _parse_ncu_metric_names(text: str) -> frozenset[str]:
    names: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("Chip ", "---", "Metric Name")):
            continue
        token = stripped.split()[0]
        if token.startswith("-"):
            token = token[1:]
        if token.startswith("arch:"):
            parts = token.split(":")
            token = parts[-1] if len(parts) >= 3 else token
        if token.startswith("breakdown:"):
            token = token.split(":", 1)[1]
        # Lines look like "<metric_name>   <description>" for query output and
        # include optional availability-qualified aliases for list output.
        if "." in token or "__" in token:
            names.add(token)
    return frozenset(names)


def _ncu_chip_for_device(device: NvidiaDevice) -> str | None:
    name = device.name.lower()
    if "v100" in name or "gv100" in name:
        return "gv100"
    if "h100" in name or "h800" in name or "gh100" in name:
        return "gh100"
    if "a100" in name or "a800" in name or "ga100" in name:
        return "ga100"
    if "l40" in name or "rtx 6000 ada" in name:
        return "ad102"
    if "l4" in name:
        return "ad104"
    if "t4" in name:
        return "tu104"
    return None


def _error_line(completed: subprocess.CompletedProcess[str]) -> str | None:
    text = "\n".join(part for part in (completed.stderr, completed.stdout) if part)
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("==ERROR==") or "ERR_NVGPUCTRPERM" in stripped:
            return stripped
    return None


def _discover_ncu_metrics(
    ncu: ToolStatus,
    devices: list[NvidiaDevice] | None = None,
) -> tuple[frozenset[str], str | None]:
    """Best-effort capture of NCU's supported metric names and discovery error."""

    if not ncu.available or not ncu.path:
        return frozenset(), "ncu not available"
    chip = _ncu_chip_for_device(devices[0]) if devices else None
    if chip:
        try:
            offline = _run(
                [ncu.path, "--query-metrics", "--chips", chip, "--query-metrics-mode", "base"],
                timeout=30,
            )
        except Exception as exc:  # pragma: no cover - defensive subprocess path
            return frozenset(), str(exc)
        offline_error = _error_line(offline)
        if offline.returncode == 0 and not offline_error:
            metrics = _parse_ncu_metric_names(offline.stdout)
            if metrics:
                return metrics, None
        if offline_error:
            return frozenset(), offline_error

    try:
        completed = _run([ncu.path, "--query-metrics"], timeout=30)
    except Exception as exc:  # pragma: no cover - defensive subprocess path
        return frozenset(), str(exc)
    query_error = _error_line(completed)
    if completed.returncode == 0 and not query_error:
        return _parse_ncu_metric_names(completed.stdout), None

    query_error = query_error or _first_line(completed.stderr) or _first_line(completed.stdout) or "ncu --query-metrics failed"
    # Some NCU versions require counter permissions for query-metrics but can
    # still list the catalog. Preserve the query error so classification can
    # report the real blocker if required counters are absent.
    try:
        listed = _run([ncu.path, "--list-metrics"], timeout=30)
    except Exception:
        return frozenset(), query_error
    if listed.returncode == 0:
        return _parse_ncu_metric_names(listed.stdout), query_error
    return frozenset(), query_error


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
    ncu_metrics, ncu_metrics_error = _discover_ncu_metrics(tools["ncu"], devices)
    return NvidiaCapabilities(
        cuda_available=cuda_available,
        gpu_available=gpu_available,
        tools=tools,
        devices=devices,
        unsupported_reasons=reasons,
        ncu_metrics=ncu_metrics,
        ncu_metrics_error=ncu_metrics_error,
    )
