"""CUDA/NVIDIA command discovery and capability checks."""

from __future__ import annotations

from dataclasses import dataclass
from shutil import which
from subprocess import CompletedProcess, run


@dataclass(frozen=True)
class ToolStatus:
    name: str
    path: str | None
    available: bool

    def to_dict(self) -> dict[str, str | bool | None]:
        return {"name": self.name, "path": self.path, "available": self.available}


class CudaToolchain:
    """Small wrapper around optional NVIDIA command-line tools."""

    REQUIRED = ("nvcc",)
    OPTIONAL = ("ncu", "nvdisasm", "cuobjdump")

    def __init__(self, search_path: str | None = None) -> None:
        self.search_path = search_path

    def find(self, tool: str) -> ToolStatus:
        path = which(tool, path=self.search_path)
        return ToolStatus(name=tool, path=path, available=path is not None)

    def inventory(self) -> dict[str, ToolStatus]:
        return {tool: self.find(tool) for tool in (*self.REQUIRED, *self.OPTIONAL)}

    def has_cuda_compiler(self) -> bool:
        return self.find("nvcc").available

    def run_version(self, tool: str) -> CompletedProcess[str] | None:
        status = self.find(tool)
        if not status.path:
            return None
        return run(
            [status.path, "--version"],
            check=False,
            capture_output=True,
            text=True,
        )

    def target_summary(self) -> dict[str, object]:
        inventory = self.inventory()
        return {
            "vendor": "nvidia",
            "tools": {name: status.to_dict() for name, status in inventory.items()},
        }
