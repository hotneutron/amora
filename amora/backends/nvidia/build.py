"""CUDA build command construction for NVIDIA probes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CudaBuildConfig:
    source: Path
    output: Path
    arch: str = "sm_80"
    nvcc: str = "nvcc"
    extra_flags: tuple[str, ...] = field(default_factory=tuple)

    def argv(self) -> list[str]:
        return [
            self.nvcc,
            "-arch",
            self.arch,
            "-cubin",
            str(self.source),
            "-o",
            str(self.output),
            *self.extra_flags,
        ]


def default_output_for(source: str | Path, build_root: str | Path = "out/build/nvidia/baseline") -> Path:
    src = Path(source)
    return Path(build_root) / f"{src.stem}.cubin"
