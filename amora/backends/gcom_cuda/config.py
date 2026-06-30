"""Configuration and SKU profiles for the gcom_cuda backend.

The backend is generic. SKU-specific constants (config file paths, hardware
reference, architecture scope) live in :data:`SKU_PROFILES`, selected by name at
runtime. Do not hardcode H100 specifics into backend APIs — add a SKU profile.

Paths resolve from ``GCOM_ROOT`` (env override) and are repo-relative for
outputs. Missing files are reported by capability discovery, never assumed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _gcom_root() -> Path:
    """GCoM simulator checkout root (env override, sensible default)."""

    env = os.environ.get("GCOM_ROOT")
    if env:
        return Path(env).expanduser()
    return Path.home() / "wk" / "modern-gpu-simulator-micro-2025" / "simulator-remodeled"


GCOM_ROOT = _gcom_root()

# Simulator binary and tracer locations within the GCoM checkout.
SIM_BIN = GCOM_ROOT / "gpu-simulator" / "bin" / "release" / "accel-sim.out"
TRACER_DIR = GCOM_ROOT / "util" / "tracer_nvbit"
SIM_SETUP_SCRIPT = GCOM_ROOT / "gpu-simulator" / "setup_environment_no_git.sh"

# Repo-relative output + report roots.
OUT_ROOT = Path(os.environ.get("AMORA_OUT_ROOT", "out")) / "gcom_cuda"
REPORTS_ROOT = Path("reports") / "gcom_cuda"

# Compilation / clock constants (generic defaults; SKU may override clocks).
NVCC_ARCH = os.environ.get("AMORA_GCOM_NVCC_ARCH", "sm_90")
DRAM_ATOM_BYTES = 32  # GPGPU-Sim DRAM transaction atom size.


@dataclass(frozen=True)
class SkuProfile:
    """A simulator SKU (e.g. gcom_h100) within a family (e.g. hopper)."""

    sku: str
    family: str
    architecture_scope: str
    gpgpusim_config: Path
    trace_config: Path
    core_clock_hz: float
    dram_clock_hz: float
    # The real-hardware reference this SKU is compared against.
    hw_backend: str = "nvidia"
    hw_family: str = "hopper"
    hw_sku: str = "h100-80g"
    hw_boost_clock_hz: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "family": self.family,
            "architecture_scope": self.architecture_scope,
            "gpgpusim_config": str(self.gpgpusim_config),
            "trace_config": str(self.trace_config),
            "core_clock_hz": self.core_clock_hz,
            "dram_clock_hz": self.dram_clock_hz,
            "hardware_reference": {
                "backend": self.hw_backend,
                "family": self.hw_family,
                "sku": self.hw_sku,
                "boost_clock_hz": self.hw_boost_clock_hz,
            },
        }

    def config_exists(self) -> bool:
        return self.gpgpusim_config.exists() and self.trace_config.exists()


_TESTED_CFGS = GCOM_ROOT / "gpu-simulator" / "gpgpu-sim" / "configs" / "tested-cfgs"
_TRACE_CFGS = GCOM_ROOT / "gpu-simulator" / "configs" / "tested-cfgs"


SKU_PROFILES: dict[str, SkuProfile] = {
    "gcom_h100": SkuProfile(
        sku="gcom_h100",
        family="hopper",
        architecture_scope="nvidia_hopper",
        gpgpusim_config=_TESTED_CFGS / "SM90_H100_L2_50MB_80GB" / "gpgpusim.config",
        trace_config=_TRACE_CFGS / "SM90_H100_L2_50MB_80GB" / "trace.config",
        core_clock_hz=1800e6,
        dram_clock_hz=8000e6,
        hw_backend="nvidia",
        hw_family="hopper",
        hw_sku="h100-80g",
        hw_boost_clock_hz=1830e6,
    ),
}

DEFAULT_SKU = "gcom_h100"


def get_sku_profile(sku: str | None = None) -> SkuProfile:
    """Return the named SKU profile (default: gcom_h100). Raises on unknown SKU."""

    name = sku or DEFAULT_SKU
    if name not in SKU_PROFILES:
        known = ", ".join(sorted(SKU_PROFILES))
        raise KeyError(f"unknown gcom_cuda SKU {name!r}; known: {known}")
    return SKU_PROFILES[name]


def run_output_dir(profile: SkuProfile, run_id: str) -> Path:
    """Per-run archival directory: out/gcom_cuda/<family>/<sku>/<run_id>/."""

    return OUT_ROOT / profile.family / profile.sku / run_id
