"""Version + provenance contract for gcom_cuda runs.

Every simulated run records enough provenance to reproduce and re-parse it: git
commits (AMORA and GCoM), simulator binary identity, tracer/CUDA/driver
versions, the SKU profile, and config-file hashes. All collection is best
effort — absent items are recorded as ``None`` rather than raising.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Any

from amora.backends.gcom_cuda import config as cfg


def _run(args: list[str], *, cwd: Path | None = None, timeout: int = 10) -> str | None:
    try:
        completed = subprocess.run(
            args, check=False, capture_output=True, text=True, timeout=timeout,
            cwd=str(cwd) if cwd else None,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    out = completed.stdout.strip()
    return out or None


def _git_info(repo: Path) -> dict[str, Any]:
    if not repo.exists():
        return {"available": False}
    commit = _run(["git", "rev-parse", "HEAD"], cwd=repo)
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo)
    status = _run(["git", "status", "--porcelain"], cwd=repo)
    return {
        "available": commit is not None,
        "commit": commit,
        "branch": branch,
        "dirty": bool(status) if status is not None else None,
    }


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _amora_repo_root() -> Path:
    # amora/backends/gcom_cuda/version.py -> repo root is three parents up.
    return Path(__file__).resolve().parents[3]


# Environment variables that affect tracing/simulation provenance.
_PROVENANCE_ENV = (
    "GCOM_ROOT", "CUDA_INSTALL_PATH", "LD_LIBRARY_PATH",
    "AMORA_GCOM_NVCC_ARCH", "OMP_NUM_THREADS",
)


def collect_version_metadata(profile: cfg.SkuProfile, *, devices: list | None = None) -> dict[str, Any]:
    """Assemble the full provenance record for a gcom_cuda run."""

    sim_bin = cfg.SIM_BIN
    sim_mtime = None
    if sim_bin.exists():
        sim_mtime = sim_bin.stat().st_mtime

    nvcc_version = _run(["nvcc", "--version"])
    driver_version = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])

    return {
        "mapping_version": None,  # filled by metrics_map consumers when relevant
        "amora_git": _git_info(_amora_repo_root()),
        "gcom_git": _git_info(cfg.GCOM_ROOT),
        "simulator": {
            "binary_path": str(sim_bin),
            "exists": sim_bin.exists(),
            "build_mtime": sim_mtime,
        },
        "toolkit": {
            "nvcc_version": (nvcc_version.splitlines()[-1] if nvcc_version else None),
            "driver_version": driver_version,
        },
        "sku_profile": profile.to_dict(),
        "config_hashes": {
            "gpgpusim_config": _sha256(profile.gpgpusim_config),
            "trace_config": _sha256(profile.trace_config),
        },
        "gpu_devices": devices if devices is not None else None,
        "environment": {k: os.environ.get(k) for k in _PROVENANCE_ENV},
    }
