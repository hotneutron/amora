"""Shared pytest configuration: auto-skip GPU-gated markers when hardware is absent."""

from __future__ import annotations

import pytest

from amora.backends.nvidia.cuda import discover_capabilities


def _capability_skip_reason() -> tuple[str | None, str | None, str | None]:
    """Return (cuda_skip_reason, ncu_skip_reason, nvbit_skip_reason) for the current host."""

    try:
        caps = discover_capabilities()
    except Exception as exc:  # pragma: no cover - defensive
        msg = f"capability discovery failed: {exc!r}"
        return msg, msg, msg

    cuda_reason = None
    if not caps.gpu_available:
        cuda_reason = "no CUDA-capable GPU reported by nvidia-smi"
    elif not caps.cuda_available:
        cuda_reason = "nvcc not available on PATH"

    ncu_reason = None
    if "ncu" not in caps.tools or not caps.tools["ncu"].available:
        ncu_reason = "Nsight Compute (ncu) is not available on PATH"

    # AMORA does not yet shell out to NVBit, so we treat it as always-skip until wired up.
    nvbit_reason = "NVBit integration not implemented; no runtime probe to gate"

    return cuda_reason, ncu_reason, nvbit_reason


_CUDA_SKIP, _NCU_SKIP, _NVBIT_SKIP = _capability_skip_reason()


def pytest_collection_modifyitems(config, items):  # noqa: D401 - pytest hook
    """Auto-skip tests carrying the cuda/ncu/nvbit markers on hosts that can't run them."""

    skip_map = {
        "cuda": pytest.mark.skip(reason=_CUDA_SKIP) if _CUDA_SKIP else None,
        "ncu": pytest.mark.skip(reason=_NCU_SKIP) if _NCU_SKIP else None,
        "nvbit": pytest.mark.skip(reason=_NVBIT_SKIP) if _NVBIT_SKIP else None,
    }
    for item in items:
        for marker_name, skip in skip_map.items():
            if skip is None:
                continue
            if marker_name in item.keywords:
                item.add_marker(skip)
