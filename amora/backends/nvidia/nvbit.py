"""NVBit integration placeholders."""

from __future__ import annotations

import os
from pathlib import Path


def discover_nvbit(path: str | None = None) -> dict[str, object]:
    candidate = path or os.environ.get("NVBIT_ROOT")
    if not candidate:
        return {
            "available": False,
            "path": None,
            "reason": "NVBIT_ROOT is not set",
        }
    root = Path(candidate)
    return {
        "available": root.exists(),
        "path": str(root),
        "reason": None if root.exists() else "NVBit path does not exist",
    }
