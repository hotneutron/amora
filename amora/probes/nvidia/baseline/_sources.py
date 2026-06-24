"""Helpers for registering CUDA kernel sources alongside probe results."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a CUDA source on disk."""

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def source_descriptor(path: Path) -> dict[str, object]:
    """Describe a CUDA source for the probe report."""

    return {
        "kind": "cuda_source",
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
