"""Exact runtime identity; package availability never implies TPU execution."""
from __future__ import annotations

import importlib.metadata
import platform
import re
from pathlib import Path

from .artifacts import digest, file_hash


def runtime_identity():
    packages = {}
    for name in ("jax", "jaxlib", "libtpu", "numpy", "xprof"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    identity = {"python": platform.python_version(), "platform": platform.platform(),
                "packages": packages,
                "runtime_lock_sha256": file_hash("runtime-lock.txt") if Path("runtime-lock.txt").is_file() else None}
    verified = False
    if identity["runtime_lock_sha256"]:
        requirements = [line for line in Path("runtime-lock.txt").read_text().splitlines()
                        if line.strip() and not line.startswith("#")]
        verified = bool(requirements)
        for line in requirements:
            match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+) --hash=sha256:[0-9a-f]{64}", line)
            if not match:
                verified = False
                continue
            try:
                verified &= importlib.metadata.version(match[1]) == match[2]
            except importlib.metadata.PackageNotFoundError:
                verified = False
    identity["runtime_lock_versions_verified"] = bool(verified)
    identity["runtime_digest"] = digest(identity)
    return identity
