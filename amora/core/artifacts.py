"""Artifact hashing and source discovery helpers."""

from __future__ import annotations

from hashlib import sha256
from importlib import resources
from pathlib import Path


def sha256_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()


def read_package_text(package: str, resource_name: str) -> str:
    return resources.files(package).joinpath(resource_name).read_text(encoding="utf-8")


def write_text_if_changed(path: Path, text: str) -> bool:
    """Write text and return True when the file contents changed."""

    if path.exists() and path.read_text(encoding="utf-8") == text:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return True
