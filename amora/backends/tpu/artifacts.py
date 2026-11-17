"""Canonical hashes and immutable, validated TPU evidence."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


def stamp():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError("identifier must contain letters, digits, '.', '_' or '-'")
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path


def finish_manifest(root, metadata):
    root = Path(root)
    artifacts = {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*"))
                 if p.is_file() and p.name != "manifest.json"}
    manifest = {"schema_version": 1, **metadata, "artifacts": artifacts}
    manifest["run_digest"] = digest(manifest)
    write_json(root / "manifest.json", manifest)
    return manifest


def validate_manifest(path):
    path = Path(path)
    manifest = json.loads(path.read_text())
    original = manifest.pop("run_digest")
    if digest(manifest) != original:
        raise ValueError("run manifest digest mismatch")
    root = path.parent.resolve()
    for name, expected in manifest["artifacts"].items():
        candidate = (root / name).resolve()
        if not candidate.is_relative_to(root) or not candidate.is_file():
            raise ValueError("missing or escaping artifact path")
        if file_hash(candidate) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    manifest["run_digest"] = original
    return manifest
