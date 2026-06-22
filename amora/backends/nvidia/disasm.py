"""Binary and disassembly helpers."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_disassembler(disassembler: str, binary: str | Path) -> tuple[int, str, str]:
    completed = subprocess.run(
        [disassembler, str(binary)],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.returncode, completed.stdout, completed.stderr
