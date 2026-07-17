"""Registry for benchmark definition providers."""

from __future__ import annotations

from typing import Any


def _definitions() -> dict[str, Any]:
    from benchmark_generators.ppp_canonical.generator import PPP_CANONICAL

    return {PPP_CANONICAL.benchmark_id: PPP_CANONICAL}


def get_benchmark(benchmark_id: str) -> Any:
    """Return a registered benchmark definition by stable ID."""

    try:
        return _definitions()[benchmark_id]
    except KeyError as exc:
        known = ", ".join(sorted(_definitions()))
        raise KeyError(f"unknown benchmark {benchmark_id!r}; known: {known}") from exc


def list_benchmarks() -> list[dict[str, Any]]:
    """Return compact benchmark definition metadata for CLI discovery."""

    return [
        definition.describe()
        for _, definition in sorted(_definitions().items())
    ]
