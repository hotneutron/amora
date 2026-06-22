"""JSON report rendering for AMORA probe results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from amora.schemas.results import ProbeResult


def render_report(results: Iterable[ProbeResult], *, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    result_list = list(results)
    return {
        "schema_version": 1,
        "metadata": metadata or {},
        "results": [result.to_dict() for result in result_list],
    }


def write_report(
    path: str | Path,
    results: Iterable[ProbeResult],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = render_report(results, metadata=metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
