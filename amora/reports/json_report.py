"""JSON report rendering for AMORA probe results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from amora.schemas.results import ProbeResult, _clean


_CAPABILITIES_REF = {"$ref": "metadata.backend_capabilities"}


def _dedupe_tool_context(
    result: dict[str, Any],
    capabilities_clean: Any,
) -> dict[str, Any]:
    """Replace duplicate tool snapshots with a `$ref` to keep reports compact."""

    if capabilities_clean is None:
        return result
    tool_context = result.get("tool_context")
    if not isinstance(tool_context, dict):
        return result
    tools = tool_context.get("tools")
    if tools == capabilities_clean:
        tool_context = dict(tool_context)
        tool_context["tools"] = dict(_CAPABILITIES_REF)
        result = dict(result)
        result["tool_context"] = tool_context
    return result


def render_report(
    results: Iterable[ProbeResult],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    capabilities = metadata.get("backend_capabilities") if metadata else None
    capabilities_clean = _clean(capabilities) if capabilities is not None else None
    rendered = [
        _dedupe_tool_context(result.to_dict(), capabilities_clean) for result in results
    ]
    return {
        "schema_version": 1,
        "metadata": metadata,
        "results": rendered,
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
