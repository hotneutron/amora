"""Markdown report rendering for AMORA probe results.

Turns an AMORA JSON report (as produced by :mod:`amora.reports.json_report`)
into a well-organized, per-vendor / per-architecture report tree:

    reports/
        README.md                       # top-level index of vendors
        <vendor>/                       # e.g. nvidia
            SUMMARY.md                  # compiled cross-SKU summary
            <family>/                   # architecture family, e.g. hopper
                README.md               # per-SKU outcome tables
                manifest.json           # run metadata, keyed by SKU
                environment.md          # host toolchain + devices, keyed by SKU
                probes-<sku>.md         # ALL probes for one SKU in one file

A "generation" is a single probe run (all probes P0-P3 together) for one GPU
SKU (model + memory, e.g. ``h100-80g``). Multiple SKUs of the same architecture
family share the family folder; each SKU gets its own consolidated
``probes-<sku>.md`` and its own entry in the shared ``manifest.json``.

The renderer is data-driven: it reflects whatever the JSON contains.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from amora.reports.probe_groups import group_for_probe, vendor_groups


# --------------------------------------------------------------------------- #
# Small formatting helpers
# --------------------------------------------------------------------------- #


def _get(d: Any, *path: str, default: Any = None) -> Any:
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _fmt_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _fmt_cell(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, sort_keys=True)}`"
    if isinstance(value, str) and not value:
        return "—"
    return _fmt_scalar(value)


def _shape(launch: dict[str, Any]) -> str:
    grid = launch.get("grid")
    block = launch.get("block")
    if grid is None and block is None:
        return "—"
    return f"grid={grid} block={block}"


def _counts_str(counts: dict[str, int]) -> str:
    return ", ".join(f"`{k}`={v}" for k, v in sorted(counts.items())) or "—"


def _anchor(probe_id: str) -> str:
    """GitHub-style anchor for a probe heading."""
    return re.sub(r"[^a-z0-9_]+", "", probe_id.lower().replace(".", ""))


def _measurement_str(value: Any, unit: str | None) -> str:
    """Compact one-line measurement for index/summary tables."""
    if value is None:
        return "—"
    if isinstance(value, dict):
        keys = ", ".join(sorted(value))
        return f"_object_ ({keys})" if keys else "_object_"
    if isinstance(value, list):
        return f"_list_ ({len(value)} items)"
    text = _fmt_scalar(value)
    return f"{text} {unit}".strip() if unit else text


def _kv_table(title: str, mapping: dict[str, Any], units: dict[str, str] | None = None, *, level: int = 3) -> list[str]:
    if not mapping:
        return []
    units = units or {}
    hashes = "#" * level
    lines = [f"{hashes} {title}", "", "| key | value | unit |", "| --- | --- | --- |"]
    for key in sorted(mapping):
        unit = units.get(key, "")
        lines.append(f"| `{key}` | {_fmt_cell(mapping[key])} | {unit or '—'} |")
    lines.append("")
    return lines


def _list_block(title: str, items: list[Any]) -> list[str]:
    if not items:
        return []
    lines = [f"**{title}:**", ""]
    lines.extend(f"- {item}" for item in items)
    lines.append("")
    return lines


def _is_table_of_rows(value: Any) -> bool:
    """True for a non-empty list of flat dicts (renderable as a table)."""
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(row, dict) for row in value)
        and all(not isinstance(v, (dict, list)) for row in value for v in row.values())
    )


def _rows_table(title: str, rows: list[dict[str, Any]], *, level: int = 4) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    hashes = "#" * level
    lines = [f"{hashes} {title} ({len(rows)} rows)", "", header, sep]
    for row in rows:
        lines.append("| " + " | ".join(_fmt_cell(row.get(col)) for col in columns) + " |")
    lines.append("")
    return lines


def _details_json(summary: str, value: Any) -> list[str]:
    return [
        f"<details><summary>{summary}</summary>",
        "",
        "```json",
        json.dumps(value, indent=2, sort_keys=True),
        "```",
        "",
        "</details>",
        "",
    ]


def _render_values(values: dict[str, Any]) -> list[str]:
    """Render the full probe-specific raw ``values`` payload.

    Scalars go into one flat table; list-of-dict sweeps become their own
    tables; large nested structures are tucked into <details> JSON blocks.
    ``registered_source`` is rendered separately by the caller.
    """
    if not values:
        return []
    scalars: dict[str, Any] = {}
    tables: list[tuple[str, list[dict[str, Any]]]] = []
    blocks: list[tuple[str, Any]] = []
    for key in sorted(values):
        if key == "registered_source":
            continue
        val = values[key]
        if _is_table_of_rows(val):
            tables.append((key, val))
        elif isinstance(val, (dict, list)):
            blocks.append((key, val))
        else:
            scalars[key] = val

    lines: list[str] = ["### Raw values", ""]
    if scalars:
        lines += ["| key | value |", "| --- | --- |"]
        for key in sorted(scalars):
            lines.append(f"| `{key}` | {_fmt_cell(scalars[key])} |")
        lines.append("")
    for name, rows in tables:
        lines += _rows_table(f"`{name}`", rows)
    for name, val in blocks:
        lines += _details_json(f"<code>{name}</code> (JSON)", val)
    return lines


# --------------------------------------------------------------------------- #
# Vendor / family / SKU derivation
# --------------------------------------------------------------------------- #


_VENDOR_BY_BACKEND = {"nvidia_cuda": "nvidia"}

# (regex on device name) -> (family, model-slug)
_MODEL_PATTERNS = [
    (re.compile(r"\bB300\b", re.I), ("blackwell-ultra", "b300")),
    (re.compile(r"\bGB300\b", re.I), ("blackwell-ultra", "gb300")),
    (re.compile(r"\bGB200\b", re.I), ("blackwell", "gb200")),
    (re.compile(r"\bB200\b", re.I), ("blackwell", "b200")),
    (re.compile(r"\bB100\b", re.I), ("blackwell", "b100")),
    (re.compile(r"\bGH200\b", re.I), ("hopper", "gh200")),
    (re.compile(r"\bH200\b", re.I), ("hopper", "h200")),
    (re.compile(r"\bH100\b", re.I), ("hopper", "h100")),
    (re.compile(r"\bH20\b", re.I), ("hopper", "h20")),
    (re.compile(r"\bL40S\b", re.I), ("ada", "l40s")),
    (re.compile(r"\bL40\b", re.I), ("ada", "l40")),
    (re.compile(r"\bL4\b", re.I), ("ada", "l4")),
    (re.compile(r"\bA800\b", re.I), ("ampere", "a800")),
    (re.compile(r"\bA100\b", re.I), ("ampere", "a100")),
    (re.compile(r"\bA30\b", re.I), ("ampere", "a30")),
    (re.compile(r"\bA10\b", re.I), ("ampere", "a10")),
    (re.compile(r"\bT4\b", re.I), ("turing", "t4")),
]

_MEM_PATTERN = re.compile(r"(\d+)\s*GB", re.I)


def derive_vendor(report: dict[str, Any]) -> str:
    backend = _get(report, "metadata", "backend_capabilities", "backend", default="")
    if backend in _VENDOR_BY_BACKEND:
        return _VENDOR_BY_BACKEND[backend]
    return backend.split("_", 1)[0] or "vendor"


def _primary_device_name(report: dict[str, Any]) -> str:
    devices = _get(report, "metadata", "backend_capabilities", "devices", default=[]) or []
    return devices[0].get("name", "") if devices else ""


def derive_family_model(name: str) -> tuple[str, str]:
    for pattern, (family, model) in _MODEL_PATTERNS:
        if pattern.search(name):
            return family, model
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return "unknown", (slug or "unknown")


def derive_sku(name: str, model: str) -> str:
    """e.g. ('NVIDIA H100 80GB HBM3', 'h100') -> 'h100-80g'."""
    mem = _MEM_PATTERN.search(name)
    return f"{model}-{mem.group(1)}g" if mem else model


def derive_layout(report: dict[str, Any]) -> tuple[str, str, str]:
    """Return (vendor, family, sku) derived from the report."""
    name = _primary_device_name(report)
    family, model = derive_family_model(name)
    return derive_vendor(report), family, derive_sku(name, model)


# --------------------------------------------------------------------------- #
# Aggregations
# --------------------------------------------------------------------------- #


def _aggregate_counts(results: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    tier: dict[str, int] = {}
    fit: dict[str, int] = {}
    mode: dict[str, int] = {}
    for res in results:
        t = _get(res, "raw_observation", "evidence_tier", default="unknown")
        f = _get(res, "normalized_measurement", "fit_status", default="unknown")
        m = _get(res, "launch", "mode", default="unknown")
        tier[t] = tier.get(t, 0) + 1
        fit[f] = fit.get(f, 0) + 1
        mode[m] = mode.get(m, 0) + 1
    return {"launch_mode": mode, "evidence_tier": tier, "fit_status": fit}


# --------------------------------------------------------------------------- #
# Per-probe section (inside the consolidated probes-<sku>.md)
# --------------------------------------------------------------------------- #


def _build_probe_section(result: dict[str, Any]) -> list[str]:
    identity = result.get("identity", {})
    probe_id = identity.get("probe_id", "unknown")
    launch = result.get("launch", {})
    raw = result.get("raw_observation", {})
    norm = result.get("normalized_measurement", {})
    interp = result.get("backend_interpretation", {})
    sim = result.get("simulator_estimate", {})

    value = norm.get("value")
    summary_rows = [
        ("launch", f"`{launch.get('mode')}`  {_shape(launch)}".strip()),
        ("evidence_tier", f"`{raw.get('evidence_tier')}`"),
        ("fit_status", f"`{norm.get('fit_status')}`"),
        ("measurement", f"`{norm.get('name')}` = {_measurement_str(value, norm.get('unit'))}"),
        ("simulator_param", f"`{sim.get('parameter')}` = {_measurement_str(sim.get('value'), sim.get('unit'))}"),
        ("concept", f"`{interp.get('concept')}`"),
    ]

    lines: list[str] = [f"## {probe_id}", ""]
    lines += ["| field | value |", "| --- | --- |"]
    for key, val in summary_rows:
        lines.append(f"| {key} | {val} |")
    lines.append("")

    if identity.get("binary_hash"):
        lines.append(f"- binary_hash: `{identity['binary_hash']}`")
    if launch.get("extras"):
        lines.append(f"- launch.extras: `{json.dumps(launch['extras'], sort_keys=True)}`")
    if interp.get("interpretation", {}).get("nvidia_backend"):
        lines.append(f"- interpretation: {interp['interpretation']['nvidia_backend']}")
    if sim.get("mapping_contract"):
        lines.append(f"- mapping_contract: {sim['mapping_contract']}")
    lines.append("")

    lines += _list_block("assumptions", norm.get("assumptions", []))

    # Structured measurement value (objects rendered as JSON, large ones folded).
    if isinstance(value, (dict, list)):
        lines += ["### Measurement value", ""]
        if _is_table_of_rows(value):
            lines += _rows_table("value", value)
        elif len(json.dumps(value)) > 600:
            lines += _details_json("value (JSON)", value)
        else:
            lines += ["```json", json.dumps(value, indent=2, sort_keys=True), "```", ""]

    # Metrics + full raw values (sweeps, cycle stats, etc.).
    lines += _kv_table("Metrics", raw.get("metrics", {}), raw.get("units", {}))
    lines += _render_values(raw.get("values", {}))

    # Registered source artifact (kernel-bound probes).
    reg = _get(raw, "values", "registered_source", default=None)
    if isinstance(reg, dict):
        lines += [
            "### Registered source",
            "",
            f"- path: `{reg.get('path')}`",
            f"- bytes: `{reg.get('bytes')}`  ·  sha256: `{reg.get('sha256')}`",
            "",
        ]

    lines.append("[↑ contents](#contents)")
    lines += ["", "---", ""]
    return lines


def _build_probes_doc(report: dict[str, Any], *, vendor: str, family: str, sku: str, generated_at: str) -> str:
    caps = _get(report, "metadata", "backend_capabilities", default={})
    results = report.get("results", [])
    device0 = (caps.get("devices") or [{}])[0]
    counts = _aggregate_counts(results)

    toc = " · ".join(
        f"[{_get(r, 'identity', 'probe_id')}](#{_anchor(_get(r, 'identity', 'probe_id', default=''))})"
        for r in results
    )

    lines: list[str] = [
        f"# {vendor} / {family} / {sku} — Probe Results",
        "",
        f"- Generated: {generated_at}",
        f"- Device: {device0.get('name', 'n/a')}  ·  Backend: `{caps.get('backend')}`  ·  Probes: {len(results)}",
        f"- `fit_status`: {_counts_str(counts['fit_status'])}",
        "- Back to [family index](README.md)",
        "",
        '<a id="contents"></a>',
        "## Contents",
        "",
        toc,
        "",
        "---",
        "",
    ]
    for res in results:
        lines += _build_probe_section(res)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Family-level docs (keyed by SKU, merged across runs)
# --------------------------------------------------------------------------- #


def _load_json(path: Path, default: Any) -> Any:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _sku_manifest_entry(
    report: dict[str, Any],
    *,
    sku: str,
    generated_at: str,
    source_path: str | None,
    source_sha256: str | None,
) -> dict[str, Any]:
    caps = _get(report, "metadata", "backend_capabilities", default={})
    results = report.get("results", [])
    device0 = (caps.get("devices") or [{}])[0]
    return {
        "sku": sku,
        "backend": caps.get("backend"),
        "primary_device": device0.get("name"),
        "device_count": len(caps.get("devices", [])),
        "driver_version": device0.get("driver_version"),
        "schema_version": report.get("schema_version"),
        "generated_at": generated_at,
        "source_json": source_path,
        "source_sha256": source_sha256,
        "probes_file": f"probes-{sku}.md",
        "probe_count": len(results),
        "probe_ids": [_get(r, "identity", "probe_id") for r in results],
        "counts": _aggregate_counts(results),
        "outcomes": _outcome_rows(report),
        "capabilities": caps,
        "scalars": [
            {
                "probe_id": _get(r, "identity", "probe_id"),
                "value": _get(r, "normalized_measurement", "value"),
                "unit": _get(r, "normalized_measurement", "unit"),
            }
            for r in results
            if not isinstance(_get(r, "normalized_measurement", "value"), (dict, list))
        ],
    }


def _build_family_readme(vendor: str, family: str, manifest: dict[str, Any]) -> str:
    skus = manifest.get("skus", {})
    lines: list[str] = [
        f"# AMORA Report — {vendor} / {family}",
        "",
        f"- SKUs: {len(skus)}",
        "- Metadata: [manifest.json](manifest.json) · environment: [environment.md](environment.md)",
        "",
        "## SKUs",
        "",
        "| sku | device | probes | fit_status | generated | report |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for sku in sorted(skus):
        entry = skus[sku]
        fit = _counts_str(_get(entry, "counts", "fit_status", default={}))
        lines.append(
            f"| `{sku}` | {entry.get('primary_device', '—')} | {entry.get('probe_count', 0)} | "
            f"{fit} | {entry.get('generated_at', '—')} | [probes]({entry.get('probes_file')}) |"
        )
    lines.append("")

    # Per-SKU probe outcome tables.
    for sku in sorted(skus):
        entry = skus[sku]
        outcomes = entry.get("outcomes", [])
        if not outcomes:
            continue
        lines += [
            f"## `{sku}` outcomes",
            "",
            "| probe_id | mode | evidence_tier | fit_status | measurement |",
            "| --- | --- | --- | --- | --- |",
        ]
        for row in outcomes:
            lines.append(
                f"| [{row['probe_id']}]({entry['probes_file']}#{_anchor(row['probe_id'])}) | "
                f"`{row['mode']}` | `{row['tier']}` | `{row['fit']}` | {row['measurement']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _outcome_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for res in report.get("results", []):
        value = _get(res, "normalized_measurement", "value")
        unit = _get(res, "normalized_measurement", "unit", default="")
        rows.append(
            {
                "probe_id": _get(res, "identity", "probe_id", default="unknown"),
                "mode": _get(res, "launch", "mode", default="—"),
                "tier": _get(res, "raw_observation", "evidence_tier", default="—"),
                "fit": _get(res, "normalized_measurement", "fit_status", default="—"),
                "measurement": _measurement_str(value, unit),
            }
        )
    return rows


def _build_family_environment(manifest: dict[str, Any], *, vendor: str, family: str) -> str:
    lines: list[str] = [f"# {vendor} / {family} — Host Environments", ""]
    for sku in sorted(manifest.get("skus", {})):
        caps = manifest["skus"][sku].get("capabilities", {})
        lines += [
            f"## `{sku}`",
            "",
            f"- Backend: `{caps.get('backend')}`  ·  CUDA: `{caps.get('cuda_available')}`  ·  GPU: `{caps.get('gpu_available')}`",
            "",
            "| tool | available | path | version |",
            "| --- | :-: | --- | --- |",
        ]
        for name in sorted(caps.get("tools", {})):
            tool = caps["tools"][name]
            mark = "+" if tool.get("available") else "-"
            lines.append(
                f"| `{name}` | `{mark}` | {_fmt_cell(tool.get('path'))} | {_fmt_cell(tool.get('version'))} |"
            )
        lines += ["", "| index | name | uuid | driver |", "| ---: | --- | --- | --- |"]
        for dev in caps.get("devices", []):
            lines.append(
                f"| {dev.get('index')} | {dev.get('name')} | `{dev.get('uuid')}` | {dev.get('driver_version')} |"
            )
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Vendor summary + root readme
# --------------------------------------------------------------------------- #


def _build_vendor_summary(vendor: str, vendor_dir: Path) -> str:
    families: list[tuple[str, dict[str, Any]]] = []
    for family_dir in sorted(p for p in vendor_dir.iterdir() if p.is_dir()):
        manifest = _load_json(family_dir / "manifest.json", None)
        if manifest:
            families.append((family_dir.name, manifest))

    lines: list[str] = [f"# AMORA Summary — {vendor}", ""]
    total_skus = sum(len(m.get("skus", {})) for _, m in families)
    lines += [f"- Families: {len(families)}  ·  SKUs: {total_skus}", "", "## SKUs", ""]
    lines += ["| family | sku | device | probes | fit_status | report |", "| --- | --- | --- | ---: | --- | --- |"]
    for family, manifest in families:
        for sku in sorted(manifest.get("skus", {})):
            entry = manifest["skus"][sku]
            fit = _counts_str(_get(entry, "counts", "fit_status", default={}))
            lines.append(
                f"| `{family}` | `{sku}` | {entry.get('primary_device', '—')} | "
                f"{entry.get('probe_count', 0)} | {fit} | "
                f"[probes]({family}/{entry.get('probes_file')}) |"
            )
    lines.append("")

    # Per-family, per-probe-group measurement tables.
    # Each table: rows = SKUs in the family, columns = scalar probes in the group.
    lines += [
        "## Measurement trends",
        "",
        "Per family, one table per probe group. Rows are SKUs; columns are the "
        "scalar probes in that group. Adding a SKU appends one row; adding a probe "
        "adds one column within its group.",
        "",
    ]
    for family, manifest in families:
        skus = sorted(manifest.get("skus", {}))
        if not skus:
            continue
        # Collect scalar measurements per sku: {probe_id: (value, unit)}.
        per_sku: dict[str, dict[str, tuple[Any, str | None]]] = {}
        for sku in skus:
            per_sku[sku] = {
                row["probe_id"]: (row["value"], row.get("unit"))
                for row in manifest["skus"][sku].get("scalars", [])
            }
        # Probes present in this family, grouped.
        present = sorted({pid for s in per_sku.values() for pid in s})
        groups: dict[str, dict[str, Any]] = {}
        for pid in present:
            key, label = group_for_probe(vendor, pid)
            g = groups.setdefault(key, {"label": label, "probes": []})
            g["probes"].append(pid)
        # Stable group order: follow the vendor registration, then extras.
        registered = [g.key for g in vendor_groups(vendor)]
        ordered_keys = [k for k in registered if k in groups]
        ordered_keys += [k for k in sorted(groups) if k not in registered]

        lines += [f"### {family}", ""]
        for key in ordered_keys:
            g = groups[key]
            probes = g["probes"]
            # Groups span multiple probe-id prefixes, so use the full probe_id
            # as the column header (with its unit when known).
            headers = []
            for pid in probes:
                unit = None
                for s in per_sku.values():
                    entry = s.get(pid)
                    if entry and entry[1]:
                        unit = entry[1]
                        break
                headers.append(f"{pid} ({unit})" if unit else pid)
            lines += [
                f"#### {g['label']}",
                "",
                "| sku | " + " | ".join(headers) + " |",
                "| --- | " + " | ".join("---:" for _ in probes) + " |",
            ]
            for sku in skus:
                cells = []
                for pid in probes:
                    entry = per_sku[sku].get(pid)
                    if entry is None or entry[0] is None:
                        cells.append("—")
                    else:
                        cells.append(_fmt_scalar(entry[0]))
                lines.append(f"| `{sku}` | " + " | ".join(cells) + " |")
            lines.append("")
    return "\n".join(lines)


def _build_root_readme(reports_root: Path) -> str:
    lines = [
        "# AMORA Reports",
        "",
        "Per-vendor, per-architecture-family probe reports.",
        "",
        "| vendor | summary |",
        "| --- | --- |",
    ]
    for vendor_dir in sorted(p for p in reports_root.iterdir() if p.is_dir()):
        if (vendor_dir / "SUMMARY.md").exists():
            lines.append(f"| `{vendor_dir.name}` | [SUMMARY]({vendor_dir.name}/SUMMARY.md) |")
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_generation(
    report: dict[str, Any],
    reports_root: str | Path,
    *,
    vendor: str | None = None,
    family: str | None = None,
    sku: str | None = None,
    source_path: str | None = None,
    source_sha256: str | None = None,
    generated_at: str | None = None,
) -> Path:
    """Write one SKU's consolidated report into its family folder.

    Updates the family-level README/manifest/environment (keyed by SKU) and
    refreshes the vendor summary + top-level README. Returns the SKU's
    ``probes-<sku>.md`` path.
    """
    d_vendor, d_family, d_sku = derive_layout(report)
    vendor = vendor or d_vendor
    family = family or d_family
    sku = sku or d_sku
    generated_at = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")

    reports_root = Path(reports_root)
    family_dir = reports_root / vendor / family
    family_dir.mkdir(parents=True, exist_ok=True)

    # 1. Consolidated probes file for this SKU.
    probes_path = family_dir / f"probes-{sku}.md"
    _write(
        probes_path,
        _build_probes_doc(report, vendor=vendor, family=family, sku=sku, generated_at=generated_at),
    )

    # 2. Merge this SKU into the family manifest.
    manifest = _load_json(family_dir / "manifest.json", {"vendor": vendor, "family": family, "skus": {}})
    manifest.setdefault("skus", {})
    entry = _sku_manifest_entry(
        report,
        sku=sku,
        generated_at=generated_at,
        source_path=source_path,
        source_sha256=source_sha256,
    )
    manifest["skus"][sku] = entry

    # 3. Family README + environment.
    _write(family_dir / "README.md", _build_family_readme(vendor, family, manifest))
    _write(family_dir / "environment.md", _build_family_environment(manifest, vendor=vendor, family=family))

    # 4. Persist manifest.
    _write(family_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))

    # 5. Vendor summary + root README.
    vendor_dir = reports_root / vendor
    _write(vendor_dir / "SUMMARY.md", _build_vendor_summary(vendor, vendor_dir))
    _write(reports_root / "README.md", _build_root_readme(reports_root))
    return probes_path


def write_reports_from_json(
    json_path: str | Path,
    reports_root: str | Path,
    *,
    vendor: str | None = None,
    family: str | None = None,
    sku: str | None = None,
) -> Path:
    """Load an AMORA JSON report and render the SKU report under ``reports_root``."""
    json_path = Path(json_path)
    raw = json_path.read_text(encoding="utf-8")
    report = json.loads(raw)
    digest = sha256(raw.encode("utf-8")).hexdigest()
    return write_generation(
        report,
        reports_root,
        vendor=vendor,
        family=family,
        sku=sku,
        source_path=str(json_path),
        source_sha256=digest,
    )
