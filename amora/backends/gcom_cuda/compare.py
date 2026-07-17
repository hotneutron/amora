"""Compare real-NVIDIA vs simulated-gcom_cuda probe reports.

Two layers, kept separate:
- **probe-level**: AMORA scalar probe values (HW vs sim), per the metrics_map
  category/state; supports the comparable/approximate/unavailable taxonomy.
- **counter-level**: real NCU counters (from the HW result) vs GCoM-derived
  logical metrics, tagged by fidelity. A proxy counter never upgrades a probe
  scalar.

Writes Markdown + JSON under the gcom_cuda report hierarchy, with a per-group
coverage rollup (reusing the NVIDIA report group taxonomy) and a validation
anchor summary.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from amora.backends.gcom_cuda.runner import STALL_REASON_KEYS
from amora.probes.gcom_cuda.baseline import metrics_map as mm

# Validation anchors (plan §Accuracy Validation Anchors): subset of canonical IDs.
ANCHORS = (
    "arithmetic_latency.dependent_chain",
    "arithmetic_throughput.independent_chains",
    "shared_memory.pointer_chase",
    "l1_cache.pointer_chase",
    "l2_cache.pointer_chase",
    "global_memory.streaming",
    "synchronization.barrier_latency",
    "tensor_core.mma_latency",
    "tensor_core.mma_throughput",
)

# Report group taxonomy (reused from reports/nvidia/SUMMARY.md).
GROUPS: dict[str, tuple[str, ...]] = {
    "Compute & Scheduling": (
        "arithmetic_latency", "arithmetic_throughput", "scheduler_policy", "topology",
    ),
    "Register, Tensor & Sync": ("register_file", "tensor_core", "synchronization"),
    "On-chip Memory": ("shared_memory", "l1_cache", "l2_cache", "memory_pipeline"),
    "Global Memory & DRAM": ("global_memory",),
    "Transfer & Interconnect": ("tma_copy", "interconnect"),
}

ANCHOR_DEFAULT_TOLERANCE = 0.5  # |pct_error| <= 50% passes by default (uncalibrated sim)


def load_backend_report(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load a backend JSON report into {probe_id: result_dict}."""

    data = json.loads(Path(path).read_text())
    out: dict[str, dict[str, Any]] = {}
    for result in data.get("results", []):
        pid = (result.get("identity") or {}).get("probe_id")
        if pid:
            out[pid] = result
    return out


def _scalar(result: dict[str, Any], layer: str) -> Any:
    node = result.get(layer) or {}
    return node.get("value")


def _group_of(probe_id: str) -> str:
    prefix = probe_id.split(".", 1)[0]
    for group, prefixes in GROUPS.items():
        if prefix in prefixes:
            return group
    return "Other"


def _pct_error(hw: float, sim: float) -> float | None:
    if hw == 0:
        return None
    return abs(sim - hw) / abs(hw)


def compare_probes(real: dict[str, dict[str, Any]],
                   sim: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Probe-level comparison rows over the canonical inventory."""

    rows: list[dict[str, Any]] = []
    for probe_id, policy in mm.METRICS_MAP.items():
        hw_result = real.get(probe_id)
        sim_result = sim.get(probe_id)
        hw_value = _scalar(hw_result, "normalized_measurement") if hw_result else None
        sim_value = _scalar(sim_result, "simulator_estimate") if sim_result else None
        unit = ((hw_result or {}).get("normalized_measurement") or {}).get("unit")
        concept = ((hw_result or {}).get("backend_interpretation") or {}).get("concept")

        state = policy.state
        if sim_result is not None:
            state = (sim_result.get("raw_observation") or {}).get("values", {}).get("gcom_state", state)

        abs_err = pct_err = None
        if isinstance(hw_value, (int, float)) and isinstance(sim_value, (int, float)):
            abs_err = abs(sim_value - hw_value)
            pct_err = _pct_error(float(hw_value), float(sim_value))

        rows.append({
            "probe_id": probe_id,
            "group": _group_of(probe_id),
            "category": policy.category,
            "state": state,
            "concept": concept,
            "unit": unit,
            "fidelity": policy.fidelity,
            "architecture_scope": policy.architecture_scope,
            "hw_value": hw_value,
            "sim_value": sim_value,
            "abs_error": abs_err,
            "pct_error": pct_err,
            "is_anchor": probe_id in ANCHORS,
            "limitations": policy.limitations,
        })
    return rows


def compare_counters(real: dict[str, dict[str, Any]],
                     sim: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Counter-level comparison: HW NCU values vs GCoM-derived logical metrics."""

    rows: list[dict[str, Any]] = []
    for probe_id, sim_result in sim.items():
        derived = (sim_result.get("backend_interpretation") or {}).get("metric_resolver") or {}
        if not isinstance(derived, dict) or not derived:
            continue
        hw_result = real.get(probe_id) or {}
        hw_metrics = (hw_result.get("raw_observation") or {}).get("metrics") or {}
        hw_resolver = (hw_result.get("backend_interpretation") or {}).get("metric_resolver") or {}
        for logical, info in derived.items():
            if isinstance(logical, str) and logical.startswith("stall_"):
                continue
            sim_val = info.get("value") if isinstance(info, dict) else info
            hw_val = _logical_ncu_value(hw_result, logical)
            pct = None
            if isinstance(hw_val, (int, float)) and isinstance(sim_val, (int, float)):
                pct = _pct_error(float(hw_val), float(sim_val))
            rows.append({
                "probe_id": probe_id,
                "logical": logical,
                "fidelity": (info.get("fidelity") if isinstance(info, dict) else None),
                "ncu_metric": (info.get("ncu_metric") if isinstance(info, dict) else None),
                "hw_ncu": hw_val,
                "sim_gcom": sim_val,
                "pct_error": pct,
            })
    return rows


def _logical_ncu_value(result: dict[str, Any], logical: str) -> Any:
    raw = result.get("raw_observation") or {}
    metrics = raw.get("metrics") or {}
    if logical in metrics:
        return metrics[logical]

    values = raw.get("values") or {}
    for key in ("gcom_counter_comparison", "ncu"):
        record = values.get(key)
        record_values = record.get("values") if isinstance(record, dict) else None
        if isinstance(record_values, dict) and logical in record_values:
            return record_values[logical]

    resolver = (result.get("backend_interpretation") or {}).get("metric_resolver") or {}
    if logical in resolver:
        return resolver[logical]
    resolver_values = resolver.get("values") if isinstance(resolver, dict) else None
    if isinstance(resolver_values, dict):
        return resolver_values.get(logical)
    return None


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict) and isinstance(value.get("value"), (int, float)):
        return float(value["value"])
    return None


def _ncu_stall_histogram(result: dict[str, Any] | None) -> dict[str, float]:
    """Return {reason: pct} from the NVIDIA report, when collected.

    NVIDIA probes currently expose stall attribution either as
    raw_observation.values.stall_attribution.stalls or as logical
    stall_<reason> values in metrics / metric_resolver. All values are treated
    as NCU percentages and normalized to the GCoM reason names.
    """

    if not result:
        return {}
    raw = result.get("raw_observation") or {}
    values = raw.get("values") or {}
    metrics = raw.get("metrics") or {}
    resolver = (result.get("backend_interpretation") or {}).get("metric_resolver") or {}
    hist: dict[str, float] = {}

    attribution = values.get("stall_attribution")
    stalls = attribution.get("stalls") if isinstance(attribution, dict) else None
    if isinstance(stalls, dict):
        for reason, value in stalls.items():
            pct = _numeric(value)
            if pct is not None:
                hist[str(reason)] = pct

    for reason in STALL_REASON_KEYS:
        for key in (f"stall_{reason}", f"stall_{reason}_pct"):
            pct = _numeric(metrics.get(key))
            if pct is None:
                pct = _numeric(resolver.get(key))
            if pct is not None:
                hist[reason] = pct
                break
    return hist


def _gcom_stall_histogram(result: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    if not result:
        return {}
    metrics = (result.get("raw_observation") or {}).get("metrics") or {}
    hist = metrics.get("gcom_stall_reason_histogram")
    return hist if isinstance(hist, dict) else {}


def compare_stall_reason_histograms(
    real: dict[str, dict[str, Any]],
    sim: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Compare NCU and GCoM stall-reason histograms per probe."""

    rows: list[dict[str, Any]] = []
    for probe_id in sorted(set(real) | set(sim)):
        hw_hist = _ncu_stall_histogram(real.get(probe_id))
        sim_hist = _gcom_stall_histogram(sim.get(probe_id))
        if not hw_hist and not sim_hist:
            continue

        reasons: dict[str, dict[str, Any]] = {}
        abs_errors: list[float] = []
        for reason in STALL_REASON_KEYS:
            hw_pct = hw_hist.get(reason)
            sim_entry = sim_hist.get(reason)
            sim_pct = None
            sim_count = None
            if isinstance(sim_entry, dict):
                sim_pct = _numeric(sim_entry.get("pct"))
                sim_count = _numeric(sim_entry.get("count"))
            abs_delta = None
            if hw_pct is not None and sim_pct is not None:
                abs_delta = abs(sim_pct - hw_pct)
                abs_errors.append(abs_delta)
            reasons[reason] = {
                "ncu_pct": hw_pct,
                "gcom_pct": sim_pct,
                "gcom_count": sim_count,
                "abs_pct_point_error": abs_delta,
            }

        metrics = (sim.get(probe_id, {}).get("raw_observation") or {}).get("metrics") or {}
        rows.append({
            "probe_id": probe_id,
            "ncu_available": bool(hw_hist),
            "gcom_available": bool(sim_hist),
            "gcom_complete": metrics.get("gcom_stall_reason_complete") if sim_hist else None,
            "gcom_denominator": metrics.get("gcom_stall_reason_denominator") if sim_hist else None,
            "mean_abs_pct_point_error": statistics.fmean(abs_errors) if abs_errors else None,
            "max_abs_pct_point_error": max(abs_errors) if abs_errors else None,
            "reasons": reasons,
        })
    return rows


def summarize_stall_reason_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate NCU-vs-GCoM stall percentages across comparable probe rows."""

    summary: list[dict[str, Any]] = []
    for reason in STALL_REASON_KEYS:
        ncu_vals: list[float] = []
        gcom_vals: list[float] = []
        abs_errors: list[float] = []
        probe_ids: list[str] = []
        for row in rows:
            entry = (row.get("reasons") or {}).get(reason) or {}
            ncu_pct = _numeric(entry.get("ncu_pct"))
            gcom_pct = _numeric(entry.get("gcom_pct"))
            if ncu_pct is None or gcom_pct is None:
                continue
            ncu_vals.append(ncu_pct)
            gcom_vals.append(gcom_pct)
            abs_errors.append(abs(gcom_pct - ncu_pct))
            probe_ids.append(row["probe_id"])
        summary.append({
            "reason": reason,
            "probe_count": len(probe_ids),
            "probes": probe_ids,
            "ncu_mean_pct": statistics.fmean(ncu_vals) if ncu_vals else None,
            "gcom_mean_pct": statistics.fmean(gcom_vals) if gcom_vals else None,
            "mean_abs_pct_point_error": statistics.fmean(abs_errors) if abs_errors else None,
            "max_abs_pct_point_error": max(abs_errors) if abs_errors else None,
        })
    return summary


def _accuracy_rollup(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_group: dict[str, list[float]] = {}
    for r in rows:
        if r["category"] == mm.COMPARABLE and isinstance(r["pct_error"], (int, float)):
            by_group.setdefault(r["group"], []).append(r["pct_error"])
    rollup = {}
    for group, errs in by_group.items():
        rollup[group] = {
            "n": len(errs),
            "mean_pct_error": statistics.fmean(errs) if errs else None,
            "median_pct_error": statistics.median(errs) if errs else None,
        }
    return rollup


def _anchor_summary(rows: list[dict[str, Any]],
                    tolerance: float = ANCHOR_DEFAULT_TOLERANCE) -> dict[str, Any]:
    passed, failed, unavailable = [], [], []
    for r in rows:
        if not r["is_anchor"]:
            continue
        if not isinstance(r["pct_error"], (int, float)):
            unavailable.append(r["probe_id"])
        elif r["pct_error"] <= tolerance:
            passed.append(r["probe_id"])
        else:
            failed.append(r["probe_id"])
    return {
        "tolerance": tolerance,
        "passed": passed,
        "failed": failed,
        "unavailable": unavailable,
        "reliable": len(failed) == 0 and len(passed) > 0,
    }


def _coverage(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for r in rows:
        key = r["state"] or r["category"]
        counts[key] = counts.get(key, 0) + 1
    with_hw = sum(1 for r in rows if isinstance(r["hw_value"], (int, float)))
    with_sim = sum(1 for r in rows if isinstance(r["sim_value"], (int, float)))
    return {
        "total_probes": len(rows),
        "with_hw_value": with_hw,
        "with_sim_value": with_sim,
        "by_state": counts,
    }


def _stall_histogram_coverage(sim: dict[str, dict[str, Any]]) -> dict[str, Any]:
    complete: list[str] = []
    partial: list[str] = []
    missing: list[str] = []
    missing_reasons: dict[str, int] = {}
    for probe_id, result in sim.items():
        metrics = (result.get("raw_observation") or {}).get("metrics") or {}
        hist = metrics.get("gcom_stall_reason_histogram")
        if not hist:
            missing.append(probe_id)
            continue
        if metrics.get("gcom_stall_reason_complete") is False:
            partial.append(probe_id)
            for reason in metrics.get("gcom_stall_reason_missing") or ():
                missing_reasons[reason] = missing_reasons.get(reason, 0) + 1
        else:
            complete.append(probe_id)
    return {
        "complete": complete,
        "partial": partial,
        "missing": missing,
        "complete_count": len(complete),
        "partial_count": len(partial),
        "missing_count": len(missing),
        "missing_reasons": missing_reasons,
    }


def build_comparison(real_path: str | Path, sim_path: str | Path) -> dict[str, Any]:
    real = load_backend_report(real_path)
    sim = load_backend_report(sim_path)
    probe_rows = compare_probes(real, sim)
    counter_rows = compare_counters(real, sim)
    stall_rows = compare_stall_reason_histograms(real, sim)
    stall_summary = summarize_stall_reason_comparison(stall_rows)
    sim_meta = json.loads(Path(sim_path).read_text()).get("metadata", {})
    return {
        "mapping_version": mm_mapping_version(),
        "probe_comparison": probe_rows,
        "counter_comparison": counter_rows,
        "stall_reason_comparison": stall_rows,
        "stall_reason_summary": stall_summary,
        "accuracy_rollup": _accuracy_rollup(probe_rows),
        "anchor_summary": _anchor_summary(probe_rows),
        "coverage": _coverage(probe_rows),
        "stall_reason_coverage": _stall_histogram_coverage(sim),
        "category_counts": mm.category_counts(),
        "sim_metadata": sim_meta,
    }


def mm_mapping_version() -> str:
    from amora.probes.gcom_cuda.baseline.gcom_metrics_map import MAPPING_VERSION
    return MAPPING_VERSION


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, (dict, list)):
        # Composite/behavioral HW values (occupancy sweeps, analyze summaries)
        # are not scalar-comparable; keep the table readable.
        return "(composite)"
    s = str(v)
    return s if len(s) <= 40 else s[:37] + "..."


def _pct(v: Any) -> str:
    """Render a fractional error (e.g. 1.517) as a percentage (e.g. 151.7%)."""
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v * 100:.1f}%"


def _pct_points(v: Any) -> str:
    """Render a percentage-valued metric as percentage points."""
    if not isinstance(v, (int, float)):
        return "—"
    return f"{v:.2f}"


def _bar(v: Any, *, width: int = 18) -> str:
    if not isinstance(v, (int, float)):
        return "—"
    filled = max(0, min(width, round((float(v) / 100.0) * width)))
    return "#" * filled + "." * (width - filled)


def _svg_escape(value: Any) -> str:
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def render_stall_summary_svg(comparison: dict[str, Any], *, sku: str) -> str:
    summary = comparison.get("stall_reason_summary") or []
    rows = [r for r in summary if r.get("probe_count")]
    if not rows:
        rows = summary

    left = 170
    right = 28
    top = 72
    row_h = 26
    axis_h = 44
    plot_w = 420
    height = top + len(rows) * row_h + axis_h
    width = left + plot_w + right
    ncu_color = "#2563eb"
    gcom_color = "#dc2626"
    grid_color = "#e5e7eb"
    text_color = "#111827"
    muted = "#6b7280"

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        f"<title id=\"title\">NCU vs GCoM stall reason summary for {_svg_escape(sku)}</title>",
        "<desc id=\"desc\">Grouped horizontal bar chart comparing average stall reason "
        "percentages across probes where both NCU and GCoM values are available.</desc>",
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="28" font-family="Arial, sans-serif" font-size="18" '
        f'font-weight="700" fill="{text_color}">NCU vs GCoM Stall Reasons</text>',
        f'<text x="20" y="50" font-family="Arial, sans-serif" font-size="12" '
        f'fill="{muted}">Average percentage across probes with both values</text>',
        f'<rect x="{left}" y="20" width="12" height="12" fill="{ncu_color}"/>',
        f'<text x="{left + 18}" y="31" font-family="Arial, sans-serif" font-size="12" '
        f'fill="{text_color}">NCU</text>',
        f'<rect x="{left + 68}" y="20" width="12" height="12" fill="{gcom_color}"/>',
        f'<text x="{left + 86}" y="31" font-family="Arial, sans-serif" font-size="12" '
        f'fill="{text_color}">GCoM</text>',
    ]

    for pct in (0, 25, 50, 75, 100):
        x = left + (pct / 100.0) * plot_w
        lines.extend([
            f'<line x1="{x:.1f}" y1="{top - 10}" x2="{x:.1f}" '
            f'y2="{height - axis_h + 4}" stroke="{grid_color}" stroke-width="1"/>',
            f'<text x="{x:.1f}" y="{height - 18}" text-anchor="middle" '
            f'font-family="Arial, sans-serif" font-size="11" fill="{muted}">{pct}%</text>',
        ])

    for i, row in enumerate(rows):
        y = top + i * row_h
        reason = row.get("reason", "")
        ncu = row.get("ncu_mean_pct")
        gcom = row.get("gcom_mean_pct")
        probes = row.get("probe_count", 0)
        ncu_w = 0 if not isinstance(ncu, (int, float)) else max(0, min(plot_w, ncu / 100.0 * plot_w))
        gcom_w = 0 if not isinstance(gcom, (int, float)) else max(0, min(plot_w, gcom / 100.0 * plot_w))
        lines.extend([
            f'<text x="{left - 10}" y="{y + 16}" text-anchor="end" '
            f'font-family="Arial, sans-serif" font-size="12" fill="{text_color}">'
            f'{_svg_escape(reason)}</text>',
            f'<rect x="{left}" y="{y + 3}" width="{ncu_w:.1f}" height="9" '
            f'rx="2" fill="{ncu_color}"/>',
            f'<rect x="{left}" y="{y + 14}" width="{gcom_w:.1f}" height="9" '
            f'rx="2" fill="{gcom_color}"/>',
        ])
        if isinstance(ncu, (int, float)):
            lines.append(
                f'<text x="{left + ncu_w + 4:.1f}" y="{y + 11}" '
                f'font-family="Arial, sans-serif" font-size="10" fill="{muted}">{ncu:.1f}</text>'
            )
        if isinstance(gcom, (int, float)):
            lines.append(
                f'<text x="{left + gcom_w + 4:.1f}" y="{y + 22}" '
                f'font-family="Arial, sans-serif" font-size="10" fill="{muted}">{gcom:.1f}</text>'
            )
        if probes:
            lines.append(
                f'<text x="{width - 6}" y="{y + 16}" text-anchor="end" '
                f'font-family="Arial, sans-serif" font-size="10" fill="{muted}">n={probes}</text>'
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def render_markdown(comparison: dict[str, Any], *, sku: str) -> str:
    lines: list[str] = [f"# gcom_cuda sim-vs-HW comparison: {sku}", ""]
    lines.append(f"- mapping version: `{comparison['mapping_version']}`")
    cov = comparison["coverage"]
    lines.append(f"- probes: {cov['total_probes']} · with HW value: {cov['with_hw_value']} · "
                 f"with sim value: {cov['with_sim_value']}")
    cc = comparison["category_counts"]
    lines.append(f"- categories: " + ", ".join(f"{k}={v}" for k, v in sorted(cc.items())))
    lines.append("")

    anc = comparison["anchor_summary"]
    lines += ["## Validation anchors", "",
              f"- passed: {len(anc['passed'])} · failed: {len(anc['failed'])} · "
              f"unavailable: {len(anc['unavailable'])} (tol {anc['tolerance']:.0%})",
              f"- broad comparison reliable: **{anc['reliable']}**", ""]

    lines += ["## Accuracy rollup (comparable probes)", "",
              "| group | n | mean |pct err| | median |pct err| |", "|---|---|---|---|"]
    for group, r in sorted(comparison["accuracy_rollup"].items()):
        lines.append(f"| {group} | {r['n']} | {_pct(r['mean_pct_error'])} | "
                     f"{_pct(r['median_pct_error'])} |")
    lines.append("")

    lines += ["## Probe-level comparison", "",
              "| probe | group | category | state | hw | sim | pct err | anchor |",
              "|---|---|---|---|---|---|---|---|"]
    for r in comparison["probe_comparison"]:
        lines.append(
            f"| {r['probe_id']} | {r['group']} | {r['category']} | {_fmt(r['state'])} | "
            f"{_fmt(r['hw_value'])} | {_fmt(r['sim_value'])} | {_pct(r['pct_error'])} | "
            f"{'✓' if r['is_anchor'] else ''} |"
        )
    lines.append("")

    stall_cov = comparison.get("stall_reason_coverage") or {}
    if stall_cov:
        lines += ["## GCoM stall-reason coverage", "",
                  f"- complete histograms: {stall_cov.get('complete_count', 0)}",
                  f"- partial histograms: {stall_cov.get('partial_count', 0)}",
                  f"- no histogram: {stall_cov.get('missing_count', 0)}"]
        missing_reasons = stall_cov.get("missing_reasons") or {}
        if missing_reasons:
            formatted = ", ".join(
                f"{reason}={count}" for reason, count in sorted(missing_reasons.items())
            )
            lines.append(f"- missing reasons in partial histograms: {formatted}")
        lines.append("")

    stall_cmp = comparison.get("stall_reason_comparison") or []
    if stall_cmp:
        lines += ["## Stall-reason histogram comparison (NCU vs GCoM)", ""]
        stall_summary = comparison.get("stall_reason_summary") or []
        if stall_summary:
            lines += [
                "### All-Probe Stall Summary",
                "",
                f"![NCU vs GCoM stall reason summary](sim-vs-hw-{sku}-stall-summary.svg)",
                "",
                "| reason | probes | ncu avg pct | ncu | gcom avg pct | gcom | mean abs pp err | max abs pp err |",
                "|---|---:|---:|---|---:|---|---:|---:|",
            ]
            for row in stall_summary:
                lines.append(
                    f"| {row['reason']} | {row['probe_count']} | "
                    f"{_pct_points(row['ncu_mean_pct'])} | {_bar(row['ncu_mean_pct'])} | "
                    f"{_pct_points(row['gcom_mean_pct'])} | {_bar(row['gcom_mean_pct'])} | "
                    f"{_pct_points(row['mean_abs_pct_point_error'])} | "
                    f"{_pct_points(row['max_abs_pct_point_error'])} |"
                )
            lines.append("")

        for row in stall_cmp:
            lines += [
                f"### {row['probe_id']}",
                "",
                f"- NCU histogram: {'yes' if row['ncu_available'] else 'no'}",
                f"- GCoM histogram: {'yes' if row['gcom_available'] else 'no'}"
                f" · complete: {_fmt(row['gcom_complete'])}"
                f" · denominator: {_fmt(row['gcom_denominator'])}",
                f"- mean absolute error: {_pct_points(row['mean_abs_pct_point_error'])} pp"
                f" · max absolute error: {_pct_points(row['max_abs_pct_point_error'])} pp",
                "",
                "| reason | ncu pct | gcom pct | abs pp err | gcom count |",
                "|---|---|---|---|---|",
            ]
            for reason in STALL_REASON_KEYS:
                entry = row["reasons"][reason]
                lines.append(
                    f"| {reason} | {_pct_points(entry['ncu_pct'])} | "
                    f"{_pct_points(entry['gcom_pct'])} | "
                    f"{_pct_points(entry['abs_pct_point_error'])} | "
                    f"{_fmt(entry['gcom_count'])} |"
                )
            lines.append("")

    if comparison["counter_comparison"]:
        lines += ["## Non-stall counter comparison (GCoM-derived vs NCU)", "",
                  "| probe | logical | fidelity | hw ncu | sim gcom | pct err |",
                  "|---|---|---|---|---|---|"]
        for r in comparison["counter_comparison"]:
            lines.append(
                f"| {r['probe_id']} | {r['logical']} | {_fmt(r['fidelity'])} | "
                f"{_fmt(r['hw_ncu'])} | {_fmt(r['sim_gcom'])} | {_pct(r['pct_error'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_outputs(comparison: dict[str, Any], out_dir: str | Path, *,
                  family: str = "hopper", sku: str = "gcom_h100") -> dict[str, str]:
    base = Path(out_dir) / family
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / f"sim-vs-hw-{sku}.json"
    md_path = base / f"sim-vs-hw-{sku}.md"
    svg_path = base / f"sim-vs-hw-{sku}-stall-summary.svg"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    svg_path.write_text(render_stall_summary_svg(comparison, sku=sku))
    md_path.write_text(render_markdown(comparison, sku=sku) + "\n")
    return {"json": str(json_path), "markdown": str(md_path), "stall_summary_svg": str(svg_path)}
