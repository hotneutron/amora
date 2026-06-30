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
            sim_val = info.get("value") if isinstance(info, dict) else info
            hw_val = hw_metrics.get(logical)
            if hw_val is None and isinstance(hw_resolver, dict):
                hw_val = hw_resolver.get(logical)
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


def build_comparison(real_path: str | Path, sim_path: str | Path) -> dict[str, Any]:
    real = load_backend_report(real_path)
    sim = load_backend_report(sim_path)
    probe_rows = compare_probes(real, sim)
    counter_rows = compare_counters(real, sim)
    sim_meta = json.loads(Path(sim_path).read_text()).get("metadata", {})
    return {
        "mapping_version": mm_mapping_version(),
        "probe_comparison": probe_rows,
        "counter_comparison": counter_rows,
        "accuracy_rollup": _accuracy_rollup(probe_rows),
        "anchor_summary": _anchor_summary(probe_rows),
        "coverage": _coverage(probe_rows),
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
    return str(v)


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
        lines.append(f"| {group} | {r['n']} | {_fmt(r['mean_pct_error'])} | "
                     f"{_fmt(r['median_pct_error'])} |")
    lines.append("")

    lines += ["## Probe-level comparison", "",
              "| probe | group | category | state | hw | sim | pct err | anchor |",
              "|---|---|---|---|---|---|---|---|"]
    for r in comparison["probe_comparison"]:
        lines.append(
            f"| {r['probe_id']} | {r['group']} | {r['category']} | {_fmt(r['state'])} | "
            f"{_fmt(r['hw_value'])} | {_fmt(r['sim_value'])} | {_fmt(r['pct_error'])} | "
            f"{'✓' if r['is_anchor'] else ''} |"
        )
    lines.append("")

    if comparison["counter_comparison"]:
        lines += ["## Counter-level comparison (GCoM-derived vs NCU)", "",
                  "| probe | logical | fidelity | hw ncu | sim gcom | pct err |",
                  "|---|---|---|---|---|---|"]
        for r in comparison["counter_comparison"]:
            lines.append(
                f"| {r['probe_id']} | {r['logical']} | {_fmt(r['fidelity'])} | "
                f"{_fmt(r['hw_ncu'])} | {_fmt(r['sim_gcom'])} | {_fmt(r['pct_error'])} |"
            )
        lines.append("")

    return "\n".join(lines)


def write_outputs(comparison: dict[str, Any], out_dir: str | Path, *,
                  family: str = "hopper", sku: str = "gcom_h100") -> dict[str, str]:
    base = Path(out_dir) / family
    base.mkdir(parents=True, exist_ok=True)
    json_path = base / f"sim-vs-hw-{sku}.json"
    md_path = base / f"sim-vs-hw-{sku}.md"
    json_path.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    md_path.write_text(render_markdown(comparison, sku=sku) + "\n")
    return {"json": str(json_path), "markdown": str(md_path)}
