"""No-GPU unit tests for the gcom_cuda backend."""

from __future__ import annotations

import json

from amora.backends.gcom_cuda import config as gcfg
from amora.backends.gcom_cuda.gcom import discover_capabilities
from amora.backends.gcom_cuda.runner import (
    STALL_REASON_KEYS,
    extract_stall_reason_histogram,
    parse_stats,
)
from amora.probes.gcom_cuda import baseline as gbaseline
from amora.probes.gcom_cuda.baseline import metrics_map as mm
from amora.probes.nvidia.baseline import PROBES as NVIDIA_PROBES


def test_inventory_single_source_of_truth():
    # gcom inventory is derived from the nvidia registry, never re-listed.
    assert gbaseline.PLANNED_PROBES == tuple(NVIDIA_PROBES)


def test_metrics_map_covers_exactly_nvidia_inventory():
    assert set(mm.METRICS_MAP) == set(NVIDIA_PROBES)


def test_category_counts_partition_inventory():
    counts = mm.category_counts()
    assert sum(counts.values()) == len(NVIDIA_PROBES)
    # All three categories are represented.
    assert counts.get(mm.COMPARABLE, 0) > 0
    assert counts.get(mm.APPROXIMATE, 0) > 0
    assert counts.get(mm.UNAVAILABLE, 0) > 0


def test_sku_profile_default():
    profile = gcfg.get_sku_profile()
    assert profile.sku == "gcom_h100"
    assert profile.family == "hopper"
    assert profile.hw_sku == "h100-80g"


def test_capabilities_shape_no_gpu():
    caps = discover_capabilities()
    d = caps.to_dict()
    assert d["backend"] == "gcom_cuda"
    assert "simulator_built" in d and "tracer_built" in d
    assert "sku_profile" in d


def test_parse_stats_extracts_numeric_keys():
    stdout = (
        "gpu_sim_cycle = 12345\n"
        "gpu_tot_sim_insn = 6789\n"
        "L2_total_cache_miss_rate = 0.25\n"
        "ncu_stall_selected = 10\n"
        "ncu_stall_selected_pct = 12.5\n"
        "not a stat line\n"
        "gpu_sim_cycle = 99999\n"  # last value wins
    )
    stats = parse_stats(stdout)
    assert stats["gpu_sim_cycle"] == 99999.0
    assert stats["gpu_tot_sim_insn"] == 6789.0
    assert stats["L2_total_cache_miss_rate"] == 0.25
    assert stats["ncu_stall_selected"] == 10.0
    assert stats["ncu_stall_selected_pct"] == 12.5


def test_extract_stall_reason_histogram_complete():
    stats = {"total_num_cycles_issue_stage_evaluated": 80.0}
    for i, reason in enumerate(STALL_REASON_KEYS, start=1):
        stats[f"ncu_stall_{reason}"] = float(i)
        stats[f"ncu_stall_{reason}_pct"] = float(i) / 80.0 * 100.0

    hist = extract_stall_reason_histogram(stats)

    assert hist is not None
    assert hist["schema"] == "ncu-stall-v1"
    assert hist["complete"] is True
    assert hist["missing_reasons"] == []
    assert hist["denominator"] == 80.0
    assert set(hist["reasons"]) == set(STALL_REASON_KEYS)
    assert hist["reasons"]["selected"]["count"] == 1.0
    assert hist["reasons"]["selected"]["pct"] == 1.25


def test_extract_stall_reason_histogram_partial_is_marked():
    stats = {
        "total_num_cycles_issue_stage_evaluated": 80.0,
        "ncu_stall_selected": 10.0,
    }

    hist = extract_stall_reason_histogram(stats)

    assert hist is not None
    assert hist["complete"] is False
    assert "not_selected" in hist["missing_reasons"]
    assert "not_selected" not in hist["reasons"]
    assert hist["reasons"]["selected"]["pct"] == 12.5


def test_trace_compat_overrides_disable_tma_real_base_without_sidecar(tmp_path):
    from amora.backends.gcom_cuda.runner import _trace_compat_overrides

    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    assert _trace_compat_overrides(trace_dir) == [
        "-tma_real_base_addr_enable", "0",
        "-tma_operand_addr_tiling_enable", "0",
    ]

    extra = trace_dir / "extra_info"
    extra.mkdir()
    (extra / "tma_pc_base_map.json").write_text("{}")
    assert _trace_compat_overrides(trace_dir) == []


def test_unavailable_probes_short_circuit_without_execution():
    # Analysis-only / unsupported probes resolve to their policy state without
    # ever attempting a trace+simulate (safe regardless of tooling presence).
    caps = discover_capabilities()
    ctx = gbaseline.RunContext()  # no hw_baseline
    for probe_id, expect in [
        ("l1_cache.analyze", mm.NOT_APPLICABLE),
        ("shared_memory.bank_stride", mm.UNSUPPORTED),
        ("memory_pipeline.lane_patterns", mm.PROXY_ONLY),
    ]:
        r = gbaseline.run_probe(probe_id, caps, ctx)[0]
        assert r.raw_observation.values.get("gcom_state") == expect


def test_comparable_probe_without_hw_baseline_is_missing_stat():
    # Without a HW denominator, a comparable per-op probe must report
    # missing_stat rather than execute or fabricate a value.
    caps = discover_capabilities()
    r = gbaseline.run_probe("arithmetic_latency.dependent_chain", caps,
                            gbaseline.RunContext())[0]
    state = r.raw_observation.values.get("gcom_state")
    # missing_stat when sim is built but denominator absent; also acceptable if
    # the simulator isn't built in this environment.
    assert state == mm.MISSING_STAT
    assert r.raw_observation.evidence_tier.value == "unsupported"


def test_run_all_parallel_preserves_order_and_inventory():
    # Parallel run_all must return every probe exactly once, in canonical NVIDIA
    # order. Force the no-execution fast path (simulator_built=False) so the test
    # is fast and GPU-independent: every probe short-circuits to a state.
    import dataclasses

    caps = dataclasses.replace(discover_capabilities(), simulator_built=False)
    ctx = gbaseline.RunContext(max_workers=4)
    results = gbaseline.run_all(caps, ctx)
    ids = [r.identity.probe_id for r in results]
    assert ids == list(gbaseline.PLANNED_PROBES)
    # Same result as the serial path (workers=1).
    serial = [r.identity.probe_id
              for r in gbaseline.run_all(caps, gbaseline.RunContext(max_workers=1))]
    assert serial == ids


def test_derive_logical_metrics_from_stats():
    from amora.backends.gcom_cuda.runner import derive_logical_metrics

    stats = {
        "gpu_sim_cycle": 1000.0,
        "gpgpu_n_dram_reads": 10.0,
        "L2_total_cache_accesses": 100.0,
        "L2_total_cache_misses": 30.0,
        "ncu_stall_selected_pct": 12.5,
    }
    derived = derive_logical_metrics(stats)
    assert derived["sm_active_cycles"]["value"] == 1000.0
    assert derived["dram_bytes_read"]["value"] == 10.0 * gcfg.DRAM_ATOM_BYTES
    assert derived["l2_sector_hits"]["value"] == 70.0
    assert derived["stall_selected_pct"]["value"] == 12.5


def test_gcom_result_attaches_stall_histogram():
    import dataclasses
    from pathlib import Path
    from amora.probes.gcom_cuda.baseline import _result

    caps = dataclasses.replace(discover_capabilities(), simulator_built=True)
    policy = mm.METRICS_MAP["arithmetic_latency.dependent_chain"]
    stats = {
        "gpu_sim_cycle": 1000.0,
        "gpu_tot_sim_insn": 20.0,
        "gpu_ipc": 0.02,
        "total_num_cycles_issue_stage_evaluated": 80.0,
    }
    for i, reason in enumerate(STALL_REASON_KEYS, start=1):
        stats[f"ncu_stall_{reason}"] = float(i)
        stats[f"ncu_stall_{reason}_pct"] = float(i) / 80.0 * 100.0

    result = _result(
        "arithmetic_latency.dependent_chain",
        10.0,
        "cycles",
        policy,
        caps,
        stats,
        None,
        Path("/tmp/gcom.log"),
    )
    metrics = result.raw_observation.metrics
    assert metrics["gcom_stall_reason_schema"] == "ncu-stall-v1"
    assert metrics["gcom_stall_reason_complete"] is True
    assert metrics["gcom_stall_reason_denominator"] == 80.0
    assert metrics["gcom_stall_reason_histogram"]["selected"]["count"] == 1.0


def _fake_report(results: dict) -> dict:
    return {"schema_version": 1, "metadata": {}, "results": [
        {
            "identity": {"probe_id": pid},
            "normalized_measurement": {"value": vals.get("hw"), "unit": "cycles"},
            "simulator_estimate": {"value": vals.get("sim")},
            "backend_interpretation": {"concept": pid, "metric_resolver": vals.get("counters", {})},
            "raw_observation": {
                "values": vals.get("raw_values", {}),
                "metrics": vals.get("hw_metrics", {}),
            },
        }
        for pid, vals in results.items()
    ]}


def test_compare_builds_rows_and_anchors(tmp_path):
    from amora.backends.gcom_cuda.compare import build_comparison

    real = _fake_report({
        "arithmetic_latency.dependent_chain": {"hw": 4.0},
        "global_memory.streaming": {"hw": 3000.0},
    })
    sim = _fake_report({
        "arithmetic_latency.dependent_chain": {"sim": 4.4},
        "global_memory.streaming": {"sim": 2700.0},
    })
    real_path = tmp_path / "real.json"
    sim_path = tmp_path / "sim.json"
    real_path.write_text(json.dumps(real))
    sim_path.write_text(json.dumps(sim))

    comparison = build_comparison(real_path, sim_path)
    rows = {r["probe_id"]: r for r in comparison["probe_comparison"]}
    fma = rows["arithmetic_latency.dependent_chain"]
    assert fma["is_anchor"] is True
    assert abs(fma["pct_error"] - 0.1) < 1e-9
    assert comparison["anchor_summary"]["reliable"] is True
    assert comparison["coverage"]["total_probes"] == len(NVIDIA_PROBES)


def test_compare_stall_histograms_per_probe(tmp_path):
    from amora.backends.gcom_cuda.compare import build_comparison

    real = _fake_report({
        "arithmetic_latency.dependent_chain": {
            "hw": 4.0,
            "raw_values": {
                "stall_attribution": {
                    "stalls": {
                        "selected": 11.0,
                        "short_scoreboard": 80.0,
                    },
                },
            },
        },
    })
    sim = _fake_report({
        "arithmetic_latency.dependent_chain": {
            "sim": 4.4,
            "hw_metrics": {
                "gcom_stall_reason_histogram": {
                    "selected": {"count": 10.0, "pct": 10.0},
                    "short_scoreboard": {"count": 85.0, "pct": 85.0},
                },
                "gcom_stall_reason_complete": True,
                "gcom_stall_reason_denominator": 100.0,
            },
            "counters": {
                "stall_selected_pct": {"value": 10.0, "fidelity": "proportional"},
                "sm_active_cycles": {"value": 200.0, "fidelity": "direct"},
            },
        },
    })
    real_path = tmp_path / "real.json"
    sim_path = tmp_path / "sim.json"
    real_path.write_text(json.dumps(real))
    sim_path.write_text(json.dumps(sim))

    comparison = build_comparison(real_path, sim_path)

    stall_rows = comparison["stall_reason_comparison"]
    assert len(stall_rows) == 1
    row = stall_rows[0]
    assert row["probe_id"] == "arithmetic_latency.dependent_chain"
    assert row["ncu_available"] is True
    assert row["gcom_available"] is True
    assert row["gcom_complete"] is True
    assert row["reasons"]["selected"]["ncu_pct"] == 11.0
    assert row["reasons"]["selected"]["gcom_pct"] == 10.0
    assert row["reasons"]["selected"]["abs_pct_point_error"] == 1.0
    assert row["reasons"]["short_scoreboard"]["abs_pct_point_error"] == 5.0
    assert row["mean_abs_pct_point_error"] == 3.0
    assert row["max_abs_pct_point_error"] == 5.0
    assert all(
        not r["logical"].startswith("stall_")
        for r in comparison["counter_comparison"]
    )
