"""No-GPU unit tests for the gcom_cuda backend."""

from __future__ import annotations

import json

from amora.backends.gcom_cuda import config as gcfg
from amora.backends.gcom_cuda.gcom import discover_capabilities
from amora.backends.gcom_cuda.runner import parse_stats
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
        "not a stat line\n"
        "gpu_sim_cycle = 99999\n"  # last value wins
    )
    stats = parse_stats(stdout)
    assert stats["gpu_sim_cycle"] == 99999.0
    assert stats["gpu_tot_sim_insn"] == 6789.0
    assert stats["L2_total_cache_miss_rate"] == 0.25


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
    }
    derived = derive_logical_metrics(stats)
    assert derived["sm_active_cycles"]["value"] == 1000.0
    assert derived["dram_bytes_read"]["value"] == 10.0 * gcfg.DRAM_ATOM_BYTES
    assert derived["l2_sector_hits"]["value"] == 70.0


def _fake_report(results: dict) -> dict:
    return {"schema_version": 1, "metadata": {}, "results": [
        {
            "identity": {"probe_id": pid},
            "normalized_measurement": {"value": vals.get("hw"), "unit": "cycles"},
            "simulator_estimate": {"value": vals.get("sim")},
            "backend_interpretation": {"concept": pid, "metric_resolver": vals.get("counters", {})},
            "raw_observation": {"values": {}, "metrics": vals.get("hw_metrics", {})},
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
