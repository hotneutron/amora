from __future__ import annotations

import json
from importlib import util

import pytest

from amora.benchmarking.materialize import materialize_benchmark, write_manifest
from amora.benchmarking.registry import get_benchmark, list_benchmarks
from benchmark_generators.ppp_canonical.cases import candidate_items
from benchmark_generators.ppp_canonical.kernels import CANONICAL_KERNELS


TARGET = {
    "vendor": "nvidia",
    "family": "hopper",
    "hardware_sku": "h100-80g",
    "arch_profile": "sm_90_h100",
}


def test_registry_exposes_local_ppp_generator():
    listed = list_benchmarks()

    assert listed == [
        {
            "benchmark_id": "ppp_canonical",
            "benchmark_revision": 1,
            "definition_kind": "generator",
            "description": "AMORA-owned canonical PPP kernel-and-shape generator",
            "kernels": [
                "aligned_gemm_fp16",
                "embedding",
                "flash_attention_fwd",
                "flashmla_dense_decode",
                "gelu",
                "gelu_gemm_fp16",
                "megamoe_fp8",
                "rmsnorm",
                "rmsnorm_gemm_fp16",
            ],
            "presets": ["h100_2500", "h100_5600"],
        }
    ]
    assert get_benchmark("ppp_canonical").benchmark_id == "ppp_canonical"
    with pytest.raises(KeyError, match="unknown benchmark"):
        get_benchmark("missing")


def test_ppp_generator_is_an_installable_package():
    assert util.find_spec("benchmark_generators.ppp_canonical") is not None


def test_materialize_exact_balanced_case_count_is_deterministic():
    first = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=2500,
        seed=20260717,
    )
    second = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=2500,
        seed=20260717,
    )

    assert first.case_set_digest == second.case_set_digest
    assert first.materialized_case_count == 2500
    assert len(first.cases) == 2500
    assert [case.case_key for case in first.cases] == sorted(
        case.case_key for case in first.cases
    )
    assert len({case.case_key for case in first.cases}) == 2500
    counts = {}
    for case in first.cases:
        counts[case.kernel_id] = counts.get(case.kernel_id, 0) + 1
        assert case.shape_class == "sweep"
        assert case.case_generation["generator_source_sha256"]
        assert case.execution_contract["replay_contract_revision"] == 1
    assert set(counts) == set(get_benchmark("ppp_canonical").describe()["kernels"])
    assert sorted(counts.values()) == [277, 277, 278, 278, 278, 278, 278, 278, 278]


def test_materialization_seed_changes_manifest_but_not_count():
    first = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=63,
        seed=1,
    )
    second = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=63,
        seed=2,
    )

    assert first.materialized_case_count == second.materialized_case_count == 63
    assert first.case_set_digest != second.case_set_digest
    assert [case.case_key for case in first.cases] != [
        case.case_key for case in second.cases
    ]


def test_h100_5600_materializes_exactly_despite_uneven_capacities():
    preset = get_benchmark("ppp_canonical").presets["h100_5600"]
    manifest = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=preset["case_count"],
        seed=preset["seed"],
    )

    counts = {}
    for case in manifest.cases:
        counts[case.kernel_id] = counts.get(case.kernel_id, 0) + 1
    assert manifest.materialized_case_count == 5600
    assert len({case.case_key for case in manifest.cases}) == 5600
    assert counts["flashmla_dense_decode"] == 370
    assert sum(counts.values()) == 5600


def test_materialize_rejects_invalid_case_count():
    with pytest.raises(ValueError, match="case_count must be positive"):
        materialize_benchmark(
            "ppp_canonical",
            target=TARGET,
            case_count=0,
            seed=1,
        )


def test_ppp_candidate_streams_have_unique_shapes():
    for spec in CANONICAL_KERNELS:
        candidates = candidate_items(spec)
        shape_keys = [tuple(sorted(item["shape"].items())) for item in candidates]
        assert len(shape_keys) == len(set(shape_keys)), spec.kernel_id


def test_write_manifest_round_trips_case_set(tmp_path):
    manifest = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=9,
        seed=7,
    )

    destination = write_manifest(manifest, tmp_path / "manifest.json")
    data = json.loads(destination.read_text())

    assert data["case_set_digest"] == manifest.case_set_digest
    assert data["case_count_requested"] == 9
    assert data["case_count_materialized"] == 9
    assert len(data["cases"]) == 9
