from __future__ import annotations

from pathlib import Path

import pytest

from amora.backends.nvidia.ncu import NcuCommand
from amora.backends.nvidia.cuda import NvidiaCapabilities, discover_capabilities
from amora.backends.nvidia.ncu_run import NcuResult, NcuUnavailable
from amora.backends.nvidia import benchmark as nvidia_benchmark
from amora.benchmarking.classification import (
    ClassificationResult,
    build_classification_manifest,
)
from amora.benchmarking.materialize import materialize_benchmark
from benchmark_generators.ppp_canonical.replay import contract_for_case


TARGET = {
    "vendor": "nvidia",
    "family": "hopper",
    "hardware_sku": "h100-80g",
    "arch_profile": "sm_90_h100",
}


def _result(case_key: str, instructions: float) -> ClassificationResult:
    return ClassificationResult(
        case_key=case_key,
        status="classified",
        total_instructions=instructions,
    )


def test_full_classification_assigns_deterministic_terciles_with_tie_breaks():
    keys = ["case-d", "case-a", "case-c", "case-b", "case-e"]
    results = [
        _result("case-d", 10),
        _result("case-a", 10),
        _result("case-c", 20),
        _result("case-b", 20),
        _result("case-e", 30),
    ]

    manifest = build_classification_manifest(
        case_set_digest="case-set",
        target=TARGET,
        results=results,
        expected_case_keys=keys,
    )

    assert manifest.case_coverage_complete is True
    assert manifest.case_count_expected == 5
    assert manifest.case_count_attempted == 5
    assert manifest.rank_assignments["case-a"]["size_rank"] == "small"
    assert manifest.rank_assignments["case-d"]["size_rank"] == "small"
    assert manifest.rank_assignments["case-b"]["size_rank"] == "medium"
    assert manifest.rank_assignments["case-c"]["size_rank"] == "medium"
    assert manifest.rank_assignments["case-e"]["size_rank"] == "large"
    assert manifest.rank_boundaries == {
        "small_max": 10.0,
        "medium_max": 20.0,
        "large_max": 30.0,
    }


def test_partial_classification_does_not_assign_ranks():
    manifest = build_classification_manifest(
        case_set_digest="case-set",
        target=TARGET,
        results=[_result("case-a", 10)],
        expected_case_keys=["case-a", "case-b"],
    )

    assert manifest.case_coverage_complete is False
    assert manifest.rank_assignments == {}
    assert manifest.rank_boundaries == {
        "small_max": None,
        "medium_max": None,
        "large_max": None,
    }


def test_ppp_replay_contracts_cover_all_materialized_kernels():
    manifest = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=9,
        seed=7,
    )

    for case in manifest.cases:
        contract = contract_for_case(case)
        assert contract.source.is_file()
        assert contract.launch_skip >= 1
        assert contract.launch_count == 1


def test_ncu_command_supports_warmup_skip_and_kernel_name_base():
    command = NcuCommand(
        executable="ncu",
        metrics=("smsp__inst_executed.sum",),
        target=("./driver",),
        launch_skip=1,
        launch_count=1,
        kernel_name="regex:amora_ppp",
        kernel_name_base="mangled",
    )

    argv = command.argv()
    assert ["--launch-skip", "1"] == argv[argv.index("--launch-skip"):argv.index("--launch-skip") + 2]
    assert ["--kernel-name-base", "mangled"] == argv[
        argv.index("--kernel-name-base"):argv.index("--kernel-name-base") + 2
    ]


def test_basic_classifier_maps_resolved_instruction_counter(monkeypatch, tmp_path):
    case = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=9,
        seed=7,
    ).cases[0]
    caps = NvidiaCapabilities(
        cuda_available=True,
        gpu_available=True,
        ncu_metrics=frozenset(
            {
                "smsp__inst_executed.sum",
                "sm__cycles_elapsed.avg",
                "gpu__time_duration.sum",
            }
        ),
    )
    observed = {}

    def fake_profiled(source, **kwargs):
        observed["source"] = source
        observed.update(kwargs)
        return NcuResult(
            metrics={
                "smsp__inst_executed.sum": 123.0,
                "sm__cycles_elapsed.avg": 45.0,
                "gpu__time_duration.sum": 67.0,
            },
            raw_rows=[{"Kernel Name": "amora_ppp_gemm"}],
            binary_path=Path("/tmp/driver"),
            binary_sha256="binary",
            source_path=source,
            source_sha256="source",
            command=("ncu", "--csv"),
            arch=kwargs["arch"],
            link_flags=kwargs["link_flags"],
        )

    monkeypatch.setattr(nvidia_benchmark, "run_kernel_profiled", fake_profiled)
    result = nvidia_benchmark.classify_case_basic(
        case,
        capabilities=caps,
        build_root=tmp_path,
    )

    assert result.status == "classified"
    assert result.total_instructions == 123.0
    assert result.kernel_name == "amora_ppp_gemm"
    assert result.metrics == {
        "inst_executed": 123.0,
        "elapsed_cycles": 45.0,
        "duration_ns": 67.0,
    }
    assert result.provenance["ncu_tool"] is None
    assert observed["launch_count"] == 1
    assert observed["launch_skip"] == 1


def test_basic_classifier_returns_unclassified_when_instruction_metric_is_missing(tmp_path):
    case = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=9,
        seed=7,
    ).cases[0]
    caps = NvidiaCapabilities(
        cuda_available=True,
        gpu_available=True,
        ncu_metrics=frozenset({"sm__cycles_elapsed.avg"}),
    )

    result = nvidia_benchmark.classify_case_basic(
        case,
        capabilities=caps,
        build_root=tmp_path,
    )

    assert result.status == "unclassified"
    assert result.total_instructions is None
    assert result.reason == "NCU instruction metric is unavailable"


@pytest.mark.cuda
@pytest.mark.ncu
def test_h100_gelu_basic_classification_smoke(tmp_path):
    capabilities = discover_capabilities()
    if not capabilities.cuda_available or not capabilities.ncu_metrics:
        pytest.skip("CUDA or NCU basic metrics are unavailable")
    manifest = materialize_benchmark(
        "ppp_canonical",
        target=TARGET,
        case_count=9,
        seed=7,
    )
    case = next(item for item in manifest.cases if item.kernel_id == "gelu")

    result = nvidia_benchmark.classify_case_basic(
        case,
        capabilities=capabilities,
        build_root=tmp_path / "build",
        timeout=600,
    )

    assert result.status == "classified"
    assert result.total_instructions and result.total_instructions > 0
    assert result.metrics["elapsed_cycles"] > 0
    assert result.metrics["duration_ns"] > 0
    assert result.provenance["source_sha256"]
    assert result.provenance["binary_sha256"]
