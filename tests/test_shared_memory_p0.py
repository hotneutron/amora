from amora.probes.nvidia.p0.shared_memory.analyze import (
    infer_shared_memory_bank_period,
    shared_memory_analysis_probe,
)
from amora.probes.nvidia.p0.shared_memory.sources import shared_memory_source_probe


def test_infer_shared_memory_bank_period_uses_stride_peaks():
    assert infer_shared_memory_bank_period([10, 30, 10, 30, 10, 30]) == 2


def test_shared_memory_analysis_probe_reports_indeterminate_without_period():
    result = shared_memory_analysis_probe([10, 11, 10, 11, 10, 11])

    assert result.status == "indeterminate"
    assert result.estimates == []
    assert "No stable" in result.warnings[0]


def test_shared_memory_source_probe_hashes_cuda_sources():
    result = shared_memory_source_probe()

    assert result.status == "planned"
    assert {artifact.path.rsplit("/", 1)[-1] for artifact in result.artifacts} == {
        "bank_stride.cu",
        "pointer_chase.cu",
    }
    assert all(artifact.sha256 for artifact in result.artifacts)
