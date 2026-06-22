from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia import baseline


def test_baseline_list_marks_all_planned_probes_with_status():
    probes = baseline.list_probes()

    assert {probe["probe_id"] for probe in probes} == set(baseline.PLANNED_PROBES)
    assert all(probe["runner_available"] for probe in probes)
    implemented = {probe["probe_id"] for probe in probes if probe["implemented"]}
    assert implemented == {"topology.device_attributes"}


def test_run_all_returns_one_result_per_planned_probe_without_gpu():
    caps = NvidiaCapabilities(
        cuda_available=False,
        gpu_available=False,
        unsupported_reasons=["test environment has no CUDA"],
    )

    results = baseline.run_all(caps)

    assert len(results) == len(baseline.PLANNED_PROBES)
    assert {result.identity.probe_id for result in results} == set(baseline.PLANNED_PROBES)
    assert all(result.raw_observation.evidence_tier.value == "unsupported" for result in results)
