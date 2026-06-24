from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia import baseline


def test_baseline_list_marks_all_planned_probes_with_status():
    probes = baseline.list_probes()

    assert {probe["probe_id"] for probe in probes} == set(baseline.PLANNED_PROBES)
    assert all(probe["runner_available"] for probe in probes)
    implemented = {probe["probe_id"] for probe in probes if probe["implemented"]}
    assert implemented == set(baseline.PLANNED_PROBES)


def test_run_all_returns_one_result_per_planned_probe_without_gpu():
    caps = NvidiaCapabilities(
        cuda_available=False,
        gpu_available=False,
        unsupported_reasons=["test environment has no CUDA"],
    )

    results = baseline.run_all(caps)

    assert len(results) == len(baseline.PLANNED_PROBES)
    assert {result.identity.probe_id for result in results} == set(baseline.PLANNED_PROBES)
    by_id = {result.identity.probe_id: result for result in results}
    # CPU-only probes do not require a GPU and stay non-unsupported.
    assert by_id["topology.occupancy"].raw_observation.evidence_tier.value == "direct_metadata"
    # Kernel-bound probes remain unsupported without a GPU but register their CUDA source hash.
    kernel_probe_ids = {
        "topology.persistent_cta",
        "arithmetic_latency.dependent_chain",
        "arithmetic_throughput.independent_chains",
        "shared_memory.pointer_chase",
        "shared_memory.bank_stride",
        # P1 kernel-bound probes
        "l1_cache.pointer_chase",
        "l1_cache.working_set",
        "l1_cache.conflict_sets",
        "scheduler_policy.ready_warps",
        "scheduler_policy.mixed_issue",
        "register_file.register_bank_sweep",
        "register_file.register_latency",
    }
    for probe_id in kernel_probe_ids:
        result = by_id[probe_id]
        assert result.raw_observation.evidence_tier.value == "unsupported"
        registered = result.raw_observation.values.get("registered_source")
        assert registered is not None
        assert registered["kind"] == "cuda_source"
        assert len(registered["sha256"]) == 64
    # Analyzers degrade to unsupported when their kernel inputs are unavailable.
    analyzer_ids = {
        "shared_memory.analyze",
        "l1_cache.analyze",
        "scheduler_policy.analyze",
        "register_file.analyze",
    }
    for probe_id in analyzer_ids:
        assert by_id[probe_id].raw_observation.evidence_tier.value == "unsupported"


def test_p1_probe_groups_are_registered():
    ids = set(baseline.PLANNED_PROBES)
    expected_p1 = {
        "l1_cache.pointer_chase",
        "l1_cache.working_set",
        "l1_cache.conflict_sets",
        "l1_cache.analyze",
        "scheduler_policy.ready_warps",
        "scheduler_policy.mixed_issue",
        "scheduler_policy.analyze",
        "register_file.register_bank_sweep",
        "register_file.register_latency",
        "register_file.analyze",
    }
    assert expected_p1 <= ids
