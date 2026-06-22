from amora.backends.nvidia.cuda import NvidiaCapabilities, NvidiaDevice
from amora.probes.nvidia.baseline.topology.device_attributes import run


def test_device_attributes_reports_unsupported_without_gpu():
    caps = NvidiaCapabilities(
        cuda_available=False,
        gpu_available=False,
        unsupported_reasons=["no gpu"],
    )

    results = run(caps)

    assert len(results) == 1
    assert results[0].identity.probe_id == "topology.device_attributes"
    assert results[0].raw_observation.unsupported_reason == "no gpu"


def test_device_attributes_emits_identity_when_gpu_is_reported():
    caps = NvidiaCapabilities(
        cuda_available=False,
        gpu_available=True,
        devices=[NvidiaDevice(index=0, name="Mock GPU", uuid="GPU-uuid")],
    )

    results = run(caps)
    data = results[0].to_dict()

    assert data["raw_observation"]["evidence_tier"] == "direct_metadata"
    assert data["raw_observation"]["values"]["device_name"] == "Mock GPU"
    assert data["simulator_estimate"]["fit_status"] == "direct"
