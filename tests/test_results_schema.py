from amora.schemas.results import EvidenceTier, HardwareProfile, ParameterEstimate, ProbeResult


def test_parameter_estimate_serializes_evidence_value():
    estimate = ParameterEstimate(
        name="shader_core_config::warp_size",
        value=32,
        evidence=EvidenceTier.DIRECT_METADATA,
        confidence=0.95,
        unit="threads",
        risk="low",
        notes=("CUDA attribute warp_size",),
    )

    as_dict = estimate.to_dict()

    assert as_dict["evidence"] == "direct_metadata"
    assert as_dict["value"] == 32
    assert as_dict["notes"] == ["CUDA attribute warp_size"]


def test_hardware_profile_collects_parameter_estimates_and_confidence():
    profile = HardwareProfile(
        target={"vendor": "nvidia"},
        raw_results=[
            ProbeResult(
                name="topology/device_attributes.py",
                tier="P0",
                status="ok",
                estimates=[
                    ParameterEstimate(
                        name="shader_core_config::warp_size",
                        value=32,
                        evidence=EvidenceTier.DIRECT_METADATA,
                        confidence=0.95,
                    )
                ],
            )
        ],
    )

    rendered = profile.to_dict()

    assert rendered["repo_parameter_estimates"]["shader_core_config::warp_size"] == 32
    assert rendered["confidence"]["shader_core_config::warp_size"] == 0.95
    assert rendered["raw_results"][0]["name"] == "topology/device_attributes.py"
