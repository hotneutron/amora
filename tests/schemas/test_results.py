from amora.schemas.evidence import EvidenceTier, FitStatus, UncertaintyCategory
from amora.schemas.results import ProbeResult


def test_unsupported_result_has_all_layers():
    result = ProbeResult.unsupported("probe.x", "not available")
    data = result.to_dict()

    assert data["identity"]["probe_id"] == "probe.x"
    assert data["raw_observation"]["evidence_tier"] == EvidenceTier.UNSUPPORTED.value
    assert data["normalized_measurement"]["fit_status"] == FitStatus.UNSUPPORTED.value
    assert data["normalized_measurement"]["uncertainty"] == UncertaintyCategory.INDETERMINATE.value
    assert data["backend_interpretation"]["downgrade_reason"] == "not available"
    assert data["simulator_estimate"]["mapping_contract"] == "unsupported"


def test_unsupported_result_serializes_without_enum_objects():
    data = ProbeResult.unsupported("probe.y", "missing").to_dict()

    assert data["raw_observation"]["evidence_tier"] == "unsupported"
    assert data["simulator_estimate"]["fit_status"] == "unsupported"
