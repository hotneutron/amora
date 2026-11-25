import json

import pytest

from amora.backends.tpu.reporting import write_tpu_report
from amora.reports.markdown_report import derive_layout


def report(target="instance1", runtime="runtime1"):
    return {"schema_version": 1, "results": [], "metadata": {
        "report_identity": {"family": "v5p", "sku": "v5p", "target_config_digest": target,
                            "runtime_digest": runtime},
        "backend_capabilities": {"backend": "tpu_jax", "runtime_available": True,
                                 "devices": [{"id": 0, "name": "TPU v5p"}]}}}


def test_reports_separate_same_sku_instances_and_runtimes(tmp_path):
    first = write_tpu_report(report(), tmp_path, run_id="run1")
    second = write_tpu_report(report("instance2"), tmp_path, run_id="run1")
    third = write_tpu_report(report(runtime="runtime2"), tmp_path, run_id="run1")
    assert len({first,second,third}) == 3
    with pytest.raises(FileExistsError): write_tpu_report(report(), tmp_path, run_id="run1")
    assert len(json.loads((tmp_path/"google_tpu/v5p/manifest.json").read_text())["runs"]) == 3
    environment = (first.parent/"environment.md").read_text()
    assert "Runtime:" in environment and "CUDA:" not in environment and "GPU:" not in environment
    assert derive_layout(report()) == ("google_tpu", "v5p", "v5p")


def test_conflicting_generation_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="profile_conflict"):
        write_tpu_report(report(), tmp_path, family="v5e")


def test_unqualified_measurement_cannot_be_curated(tmp_path):
    source=report()
    source["metadata"]["qualification"]={"status":"inconclusive"}
    with pytest.raises(ValueError,match="unqualified"):
        write_tpu_report(source,tmp_path)


def test_curated_report_binds_verified_source_artifacts(tmp_path):
    from amora.backends.tpu.artifacts import finish_manifest,write_json
    source=tmp_path/"source";write_json(source/"report.json",report())
    manifest=finish_manifest(source,{"status":"passed"})
    destination=write_tpu_report(report(),tmp_path/"reports",source_report=source/"report.json")
    copied=json.loads((destination.parent/"report.json").read_text())
    assert copied["metadata"]["source_evidence"]["run_digest"]==manifest["run_digest"]
    assert "Verified source manifest" in destination.read_text()
    (source/"report.json").write_text('{}')
    with pytest.raises(ValueError,match="artifact hash mismatch"):
        write_tpu_report(report(),tmp_path/"reports",run_id="tampered",source_report=source/"report.json")
