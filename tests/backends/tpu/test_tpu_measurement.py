"""CPU-synthetic orchestration checks; never qualify TPU hardware."""
import json

import pytest

from amora.backends.tpu.measurement import execute_in_process


def test_compiled_campaign_keeps_time_domains_and_artifacts(tmp_path,monkeypatch):
    jax=pytest.importorskip("jax")
    original=jax.local_devices
    monkeypatch.setattr(jax,"local_devices",lambda *a,**k: original(backend="cpu"))
    caps={"status":"ok","backend":"tpu_jax","devices":[{"name":"synthetic CPU fixture","id":0}],
          "selected_device":{"name":"synthetic CPU fixture","id":0},
          "target":{"target_config_digest":"synthetic-target"},
          "runtime":{"runtime_digest":"synthetic-runtime","runtime_lock_sha256":"synthetic-lock","runtime_lock_versions_verified":True},"architecture":{"generation":"unknown"}}
    monkeypatch.setattr("amora.backends.tpu.capabilities.discover_in_process",lambda *a: caps)
    monkeypatch.setattr("amora.backends.tpu.ownership.snapshot",lambda **k:{"state":"exclusive","holders":[]})
    recipe={"schema_version":1,"recipe_id":"synthetic","warmup":1,"samples":3,"repeats":1,
            "max_relative_mad":1,"cases":[{"probe_id":"vector.elementwise_throughput","shape":[7,17]}]}
    result=execute_in_process(recipe,tmp_path)
    assert result["status"] == "passed", result
    assert (tmp_path/"semantic-verification.json").exists()
    correctness=json.loads((tmp_path/"correctness.json").read_text())["results"][0]
    assert correctness["metadata"]["compile_ns"] > 0
    assert correctness["metadata"]["staging_ns"] > 0
    assert correctness["timing"]["time_domain"] == "synchronized_host"
    assert "compile_ns" not in json.loads((tmp_path/"samples.jsonl").read_text().splitlines()[0])
    assert result["compiler_artifacts"]


def test_portable_shard_has_complete_output_checks(tmp_path,monkeypatch):
    jax=pytest.importorskip("jax")
    original=jax.local_devices
    monkeypatch.setattr(jax,"local_devices",lambda *a,**k:original(backend="cpu"))
    caps={"status":"ok","backend":"tpu_jax","devices":[],"selected_device":{},
          "target":{"target_config_digest":"synthetic-target"},"runtime":{"runtime_digest":"synthetic-runtime","runtime_lock_sha256":"synthetic-lock","runtime_lock_versions_verified":True},
          "architecture":{"generation":"unknown"}}
    monkeypatch.setattr("amora.backends.tpu.capabilities.discover_in_process",lambda *a:caps)
    monkeypatch.setattr("amora.backends.tpu.ownership.snapshot",lambda **k:{"state":"exclusive","holders":[]})
    recipe={"schema_version":1,"recipe_id":"synthetic-portable","cases":[{"kind":"portable_core",
            "tier":28,"profile":"smoke","start":0,"stop":4,"measure":False}]}
    result=execute_in_process(recipe,tmp_path)
    assert result["status"] == "passed" and result["cases"] == 4
    records=json.loads((tmp_path/"correctness.json").read_text())["results"]
    assert all(r["correctness"]["passed"] and "timing" not in r for r in records)


def test_missing_optional_feature_stays_unsupported(tmp_path,monkeypatch):
    jax=pytest.importorskip("jax")
    original=jax.local_devices
    monkeypatch.setattr(jax,"local_devices",lambda *a,**k:original(backend="cpu"))
    caps={"status":"ok","backend":"tpu_jax","devices":[],"selected_device":{},
          "target":{"target_config_digest":"synthetic-target"},"runtime":{"runtime_digest":"synthetic-runtime","runtime_lock_sha256":"synthetic-lock","runtime_lock_versions_verified":True},
          "architecture":{"generation":"unknown"}}
    monkeypatch.setattr("amora.backends.tpu.capabilities.discover_in_process",lambda *a:caps)
    monkeypatch.setattr("amora.backends.tpu.ownership.snapshot",lambda **k:{"state":"exclusive","holders":[]})
    def missing(*args):raise NotImplementedError("Pallas unavailable in fixture")
    monkeypatch.setattr("amora.backends.tpu.probes.build_probe",missing)
    result=execute_in_process({"schema_version":1,"recipe_id":"missing-feature",
        "cases":[{"probe_id":"global_memory.streaming"}]},tmp_path)
    assert result["status"] == "inconclusive"
    report=json.loads((tmp_path/"report.json").read_text())
    assert report["results"][0]["raw_observation"]["evidence_tier"] == "unsupported"
