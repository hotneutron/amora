import builtins
import json
from pathlib import Path
import time

import pytest

from amora.backends.tpu.architecture import profiles, resolve_profile
from amora.backends.tpu.artifacts import finish_manifest, validate_manifest, write_json
from amora.backends.tpu.capabilities import device_index, discover_capabilities
from amora.backends.tpu.measurement import validate_recipe
from amora.backends.tpu.runner import drain, time_program


def test_generation_variants_and_unknown():
    assert set(profiles()) == {"v2", "v3", "v4", "v5e", "v5p", "v6e", "tpu7x"}
    for name, profile in profiles().items():
        assert resolve_profile(profile["aliases"][0])["generation"] == name
        assert profile["sources"] and profile["facts"]["matrix_tile"]["value"] is None
    assert resolve_profile("TPU v5e")["generation"] != resolve_profile("TPU v5p")["generation"]
    assert resolve_profile("TPU v99")["generation"] == "unknown"
    assert resolve_profile("Trillium")["generation"] == "v6e"
    with pytest.raises(ValueError, match="profile_conflict"):
        resolve_profile("TPU v5e", "v5p")
    with pytest.raises(ValueError, match="profile_conflict"):
        resolve_profile("TPU v4", chip_version="v5p")


@pytest.mark.parametrize("device", ["cpu", "tpu:-1", "tpu:x", "tpu:1/../../"])
def test_invalid_device(device):
    with pytest.raises(ValueError): device_index(device)


def test_missing_runtime_is_explicit(monkeypatch):
    monkeypatch.setattr("amora.backends.tpu.capabilities.runtime_identity",
                        lambda: {"packages": {"jax": None, "libtpu": None}})
    assert discover_capabilities()["unsupported_reasons"] == ["runtime_missing"]


def test_recipe_rejects_invalid_measurement_units():
    recipe = {"schema_version": 1, "recipe_id": "test", "cases": [{"probe_id": "runtime.dispatch_latency"}]}
    for change in ({"samples": True}, {"batch": 0}, {"warmup": -1}, {"max_relative_mad": float("nan")},
                   {"recipe_id": "../escape"}, {"cases": [{"probe_id": "unknown"}]},
                   {"cases": [{"probe_id": "matrix.matmul_throughput", "shape": [2, -1, 2]}]}):
        with pytest.raises(ValueError): validate_recipe({**recipe, **change})


def test_manifest_tamper_and_overwrite(tmp_path):
    write_json(tmp_path / "correctness.json", {"passed": True})
    finish_manifest(tmp_path, {"status": "passed"})
    assert validate_manifest(tmp_path / "manifest.json")["status"] == "passed"
    with pytest.raises(FileExistsError): write_json(tmp_path / "correctness.json", {})
    (tmp_path / "correctness.json").write_text("{}")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        validate_manifest(tmp_path / "manifest.json")


def test_timing_waits_for_all_outputs():
    class Async:
        def block_until_ready(self): time.sleep(0.002)
    result = time_program(lambda: (Async(), {"other": Async()}),
                          warmup=0, samples=2, repeats=1, batch=2, max_relative_mad=1)
    assert all(item["dispatch_to_completion_ns"] >= 7_000_000 for item in result["samples"])
    assert result["time_domain"] == "synchronized_host"
    assert result["batch_size"] == 2
    assert result["median_ns_per_invocation"] == result["summary"]["median"] / 2


def test_busy_discovery_never_initializes_jax(monkeypatch):
    from amora.backends.tpu.capabilities import discover_in_process
    monkeypatch.setattr("amora.backends.tpu.ownership.snapshot",
                        lambda: {"state": "busy", "holders": [{"pid": 99}]})
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] == "jax": raise AssertionError("JAX initialized on busy host")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert discover_in_process()["unsupported_reasons"] == ["resource_busy"]


def test_worker_timeout_terminates_only_child_group(monkeypatch):
    import subprocess
    from amora.backends.tpu.worker import run_worker
    signals = []
    class Child:
        pid = 123456
        returncode = -15
        calls = 0
        def communicate(self, timeout=None):
            self.calls += 1
            if self.calls == 1: raise subprocess.TimeoutExpired("worker", timeout)
            return "", "timed out"
    monkeypatch.setattr("amora.backends.tpu.worker.subprocess.Popen", lambda *a,**k: Child())
    monkeypatch.setattr("amora.backends.tpu.worker.os.killpg", lambda pid,sig: signals.append((pid,sig)))
    result = run_worker("measure", {}, timeout=0.1)
    assert result["reason"] == "measurement_timeout"
    assert len(signals) == 1 and signals[0][0] == 123456


def test_busy_measurement_emits_unsupported_report_without_jax(tmp_path,monkeypatch):
    from amora.backends.tpu.measurement import execute_in_process
    caps = {"status":"unsupported","backend":"tpu_jax","devices":[],
            "unsupported_reasons":["resource_busy"]}
    monkeypatch.setattr("amora.backends.tpu.capabilities.discover_in_process",lambda *a: caps)
    recipe={"schema_version":1,"recipe_id":"busy","cases":[{"probe_id":"runtime.dispatch_latency"}]}
    result=execute_in_process(recipe,tmp_path)
    assert result["status"] == "failed" and result["reason"] == "resource_busy"
    report=json.loads((tmp_path/"report.json").read_text())
    assert report["results"][0]["raw_observation"]["evidence_tier"] == "unsupported"


def test_runtime_lock_must_match_installed_versions(tmp_path,monkeypatch):
    from amora.backends.tpu.compatibility import runtime_identity
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("amora.backends.tpu.compatibility.importlib.metadata.version",lambda name:"1.0")
    lock=tmp_path/"runtime-lock.txt"
    lock.write_text("jax==2.0 --hash=sha256:" + "a"*64 + "\n")
    assert not runtime_identity()["runtime_lock_versions_verified"]
    lock.write_text("jax==1.0 --hash=sha256:" + "a"*64 + "\n")
    assert runtime_identity()["runtime_lock_versions_verified"]


def test_warmup_uses_the_measured_batch_unit():
    calls=[]
    result=time_program(lambda:calls.append(1),warmup=2,samples=2,repeats=1,batch=3,max_relative_mad=1)
    assert len(calls)==12
    assert result["warmup_invocations"]==6
