"""Behavioral checks for feature validation receipts and result accounting."""
import json
import os
from pathlib import Path
import sys
import time

import pytest

from amora.backends.flashgpu_sim.runner import capture, clean_environment, run_test, summarize
from amora.cli import build_parser
from amora.probes.nvidia.feature_validation import list_probes

def test_cli_and_probe_sources():
    args = build_parser().parse_args(["flashgpu_sim", "list"])
    assert args.backend == "flashgpu_sim"
    probes = list_probes()
    assert {p["probe_id"] for p in probes} == {"dsm", "tcgen05", "tmem-capacity"}
    assert all(len(p["source_sha256"]) == 64 for p in probes)

def test_receipt_preserves_failure_and_rejects_reuse(tmp_path):
    record = capture([sys.executable, "-c", "print('diagnostic'); raise SystemExit(3)"],
                     label="failed", evidence_root=tmp_path, cwd=tmp_path)
    assert record["exit_code"] == 3
    assert (tmp_path / "failed.txt").read_text() == "diagnostic\n"
    assert json.loads((tmp_path / "failed.receipt.json").read_text())["sha256"] == record["sha256"]
    with pytest.raises(FileExistsError):
        capture([sys.executable, "-c", "print('overwrite')"],
                label="failed", evidence_root=tmp_path, cwd=tmp_path)
    assert (tmp_path / "failed.txt").read_text() == "diagnostic\n"

def test_timeout_terminates_spawned_process_group(tmp_path):
    marker = tmp_path / "child-finished"
    script = (
        "import subprocess,sys,time; "
        "subprocess.Popen([sys.executable,'-c',"
        + repr("import time,pathlib; time.sleep(0.5); pathlib.Path(" + repr(str(marker)) + ").write_text('done')")
        + "]); print('started',flush=True); time.sleep(10)"
    )
    record = capture([sys.executable, "-c", script], label="timeout",
                     evidence_root=tmp_path, cwd=tmp_path, timeout=0.15)
    assert record["timed_out"] is True
    assert record["exit_code"] == 124
    time.sleep(0.55)
    assert not marker.exists()
    assert "started" in (tmp_path / "timeout.txt").read_text()

def test_missing_command_retains_receipt(tmp_path):
    record = capture([str(tmp_path / "absent")], label="absent",
                     evidence_root=tmp_path, cwd=tmp_path)
    assert record["exit_code"] == 127
    assert record["launch_error"]
    assert (tmp_path / "absent.receipt.json").exists()

def test_standalone_driver_receipt_uses_only_program_arguments(tmp_path, monkeypatch):
    sim = tmp_path / "sim"
    config = sim / "configs/SM90_H100"
    config.mkdir(parents=True)
    (config / "gpgpusim.config").write_text("-gpgpu_compute_capability_major 9\n")
    monkeypatch.setattr("amora.backends.flashgpu_sim.runner.subprocess.check_output",
                        lambda *a, **k: "0123456789abcdef\n")
    record = run_test(binary=sys.executable, sim_root=sim, cuda_root=tmp_path / "cuda",
                      evidence_root=tmp_path / "evidence", label="standalone", mode="native",
                      program_arguments=["-c", "print('metric=17')"])
    assert record["exit_code"] == 0
    assert record["metadata"]["kind"] == "program"
    assert record["metadata"]["xml"] is None
    assert (tmp_path / "evidence/standalone.txt").read_text() == "metric=17\n"
    assert summarize(tmp_path / "evidence")["runs"][0]["status"] == "passed"

def test_native_environment_removes_simulator_state(monkeypatch):
    monkeypatch.setenv("LD_LIBRARY_PATH", "/sim/lib/release:/native/lib:/other/FlashGPU-Sim/lib/debug")
    monkeypatch.setenv("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN", "1")
    monkeypatch.setenv("FLASHGPU_SASS_AUTO", "1")
    monkeypatch.setenv("PTX_SIM_MODE_FUNC", "1")
    env = clean_environment(Path("/sim"), Path("/cuda"))
    assert env["LD_LIBRARY_PATH"] == "/cuda/lib64:/native/lib"
    assert all(key not in env for key in
               ("GPGPUSIM_SETUP_ENVIRONMENT_WAS_RUN", "FLASHGPU_SASS_AUTO", "PTX_SIM_MODE_FUNC"))

def test_result_accounting_keeps_failures_skips_and_crashes(tmp_path):
    records = [
        dict(label="suite", exit_code=1, elapsed_seconds=1),
        dict(label="crash", exit_code=-11, elapsed_seconds=2),
    ]
    (tmp_path / "commands.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    (tmp_path / "suite.xml").write_text(
        '<testsuites><testsuite>'
        '<testcase classname="S" name="Pass"/>'
        '<testcase classname="S" name="Skip"><skipped/></testcase>'
        '<testcase classname="S" name="Fail"><failure message="a"/><failure message="b"/></testcase>'
        '</testsuite></testsuites>'
    )
    results = {r["label"]: r for r in summarize(tmp_path)["runs"]}
    assert results["suite"]["passed"] == 1
    assert results["suite"]["failed"] == 1
    assert results["suite"]["skipped"] == 1
    assert results["suite"]["cases"][-1]["failure_messages"] == ["a", "b"]
    assert results["crash"]["status"] == "crashed"

