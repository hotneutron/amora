"""Bounded worker and versioned receipts for Portable-core V2 device correctness."""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys

from benchmark_generators.portable_core.expanded.v2.runner import verify
from benchmark_generators.portable_core.expanded.v2.spec import Case, Input, Node, Program, Ref, Tensor, digest

from .contracts import Blocked, MAX_CASES, MODE, PROTOCOL, check_case, device_index, prepare_case


def _write(path, data):
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def _load(frozen, requested, manifest_sha256=None):
    receipt, cases = verify(frozen, return_cases=True)
    manifest = receipt["files"]["manifest.json"]
    if manifest_sha256 is not None and manifest_sha256 != manifest:
        raise ValueError("V2 manifest does not match the pinned digest")
    if requested is not None:
        by_id = {c.id: c for c in cases}
        if len(requested) != len(set(requested)) or not set(requested) <= set(by_id):
            raise ValueError("unknown or duplicate frozen case ID")
        cases = [by_id[k] for k in requested]
    if not 1 <= len(cases) <= MAX_CASES:
        raise ValueError(f"select 1–{MAX_CASES} frozen cases per qualification shard with --case")
    return receipt, cases


def _source_hashes(backend):
    root = Path(__file__).resolve().parents[2]
    vendor = "nvidia" if backend == "cuda_torch" else "tpu"
    paths = ["benchmarking/portable_v2/runner.py", "benchmarking/portable_v2/contracts.py",
             f"backends/{vendor}/portable_core_v2.py"]
    if vendor == "tpu":
        paths += [f"backends/tpu/{n}.py" for n in ("architecture", "capabilities", "compatibility", "ownership")]
    return {p: sha256((root / p).read_bytes()).hexdigest() for p in paths}


def _boundary_controls(target, output):
    import numpy as np

    output.mkdir()
    values = {"x": np.array([[1., 1., 1., 1.0018]], dtype=np.float32)}
    observations = {}
    for dtype in ("fp16", "fp32"):
        id_ = f"v2-boundary-control-{dtype}"
        graph = Program((Input("x", "x", Tensor((1, 4), "fp32")),), (
            Node("r", "row_stat_broadcast", (Ref("x"),), (("stat", "mean"), ("temporary_dtype", dtype))),
            Node("y", "add", (Ref("r"), Ref("r"))),
        ), (Ref("y"),))
        case = Case(id_, "boundary-control", graph, ())
        here = output / dtype
        here.mkdir()
        types, raw, binding = prepare_case(case, values)
        error, compiler = check_case(target, case, types, raw, here)
        observations[dtype] = {"case_id": id_, "case_sha256": case.case_sha256, "binding": binding,
                               "passed": error is None, "error": error, "compiler": compiler}
    # The two exact oracle outputs differ by ~0.0009, greater than their FP32 comparison window.
    return observations


def _target(backend, device, generation):
    if backend == "cuda_torch":
        from amora.backends.nvidia.portable_core_v2 import Target
    else:
        from amora.backends.tpu.portable_core_v2 import Target
    return Target(device, expected_generation=generation)


def _result(request, rows, *, status, reason=None, **extra):
    return {"schema": PROTOCOL, "backend": request["backend"], "device": request["device"],
            "manifest_sha256": request["manifest_sha256"], "case_ids": request["case_ids"],
            "status": status, "reason": reason, "cases": rows,
            "native_performance_qualified": False, "cpu_fallback": False, **extra}


def _run(request, output):
    _, cases = _load(request["frozen"], request["case_ids"], request["manifest_sha256"])
    backend, device = request["backend"], request["device"]
    device_index(backend, device)
    sources = _source_hashes(backend)
    if sources != request["adapter_sources_sha256"]:
        raise ValueError("worker adapter source changed")
    rows = {c.id: {"case_id": c.id, "case_sha256": c.case_sha256, "program_sha256": c.program_sha256,
                   "upstream_id": c.upstream_id, "upstream_sha256": c.upstream_sha256,
                   "status": "blocked", "qualified": False} for c in cases}
    bindings = {}
    for c in cases:
        try:
            _, _, record = prepare_case(c)
            bindings[c.id] = record
        except Blocked as error:
            rows[c.id].update(reason=error.reason, detail=str(error))
        except ValueError as error:
            rows[c.id].update(status="failed", reason="invalid_input", detail=str(error))
    if not bindings:
        return _result(request, list(rows.values()), status="blocked", reason="no_admissible_cases")
    try:
        target = _target(backend, device, request["expected_generation"])
    except Blocked as error:
        for c in bindings:
            rows[c].update(reason=error.reason, detail=str(error))
        return _result(request, list(rows.values()), status="blocked", reason=error.reason)
    identity = target.identity
    hardware = {"target": identity, "target_sha256": digest(identity), **target.evidence}
    controls = {}
    failure = None
    try:
        controls = _boundary_controls(target, output / "boundary-controls")
        if not all(c["passed"] for c in controls.values()):
            failure = "rounding_control_failed"
        else:
            for i, case in enumerate(cases):
                if case.id not in bindings:
                    continue
                row = rows[case.id]
                row.update(binding=bindings[case.id], artifact_directory=f"case-{i:03d}")
                path = output / row["artifact_directory"]
                path.mkdir()
                try:
                    types, raw, replay = prepare_case(case)
                    if replay != bindings[case.id]:
                        raise ValueError("prepared input identity drift")
                    error, compiler = check_case(target, case, types, raw, path)
                    row.update(status="passed" if error is None else "failed", reason=error, compiler=compiler,
                               realization_sha256=digest({"protocol": PROTOCOL, "mode": MODE[backend],
                                   "case_sha256": case.case_sha256, "manifest_sha256": request["manifest_sha256"],
                                   "binding": replay, "target_sha256": hardware["target_sha256"],
                                   "sources": sources, "compiler": compiler}))
                except Blocked as error:
                    row.update(reason=error.reason, detail=str(error))
                except (ValueError, RuntimeError, TypeError) as error:
                    row.update(status="failed", reason="adapter_error", detail=str(error))
    finally:
        after, valid = target.finish()
        hardware.update(after)
    if not valid:
        failure = "ownership_control_failed"
    for row in rows.values():
        if "reason" not in row:
            row["reason"] = failure or "no_result"
        row["qualified"] = row["status"] == "passed" and failure is None
    status = "passed" if all(r["qualified"] for r in rows.values()) else "not_qualified"
    return _result(request, list(rows.values()), status=status, reason=failure, hardware=hardware,
                   adapter_sources_sha256=sources, controls=controls)


def qualify(frozen, output, backend, device, *, cases=None, manifest_sha256=None, expected_generation=None, timeout=120):
    """Validate V2 snapshot and run one bounded isolated accelerator worker."""
    device_index(backend, device)
    if backend != "tpu_jax" and expected_generation is not None:
        raise ValueError("expected generation applies to TPU only")
    if not math.isfinite(timeout) or not 0 < timeout <= 600:
        raise ValueError("timeout must be in (0, 600] seconds")
    receipt, pins = _load(frozen, cases, manifest_sha256)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    request = {"backend": backend, "device": device, "expected_generation": expected_generation,
               "frozen": str(Path(frozen).resolve()), "case_ids": [c.id for c in pins],
               "manifest_sha256": receipt["files"]["manifest.json"], "timeout_seconds": timeout,
               "adapter_sources_sha256": _source_hashes(backend)}
    msg = output / "request.json"
    _write(msg, request)
    env = dict(os.environ)
    if backend == "tpu_jax":
        env["JAX_PLATFORMS"] = "tpu"
    else:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    reason = None
    with (output/"worker.stdout.log").open("xb") as stdout, (output/"worker.stderr.log").open("xb") as stderr:
        proc = subprocess.Popen([sys.executable, "-m", __name__, str(msg)], stdout=stdout, stderr=stderr,
                                env=env, start_new_session=True)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            reason = "worker_timeout"
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    try:
        worker = json.loads((output/"worker.json").read_text()) if reason is None else {}
    except (OSError, ValueError):
        worker = {}
    expected = [(c.id, c.case_sha256) for c in pins]
    if (worker.get("schema") != PROTOCOL or worker.get("manifest_sha256") != request["manifest_sha256"]
            or worker.get("backend") != backend or worker.get("device") != device
            or worker.get("native_performance_qualified") is not False or worker.get("cpu_fallback") is not False
            or [(r.get("case_id"), r.get("case_sha256")) for r in worker.get("cases", [])] != expected
            or (proc.returncode == 0) != (worker.get("status") == "passed")):
        worker = _result(request, [{"case_id": c.id, "case_sha256": c.case_sha256,
            "status": "blocked", "qualified": False, "reason": reason or "worker_failed"} for c in pins],
            status="blocked", reason=reason or "worker_failed")
    worker["worker_exit_code"] = proc.returncode
    _write(output/"result.json", worker)
    return worker


def main():
    if len(sys.argv) != 2:
        raise SystemExit("internal worker expects one qualification request")
    request_path = Path(sys.argv[1])
    request = json.loads(request_path.read_text())
    result = _run(request, request_path.parent)
    _write(request_path.parent/"worker.json", result)
    return int(result["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
