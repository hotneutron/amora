"""Bounded subprocess boundary for both discovery and measurement."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def run_worker(action, payload, timeout=120):
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("worker timeout must be positive")
    with tempfile.TemporaryDirectory(prefix="amora-tpu-") as directory:
        root = Path(directory)
        request, response = root / "request.json", root / "response.json"
        request.write_text(json.dumps(payload))
        env = dict(os.environ, JAX_PLATFORMS="tpu")
        started = time.perf_counter_ns()
        process = subprocess.Popen(
            [sys.executable, "-m", "amora.backends.tpu.worker", action,
             str(request), str(response)], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            return {"status": "failed", "reason": "initialization_timeout" if action == "discover"
                    else "measurement_timeout", "stderr": stderr[-16000:],
                    "worker_wall_ns": time.perf_counter_ns() - started}
        if response.exists():
            result = json.loads(response.read_text())
        else:
            result = {"status": "failed", "reason": "worker_failed",
                      "exit_code": process.returncode}
        result["worker_stderr"] = stderr[-16000:]
        result["worker_stdout"] = stdout[-4000:]
        result["worker_wall_ns"] = time.perf_counter_ns() - started
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("discover", "measure"))
    parser.add_argument("request", type=Path)
    parser.add_argument("response", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.request.read_text())
    try:
        if args.action == "discover":
            from .capabilities import discover_in_process
            result = discover_in_process(**payload)
        else:
            from .measurement import execute_in_process
            result = execute_in_process(**payload)
    except Exception as error:
        import traceback
        result = {"status": "failed", "reason": type(error).__name__ + ": " + str(error),
                  "traceback": traceback.format_exc()}
    args.response.write_text(json.dumps(result, indent=2, default=str))
    return 0 if result.get("status") != "failed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
