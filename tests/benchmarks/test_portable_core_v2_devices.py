"""Local controls for V2's device-check protocol; no native performance claims."""
import json
from pathlib import Path
import signal
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
from amora.benchmarking.portable_v2 import runner
from amora.benchmarking.portable_v2.contracts import Blocked, compare, device_index, prepare_case
from benchmark_generators.portable_core.expanded.v2 import suites
from benchmark_generators.portable_core.expanded.v2.reference import oracle
from benchmark_generators.portable_core.expanded.v2.runner import freeze


def test_inspect_v2_without_frameworks():
    script = '''import builtins
old = builtins.__import__
def check(name, *args, **kw):
    if name.partition('.')[0] in {'jax','torch','numpy'}: raise AssertionError(name)
    return old(name, *args, **kw)
builtins.__import__ = check
from amora.cli import main
raise SystemExit(main(['benchmarks','inspect-v2']))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, check=True)
    value = json.loads(result.stdout)
    assert not value["hardware_opened"] and not value["cpu_fallback"]
    assert set(value["backends"]) == {"cuda_torch", "tpu_jax"}


@pytest.mark.parametrize("backend, device", [("tpu_jax", "cpu:0"), ("cuda_torch", "tpu:0"), ("tpu_jax", "tpu")])
def test_device_requires_explicit_matching_platform(backend, device):
    with pytest.raises(ValueError, match="explicit"):
        device_index(backend, device)


def test_wrong_manifest_and_duplicate_cases_fail_before_worker(tmp_path):
    root = tmp_path / "freeze"
    freeze(root, "h3-programs")
    case = suites.generate("h3-programs")[0]
    with pytest.raises(ValueError, match="pinned digest"):
        runner.qualify(root, tmp_path / "output", "tpu_jax", "tpu:0", manifest_sha256="0"*64)
    with pytest.raises(ValueError, match="duplicate"):
        runner.qualify(root, tmp_path / "output", "tpu_jax", "tpu:0", cases=[case.id, case.id])
    assert not (tmp_path / "output").exists()


def test_packed_unique_index_admission_and_repeatable_binding():
    cases = suites.generate("h3-semantics", "development")
    scatter = next(c for c in cases if c.id == "control-indexed_scatter_add-distinct-64x96")
    types, raw, a = prepare_case(scatter)
    assert types["z"][0].shape == (64, 96)
    assert a == prepare_case(scatter)[2]
    raw["idx"][1] = raw["idx"][0]
    with pytest.raises(Blocked) as denied:
        prepare_case(scatter, raw)
    assert denied.value.reason == "unsupported_duplicate_indices"
    raw["idx"][1] = 64
    with pytest.raises(ValueError, match="index outside"):
        prepare_case(scatter, raw)


@pytest.mark.parametrize("variant", ["strided", "offset"])
def test_physical_layout_never_silently_repacks(variant):
    case = next(c for c in suites.generate("h3-semantics", "development")
                if c.id == f"control-indexed_embedding_sum-{variant}-64x96")
    with pytest.raises(Blocked) as denied:
        prepare_case(case)
    assert denied.value.reason == "unsupported_layout"


def test_oracle_requires_exact_discrete_values_and_exact_dtype():
    value = np.array([2], dtype=np.float32)
    almost = np.array([2.0001], dtype=np.float32)
    assert compare([almost], [value], [False]) is None
    assert compare([almost], [value], [True]) == "oracle value mismatch on port 0"
    assert "dtype" in compare([value.astype(np.float16)], [value], [False])
    assert compare([], [value], [False]) == "arity mismatch"


def test_worker_timeout_has_no_qualified_rows(tmp_path, monkeypatch):
    calls = []

    class Process:
        pid = 4242
        returncode = -15

        def __init__(self, argv, **kw):
            assert argv[1:3] == ["-m", "amora.benchmarking.portable_v2.runner"]
            assert kw["start_new_session"] and kw["env"]["JAX_PLATFORMS"] == "tpu"

        def wait(self, timeout=None):
            if not calls:
                raise subprocess.TimeoutExpired("fake", timeout)
            return self.returncode

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(runner.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    frozen = tmp_path / "frozen"
    freeze(frozen, "h3-programs")
    result = runner.qualify(frozen, tmp_path / "out", "tpu_jax", "tpu:0")
    assert calls == [(4242, signal.SIGTERM)]
    assert result["status"] == "blocked" and result["reason"] == "worker_timeout"
    assert len(result["cases"]) == 6 and not any(x["qualified"] for x in result["cases"])


def test_torch_device_array_dispatch_on_cpu_only():
    torch = pytest.importorskip("torch")
    from amora.backends.nvidia.portable_core_v2 import _DeviceArrayFunctions
    arrays = _DeviceArrayFunctions(torch, torch.device("cpu"))
    assert arrays.arange(3).tolist() == [0, 1, 2]
    assert arrays.eye(2, dtype=torch.float32).tolist() == [[1., 0.], [0., 1.]]


def test_jax_jit_control_preserves_ordered_topk_ports_on_cpu():
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    from amora.backends.tpu.portable_core_v2 import _JaxMath, build_program
    cpu = jax.devices("cpu")[0]
    case = next(c for c in suites.generate("mechanism-controls") if c.family == "row_topk_select")
    types, raw, _ = prepare_case(case)
    function = jax.jit(build_program(case, types, _JaxMath(jnp, jax.lax)))
    leaves = function(*(jax.device_put(raw[i.id], cpu) for i in case.program.inputs))
    assert all(a.devices() == {cpu} for a in leaves)
    assert compare(list(map(np.asarray, leaves)), oracle(case, raw), [True, True]) is None
