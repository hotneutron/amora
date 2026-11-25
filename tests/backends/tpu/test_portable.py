import numpy as np
import pytest

from benchmark_generators import portable_core as core
from amora.backends.tpu.portable_core import check_outputs, prepare, program, verify_semantics


def test_all_semantic_fingerprints():
    assert verify_semantics()["passed"]


def test_all_portable_jit_cpu_semantics():
    jax = pytest.importorskip("jax")
    device = jax.devices("cpu")[0]
    for case in core.generate_cases(28, "smoke"):
        args, expected, _ = prepare(case, device)
        function = jax.jit(program(case))
        actual = function(*args)
        types = {**case["inputs"], **{n["id"]: n for n in case["nodes"]}}
        dtypes = [types[n]["dtype"] for n in case.get("outputs", [case["output"]])]
        matrix = any(n["op"] == "matmul" for n in case["nodes"])
        result = check_outputs(actual, expected, matrix=matrix, output_dtypes=dtypes)
        assert result["passed"], (case["case_id"], result)


def test_gather_inputs_stay_dynamic():
    jax = pytest.importorskip("jax")
    case = next(c for c in core.generate_cases(20) if any(n["op"] == "gather" for n in c["nodes"]))
    raw = core.make_inputs(case)
    args, expected, _ = prepare(case, jax.devices("cpu")[0], raw)
    compiled = jax.jit(program(case)).lower(*args).compile()
    original = compiled(*args)
    changed = {name: value.copy() for name, value in raw.items()}
    name = next(name for name,spec in case["inputs"].items() if spec["domain"] == "index")
    changed[name] = (changed[name]+1) % case["inputs"][name]["upper"]
    other, _, _ = prepare(case, jax.devices("cpu")[0], changed)
    assert any(not np.array_equal(a,b) for a,b in zip(original,compiled(*other)))
