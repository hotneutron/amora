"""Portable v2 contracts; only CPU correctness is covered here."""
from dataclasses import replace
import hashlib
import json
import subprocess
import sys

import pytest

np = pytest.importorskip("numpy")
from benchmark_generators import portable_core as portable
from benchmark_generators.portable_core.__main__ import main
from benchmark_generators.portable_core.expanded.v2 import runner, suites
from benchmark_generators.portable_core.expanded.v2.execution import Backend, BackendUnavailable, execute_prepared, make_inputs, prepare
from benchmark_generators.portable_core.expanded.v2.reference import oracle, check
from benchmark_generators.portable_core.expanded.v2.spec import Case, Input, Node, Program, Ref, Tensor


def program(nodes, outputs, *, inputs=None):
    return Program(inputs or (Input("x", "data", Tensor(("M", "N"), "fp32")),), tuple(nodes), tuple(outputs))


def case(graph, m=2, n=4):
    return Case("case", "test", graph, (("M", m), ("N", n)))


def test_canonical_renaming_and_reordered_independent_work():
    first = case(program((Node("a", "relu", (Ref("x"),)), Node("b", "row_roll", (Ref("x"),)),
                          Node("c", "add", (Ref("a"), Ref("b")))), (Ref("c"),)))
    second = case(program((Node("r", "row_roll", (Ref("x"),)), Node("q", "relu", (Ref("x"),)),
                           Node("s", "add", (Ref("q"), Ref("r")))), (Ref("s"),)))
    assert first.program_sha256 == second.program_sha256
    assert first.case_sha256 == second.case_sha256
    assert case(first.program, n=8).program_sha256 == first.program_sha256
    assert case(first.program, n=8).case_sha256 != first.case_sha256
    assert replace(first, seed=1).case_sha256 != first.case_sha256


def test_fork_join_and_same_order_chain_have_different_edges():
    chain, fork = suites.generate("h3-programs")[:2]
    assert [n.op for n in chain.program.nodes] == [n.op for n in fork.program.nodes]
    assert chain.program_sha256 != fork.program_sha256
    edges = suites.generate("h3-programs")[2:4]
    assert edges[0].program_sha256 != edges[1].program_sha256


def test_fanout_and_duplicating_equal_producers_have_distinct_identity():
    a = Node("a", "relu", (Ref("x"),))
    shared = case(program((a, Node("b", "add", (Ref("a"), Ref("a")))), (Ref("b"),)))
    duplicated = case(program((a, Node("c", "relu", (Ref("x"),)),
                               Node("b", "add", (Ref("a"), Ref("c")))), (Ref("b"),)))
    assert shared.program_sha256 != duplicated.program_sha256


@pytest.mark.parametrize("nodes,outputs,match", [
    ((Node("a", "relu", (Ref("x", 1),)),), (Ref("a"),), "input port"),
    ((Node("a", "relu", (Ref("a"),)),), (Ref("a"),), "input port"),
    ((Node("a", "relu", (Ref("x"),)),), (Ref("a", 1),), "output port"),
    ((Node("a", "relu", (Ref("x"),)), Node("unused", "relu", (Ref("x"),))), (Ref("a"),), "unreachable"),
])
def test_validation_rejects_invalid_ports_cycles_and_dead_work(nodes, outputs, match):
    with pytest.raises(ValueError, match=match):
        case(program(nodes, outputs))


@pytest.mark.parametrize("op,n,attrs", [("segment_sum4", 3, ()),
    ("row_topk_select", 4, (("k", 2), ("ties", "arbitrary"))),
    ("cast", 4, (("dtype", "int32"),))])
def test_invalid_op_contracts_fail_closed(op, n, attrs):
    with pytest.raises(ValueError):
        case(program((Node("a", op, (Ref("x"),), attrs),), (Ref("a"),)), n=n)


def test_roundtrip_checks_hash_and_metadata():
    original = suites.generate("operation-gate-v2")[0]
    assert Case.read(original.record()) == original
    record = original.record()
    record["program_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        Case.read(record)


def test_historical_cohorts_and_torch_witness_quota():
    reference = portable.generate_cases(suite="operation-gate-v2", profile="reference")
    controls = portable.generate_cases(suite="mechanism-controls", profile="reference")
    assert len(reference) == 720 and len({c.upstream_id for c in reference}) == 720
    assert len(controls) == 288 and all(c.intent == "development" for c in controls)
    assert {c.intent for c in reference} == {"reference"}
    witnesses = [c for c in reference if "torch" in suites.required(c, "operation-gate-v2", "reference")]
    assert len(witnesses) == 240 and {c.family for c in witnesses} == {c.family for c in reference}
    recurrent = next(c for c in reference if c.program.unroll == 2)
    assert len(recurrent.program.state_edges) == 1
    assert recurrent.program.state_edges[0][0].producer.endswith("_u0")


def test_fp16_temporary_rounding_is_observable_in_later_operation():
    def fixture(temp):
        return case(program((Node("reduce", "row_stat_broadcast", (Ref("x"),),
                                  (("stat", "mean"), ("temporary_dtype", temp))),
                              Node("z", "add", (Ref("reduce"), Ref("reduce")))), (Ref("z"),)), m=1, n=4)
    value = np.array([[1., 1., 1., 1.001]], dtype=np.float32)
    a, b = fixture("fp16"), fixture("fp32")
    assert a.program_sha256 != b.program_sha256
    obj = prepare(a, Backend(), {"x": value})
    got, = execute_prepared(obj)
    np.testing.assert_array_equal(got, [[2., 2., 2., 2.]])
    assert (oracle(b, {"x": value})[0] > got).all()
    assert check(obj, {"x": value}) == (True, None)


def test_rotary_uses_source_pair_order_and_per_column_angles():
    inputs = (Input("x", "data", Tensor(("M", "N"), "fp32")), Input("a", "angle", Tensor(("M", "N"), "fp32")))
    item = case(program((Node("z", "rotary_pair_transform", (Ref("x"), Ref("a"))),), (Ref("z"),), inputs=inputs), m=1, n=2)
    raw = {"x": np.array([[3, 7]], dtype=np.float32), "a": np.zeros((1, 2), dtype=np.float32)}
    want = [[3, 3]]
    np.testing.assert_array_equal(oracle(item, raw)[0], want)
    np.testing.assert_array_equal(execute_prepared(prepare(item, Backend(), raw))[0], want)


def test_topk_ties_and_output_port_order():
    item = case(program((Node("z", "row_topk_select", (Ref("x"),), (("k", 2), ("ties", "lowest_index"))),),
                        (Ref("z", 1), Ref("z"))), m=1)
    raw = {"x": np.array([[2, 9, 9, -1]], dtype=np.float32)}
    idx, got = execute_prepared(prepare(item, Backend(), raw))
    assert idx.dtype == np.int32 and got.dtype == np.float32
    np.testing.assert_array_equal(idx, [[1, 2]])
    np.testing.assert_array_equal(got, [[9, 9]])


def test_prepare_enforces_layout_and_external_indices():
    cases = suites.generate("h3-semantics", "development")
    item = next(c for c in cases if "embedding" in c.id and "strided" in c.id)
    raw = make_inputs(item)
    assert raw["x"].strides == ((96+13)*4, 4)
    obj = prepare(item, Backend(), raw)
    assert np.shares_memory(obj.inputs["x"][0], raw["x"])
    with pytest.raises(ValueError, match="strides"):
        prepare(item, Backend(), {**raw, "x": raw["x"].copy()})
    raw["idx"][0, 0] = 64
    with pytest.raises(ValueError, match="index outside"):
        prepare(item, Backend(), raw)


def test_backend_unsupported_device_has_no_fallback():
    with pytest.raises(BackendUnavailable, match="not qualified"):
        Backend("numpy", "tpu:0")
    item = suites.generate("h3-semantics")[0]
    with pytest.raises(ValueError, match="v2 cases require"):
        portable.evaluate(item, portable.create_backend("numpy"))


def test_freeze_reproducible_and_rejects_tamper_even_if_rehashed(tmp_path):
    out, other = tmp_path/"a", tmp_path/"b"
    runner.freeze(out, "mechanism-controls")
    runner.freeze(other, "mechanism-controls")
    for name in ("freeze_receipt.json", "manifest.json", "coverage.json"):
        assert (out/name).read_bytes() == (other/name).read_bytes()
    with pytest.raises(RuntimeError, match="overwrite"):
        runner.freeze(out, "mechanism-controls")
    receipt = runner.verify(out)
    manifest = out/"manifest.json"
    doc = json.loads(manifest.read_text())
    doc["cases"][0]["required_backends"] = []
    data = json.dumps(doc).encode()
    manifest.write_bytes(data)
    receipt["files"]["manifest.json"] = hashlib.sha256(data).hexdigest()
    (out/"freeze_receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="canonical"):
        runner.verify(out)


def test_required_torch_cannot_be_excluded_to_pass(tmp_path):
    frozen = tmp_path/"frozen"
    runner.freeze(frozen, "h3-semantics")
    result = runner.check(frozen, tmp_path/"rows", backends=["numpy"], start=0, stop=1)
    assert result["status"] == "FAIL" and result["unavailable"] == {"torch": "excluded by caller"}
    assert result["passed_required"] == 1 and result["required_checks"] == 2


def test_shard_aggregation_rejects_missing_and_duplicate_checks(tmp_path):
    frozen = tmp_path/"frozen"
    runner.freeze(frozen, "mechanism-controls")
    a, b = tmp_path/"a", tmp_path/"b"
    assert runner.check(frozen, a, start=0, stop=1)["status"] == "PASS"
    assert runner.aggregate(frozen, [a])["status"] == "FAIL"
    with pytest.raises(ValueError, match="duplicate"):
        runner.aggregate(frozen, [a, a])
    assert runner.check(frozen, b, start=1)["status"] == "PASS"
    assert runner.aggregate(frozen, [a, b])["status"] == "PASS"


def test_reference_h3_fails_closed_before_writing(tmp_path, capsys):
    folder = tmp_path/"not-created"
    assert main(["freeze", "--suite", "h3-programs", "--profile", "reference", "--out", str(folder)]) == 1
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "BLOCKED" and "blocked_missing_source" in result["reason"]
    assert not folder.exists()


def test_core20_schema_still_uses_legacy_cli_check(tmp_path, capsys):
    frozen = tmp_path/"core20"
    portable.freeze(frozen, tier=20)
    assert main(["check", "--frozen", str(frozen), "--out", str(tmp_path/"checked"),
                 "--backends", "numpy"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "PASS" and result["tier"] == 20 and result["checks"] == 152


def test_v2_discovery_is_array_library_free():
    script = '''import builtins
original = builtins.__import__
def blocked(name, *args, **kwargs):
    if name.split('.')[0] in ('numpy', 'torch', 'jax'):
        raise ImportError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = blocked
from benchmark_generators.portable_core import describe
assert describe(suite='operation-gate-v2')['revision'] == 2
'''
    subprocess.run([sys.executable, "-c", script], check=True)
