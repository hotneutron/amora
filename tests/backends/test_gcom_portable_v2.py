"""Evidence admission faults for the portable GCoM lane; no GPU required."""
import json
import os
from pathlib import Path
import pytest

from amora.backends.gcom_cuda import portable_v2 as lane


def device(**changes):
    value = dict(name="NVIDIA H100 80GB HBM3", compute_capability=[9, 0],
                 mig=0, processes=[], gpu_utilization=0, memory_utilization=0,
                 memory_used=0, memory_reserved=501219328)
    return dict(value, **changes)


def test_driver_reserved_memory_does_not_disqualify_idle_card():
    assert lane.idle_candidate(device())


@pytest.mark.parametrize("changes", [dict(processes=[123]), dict(gpu_utilization=1),
    dict(memory_utilization=1), dict(memory_used=65*1024*1024), dict(mig=1),
    dict(name="NVIDIA H100 94GB"), dict(compute_capability=[8, 0])])
def test_foreign_context_or_nonmatching_hardware_is_rejected(changes):
    assert not lane.idle_candidate(device(**changes))


def test_parallel_child_visibility_is_isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "parent")
    monkeypatch.setenv("LD_PRELOAD", "/foreign.so")
    monkeypatch.setenv("CUDA_INJECTION64_PATH", "/foreign.so")
    monkeypatch.setenv("FLASHGPU_TEST", "1")
    a = lane.hardware_env("GPU-a", "/cuda")
    b = lane.hardware_env("GPU-b", "/cuda", tmp_path/"tracer.so", tmp_path)
    assert a["CUDA_VISIBLE_DEVICES"] == "GPU-a"
    assert b["CUDA_VISIBLE_DEVICES"] == "GPU-b"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "parent"
    assert "LD_PRELOAD" not in a and "FLASHGPU_TEST" not in a
    assert b["ACTIVE_FROM_START"] == "1" and b["DYNAMIC_KERNEL_LIMIT_START"] == "0"


def test_lease_rechecks_idleness_and_releases_lock_on_failure(tmp_path, monkeypatch):
    d = dict(device(), uuid="GPU-test", index=0)
    monkeypatch.setattr(lane, "gpu_snapshot", lambda uuid: [dict(d, processes=[123])])
    with pytest.raises(RuntimeError, match="no longer idle"):
        with lane.GpuLease(tmp_path, d): pass
    monkeypatch.setattr(lane, "gpu_snapshot", lambda uuid: [d])
    with lane.GpuLease(tmp_path, d):
        with pytest.raises(BlockingIOError):
            with lane.GpuLease(tmp_path, d): pass
    assert json.loads((tmp_path/"GPU-test.json").read_text())["state"] == "released"


def fixture_row(tmp_path):
    manifest = dict(cases=[dict(id="sealed", case_sha256="case", program_sha256="program",
                               bindings=[dict(input=True), dict(input=False)])])
    row = dict(index=0, case_id="sealed", passed=True, node_ports_checked=1,
               samples_ms=[.02], mismatches=0, first_failed_value=-1, max_abs_error=0)
    path = tmp_path/"driver.jsonl"
    return manifest, row, path


@pytest.mark.parametrize("changes", [dict(index=1), dict(case_id="other"),
    dict(node_ports_checked=0), dict(passed="true"), dict(samples_ms=[float("nan")]),
    dict(samples_ms=[0]), dict(passed=False)])
def test_driver_mislabeled_or_invalid_rows_are_rejected(tmp_path, changes):
    manifest, row, path = fixture_row(tmp_path)
    path.write_text(json.dumps(dict(row, **changes))+"\n")
    with pytest.raises((ValueError, IndexError)):
        lane.driver_rows(path, manifest, [0])


def test_nvbit_event_spans_cannot_become_hardware_ground_truth(tmp_path):
    manifest, row, path = fixture_row(tmp_path)
    path.write_text(json.dumps(dict(row, samples_ms=[0, 120]))+"\n")
    result = lane.driver_rows(path, manifest, [0], instrumented=True)[0]
    assert result["samples_ms"] == []
    assert result["instrumented_event_samples_ms"] == [0, 120]


def activity(tmp_path, *, dropped=0, error=0, missing=False):
    records = [dict(kind="kernel", name="node_kernel", start_ns=1000+i*100,
                    end_ns=1040+i*100, device=0, context=1, stream=1) for i in range(12 if not missing else 11)]
    records.append(dict(kind="complete", error_code=error, dropped_records=dropped))
    path = tmp_path/"activity.jsonl"
    path.write_text("\n".join(map(json.dumps, records))+"\n")
    return path


def test_cupti_uses_only_target_bodies_after_exact_prefix(tmp_path):
    result = lane.activity_samples(activity(tmp_path), nodes=2, samples=2)
    assert result["samples_ms"] == [.00008, .00008]


@pytest.mark.parametrize("fault", [dict(dropped=1), dict(error=1), dict(missing=True)])
def test_cupti_loss_or_launch_mismatch_is_rejected(tmp_path, fault):
    with pytest.raises(ValueError):
        lane.activity_samples(activity(tmp_path, **fault), nodes=2, samples=2)


def test_all_kernel_cycle_blocks_are_preserved():
    text = """launching kernel name: node_kernel trace_kernel_id: 0 launch_uid: 1
gpu_sim_cycle = 100
gpu_tot_sim_cycle = 100
launching kernel name: node_kernel trace_kernel_id: 1 launch_uid: 2
gpu_sim_cycle = 220
gpu_tot_sim_cycle = 320
"""
    assert lane.parse_kernel_cycles(text) == [dict(trace_id=0, cycles=100, cumulative_cycles=100),
                                            dict(trace_id=1, cycles=220, cumulative_cycles=320)]


@pytest.mark.parametrize("text", ["gpu_sim_cycle = 0\ngpu_tot_sim_cycle = 1",
    "gpu_sim_cycle = 5\ngpu_tot_sim_cycle = 20\ngpu_sim_cycle = 5\ngpu_tot_sim_cycle = 10"])
def test_invalid_completed_cycles_are_rejected(text):
    with pytest.raises(ValueError): lane.parse_kernel_cycles(text)


def fake_replay(tmp_path):
    """A stand-in executable tests job ownership without launching GCoM."""
    binary = tmp_path/"fake-sim"
    binary.write_text("#!/usr/bin/python3\nfrom pathlib import Path\n"
        "p=Path('invocations'); p.write_text(p.read_text()+'x' if p.exists() else 'x')\n"
        "print('launching kernel name: node_kernel trace_kernel_id: 0 launch_uid: 1')\n"
        "print('gpu_sim_cycle = 100\\ngpu_tot_sim_cycle = 100')\n"
        "print('GPGPU-Sim: *** simulation thread exiting ***')\n")
    binary.chmod(0o755)
    library = tmp_path/"lib"
    library.mkdir()
    (library/"libcudart.so").write_bytes(b"library identity")
    config = tmp_path/"config"
    config.write_text("fixed parameters")
    trace = tmp_path/"trace"
    (trace/"extra_info").mkdir(parents=True)
    (trace/"dynamic_trace.pb").write_bytes(b"opaque admitted test data")
    (trace/"extra_info/enhanced_execution_info.json").write_text('{"opcode":"LDG"}')
    files = {str(p.relative_to(trace)):lane.sha(p) for p in trace.rglob("*") if p.is_file()}
    info = dict(trace_sha256="synthetic", files=files, launches=1, kernels=[dict(trace_id=0)])
    def execute():
        return lane.simulate_trace(binary, library, [config], trace, info,
            root=tmp_path/"cache", cpus=[min(os.sched_getaffinity(0))], timeout=10)
    return execute, trace, info


def test_replay_cache_reuses_evidence_and_detects_output_tampering(tmp_path):
    execute, trace, _ = fake_replay(tmp_path)
    first = execute()
    assert first["complete"] and first["max_rss_kib"] > 0
    assert execute() == first and (trace/"invocations").read_text() == "x"
    (Path(first["folder"])/"stdout.txt").write_text("changed")
    with pytest.raises(ValueError, match="cache identity"):
        execute()


def test_trace_mutation_cannot_reuse_replay_cache(tmp_path):
    execute, trace, _ = fake_replay(tmp_path)
    execute()
    (trace/"dynamic_trace.pb").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="trace changed"):
        execute()


def test_attempted_key_without_result_does_not_auto_retry(tmp_path):
    execute, trace, _ = fake_replay(tmp_path)
    result = execute()
    (Path(result["folder"])/"result.json").unlink()
    with pytest.raises(RuntimeError, match="already attempted"):
        execute()
    assert (trace/"invocations").read_text() == "x"


def test_lossless_compressed_log_preserves_replay_cache_identity(tmp_path):
    execute,trace,_=fake_replay(tmp_path)
    first=execute()
    lane.compress_replay_log(first)
    assert not (Path(first["folder"])/"stdout.txt").exists()
    assert execute()==first and (trace/"invocations").read_text()=="x"


@pytest.mark.parametrize("opcode",["BAR.SYNC.DEFER_BLOCKING","BAR.ARV","MEMBAR.SC.SYS"])
def test_ordinary_barriers_do_not_require_tma_waitblock_sidecars(tmp_path,opcode):
    execute,trace,info=fake_replay(tmp_path)
    path=trace/"extra_info/enhanced_execution_info.json"
    path.write_text(json.dumps(dict(kernels=[dict(instructions=[dict(op_code=opcode)])])))
    info["files"][str(path.relative_to(trace))]=lane.sha(path)
    assert execute()["complete"]


@pytest.mark.parametrize("opcode",["SYNCS.PHASECHK","SYNCS.ARRIVE","UTMALDG.2D","HGMMA.64x8x16","UBLKCP"])
def test_async_consumers_cannot_disable_their_required_sidecars(tmp_path,opcode):
    execute,trace,info=fake_replay(tmp_path)
    path=trace/"extra_info/enhanced_execution_info.json"
    path.write_text(json.dumps(dict(kernels=[dict(instructions=[dict(op_code=opcode)])])))
    info["files"][str(path.relative_to(trace))]=lane.sha(path)
    with pytest.raises(ValueError):execute()
    assert not (trace/"invocations").exists()


@pytest.fixture
def protobuf_trace(tmp_path):
    trace_pb = pytest.importorskip("trace_pb2")
    blocks_pb = pytest.importorskip("threadblock_pb2")
    trace = trace_pb.Trace()
    trace.binary_version=90
    d = trace.gpu_device[0]
    d.id = 0
    s = d.streams[1]
    s.id = 1
    k = s.kernels.add(id=0, name="node_kernel",function_unique_id=1)
    k.grid_dim.x=k.grid_dim.y=k.grid_dim.z=1
    k.block_dim.x=32
    k.block_dim.y=k.block_dim.z=1
    block = blocks_pb.threadblock()
    w = block.warps[0]
    w.id = 0
    w.instructions.add(pc=0, active_mask=0xffffffff,function_unique_id=1)
    root = tmp_path/"real-protobuf"
    folder=root/"threadblocks/device_0/stream_1/kernel_0"
    folder.mkdir(parents=True)
    path=folder/"d_0_s_1_k_0_0,0,0.pb"
    path.write_bytes(block.SerializeToString())
    (root/"dynamic_trace.pb").write_bytes(trace.SerializeToString())
    (root/"extra_info").mkdir()
    (root/"extra_info/enhanced_execution_info.json").write_text(json.dumps(dict(kernels=[dict(
        unique_function_id=1,architecture_version=90,is_captured_from_binary=True,
        instructions=[dict(pc_num_dec=0,op_code="EXIT",encoded_instruction=["one","two"],control_bits={})])])))
    return root, trace_pb, path, block


def test_trace_requires_all_ctas_and_real_dynamic_warp_instructions(protobuf_trace):
    root, module, path, _ = protobuf_trace
    result=lane.inspect_trace(root,module,1)
    assert result["dynamic_warp_instructions"] == 1 and result["ctas"] == 1
    path.unlink()
    with pytest.raises(ValueError,match="threadblock"):
        lane.inspect_trace(root,module,1)


@pytest.mark.parametrize("fault", ["coordinates", "warps", "empty", "corrupt", "launches"])
def test_corrupt_or_partial_protobuf_is_rejected(protobuf_trace, fault):
    root, module, path, block = protobuf_trace
    if fault == "coordinates": block.block_id.x=1
    if fault == "warps": block.warps[1].instructions.add(pc=0)
    if fault == "empty": del block.warps[0].instructions[:]
    path.write_bytes(b"invalid" if fault=="corrupt" else block.SerializeToString())
    with pytest.raises(Exception): lane.inspect_trace(root,module,2 if fault=="launches" else 1)


@pytest.mark.parametrize("fault", ["missing_cubin", "missing_encoding", "architecture", "dynamic_pc"])
def test_static_and_dynamic_metadata_must_agree(protobuf_trace,fault):
    root,module,path,block=protobuf_trace
    metadata_path=root/"extra_info/enhanced_execution_info.json"
    metadata=json.loads(metadata_path.read_text())
    function=metadata["kernels"][0]
    if fault=="missing_cubin":function["is_captured_from_binary"]=False
    if fault=="missing_encoding":function["instructions"][0]["encoded_instruction"]=[]
    if fault=="architecture":function["architecture_version"]=80
    if fault=="dynamic_pc":
        block.warps[0].instructions[0].pc=16;path.write_bytes(block.SerializeToString())
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):lane.inspect_trace(root,module,1)


def test_traced_operator_order_is_bound_to_case_nodes(protobuf_trace):
    root,module,_,_=protobuf_trace
    trace=module.Trace();trace.ParseFromString((root/"dynamic_trace.pb").read_bytes())
    trace.gpu_device[0].streams[1].kernels[0].name="void node_kernel<13>(arguments)"
    (root/"dynamic_trace.pb").write_bytes(trace.SerializeToString())
    assert lane.inspect_trace(root,module,1,expected_operator_ids=[13])["launches"]==1
    with pytest.raises(ValueError,match="operator launch order"):
        lane.inspect_trace(root,module,1,expected_operator_ids=[14])


def test_cupti_batches_follow_one_validation_graph_and_batched_prefix(tmp_path):
    records=[dict(kind="kernel",name="node_kernel",start_ns=1000+i*100,end_ns=1040+i*100,
                  device=0,context=1,stream=1) for i in range(1+(3+2)*64)]
    records.append(dict(kind="complete",error_code=0,dropped_records=0))
    path=tmp_path/"batch.jsonl";path.write_text("\n".join(map(json.dumps,records))+"\n")
    result=lane.activity_samples(path,nodes=1,samples=2,repetitions=64)
    assert result["samples_ms"]==[.00256,.00256]
