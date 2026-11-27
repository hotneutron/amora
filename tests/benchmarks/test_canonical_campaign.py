"""Canonical timing admission checks without GPUs or model execution."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from amora.benchmarking.canonical_ppp.campaign import activity, matched_native, work, flash_completion
from amora.backends.gcom_cuda import portable_v2 as lane


def records(tmp_path, **fault):
    rows=[dict(kind="kernel",name="canonical",start_ns=1000,end_ns=21000,
               device=0,context=1,stream=1,grid=[1,1,1],block=[256,1,1])]
    rows.append(dict(kind="complete",error_code=0,dropped_records=0,**fault))
    path=tmp_path/"activity.jsonl"
    path.write_text("\n".join(map(json.dumps,rows)))
    return path


def test_native_span_and_body_keep_component_gaps(tmp_path):
    path=records(tmp_path)
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows.insert(1,dict(rows[0],name="component2",start_ns=31000,end_ns=51000))
    path.write_text("\n".join(map(json.dumps,rows)))
    parsed=activity(path)
    assert parsed["body_ms"] == .04 and parsed["span_ms"] == .05
    assert matched_native([parsed]*3)["timing_qualified"]


@pytest.mark.parametrize("field,value",[("error_code",1),("dropped_records",1)])
def test_lost_or_incomplete_native_capture_cannot_be_scored(tmp_path,field,value):
    path=records(tmp_path)
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[-1][field]=value
    path.write_text("\n".join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match="incomplete or lost"):activity(path)


def test_repeat_launch_change_is_rejected(tmp_path):
    a=activity(records(tmp_path))
    b=dict(a,kernels=[dict(a["kernels"][0],grid=[2,1,1])])
    with pytest.raises(ValueError,match="changed kernel identities"):matched_native([a,b,a])


def test_short_or_variable_native_results_are_retained_but_unqualified(tmp_path):
    a=activity(records(tmp_path))
    assert not matched_native([dict(a,span_ms=.005)]*3)["timing_qualified"]
    assert not matched_native([a,dict(a,span_ms=.1),a])["timing_qualified"]


def test_work_estimate_does_not_include_fixed_embedding_vocabulary():
    assert work(dict(kernel_id="embedding",shape=dict(B=1,S=512,D=64,V=50000,U=128))) == 32768


def test_flash_runtime_exit_requires_full_launch_accounting_and_success():
    text="gpu_sim_cycle = 50\ngpu_tot_sim_cycle = 50\nGPGPU-Sim: *** exit detected ***\n"
    assert flash_completion(text,0,1)[0]
    assert not flash_completion(text,0,2)[0]
    assert not flash_completion(text,1,1)[0]
    assert not flash_completion(text.replace("GPGPU-Sim: *** exit detected ***", ""),0,1)[0]


def test_canonical_trace_requires_exact_native_kernel_order(tmp_path):
    (tmp_path/"dynamic_trace.pb").write_bytes(b"fixture")
    (tmp_path/"extra_info").mkdir()
    (tmp_path/"extra_info/enhanced_execution_info.json").write_text("{}")
    dim=SimpleNamespace(x=1,y=1,z=1)
    kernels=[SimpleNamespace(id=i,name=name,grid_dim=dim,block_dim=dim)
             for i,name in enumerate(("first","second"))]
    class Trace:
        def ParseFromString(self, data):
            self.binary_version=90
            self.gpu_device={0:SimpleNamespace(streams={0:SimpleNamespace(kernels=kernels)})}
    with pytest.raises(ValueError,match="identity or order"):
        lane.inspect_trace(tmp_path,SimpleNamespace(Trace=Trace),2,
                           expected_kernel_names=["second","first"])
    for kernel in kernels:
        kernel.name += "___0"
        kernel.function_unique_id=0
    # Matching native names pass the identity guard; this deliberately minimal
    # fixture then fails the independent cubin metadata check.
    with pytest.raises(ValueError,match="missing cubin"):
        lane.inspect_trace(tmp_path,SimpleNamespace(Trace=Trace),2,
                           expected_kernel_names=["first","second"])
