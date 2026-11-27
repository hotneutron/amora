"""Resource isolation, balanced coverage and lossless trace storage checks."""
from pathlib import Path
import os
import tarfile
import pytest

from amora.backends.gcom_cuda import portable_v2 as lane
from amora.backends.gcom_cuda import portable_v2_resources as resources
from amora.benchmarking.portable_v2 import gcom_campaign as campaign


def test_cpu_slots_and_acquisition_cores_never_overlap(tmp_path,monkeypatch):
    allowed=os.sched_getaffinity(0)
    monkeypatch.setattr(resources,"cpu_counters",lambda:{c:(100,100) for c in allowed})
    monkeypatch.setattr(resources.time,"sleep",lambda _:None)
    plan=resources.cpu_admission(tmp_path,gpu_count=2,point_count=24)
    reserved={c for core in plan["acquisition_cpus"] for c in core}
    cpus=[c for slot in plan["slots"] for c in slot["cpus"]]
    assert len(cpus)==len(set(cpus))
    assert not reserved & set(cpus)
    assert set(cpus)<=allowed
    assert sum(s["omp_threads"] for s in plan["slots"])<=plan["quota_cpus"]-len(reserved)


def test_busy_core_and_its_smt_sibling_are_excluded(tmp_path,monkeypatch):
    allowed=os.sched_getaffinity(0)
    cpu=min(allowed)
    siblings=resources.parse_cpu_list(Path(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list").read_text())
    snapshots=iter([{c:(100,100) for c in allowed},
                    {c:(200,100 if c==cpu else 200) for c in allowed}])
    monkeypatch.setattr(resources,"cpu_counters",lambda:next(snapshots))
    monkeypatch.setattr(resources.time,"sleep",lambda _:None)
    plan=resources.cpu_admission(tmp_path,point_count=100)
    assert not siblings & {c for slot in plan["slots"] for c in slot["cpus"]}


def trace_files(tmp_path):
    root=tmp_path/"raw";root.mkdir()
    (root/"dynamic.pb").write_bytes(bytes(range(256))*10)
    (root/"metadata").mkdir();(root/"metadata/static.json").write_text('{"opcode":"LDG"}')
    info=dict(trace_sha256="sealed",files={str(p.relative_to(root)):lane.sha(p) for p in root.rglob("*") if p.is_file()})
    return root,info


def test_only_verified_lossless_archive_replaces_raw_duplicate(tmp_path):
    root,info=trace_files(tmp_path)
    archive=tmp_path/"archive/trace.tar.gz"
    receipt=resources.archive_trace(root,info,archive)
    assert not root.exists() and receipt["archive_sha256"]==lane.sha(archive)
    restored=resources.restore_trace(archive,info,tmp_path/"restored")
    assert {r:lane.sha(restored/r) for r in info["files"]}==info["files"]


def test_changed_trace_is_preserved_when_archival_fails(tmp_path):
    root,info=trace_files(tmp_path)
    (root/"dynamic.pb").write_bytes(b"different")
    with pytest.raises(ValueError,match="changed"):
        resources.archive_trace(root,info,tmp_path/"trace.tar.gz")
    assert root.exists()


def test_archive_member_path_and_file_set_are_checked(tmp_path):
    root,info=trace_files(tmp_path)
    archive=tmp_path/"unsafe.tar.gz"
    with tarfile.open(archive,"w:gz") as out:
        out.add(root/"dynamic.pb",arcname="../escape")
    with pytest.raises(ValueError):resources.verify_archive(archive,info)


def test_case_assignments_cover_panel_once_and_balance_suite_strata():
    cases=[dict(index=i,aliases=[["op" if i<12 else "h3","development"]],
                shape=dict(M=16,N=32),program_sha256=str(i)) for i in range(24)]
    devices=[dict(uuid="GPU-b"),dict(uuid="GPU-a"),dict(uuid="GPU-c")]
    assignment=campaign.balanced_assignments(dict(cases=cases),list(range(24)),devices)
    assert sorted(i for values in assignment.values() for i in values)==list(range(24))
    for values in assignment.values():
        assert len(values)==8 and sum(i<12 for i in values)==4


def test_timing_quality_does_not_promote_short_or_noisy_rows():
    assert campaign.repeat_quality([[.02]*30]*3)["qualified"]
    assert not campaign.repeat_quality([[.001]*30]*3)["qualified"]
    assert not campaign.repeat_quality([[.02]*15+[.2]*15]*3)["qualified"]
