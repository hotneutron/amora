"""Portable CUDA corpus and result protocol checks without accelerator inference."""
import hashlib
import json
from pathlib import Path
import pytest

np=pytest.importorskip("numpy")
from amora.benchmarking.portable_v2.corpus import prepare_corpus,verify_corpus
from amora.backends.nvidia import portable_core_v2 as cuda

def test_portable_campaign_cli_is_isolated_from_cpu_release_checker():
    from amora.cli import build_parser
    args=build_parser().parse_args(["flashgpu_sim","portable-v2","report","--root","example"])
    assert args.backend=="flashgpu_sim" and args.command=="portable-v2"
    assert args.action=="report" and args.gpu=="7"

@pytest.fixture
def tiny(tmp_path):
    root=tmp_path/"corpus"
    manifest=prepare_corpus(root,selections=(("h3-semantics","smoke"),))
    return root,manifest

def test_source_cases_are_sealed_without_changing_identity(tiny):
    root,manifest=tiny
    checked=verify_corpus(root)
    assert len(checked["cases"])==7
    assert all(c["case_sha256"] and c["program_sha256"] for c in checked["cases"])
    assert all(c["aliases"]==[["h3-semantics","smoke"]] for c in checked["cases"])
    assert set(checked["banks"])=={p.name for p in (root/"banks").glob("*.bin")}

def test_case_payload_tampering_is_rejected(tiny):
    root,_=tiny
    with (root/"cases.txt").open("a") as stream: stream.write("changed\n")
    with pytest.raises(ValueError,match="changed corpus file"):
        verify_corpus(root)

def test_bank_tampering_is_rejected(tiny):
    root,_=tiny
    path=next((root/"banks").glob("*.bin"))
    path.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed input/oracle bank"):
        verify_corpus(root)

def test_physical_stride_and_offset_banks_are_preserved(tmp_path):
    m=prepare_corpus(tmp_path/"layout",selections=(("h3-semantics","development"),))
    strided=next(c for c in m["cases"] if "embedding" in c["id"] and "strided" in c["id"])
    x=next(b for b in strided["bindings"] if b["producer"]=="x")
    assert x["strides"]==[109*4,4]
    bank=tmp_path/"layout/banks"/(x["physical_sha256"]+".bin")
    assert bank.stat().st_size==64*109*4
    backing=np.fromfile(bank,dtype=np.float32).reshape(64,109)
    assert np.array_equal(backing[:,96:],np.zeros((64,13),np.float32))
    offset=next(c for c in m["cases"] if "embedding" in c["id"] and "offset" in c["id"])
    x=next(b for b in offset["bindings"] if b["producer"]=="x")
    assert x["offset"]==3
    data=np.fromfile(tmp_path/"layout/banks"/(x["physical_sha256"]+".bin"),dtype=np.float32)
    assert data.size==64*96+3 and np.array_equal(data[:3],np.zeros(3,np.float32))

@pytest.mark.parametrize("fault,match",[
    ("outside","outside frozen selection"),("duplicate","duplicate driver result"),
    ("failed_timing","failed correctness"),("ports","missing node-port"),
    ("binary","executed binary changed"),
])
def test_result_protocol_rejects_incomplete_or_mislabeled_evidence(tiny,tmp_path,monkeypatch,fault,match):
    root,m=tiny;case=m["cases"][0];out=tmp_path/"results"
    build=dict(binary="/test/binary",binary_sha256="a"*64,
               realization_sha256="b"*64,contract={"operator_mask":(1<<31)-1})
    def run(**kwargs):
        out.mkdir(exist_ok=True)
        result=Path(next(x.split("=",1)[1] for x in kwargs["program_arguments"] if x.startswith("--result=")))
        row=dict(index=0,case_id=case["id"],passed=True,
                 node_ports_checked=sum(not b["input"] for b in case["bindings"]),samples_ms=[0.1])
        if fault=="outside":row["index"]=1
        if fault=="failed_timing":row["passed"]=False
        if fault=="ports":row["node_ports_checked"]=0
        rows=[row,row] if fault=="duplicate" else [row]
        result.write_text("\n".join(json.dumps(x) for x in rows)+"\n")
        return dict(exit_code=0,metadata={"binary_sha256":"c"*64 if fault=="binary" else "a"*64},output="mock.txt")
    monkeypatch.setattr(cuda,"run_test",run)
    with pytest.raises(ValueError,match=match):
        cuda.run_group([0],manifest=m,builds={"all":build},corpus_root=root,evidence_root=out,
                       sim_root="/sim",cuda_root="/cuda",mode="ptx",label="protocol")

@pytest.mark.parametrize("reverse_creation", [False, True])
def test_report_keeps_completed_evidence_when_newer_singleton_is_incomplete(tmp_path,monkeypatch,reverse_creation):
    from amora.benchmarking.portable_v2 import campaign
    case=dict(index=0,id="sealed-case",family="add",shape={"M":16,"N":32},aliases=[],
              case_sha256="a"*64,program_sha256="b"*64,
              program=dict(nodes=[dict(op="add")],state_edges=[],unroll=1))
    monkeypatch.setattr(campaign,"verify_corpus",lambda root:dict(cases=[case],blocked={}))
    native=tmp_path/"native";native.mkdir()
    simulation=tmp_path/"simulation";simulation.mkdir()
    for repeat in range(3):
        (native/f"native-full-r{repeat}.validation.json").write_text(json.dumps(dict(
            rows=[dict(index=0,passed=True,samples_ms=[.1]*30)])))
    docs=[("ptx-full-batch",dict(receipt=dict(started_utc="2026-11-24T01:00:00Z"),
          rows=[dict(index=0,mode="ptx",passed=True,samples_ms=[.2])])),
          ("ptx-full-isolated",dict(receipt=dict(started_utc="2026-11-24T02:00:00Z"),
          rows=[dict(index=0,mode="ptx",passed=False,status="not_completed",samples_ms=[])]))]
    for label,doc in reversed(docs) if reverse_creation else docs:
        (simulation/(label+".validation.json")).write_text(json.dumps(doc))
    (simulation/"functional.validation.json").write_text(json.dumps(dict(rows=[
        dict(index=0,mode="ptx-functional",passed=True,samples_ms=[0])
    ])))
    report=campaign.report(tmp_path);row=report["rows"][0]
    assert row["ptx"]["passed"] and row["ptx"]["difference_percent"]==100
    assert row["ptx"]["source_validation"]=="simulation/ptx-full-batch.validation.json"
    assert len(row["ptx"]["observations"])==2
    assert len(row["native_observations"])==3
    assert row["ptx-functional"]["samples_ms"]==[]
    assert "difference_percent" not in row["ptx-functional"]
    assert report["summary"]["overall"]["eligible_comparisons"]==1
    assert report["summary"]["by_operator"]["add"]["median_ratio"]==2

