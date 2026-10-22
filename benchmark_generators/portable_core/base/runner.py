# SPDX-License-Identifier: Apache-2.0
# Adapted from Accorde: package namespace relocation only (2026-10-22).
"""Finite manifest generation and portable CPU preflight; no autoresearch."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

from .core import SCHEMA, FAMILIES, VARIANTS, digest, edge_plans, evaluate, features, generate_cases, make_inputs, oracle, split_manifest


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_new(path: Path, value, text: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value.encode() if text else json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
    with path.open("xb") as f:
        f.write(data)
    return hashlib.sha256(data).hexdigest()


def runtime_source_hashes() -> dict:
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}


def verify_runtime_sources(frozen: Path) -> dict:
    receipt=json.loads((frozen/'freeze_receipt.json').read_text())
    if receipt['runtime_source_sha256']!=runtime_source_hashes():
        raise RuntimeError('runtime source changed; create a new reviewed freeze, do not reuse this snapshot')
    return receipt


def verify_manifest(path: Path) -> dict:
    value = json.loads(path.read_text())
    expected = value.pop("sha256")
    if digest(value) != expected:
        raise RuntimeError(f"manifest digest mismatch: {path}")
    for case in value["cases"]:
        expected_case = case["sha256"]
        if digest({k:v for k,v in case.items() if k != "sha256"}) != expected_case:
            raise RuntimeError(f"case digest mismatch: {case['case_id']}")
    value["sha256"] = expected
    return value


def audit_splits(cases: list[dict], splits: dict, edges: list[dict]) -> dict:
    by_id = {c['case_id']:c for c in cases}
    if set(c['family'] for c in cases) != set(FAMILIES):
        raise RuntimeError("core20 family coverage incomplete")
    if len(by_id) != len(cases):
        raise RuntimeError("duplicate case identity")
    summaries = {}
    for held,fold in splits['folds'].items():
        train = set(fold['train'])
        tests = set(fold['operation_only'] + fold['shape_only'] + fold['joint'])
        if train & tests or not train or not fold['joint']:
            raise RuntimeError(f"invalid split support: {held}")
        if any(held in by_id[c]['semantic_tags'] for c in train):
            raise RuntimeError(f"nested semantic leakage: {held}")
        kept_edges = [e for e in edges if e['parent_case_id'] in train]
        if any(held in e['parent_semantic_tags'] for e in kept_edges):
            raise RuntimeError(f"parent-edge leakage: {held}")
        summaries[held] = {"train_cases":len(train), "operation_only_cases":len(fold['operation_only']),
                           "shape_only_cases":len(fold['shape_only']), "joint_cases":len(fold['joint']),
                           "allowed_parent_edges":len(kept_edges)}
    return {"status":"PASS", "family_count":len(FAMILIES), "case_count":len(cases),
            "folds":summaries, "native_performance_qualified":False}


def freeze(out: Path) -> dict:
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("refuse to overwrite a frozen campaign directory")
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    summaries = {}
    for profile in ('smoke','native'):
        cases = generate_cases(profile)
        manifest = {"schema":SCHEMA,"profile":profile,"cases":cases,
                    "family_count":20,"case_count":len(cases),
                    "status":"FROZEN_SEMANTIC_MANIFEST_NOT_HARDWARE_QUALIFIED"}
        manifest['sha256'] = digest(manifest)
        splits = split_manifest(cases)
        edges = edge_plans(cases)
        audit = audit_splits(cases,splits,edges)
        for name,data in [(f'{profile}_manifest.json',manifest),(f'{profile}_splits.json',splits),
                          (f'{profile}_edges.json',edges),(f'{profile}_split_audit.json',audit),
                          (f'{profile}_features.json',{'schema':SCHEMA,'features':{c['case_id']:features(c) for c in cases}})]:
            files[name] = write_new(out/name,data)
        summaries[profile] = {"cases":len(cases),"families":len(FAMILIES),"variants":dict(Counter(c['family'] for c in cases)),
                              "manifest_sha256":manifest['sha256'],"planned_edges":len(edges)}
    generated = '''"""Generated portable semantic entrypoint; no ISA-specific schedule."""\nfrom pathlib import Path\nfrom benchmark_generators.portable_core.base.backends import create_backend\nfrom benchmark_generators.portable_core.base.core import evaluate\nfrom benchmark_generators.portable_core.base.runner import verify_manifest\n\ndef run(case_id, backend_name="numpy", device="cpu", inputs=None, profile="smoke"):\n    manifest = verify_manifest(Path(__file__).with_name(profile + "_manifest.json"))\n    cases = {c["case_id"]: c for c in manifest["cases"]}\n    return evaluate(cases[case_id], create_backend(backend_name, device), inputs)\n'''
    generated=generated.replace('import verify_manifest', 'import verify_manifest, verify_runtime_sources')
    generated=generated.replace('    manifest = verify_manifest',
                                '    verify_runtime_sources(Path(__file__).parent)\n    manifest = verify_manifest')
    for family in FAMILIES:
        default=VARIANTS.get(family,('base',))[0]
        generated+=f'\ndef {family}(backend_name="numpy", device="cpu", inputs=None, profile="smoke", variant="{default}", scenario="dev_small"):\n'
        generated+=f'    return run(f"{{profile}}/{family}/{{variant}}/{{scenario}}", backend_name, device, inputs, profile)\n'
    files['portable_kernels.py'] = write_new(out/'portable_kernels.py',generated,text=True)
    result = {"schema":SCHEMA,"created_at":stamp(),"status":"FROZEN", "profiles":summaries,
              "files":files,"runtime_source_sha256":runtime_source_hashes(),
              "correctness_tolerances":{"matrix":{"atol":5e-4,"rtol":5e-3},
                                         "fp32":{"atol":2e-5,"rtol":2e-4}},
              "native_labels_opened":0,"autonomous_research":False}
    write_new(out/'freeze_receipt.json',result)
    return result


def inventory(out: Path) -> dict:
    modules = {name:importlib.util.find_spec(name) is not None for name in ('numpy','torch','jax','ml_dtypes','torch_npu')}
    try:
        nvidia = subprocess.run(['nvidia-smi','--query-gpu=name,uuid','--format=csv,noheader'],text=True,capture_output=True,timeout=10)
        gpu = {"returncode":nvidia.returncode,"stdout":nvidia.stdout.strip(),"stderr":nvidia.stderr.strip()}
    except (OSError,subprocess.TimeoutExpired) as e:
        gpu={"error":str(e)}
    result={"created_at":stamp(),"python":sys.version,"executable":sys.executable,"platform":platform.platform(),
            "modules":modules,"nvidia_inventory":gpu,"native_timing_started":False,
            "target_endpoints_configured":False,"status":"INVENTORY_ONLY_NOT_HARDWARE_ADMISSION"}
    write_new(out,result)
    return result


def preflight(frozen: Path, out: Path, names: list[str]) -> dict:
    from .backends import BackendUnavailable, create_backend
    start=time.monotonic()
    freeze_receipt=verify_runtime_sources(frozen)
    manifest=verify_manifest(frozen/'smoke_manifest.json')
    cases=manifest['cases']
    rows=[]; backends={}; unsupported={}
    for name in names:
        try:
            backend=create_backend(name,'cpu')
        except BackendUnavailable as e:
            unsupported[name]=str(e)
            continue
        backends[name]=backend.describe()
        for case in cases:
            raw=make_inputs(case)
            reference=oracle(case,raw)
            got=backend.numpy(evaluate(case,backend,raw))
            has_matrix=any(n['op']=='matmul' for n in case['nodes'])
            # Fixed before measurement: BF16 composite boundaries are less tight
            # than pure FP32 arithmetic. This is correctness, not a latency gate.
            tolerances=freeze_receipt['correctness_tolerances']['matrix' if has_matrix else 'fp32']
            atol,rtol=tolerances['atol'],tolerances['rtol']
            finite=bool(np.all(np.isfinite(got)))
            ok=finite and got.shape==reference.shape and bool(np.allclose(got,reference,atol=atol,rtol=rtol))
            error=float(np.max(np.abs(got.astype(np.float64)-reference.astype(np.float64))))
            input_hash=hashlib.sha256(b''.join(raw[k].tobytes() for k in sorted(raw))).hexdigest()
            rows.append({"case_id":case['case_id'],"family":case['family'],"backend":name,"passed":ok,
                         "output_shape":list(got.shape),"max_abs_error":error,"atol":atol,"rtol":rtol,
                         "input_sha256":input_hash,"output_sha256":hashlib.sha256(got.tobytes()).hexdigest()})
    result={"schema":SCHEMA,"created_at":stamp(),"manifest_sha256":manifest['sha256'],
            "status":"PASS" if rows and all(r['passed'] for r in rows) else "FAIL",
            "case_count":len(cases),"family_count":len(FAMILIES),"backend_checks":len(rows),
            "passed_checks":sum(r['passed'] for r in rows),"failed_checks":sum(not r['passed'] for r in rows),
            "backends":backends,"unsupported":unsupported,"elapsed_process_seconds":time.monotonic()-start,
            "native_labels_opened":0,"hardware_qualification":False,
            "scope":"CPU semantic smoke preflight; not native full-shape correctness, timing, or OOD prediction"}
    files={}
    files['checks.json']=write_new(out/'checks.json',rows)
    files['result.json']=write_new(out/'result.json',result)
    report = f'''# Core20 portable CPU semantic preflight\n\nStatus: **{result['status']}**.\n\n- Cases: {len(cases)} across20 families.\n- Backend checks: {len(rows)}; passed{result['passed_checks']}, failed{result['failed_checks']}.\n- Actual tested CPU adapters: {', '.join(backends)}.\n- Unavailable adapters: {json.dumps(unsupported,sort_keys=True)}.\n- Native performance labels opened:0.\n- Scope: semantic correctness on smoke shapes only; no GPU/TPU/NPU timing or OOD gate promotion.\n- Production/native shapes still need actual target correctness and qualified measurement.\n\n## Revision History\n\n|Revision|Timestamp|Change|\n|---|---|---|\n|r1|{result['created_at']}|Recorded actual bounded CPU semantic preflight; no native timing, source performance fitting, or autonomous research.|\n'''
    files['REPORT.md']=write_new(out/'REPORT.md',report,text=True)
    write_new(out/'receipt.json',{"created_at":stamp(),"files":files,"manifest_sha256":manifest['sha256'],
                                  "status":result['status'],"native_labels_opened":0})
    return result


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    f=commands.add_parser('freeze'); f.add_argument('--out',type=Path,required=True)
    i=commands.add_parser('inventory'); i.add_argument('--out',type=Path,required=True)
    t=commands.add_parser('preflight'); t.add_argument('--frozen',type=Path,required=True); t.add_argument('--out',type=Path,required=True)
    t.add_argument('--backends',default='numpy,torch,jax')
    a=p.parse_args()
    if a.command=='freeze': result=freeze(a.out)
    elif a.command=='inventory': result=inventory(a.out)
    else: result=preflight(a.frozen,a.out,a.backends.split(','))
    print(json.dumps(result,indent=2,sort_keys=True))
    return 1 if result.get('status')=='FAIL' else 0


if __name__=='__main__':
    raise SystemExit(main())
