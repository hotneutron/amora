# SPDX-License-Identifier: Apache-2.0
# Adapted from Accorde: package namespace relocation only (2026-10-22).
"""Write-once staged expansion and correctness; never a timing/model verdict."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from benchmark_generators.portable_core.base import core as base
from benchmark_generators.portable_core.base.runner import runtime_source_hashes, stamp, verify_manifest, write_new
from .core import generate_cases, make_inputs, oracle, evaluate, features, splits


def source_hashes():
    return {'core20':runtime_source_hashes(), 'expansion':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob('*.py'))}}


def verify_sources(folder):
    value=json.loads((folder/'freeze_receipt.json').read_text())
    if value['source_hashes']!=source_hashes(): raise RuntimeError('source changed; use a new reviewed snapshot')
    for name,sha in value['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=sha: raise RuntimeError('frozen file changed')
    return value


def audit(cases,folds,edges,tier):
    byid={c['case_id']:c for c in cases}
    if len(byid)!=len(cases) or len({c['family'] for c in cases})!=tier: raise RuntimeError('family/identity coverage')
    rows={}
    for held,f in folds['folds'].items():
        train=set(f['train']); test=set(f['operation_only']+f['joint']+f['shape_only'])
        if not train or not f['joint'] or train & test: raise RuntimeError('split support/overlap')
        if any(held in byid[c]['semantic_tags'] for c in train): raise RuntimeError('nested leakage')
        allowed=[e for e in edges if e['parent_case_id'] in train]
        if any(held in e['parent_semantic_tags'] for e in allowed): raise RuntimeError('parent edge leakage')
        rows[held]={'train':len(train),'operation_only':len(f['operation_only']),'joint':len(f['joint']),
                    'shape_only':len(f['shape_only']),'allowed_edges':len(allowed)}
    return {'status':'PASS','tier':tier,'case_count':len(cases),'folds':rows,'performance_qualified':False}


def freeze(folder,tier):
    if folder.exists() and any(folder.iterdir()): raise RuntimeError('refuse frozen overwrite')
    files={}; profiles={}
    for profile in ('smoke','native'):
        cases=generate_cases(tier,profile)
        manifest={'schema':'cross-isa-expansion-v1','tier':tier,'profile':profile,'cases':cases,
                  'status':'SEMANTIC_SPECIFICATION_ONLY','core20_preserved':True}
        manifest['sha256']=base.digest(manifest)
        f=splits(cases); edges=base.edge_plans(cases); a=audit(cases,f,edges,tier)
        for name,data in [(f'{profile}_manifest.json',manifest),(f'{profile}_splits.json',f),
                          (f'{profile}_edges.json',edges),(f'{profile}_audit.json',a),
                          (f'{profile}_features.json',{c['case_id']:features(c) for c in cases})]:
            files[name]=write_new(folder/name,data)
        profiles[profile]={'cases':len(cases),'families':tier,'manifest_sha256':manifest['sha256'],'edges':len(edges)}
    generated='''"""Generated portable expansion entrypoints; no timing qualification."""\nfrom pathlib import Path\nfrom benchmark_generators.portable_core.expanded.backends import create_backend\nfrom benchmark_generators.portable_core.expanded.core import evaluate\nfrom benchmark_generators.portable_core.expanded.runner import verify_sources\nfrom benchmark_generators.portable_core.base.runner import verify_manifest\n\ndef run(case_id,backend_name="numpy",device="cpu",profile="smoke",raw=None):\n    folder=Path(__file__).parent\n    verify_sources(folder)\n    manifest=verify_manifest(folder/(profile+"_manifest.json"))\n    case=next(c for c in manifest["cases"] if c["case_id"]==case_id)\n    return evaluate(case,create_backend(backend_name,device),raw)\n'''
    for family in dict.fromkeys(c['family'] for c in generate_cases(tier,'smoke')):
        variant=base.VARIANTS.get(family,('base',))[0]
        generated+=f'\ndef {family}(backend_name="numpy",device="cpu",profile="smoke",raw=None,variant="{variant}",scenario="dev_small"):\n'
        generated+=f'    return run(f"{{profile}}/{family}/{{variant}}/{{scenario}}",backend_name,device,profile,raw)\n'
    files['portable_kernels.py']=write_new(folder/'portable_kernels.py',generated,text=True)
    result={'created_at':stamp(),'status':'FROZEN','tier':tier,'profiles':profiles,'files':files,
            'source_hashes':source_hashes(),'native_performance_labels_opened':0,
            'correctness_tolerances':{'matrix':{'atol':5e-4,'rtol':5e-3},'fp32':{'atol':2e-5,'rtol':2e-4}}}
    write_new(folder/'freeze_receipt.json',result)
    return result


def snapshot_inputs(folder,output,profile):
    verify_sources(folder); m=verify_manifest(folder/(profile+'_manifest.json')); records=[]
    output.mkdir(parents=True,exist_ok=True)
    for c in m['cases']:
        raw=make_inputs(c); path=output/(c['sha256']+'.npz')
        with path.open('xb') as h: np.savez_compressed(h,**raw)
        records.append({'case_id':c['case_id'],'case_sha256':c['sha256'],'file':path.name,
                        'file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'input_sha256':hashlib.sha256(b''.join(raw[k].tobytes() for k in sorted(raw))).hexdigest()})
    result={'created_at':stamp(),'manifest_sha256':m['sha256'],'records':records,
            'status':'PASS','native_performance_labels_opened':0}
    write_new(output/'index.json',result)
    return {'status':'PASS','cases':len(records),'manifest_sha256':m['sha256']}


def preflight(folder,output,profile,device,names,max_seconds):
    from .backends import create_backend
    from benchmark_generators.portable_core.base.backends import BackendUnavailable
    receipt=verify_sources(folder); m=verify_manifest(folder/(profile+'_manifest.json'))
    start=time.monotonic(); rows=[]; unavailable={}; descriptions={}; hardware=None
    if device.startswith('cuda'):
        import torch
        index=int(device.partition(':')[2] or 0)
        torch.cuda.set_device(index); torch.backends.cuda.matmul.allow_tf32=False
        props=torch.cuda.get_device_properties(index)
        if 'H100' not in props.name: raise RuntimeError('source CUDA phase requires the selected H100')
        hardware={'name':props.name,'compute_capability':[props.major,props.minor],
                  'visible_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),'torch_version':torch.__version__,
                  'cuda_runtime':torch.version.cuda}
    for name in names:
        try: backend=create_backend(name,device)
        except BackendUnavailable as e:
            unavailable[name]=str(e); continue
        descriptions[name]=backend.describe()
        for c in m['cases']:
            if time.monotonic()-start>max_seconds: raise RuntimeError('bounded preflight exceeded')
            raw=make_inputs(c); reference=oracle(c,raw); result=evaluate(c,backend,raw)
            tol=receipt['correctness_tolerances']['matrix' if any(n['op']=='matmul' for n in c['nodes']) else 'fp32']
            ok=True; err=0.0; shapes=[]
            for got,expected in zip(result,reference):
                actual=backend.numpy(got); shapes.append(list(actual.shape))
                ok &= bool(actual.shape==expected.shape and np.all(np.isfinite(actual)) and np.allclose(actual,expected,**tol))
                if actual.size: err=max(err,float(np.max(np.abs(actual.astype(np.float64)-expected.astype(np.float64)))))
            ok &= len(result)==len(reference)
            rows.append({'case_id':c['case_id'],'family':c['family'],'backend':name,'passed':bool(ok),
                         'max_abs_error':err,'output_shapes':shapes})
    result={'created_at':stamp(),'status':'PASS' if rows and all(r['passed'] for r in rows) else 'FAIL',
            'tier':m['tier'],'profile':profile,'device':device,'cases':len(m['cases']),
            'checks':len(rows),'passed':sum(r['passed'] for r in rows),'failed':sum(not r['passed'] for r in rows),
            'backend_descriptions':descriptions,'hardware':hardware,'unavailable':unavailable,'manifest_sha256':m['sha256'],
            'elapsed_process_seconds':time.monotonic()-start,'native_performance_labels_opened':0,
            'scope':'numerical correctness only; no timing, atomic/carry instruction or prediction qualification'}
    write_new(output/'checks.json',rows); write_new(output/'result.json',result)
    report=f"# Core{m['tier']} {device} correctness\n\nStatus: **{result['status']}**.\n\nChecks: {result['passed']}/{len(rows)}, failures {result['failed']}. Cases: {result['cases']}.\n\nNo performance labels or native atomic/carry/predictor qualification.\n\n## Revision History\n\n|Revision|Timestamp|Change|\n|---|---|---|\n|r1|{result['created_at']}|Recorded staged expansion numerical checks and backend realization limits.|\n"
    write_new(output/'REPORT.md',report,text=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('freeze'); a.add_argument('--tier',type=int,choices=(24,28),required=True);a.add_argument('--out',type=Path,required=True)
    b=sub.add_parser('snapshot-inputs');b.add_argument('--frozen',type=Path,required=True);b.add_argument('--out',type=Path,required=True);b.add_argument('--profile',choices=('smoke','native'),default='native')
    c=sub.add_parser('preflight');c.add_argument('--frozen',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    c.add_argument('--profile',choices=('smoke','native'),default='smoke');c.add_argument('--device',default='cpu');c.add_argument('--backends',default='numpy,torch,jax');c.add_argument('--max-seconds',type=float,default=120)
    args=p.parse_args()
    if args.cmd=='freeze': result=freeze(args.out,args.tier)
    elif args.cmd=='snapshot-inputs': result=snapshot_inputs(args.frozen,args.out,args.profile)
    else: result=preflight(args.frozen,args.out,args.profile,args.device,args.backends.split(','),args.max_seconds)
    print(json.dumps(result,indent=2,sort_keys=True));return int(result.get('status')=='FAIL')


if __name__=='__main__': raise SystemExit(main())
