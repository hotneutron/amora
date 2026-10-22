# SPDX-License-Identifier: Apache-2.0
# Adapted from Accorde: package namespace relocation only (2026-10-22).
"""Portable expansion contracts; imported core20 is never mutated."""
from __future__ import annotations
import hashlib
import math
from typing import Any
import numpy as np
from benchmark_generators.portable_core.base import core as base

ADDITIONS_24=('N21','N22','N23','N24')
ADDITIONS_28=ADDITIONS_24+('N25','N26','N27','N28')


class Builder(base.Builder):
    def op(self,op,*args,**attrs):
        if op not in ('scan','segment_sum','scatter_add','stencil'):
            return super().op(op,*args,**attrs)
        specs=[self.specs[a] for a in args]
        if specs[0]['dtype']!='fp32': raise ValueError('probe arithmetic is FP32')
        if op=='scan': shape=specs[0]['shape']
        elif op=='segment_sum': shape=[len(attrs['offsets'])-1,specs[0]['shape'][1]]
        elif op=='scatter_add': shape=[attrs['rows'],specs[0]['shape'][1]]
        else: shape=specs[0]['shape']
        name=f'n{len(self.nodes)}'
        node={'id':name,'op':op,'inputs':list(args),'attrs':attrs,'shape':list(shape),
              'dtype':'fp32','materialized':True}
        self.nodes.append(node); self.specs[name]=node
        return name

    def softmax(self,x,rows,cols):
        centered=self.op('sub',x,self.broad(self.op('max',x,axis=1),(rows,cols),axis=0))
        exp=self.op('exp',centered)
        return self.op('div',exp,self.broad(self.op('sum',exp,axis=1),(rows,cols),axis=0))


def lengths(rows):
    a,b,c=max(1,rows//8),max(1,rows//4),max(1,rows//2)
    return [0,a,b,c,rows-a-b-c]


def build_addition(family,scenario,rows,cols,role,profile):
    cid=f'{profile}/{family}/base/{scenario}'
    b=Builder(cid); tags={family}; outputs=[]
    r,c=rows,cols
    d=16 if profile=='smoke' else 64
    if family=='N21':
        q=b.inp('Q',(r,d),'bf16')
        kt=b.inp('KT',(d,c),'bf16')
        v=b.inp('V',(c,d),'bf16')
        scores=b.op('div',b.op('matmul',q,kt),b.scalar('scale',math.sqrt(d)))
        p=b.softmax(scores,r,c)
        outputs=[b.op('matmul',b.op('cast',p,dtype='bf16'),v)]
        tags.add('C01')
    elif family=='N22':
        a=b.inp('A',(r,c),'bf16')
        wgate=b.inp('Wgate',(c,2*c),'bf16','weight',scale=0.5/math.sqrt(c))
        wup=b.inp('Wup',(c,2*c),'bf16','weight',scale=0.5/math.sqrt(c))
        down=b.inp('Wdown',(2*c,c),'bf16','weight',scale=0.5/math.sqrt(2*c))
        gate=b.op('silu',b.op('matmul',a,wgate))
        up=b.op('matmul',a,wup)
        outputs=[b.op('matmul',b.op('cast',b.op('mul',gate,up),dtype='bf16'),down)]
    elif family=='N23':
        q=b.inp('Q',(r,d),'bf16')
        key=b.inp('Ktable',(2*c+3,d))
        value=b.inp('Vtable',(2*c+3,d))
        index=b.inp('index',(c,),'int32','index',upper=2*c+3)
        kg=b.op('gather',key,index); vg=b.op('gather',value,index)
        kt=b.op('cast',b.op('transpose',kg),dtype='bf16')
        scores=b.op('div',b.op('matmul',q,kt),b.scalar('scale',math.sqrt(d)))
        p=b.softmax(scores,r,c)
        outputs=[b.op('matmul',b.op('cast',p,dtype='bf16'),b.op('cast',vg,dtype='bf16'))]
        tags.update(('C01','N21'))
    elif family=='N24':
        group_lengths=lengths(r)
        for g,m in enumerate(group_lengths):
            k=c if g%2 else max(1,c//2)
            n=c if g%3 else max(1,c//2)
            if m==0:
                outputs.append(b.inp(f'empty{g}',(0,n)))
            else:
                a=b.inp(f'A{g}',(m,k),'bf16')
                w=b.inp(f'W{g}',(k,n),'bf16','weight',scale=0.5/math.sqrt(k))
                outputs.append(b.op('matmul',a,w))
    elif family=='N25':
        outputs=[b.op('scan',b.inp('X',(r,c)),axis=1)]
    elif family=='N26':
        offsets=np.cumsum([0]+lengths(r)).astype(np.int32).tolist()
        x=b.inp('X',(r,c))
        o=b.inp('offsets',(len(offsets),),'int32','offsets',value=offsets)
        outputs=[b.op('segment_sum',x,o,offsets=offsets)]
    elif family=='N27':
        size=max(2,r//4)
        x=b.inp('X',(r,c),domain='positive')
        i=b.inp('index',(r,),'int32','index',upper=size)
        outputs=[b.op('scatter_add',x,i,rows=size,initialization_included=True)]
    elif family=='N28':
        x=b.inp('X',(r,c,4))
        w=b.inp('W',(4,3,3),domain='weight',scale=0.2)
        outputs=[b.op('stencil',x,w,padding='zero',radius=1)]
    else: raise ValueError(family)
    leaves={n['op'] for n in b.nodes}
    if 'silu' in leaves: leaves.update(('exp','div','add','mul'))
    case={'schema':'cross-isa-expansion-v1','case_id':cid,'family':family,'variant':'base',
          'scenario':scenario,'role':role,'profile':profile,'rows':r,'width':c,
          'inputs':b.inputs,'nodes':b.nodes,'outputs':outputs,'output':outputs[0],
          'semantic_tags':sorted(tags),'semantic_leaves':sorted(leaves),
          'realization':'portable-array-program-with-explicit-probe-metadata'}
    if family=='N24': case['group_lengths']=lengths(r)
    case['sha256']=base.digest(case)
    return case


def generate_cases(tier=24,profile='smoke'):
    if tier not in (24,28): raise ValueError('tier is24 or28')
    cases=base.generate_cases(profile)
    additions=ADDITIONS_24 if tier==24 else ADDITIONS_28
    return cases+[build_addition(f,s,r,c,role,profile) for f in additions
                  for s,r,c,role in base.PROFILES[profile]]


def make_inputs(case):
    if case['family'] in base.FAMILIES: return base.make_inputs(case)
    seed=int.from_bytes(hashlib.sha256(case['case_id'].encode()).digest()[:8],'little')
    rng=np.random.default_rng(seed); result={}
    for name,s in case['inputs'].items():
        shape=tuple(s['shape']); domain=s['domain']
        if domain=='offsets': a=np.asarray(s['value'],np.int32)
        elif domain=='constant': a=np.asarray(s['value'],np.float32)
        elif domain=='index': a=rng.integers(0,s['upper'],size=shape,dtype=np.int32)
        elif domain=='positive': a=rng.uniform(0.25,1.25,size=shape).astype(np.float32)
        elif domain=='weight': a=rng.normal(0,s['scale'],size=shape).astype(np.float32)
        else: a=rng.uniform(-1,1,size=shape).astype(np.float32)
        result[name]=base._bf16(a) if s['dtype']=='bf16' else a
    return result


def prepare(case,backend,raw=None):
    data=make_inputs(case) if raw is None else raw
    for n in case['nodes']:
        if n['op']=='segment_sum':
            o=data[n['inputs'][1]]
            if not np.array_equal(o,np.asarray(n['attrs']['offsets'],np.int32)):
                raise ValueError('segment offsets disagree with pre-execution metadata')
            if o[0]!=0 or o[-1]!=data[n['inputs'][0]].shape[0] or np.any(np.diff(o)<0):
                raise ValueError('invalid segment offsets')
        if n['op']=='scatter_add':
            i=data[n['inputs'][1]]
            if np.any(i<0) or np.any(i>=n['attrs']['rows']): raise ValueError('scatter index out of range')
    prepared=base.prepare_inputs(case,backend,data)
    prepared.expansion_validation_sha256=case['sha256']
    return prepared


def execute_prepared(case,backend,prepared):
    if prepared.case_sha256!=case['sha256'] or getattr(prepared,'expansion_validation_sha256',None)!=case['sha256']:
        raise ValueError('wrong or unvalidated expansion case')
    values=dict(prepared.values)
    for n in case['nodes']:
        a=[values[x] for x in n['inputs']]
        attrs=dict(n['attrs']); attrs['indices_validated']=True
        if n['op']=='segment_sum': attrs['offsets_validated']=True
        out=a[0].reshape(tuple(n['shape'])) if n['op']=='reshape' else backend.execute(n['op'],a,attrs)
        if tuple(out.shape)!=tuple(n['shape']): raise ValueError('output shape mismatch')
        values[n['id']]=out
    return tuple(values[x] for x in case.get('outputs',[case['output']]))


def evaluate(case,backend,raw=None):
    return execute_prepared(case,backend,prepare(case,backend,raw))


def oracle(case,raw):
    if case['family'] in base.FAMILIES:
        if case['family']=='C04' and case['profile']=='native':
            x=raw; output=np.empty((x['index'].shape[0],case['width']),np.float32)
            for g in range(len(output)):
                z=x['table'][x['index'][g]]
                prod=(z.astype(np.float64)*x['a'][g].astype(np.float64)[:,None]).astype(np.float32)
                num=np.sum(prod.astype(np.float64),axis=0).astype(np.float32)
                den=np.sum(x['a'][g].astype(np.float64)).astype(np.float32)
                output[g]=(num.astype(np.float64)/den.astype(np.float64)).astype(np.float32)
            return (output,)
        return (base.oracle(case,raw),)
    f=lambda x:np.asarray(x,np.float32)
    d=lambda x:np.asarray(x,np.float64)
    mm=lambda a,b:f(d(a)@d(b))
    silu=lambda x:f(d(x)/(1+np.exp(-d(x))))
    def attention(q,kt,v):
        scores=f(d(mm(q,kt))/np.float64(raw['scale']))
        center=f(d(scores)-d(np.max(scores,axis=1))[:,None])
        exp=f(np.exp(d(center))); denom=f(np.sum(d(exp),axis=1))
        prob=f(d(exp)/d(denom)[:,None])
        return mm(base._bf16(prob),v)
    family=case['family']; x=raw
    if family=='N21': out=attention(x['Q'],x['KT'],x['V'])
    elif family=='N22': out=mm(base._bf16(f(d(silu(mm(x['A'],x['Wgate'])))*d(mm(x['A'],x['Wup'])))),x['Wdown'])
    elif family=='N23': out=attention(x['Q'],base._bf16(x['Ktable'][x['index']].T),base._bf16(x['Vtable'][x['index']]))
    elif family=='N24':
        result=[]
        for g,m in enumerate(case['group_lengths']):
            result.append(x[f'empty{g}'] if m==0 else mm(x[f'A{g}'],x[f'W{g}']))
        return tuple(result)
    elif family=='N25': out=f(np.cumsum(d(x['X']),axis=1))
    elif family=='N26':
        o=x['offsets']; out=np.zeros((len(o)-1,x['X'].shape[1]),np.float32)
        for g in range(len(out)): out[g]=f(np.sum(d(x['X'][o[g]:o[g+1]]),axis=0))
    elif family=='N27':
        out64=np.zeros((case['nodes'][-1]['attrs']['rows'],x['X'].shape[1]),np.float64)
        for row,index in enumerate(x['index']): out64[index]+=d(x['X'][row])
        out=f(out64)
    elif family=='N28':
        h,w,c=x['X'].shape; padded=np.pad(d(x['X']),((1,1),(1,1),(0,0)))
        result=np.zeros((h,w,c),np.float64)
        for i in range(3):
            for j in range(3): result+=padded[i:i+h,j:j+w]*d(x['W'][:,i,j])[None,None,:]
        out=f(result)
    else: raise ValueError(family)
    return (out,)


def features(case):
    result=base.features(case)
    kinds={'scan':'ordered_scan','segment_sum':'segmented_ownership','scatter_add':'indexed_accumulation','stencil':'local_neighborhood'}
    for record,node in zip(result['nodes'],case['nodes']):
        if node['op'] in kinds:
            record['kind']=kinds[node['op']]
            record['mechanism_metadata']=node['attrs']
    result['output_count']=len(case.get('outputs',[case['output']]))
    if case['family']=='N24': result['group_lengths']=case['group_lengths']
    result['native_mechanism_verified']=False
    return result


def splits(cases):
    held_tags=('C01','C02','C03','C04','C07','C08','D02','D04')+tuple(f for f in ADDITIONS_28 if any(c['family']==f for c in cases))
    folds={}
    for tag in held_tags:
        train=[c['case_id'] for c in cases if c['role']=='source_development' and tag not in c['semantic_tags']]
        op=[c['case_id'] for c in cases if c['role']=='source_development' and tag in c['semantic_tags']]
        joint=[c['case_id'] for c in cases if c['role']=='sealed_shape' and tag in c['semantic_tags']]
        shape=[c['case_id'] for c in cases if c['role']=='sealed_shape' and tag not in c['semantic_tags']]
        folds[tag]={'excluded_tag':tag,'train':train,'operation_only':op,'joint':joint,'shape_only':shape}
    return {'schema':'cross-isa-expansion-splits-v1','folds':folds,'native_performance_labels_opened':0}
