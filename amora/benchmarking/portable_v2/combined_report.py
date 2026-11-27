"""Build the FlashGPU/GCoM H100 report from existing, identified evidence."""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime
import argparse,csv,hashlib,json,math
import numpy as np
from .gcom_report import scores

REQUEST='also generate a report with flashGPU-Sim and GCOM results for both canonical and portable_core v2 kernels. if canonical kernels results are not avaiable, let me know.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(pairs,lane):
    return scores([dict(timing_qualified=True,native_ms=p['gcom_native_ms'] if lane=='gcom' else p['flash_native_ms'],
                       prediction_ms=p[lane+'_ms']) for p in pairs])


def table_figure(groups,overall,path,kind, *, workload='Portable core v2', group_label=None,
                 note='Native devices/epochs, model clocks and prefix policies differ. This is a descriptive comparison.'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    matplotlib.rcParams.update({'svg.fonttype':'none','svg.hashsalt':'flashgpu-gcom-comparison'})
    keys=sorted(groups)
    if kind=='shape' and all(all(part.strip().isdigit() for part in key.split(' × ')) for key in keys):
        keys.sort(key=lambda key:(math.prod(map(int,key.split(' × '))),*map(int,key.split(' × '))))
    rows=[('All common cases',overall),*[(key,groups[key]) for key in keys]]
    peak=max([v[lane].get(metric,0) for _,v in rows for lane in ('aligned','gcom') for metric in ('mdape','p95_ape')]+[0])
    limit=max(100,math.ceil((peak+5)/20)*20)
    width,top,bottom,row_height=14.5,1.30,.80,.29;height=top+bottom+row_height*len(rows);left,right=3.20,9.00
    fig=plt.figure(figsize=(width,height),facecolor='white');ax=fig.add_axes([left/width,bottom/height,(right-left)/width,row_height*len(rows)/height])
    def text(x,y,value,**kwargs):ax.text((x-left)/(right-left),y,value,transform=ax.get_yaxis_transform(),va='center',clip_on=False,fontsize=9,**kwargs)
    for i,(name,v) in enumerate(rows):
        if i%2==0:ax.axhspan(i-.5,i+.5,color='#edf3f8' if not i else '#f5f7fa',zorder=0)
        text(.20,i,name,weight='bold' if not i else 'normal');n=v['gcom']['eligible_cases'];text(14.22,i,f"{n:,} / {v['denominator']:,}",ha='right')
        if not n:
            ax.text(2,i,'No common qualified pairs',va='center',fontsize=9,color='#637489',style='italic')
            for x in (10.65,12.10):text(x,i,'—',ha='right',color='#637489')
            continue
        for metric,marker,offset in [('mdape','o',-.14),('p95_ape','D',.14)]:
            f,g=v['aligned'][metric],v['gcom'][metric]
            ax.hlines(i+offset,min(f,g),max(f,g),color='#a6b4c4',lw=1.2)
            ax.scatter(f,i+offset,marker=marker,facecolors='white',edgecolors='#778697',s=31,zorder=3)
            ax.scatter(g,i+offset,marker=marker,color='#195c96',s=18,zorder=4)
        for lane,x,color in [('aligned',10.65,'#637489'),('gcom',12.10,'#195c96')]:text(x,i,f"{v[lane]['mdape']:.1f} / {v[lane]['p95_ape']:.1f}",ha='right',color=color)
    for x,title in [(.20,group_label or ('Operator' if kind=='operator' else 'Shape (M × N)')),(10.65,'FlashGPU PTX (%)\nMdAPE / P95'),(12.10,'GCoM (%)\nMdAPE / P95'),(14.22,'Pairs / cases')]:text(x,-.87,title,weight='bold',ha='left' if x==.20 else 'right')
    ax.set(xlim=(0,limit),ylim=(len(rows)-.5,-.5),yticks=[],xticks=range(0,limit+1,20));ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.grid(axis='x',color='#d9e3ed',lw=.7);ax.set_axisbelow(True);ax.tick_params(axis='x',length=0,labelsize=9,pad=7)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xlabel('Absolute percentage error against each lane’s native reference',fontsize=10,labelpad=8)
    fig.text(.016,1-.14/height,f'{workload}: FlashGPU-Sim vs. GCoM by {kind}',fontsize=18,weight='bold',va='top')
    fig.text(.016,1-.54/height,'Same qualified case identities · FlashGPU residency-aligned PTX vs. GCoM SASS service',fontsize=10.5,va='top',color='#526578')
    fig.text(.016,1-.82/height,note,fontsize=9,va='top',color='#526578')
    fig.legend(handles=[Line2D([],[],marker='o',ls='none',mfc='white',mec='#778697',label='FlashGPU PTX'),Line2D([],[],marker='o',ls='none',color='#195c96',label='GCoM'),Line2D([],[],marker='o',ls='none',color='black',label='MdAPE'),Line2D([],[],marker='D',ls='none',color='black',label='P95 APE')],loc='upper right',bbox_to_anchor=(.983,1-.70/height),ncol=4,fontsize=9,frameon=False)
    fig.text(.016,.11/height,'APE = 100 × |prediction / native − 1|. Equal case weights; linear percentiles. Operator groups overlap; missing estimates are not zero error.',fontsize=9,color='#526578')
    for ext in ('svg','png'):fig.savefig(path.with_suffix('.'+ext),dpi=170)
    plt.close(fig)


def generate(project):
    project=Path(project).resolve();root=project/'local_artifacts/flashgpu-gcom-comparison';identity=json.loads((root/'report-target.json').read_text());tag=identity['tag']
    sources=dict(flash_stock=project/'local_artifacts/flashgpu-portable-v2/comparison.json',flash_aligned=project/'local_artifacts/flashgpu-h100-residency/comparison-aligned-vs-stock.json',gcom=project/'local_artifacts/gcom-portable-v2/case-ledger.json',gcom_audit=project/'local_artifacts/gcom-portable-v2/final-audit.json',microbench=project/'local_artifacts/stock-flashgpu-h100/timing-comparison.json',canonical_specs=Path('/home/cliu/wk/amora/benchmark_generators/ppp_canonical/kernels.py'),archived_gcom_profile=Path('/home/cliu/wk/amora/out/gcom_cuda/hopper/gcom_h100-rerun-20260717-0300.json'),archived_hw_profile=Path('/home/cliu/wk/amora/out/nvidia/hopper/h100-80g.json'))
    flash=json.loads(sources['flash_stock'].read_text());aligned=json.loads(sources['flash_aligned'].read_text());gcom=json.loads(sources['gcom'].read_text());audit=json.loads(sources['gcom_audit'].read_text())
    frows={p['index']:p for p in flash['rows']};grows={p['index']:p for p in gcom['rows']};fpaired={p['index']:p for p in aligned['portable']['case_pairs'] if p['lane']=='ptx'}
    for i,p in grows.items():
        assert p['case_id']==frows[i]['case_id'] and p['case_sha256']==frows[i]['case_sha256'] and p['program_sha256']==frows[i]['program_sha256'],'case identity mismatch'
    eligible_g={i:p for i,p in grows.items() if p['timing_qualified'] and p['prediction_ms']}
    common=sorted(set(eligible_g)&set(fpaired));pairs=[]
    for i in common:
        f,g=fpaired[i],eligible_g[i];assert set(f['operators'])==set(g['operators'])
        pairs.append(dict(index=i,case_id=g['case_id'],case_sha256=g['case_sha256'],program_sha256=g['program_sha256'],operators=g['operators'],shape=g['shape'],aliases=g['aliases'],flash_native_ms=f['native_ms'],gcom_native_ms=g['native_ms'],stock_ms=f['stock_ms'],aligned_ms=f['aligned_ms'],gcom_ms=g['prediction_ms'],gcom_gpu=g['physical_gpu']))
    def group_stats(items,denominator):return dict(denominator=denominator,**{lane:summarize(items,lane) for lane in ('stock','aligned','gcom')})
    overall=group_stats(pairs,1155);groups={}
    for kind in ('operator','shape','suite_profile'):
        grouped=defaultdict(list);den=Counter()
        for p in grows.values():
            keys=p['operators'] if kind=='operator' else [p['shape']] if kind=='shape' else [a+'/'+b for a,b in p['aliases']]
            den.update(keys)
        for p in pairs:
            keys=p['operators'] if kind=='operator' else [p['shape']] if kind=='shape' else [a+'/'+b for a,b in p['aliases']]
            for key in keys:grouped[key].append(p)
        groups[kind]={key:group_stats(grouped[key],total) for key,total in den.items()}
    sass_pairs=[p for p in aligned['portable']['case_pairs'] if p['lane']=='sass-timing']
    sass_metrics=scores([dict(native_ms=p['native_ms'],prediction_ms=p['aligned_ms'],timing_qualified=True) for p in sass_pairs])
    from benchmark_generators.ppp_canonical.kernels import CANONICAL_KERNELS
    canonical=[dict(kernel_id=k.kernel_id,flashgpu='not_available',gcom='not_available',native='not_available',semantics=k.measurement_semantics) for k in CANONICAL_KERNELS]
    canonical_path=project/'local_artifacts/canonical-ppp-h100/case-ledger.json'
    canonical_data=json.loads(canonical_path.read_text()) if canonical_path.exists() else None
    if canonical_data:
        sources['canonical_campaign']=canonical_path
        def available(family,lane):
            counts=family['outcomes'][lane]
            return f"{counts.get('completed',0)} complete / {family['metrics'][lane]['eligible_cases']} qualified"
        for row in canonical:
            family=canonical_data['families'][row['kernel_id']]
            row.update(flashgpu='PTX '+available(family,'flash_ptx')+'; SASS '+available(family,'flash_sass'),
                       gcom=available(family,'gcom'),native=f"{family['native_completed']} complete / {family['planned']} planned")
    micro=aligned['stock'];profiles=[]
    archived=json.loads(sources['archived_gcom_profile'].read_text())
    for p in archived['results']:
        e=p['simulator_estimate']
        if e.get('value') is not None:profiles.append(dict(probe_id=p['identity']['probe_id'],value=e['value'],unit=e.get('unit'),fit_status=e.get('fit_status'),mapping=e.get('mapping_contract')))
    result=dict(timestamp=datetime.now().astimezone().isoformat(timespec='seconds'),user_prompt=REQUEST,sources={k:dict(path=str(p),sha256=sha(p)) for k,p in sources.items()},canonical_availability=canonical,canonical_inventory_checked=['gpu-perf reports/local_artifacts','amora out/local_artifacts/docs','modern-gpu-simulator-micro-2025','accorde local_artifacts/.dbg'],archived_diagnostic_profiles=profiles,independent_gcom=gcom['summary']['overall'],independent_gcom_h3_batch=gcom['summary']['h3_batch'],common=overall,common_groups=groups,pairs=pairs,flash_sass_pilot=sass_metrics,canonical_microbenchmarks=micro,comparison_status='same identities, lane-specific native references and distinct prefix/clock/estimand; descriptive only',data_audit=audit['state'])
    if canonical_data:
        result['canonical_campaign']=dict(status=canonical_data['status'],overall=canonical_data['overall'],
            common=canonical_data['common'],native_completed=canonical_data['native_completed'],report=canonical_data['report'])
        result['user_prompt_history']=[REQUEST,canonical_data['user_prompt']]
    (root/'comparison.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    with (root/'common-portable-pairs.csv').open('w') as f:
        fields=['index','case_id','flash_native_ms','gcom_native_ms','stock_ms','aligned_ms','gcom_ms','gcom_gpu','case_sha256','program_sha256'];w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(pairs)
    assets=project/'reports/assets'
    for kind in ('operator','shape'):table_figure(groups[kind],overall,assets/(tag+'-portable-ape-by-'+kind),kind)
    def n(v):return f'{v:.2f}' if v is not None else '—'
    lines=['# FlashGPU-Sim and GCoM against H100: canonical and portable core v2','',f"Updated: {result['timestamp']}",'','## User request','','```text',REQUEST,'```','','## Result and availability','',
    ('**Portable_core v2 results exist for both simulators. The dedicated canonical PPP H100 campaign now has retained results.** Canonical completion and timing-qualified counts below come from its case ledger; this is not full-manifest closeout.' if canonical_data else '**Portable_core v2 results exist for both simulators. Qualified PPP canonical benchmark results were not found for either simulator.** The canonical section is explicitly unavailable; the available instruction microbenchmarks and archived diagnostic profiles are identified separately.'),'',
    'FlashGPU residency-aligned PTX has 1,155/1,155 numerical passes and 1,041 timing-qualified pairs. GCoM has 1,141 successful timing replays, 14 incomplete hardware acquisitions and 823 qualified single-program comparisons. Its independent batch64 H3 lane has 47 complete replays, 6 incomplete runs and 37 qualified comparisons.','',
    'The primary two-lane comparison uses **790 identical qualified case/program identities**. Each simulator retains its own native reference. Hardware devices/epochs, prefix policies and model clock domains differ, so the difference between error distributions is descriptive and cannot be attributed solely to simulator quality.','',
    '## Canonical benchmark kernels','',
    ('Canonical means AMORA’s `benchmark_generators/ppp_canonical` workload families. The campaign now retains the standard 2,500-case H100 manifest plus nine minimum-work source pilots. It uses the unchanged canonical templates and a consistently warmup-disabled binary. Numerical correctness remains unevaluated because these templates have no independent output oracle.' if canonical_data else 'Canonical means AMORA’s `benchmark_generators/ppp_canonical` workload families. The README explicitly separates these benchmarks from the 36 diagnostic probes. No materialized canonical manifest, per-case H100 labels or paired FlashGPU/GCoM canonical comparison was located in the checked project, Amora or Accorde artifact paths.'),'',
    '| Canonical family | FlashGPU-Sim result | GCoM result | Measurement contract |','| --- | --- | --- | --- |']
    for p in canonical:lines.append(f"| {p['kernel_id']} | {p['flashgpu']} | {p['gcom']} | `{p['semantics']}` |")
    if canonical_data:
        lines += ['','### Canonical timing-qualified results','',
            '| Lane | Qualified pairs | MdAPE (%) | P95 APE (%) |','| --- | ---: | ---: | ---: |']
        for lane,label in [('flash_ptx','FlashGPU aligned PTX'),('flash_sass','FlashGPU aligned SASS timing'),('gcom','GCoM')]:
            m=canonical_data['overall'][lane];lines.append(f"| {label} | {m['eligible_cases']} | {n(m.get('mdape'))} | {n(m.get('p95_ape'))} |")
        lines += ['',f"Native completed: {canonical_data['native_completed']} unique cases. Full family/shape outcomes, raw pilot predictions, resource-deferred cases and timing limits are in the [canonical report]({Path(canonical_data['report']).name}).",
            'Native spans include component gaps; models retain their original service/launch policies and clock domains. Attention/MLA/MoE are the declared bounded templates; their names do not establish production kernel or native FP8 Tensor Core coverage. Do not pool canonical and portable errors.','',
            'Canonical follow-up request:','', '```text',canonical_data['user_prompt'],'```']
    lines += ['',('Canonical unavailable, unsupported, incomplete and resource-deferred cases remain explicit in the linked campaign ledger.' if canonical_data else 'These rows have no timing accuracy estimate and no fabricated zero-error value. No new canonical experiment was launched for this report. The `h100_2500` and `h100_5600` presets are definitions, not result evidence.'),'',
    '## Portable core v2: independent reported cohorts','',
    '| Lane | Qualified pairs / planned cases | MdAPE (%) | P95 APE (%) | Reference / scope |','| --- | ---: | ---: | ---: | --- |',
    '| FlashGPU stock PTX | 1,041 / 1,155 | 63.01 | 100.72 | Historical native H100 GPU 7 |',
    '| FlashGPU residency-aligned PTX | 1,041 / 1,155 | 61.39 | 96.09 | Same historical native GPU 7 references |',
    f"| FlashGPU aligned SASS pilot | {sass_metrics['eligible_cases']} / 35 pilot cases | {n(sass_metrics['mdape'])} | {n(sass_metrics['p95_ape'])} | 12 passes, 23 unsupported; only 3 timing pairs |",
    f"| GCoM single-program | 823 / 1,155 | {n(gcom['summary']['overall']['mdape'])} | {n(gcom['summary']['overall']['p95_ape'])} | Fresh per-device references on dedicated H100s |",
    f"| GCoM batch64 H3 | 37 / 53 | {n(gcom['summary']['h3_batch']['mdape'])} | {n(gcom['summary']['h3_batch']['p95_ape'])} | Separate executable/protocol; excluded from single-program pool |",'',
    'These rows use different eligible populations. They are coverage summaries, not a paired ranking of the simulators. FlashGPU SASS has no full portable panel or comparable GCoM batch lane.','',
    '## Portable core v2: common case identities','',
    '| Lane on the common cohort | Cases | MdAPE (%) | P95 APE (%) | Signed median bias (%) | Pearson | Spearman |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for lane,label in [('stock','FlashGPU stock PTX'),('aligned','FlashGPU residency-aligned PTX'),('gcom','GCoM')]:
        s=overall[lane];lines.append(f"| {label} | {s['eligible_cases']} | {n(s['mdape'])} | {n(s['p95_ape'])} | {n(s['signed_median_bias'])} | {n(s['pearson'])} | {n(s['spearman'])} |")
    lines += ['','The join validates case ID, case SHA-256, program SHA-256 and contained operators for every row. The unchanged CUDA realization is shared; inputs/oracle banks are the sealed portable corpus. Global weighting is one per unique case, not aliases or repeated samples. P95 uses linear interpolation.','',
    '### Operator and shape figures','',
    f"![Common-cohort error by operator](assets/{tag}-portable-ape-by-operator.svg)",f"[Operator PNG](assets/{tag}-portable-ape-by-operator.png)",'',
    f"![Common-cohort error by shape](assets/{tag}-portable-ape-by-shape.svg)",f"[Shape PNG](assets/{tag}-portable-ape-by-shape.png)",'',
    'The figures use the requested striped-row/table layout. FlashGPU PTX uses hollow gray markers and GCoM solid blue markers; circles denote MdAPE and diamonds P95. Counts use the common eligible intersection over the full group denominator. Missing groups are labeled.','',
    '### Operation and H3 suite breakdown','',
    '| Suite / profile | Common pairs / total | FlashGPU aligned MdAPE / P95 (%) | GCoM MdAPE / P95 (%) |','| --- | ---: | ---: | ---: |']
    for key,v in sorted(groups['suite_profile'].items()):
        lines.append(f"| {key} | {v['gcom']['eligible_cases']} / {v['denominator']} | {n(v['aligned'].get('mdape'))} / {n(v['aligned'].get('p95_ape'))} | {n(v['gcom'].get('mdape'))} / {n(v['gcom'].get('p95_ape'))} |")
    lines += ['','H3 development/smoke controls are covered. Both exact historical H3 reference profiles remain `blocked_missing_source`, and formal operation/H3 gate decisions remain `gate_not_evaluated` without approved splits and frozen acceptance criteria.','',
    '## Available H100 instruction microbenchmarks','',
    'These are kernel/window cycle probes from the earlier FlashGPU study, not PPP canonical application results. No corresponding admitted GCoM runs for these exact same-binary windows were found. The values are unchanged by the FlashGPU residency alignment.','',
    '| Probe / window | Native cycles | FlashGPU aligned PTX cycles / signed difference (%) | FlashGPU aligned SASS cycles / signed difference (%) | GCoM |','| --- | ---: | ---: | ---: | --- |']
    for p in micro:
        vals=[]
        for lane in ('ptx','sass-timing'):
            x=p[lane];vals.append(f"{n(x.get('aligned_cycles'))} / {n(x.get('aligned_error_percent'))}" if x.get('aligned_cycles') is not None else 'Unavailable / abort')
        lines.append(f"| {p['case']} · `{p['metric']}` | {n(p['native'])} | {vals[0]} | {vals[1]} | Not available |")
    lines += ['','WGMMA whole-region, issue and wait windows overlap and must not be pooled into a single application MdAPE. TMA/cp.async completion windows are separate protocols with their own native repeats.','',
    '## Archived AMORA diagnostic profile availability','',
    'The existing `out/gcom_cuda/hopper/gcom_h100-rerun-20260717-0300.json` and `out/nvidia/hopper/h100-80g.json` each contain 36 diagnostic probe records, not canonical workload cases. Five GCoM entries publish direct/bounded estimates. Their source/binary/native-reference provenance is not admitted by the current portable campaign, so they do not supply canonical benchmark accuracy.','',
    '| Diagnostic probe | Archived GCoM estimate | Unit | Fit status |','| --- | ---: | --- | --- |']
    for p in profiles:lines.append(f"| {p['probe_id']} | {n(p['value'])} | {p['unit']} | {p['fit_status']} |")
    lines += ['','## Target, timing and limitations','',
    '- H100 target: CC 9.0, 132 SMs, 2,048 threads, 64 warps and 32 CTAs per SM for the aligned FlashGPU and GCoM lanes. Stock FlashGPU limits remain 1,536 threads/48 warps/24 CTAs in its separately identified baseline.',
    '- FlashGPU clock domains retain 1500:1700:1700:2617 MHz; GCoM retains 1800:1800:1800:8000 MHz. Raw cycles from different core clocks are not interchangeable.',
    '- FlashGPU uses historical free-running GPU 7 CUDA program spans. GCoM uses fresh device-matched spans from dedicated H100s; 99 common cases use physical GPU 7, but their native measurement epoch still differs.',
    '- GCoM replays one validation graph, three warmup graphs and one target graph. Its scored service includes the configured 1,500-cycle CTA launch delay and models zero host submission gaps. The earlier FlashGPU PTX capture/run uses its own validation/sample prefix and process sharing; the two timing/cache policies are disclosed rather than treated as identical.',
    '- GCoM’s five absent-feature sidecar overrides are documented in its report. Ordinary Hopper CTA barriers remain modeled. The CUDA executable is unchanged; the NVBit loader/handle adapter is identified separately from the untouched simulator timing model.',
    '- Numerical correctness is checked natively against the independent oracle. GCoM replay admission is a timing-trace outcome, not a simulator arithmetic validation.',
    '- The portable half-identity contraction is realized as vector scaling. It does not establish dense GEMM, Tensor Core, TMA, WGMMA or DSM application coverage.',
    '- GCoM’s 14 hardware acquisition failures remain visible. Six batch runs contain 500 kernel counter blocks and are incomplete for their declared full batches; they do not contribute timing scores. Neither model was retuned to reduce error.','',
    '## Evidence and reproduction','',
    f"- [Combined machine-readable comparison](../local_artifacts/flashgpu-gcom-comparison/comparison.json)",
    '- [Common case/native/prediction crosswalk](../local_artifacts/flashgpu-gcom-comparison/common-portable-pairs.csv)',
    '- [FlashGPU residency-aligned report](261124-2322-flashgpu-h100-residency-report.md)',
    '- [GCoM report](261125-1804-gcom-portable-v2-h100-report.md)',
    '- [Stock H100 probe report](261120-2307-stock-flashgpu-h100-report.md)',
    '- [Canonical definitions](../../amora/benchmark_generators/ppp_canonical/kernels.py)',
    '- [AMORA benchmark/probe distinction](../../amora/README.md)',
    '- [GCoM audit](../local_artifacts/gcom-portable-v2/final-audit.json)','',
    'Reproduce from the Amora checkout with `PYTHONPATH=. <execution-env>/bin/python -m amora.benchmarking.portable_v2.combined_report --project /home/cliu/wk/gpu-perf`. This reparses saved data and does not run a simulator.','',
    'Source file hashes are included in the combined JSON. GCoM’s MICRO 2025 architectural model, ISPASS 2025 parallelization work and Accel-Sim ISCA 2020 citations are listed in the linked GCoM report.','',
    '## Revision History','',f"- {result['timestamp']}: Created the combined report from retained evidence; verified the 790-case identity intersection and reported missing canonical cohorts explicitly. Generated figures in the requested reference layout; no simulation or hardware launch.",'']
    report_path=Path(identity['report'])
    if report_path.exists():
        old=report_path.read_text()
        if '\n## Revision History\n' in old:
            history=old.split('\n## Revision History\n',1)[1].strip()
            marker=lines.index('## Revision History')
            lines=lines[:marker+1]+['',history,
                f"- {result['timestamp']}: Updated canonical availability and scores from the retained campaign ledger; portable identities and score arithmetic unchanged. No simulator was launched by report generation.", '']
    report_path.write_text('\n'.join(lines))
    print(identity['report']);print('Common cases',len(pairs),'scores',overall)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,required=True)
    generate(parser.parse_args().project)
