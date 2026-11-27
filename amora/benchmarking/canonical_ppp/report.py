"""Reparse canonical evidence without launching a workload or simulator."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import statistics
import fcntl

from amora.backends.gcom_cuda.portable_v2 import write
from amora.benchmarking.portable_v2.gcom_report import scores


def generate(root, *, figures=False):
    root=Path(root)
    with (root/'report.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _generate(root,figures=figures)


def _generate(root, *, figures=False):
    root=Path(root)
    if not (root/"manifest.json").exists():return
    manifest=json.loads((root/"manifest.json").read_text())
    target=json.loads((root/"campaign-target.json").read_text())
    tag=datetime.fromisoformat(target["started"]).strftime("%y%m%d-%H%M")+"-canonical-ppp-h100"
    project=Path(target["project"])
    report=project/"reports"/(tag+"-report.md")
    lanes={"flash_ptx":"flash-ptx.json","flash_sass":"flash-sass-timing.json","gcom":"gcom.json"}
    rows=[]
    for case in manifest["cases"]:
        folder=root/"points"/case["case_key"]
        row=dict(case_key=case["case_key"],kernel_id=case["kernel_id"],shape=case["shape"],
                 primary=case["case_key"] in manifest["primary_keys"],pilot=case["case_key"] in manifest["pilot_keys"],
                 hardware_status="not_executed",numerical_correctness="not_evaluated_no_template_oracle")
        if (folder/"hardware.json").exists():
            hardware=json.loads((folder/"hardware.json").read_text())
            for field in ("hardware_status","native_ms","native_body_ms","timing_qualified","samples_ms","cv","physical_gpu","gpu_uuid","reason","launches","trace_status"):
                if field in hardware:row[field]=hardware[field]
        for lane,file in lanes.items():
            row[lane]=json.loads((folder/file).read_text()) if (folder/file).exists() else dict(status="not_executed",complete=False)
        rows.append(row)
    def metric(items,lane):
        return scores([dict(native_ms=r.get("native_ms"),prediction_ms=r[lane].get("prediction_ms"),
                            timing_qualified=r.get("timing_qualified",False) and r[lane].get("complete",False)) for r in items])
    overall={lane:metric(rows,lane) for lane in lanes}
    families={k:dict(planned=sum(r["kernel_id"]==k for r in rows),
                    native_completed=sum(r["kernel_id"]==k and r["hardware_status"]=="completed" for r in rows),
                    metrics={lane:metric([r for r in rows if r["kernel_id"]==k],lane) for lane in lanes},
                    outcomes={lane:dict(Counter(r[lane]["status"] for r in rows if r["kernel_id"]==k)) for lane in lanes})
              for k in manifest["generator"]["kernel_ids"]}
    common=[r for r in rows if r.get("timing_qualified") and r["flash_ptx"].get("complete") and r["gcom"].get("complete")]
    def paired(items,denominator):
        selected=[r for r in items if r in common]
        return dict(denominator=denominator,aligned=metric(selected,'flash_ptx'),gcom=metric(selected,'gcom'))
    assets=project/'reports/assets'
    if figures:
        from amora.benchmarking.portable_v2.combined_report import table_figure
        groups={k:paired([r for r in rows if r['kernel_id']==k],v['planned']) for k,v in families.items()}
        table_figure(groups,paired(rows,len(rows)),assets/(tag+'-ape-by-family'),'family',workload='Canonical PPP',
            group_label='Kernel family',note='Same per-case native reference; clocks and service/prefix policies differ. Missing pairs remain explicit.')
        for family in families:
            measured=[r for r in rows if r['kernel_id']==family and r['hardware_status']=='completed']
            if not measured:continue
            groups={','.join(f'{k}{v}' for k,v in r['shape'].items()):paired([r],1) for r in measured}
            table_figure(groups,paired([r for r in rows if r['kernel_id']==family],families[family]['planned']),
                assets/(tag+'-'+family+'-ape-by-shape'),'shape',workload='Canonical PPP: '+family,
                group_label='Shape',note='Measured shapes; all-case row retains the sealed family denominator. Only common qualified pairs are plotted.')
    data=dict(timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),user_prompt=target["user_prompt"],
              status=json.loads((root/"status.json").read_text()) if (root/"status.json").exists() else {},
              primary_manifest_cases=len(manifest["primary_keys"]),pilot_cases=len(manifest["pilot_keys"]),
              unique_cases=len(rows),native_completed=sum(r["hardware_status"]=="completed" for r in rows),
              overall=overall,families=families,common={lane:metric(common,lane) for lane in ("flash_ptx","gcom")},
              rows=rows,report=str(report))
    primary=[r for r in rows if r['primary']]
    data['primary_native_completed']=sum(r['hardware_status']=='completed' for r in primary)
    data['primary_native_outcomes']=sum(r['hardware_status']!='not_executed' for r in primary)
    data['primary_overall']={lane:metric(primary,lane) for lane in lanes}
    data['full_primary_outcomes_recorded']=all(r['hardware_status']!='not_executed' and
        all(r[lane]['status']!='not_executed' for lane in lanes) for r in primary)
    write(root/"case-ledger.json",data)
    def n(v):return "—" if v is None else f"{v:.2f}"
    def pair(m):return f"{n(m.get('mdape'))} / {n(m.get('p95_ape'))}"
    state=data["status"].get("state","prepared")
    lines=["# Canonical PPP on H100: FlashGPU-Sim and GCoM","",f"Updated: {data['timestamp']}","",
        "## User request","","```text",target["user_prompt"],"```","","## Campaign state","",
        f"State: `{state}`. Native completion: **{data['native_completed']} / {len(rows)} unique planned cases**. "
        f"The standard H100 manifest has 2,500 cases; the family pilot adds source-generator minimum-work shapes where absent.","",
        f"Primary native completion: **{data['primary_native_completed']} / {len(primary)}**. Overall timing tables include supplemental source pilots; the primary population is also retained separately in the ledger. "
        "The full runner covers every primary identity; CPU/RAM/storage control in-flight work. GPU jobs require explicit user approval.","",
        "This is a canonical **timing** campaign. The current templates have no independent output oracle; native completion and simulator completion do not establish numerical correctness.","",
        "## Timing-qualified results","",
        "| Lane | Complete runs | Qualified pairs | MdAPE (%) | P95 APE (%) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for lane,label in [("flash_ptx","FlashGPU aligned PTX"),("flash_sass","FlashGPU aligned SASS timing"),("gcom","GCoM")]:
        m=overall[lane]
        lines.append(f"| {label} | {sum(r[lane].get('complete',False) for r in rows)} | {m['eligible_cases']} | {n(m.get('mdape'))} | {n(m.get('p95_ape'))} |")
    lines += ["","APE = 100 × abs(prediction/native − 1). Cases have equal weights; P95 uses linear interpolation. "
        "Native references require complete, dedicated captures, a median span of at least 10 µs and process-repeat CV at most 10%. Short or variable cases retain raw predictions without entering the accuracy summary.","",
        "## Common PTX and GCoM cases","",
        "| Lane | Common pairs | MdAPE (%) | P95 APE (%) | Pearson | Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for lane,label in (("flash_ptx","FlashGPU aligned PTX"),("gcom","GCoM")):
        m=data['common'][lane]
        lines.append(f"| {label} | {m['eligible_cases']} | {n(m.get('mdape'))} | {n(m.get('p95_ape'))} | {n(m.get('pearson'))} | {n(m.get('spearman'))} |")
    additional=[r for r in rows if r.get('timing_qualified') and r['gcom'].get('complete')
                and not r['flash_ptx'].get('complete') and r.get('launches',0)>1]
    if additional:
        lines += ['', '### Additional GCoM composite timing window', '',
            '| Case | Native span (ms) | Median native body sum (µs) | GCoM service (µs) | APE (%) |',
            '| --- | ---: | ---: | ---: | ---: |']
        for row in additional:
            prediction=row['gcom']['prediction_ms'];error=100*abs(prediction/row['native_ms']-1)
            lines.append(f"| {row['case_key']} | {row['native_ms']:.3f} | {row['native_body_ms']*1000:.3f} | {prediction*1000:.3f} | {error:.2f} |")
            native=json.loads((root/'points'/row['case_key']/'native.json').read_text())
            gap=max((b['start_ns']-a['end_ns'] for a,b in zip(native['kernels'],native['kernels'][1:])),default=0)/1e6
            lines += ['',f"A retained native capture has a {gap:.3f} ms gap between its selected kernels. The native span includes that gap; GCoM's service timeline models zero host submission gaps. This case remains in the independent GCoM cohort and is absent from the common PTX/GCoM intersection because PTX does not complete it. Its timing-domain mismatch explains the large error and the negative independent-cohort Pearson correlation. Kernel body sums and model service retain their distinct meanings.",'']
    lines += ["",
        "## Family coverage","",
        "| Family | Native completed / planned | PTX pairs: MdAPE / P95 (%) | SASS pairs: MdAPE / P95 (%) | GCoM pairs: MdAPE / P95 (%) |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for k,v in families.items():
        cells=[f"{v['metrics'][l]['eligible_cases']}: {pair(v['metrics'][l])}" for l in lanes]
        lines.append(f"| {k} | {v['native_completed']} / {v['planned']} | "+" | ".join(cells)+" |")
    lines += ["","## Family pilot measurements","",
        "| Family / shape | Native span (µs) / CV (%) | PTX prediction (µs) / status | SASS prediction (µs) / status | GCoM prediction (µs) / status |",
        "| --- | ---: | --- | --- | --- |"]
    for r in rows:
        if not r["pilot"]:continue
        shape=", ".join(f"{k}={v}" for k,v in r["shape"].items())
        cells=[f"{n(r[l].get('prediction_ms')*1000 if r[l].get('prediction_ms') is not None else None)} / {r[l]['status']}" for l in lanes]
        lines.append(f"| {r['kernel_id']} · {shape} | {n(r.get('native_ms',0)*1000) if 'native_ms' in r else '—'} / {n(r.get('cv',0)*100) if 'cv' in r else '—'} | "+" | ".join(cells)+" |")
    family_figure=assets/(tag+'-ape-by-family.svg')
    if family_figure.exists():
        lines += ['', '## Figures', '', f'![Common qualified error by canonical family](assets/{family_figure.name})',
                  '', 'These use the striped-row MdAPE/P95 layout of the portable comparison and plot the common qualified PTX/GCoM intersection. Independent SASS and GCoM-only results remain in the tables above.', '']
        for family in families:
            path=assets/(tag+'-'+family+'-ape-by-shape.svg')
            if path.exists():lines += [f'[{family} shape figure](assets/{path.name})','']
    lines += ["","## Timing and implementation limits","",
        "- Identical source and binary are used for the native/SASS/trace lanes. The canonical `AMORA_GCOM_TRACE` build removes warmup launches and embedding's explicit L2 flush in every lane. These are cold-process measurements and must not be pooled with the earlier warmup-enabled canonical rank workflow.",
        "- Native reference is the span from the first selected launch start to the last launch end, including fused-component submission gaps. Per-kernel body sums are retained separately. Models include their shipped service/CTA launch policies and model zero host gaps; raw body and service counters are not relabeled as identical estimands.",
        "- Aligned H100 residency is 132 SMs, 2,048 threads, 64 warps and 32 CTAs per SM. FlashGPU retains a 1,500 MHz core model; GCoM retains 1,800 MHz. Neither model is fitted to these observations.",
        "- `aligned_gemm_fp16` calls cuBLAS with uninitialized A/B buffers. Numerical and input-value equivalence cannot be claimed for that template. Fused GEMM templates use their defined byte initializations.",
        "- Attention/MLA are bounded sampled templates. `megamoe_fp8` uses unsigned byte arithmetic and bounded routing; it is not evidence of production FP8 Tensor Core MoE execution.",
        "- Real asynchronous/tensor instructions require their exact sidecar contract. The portable absent-feature override is refused for such traces. Unsupported sidecars remain explicit; missing results are not zero error.",
        ("- Every primary case has a retained native and simulator outcome; failed acquisitions, unsupported contracts and unqualified timings remain explicit." if data['full_primary_outcomes_recorded'] else
         "- The 2,500-case primary study remains incomplete. The initial 128-case stage is historical; the full continuation removes case/work-score cutoffs and waits on explicit GPU approval and actual CPU/RAM/disk admission."),"",
        "## Evidence","",
        f"- [Case ledger](../local_artifacts/canonical-ppp-h100/case-ledger.json)",
        f"- [Sealed manifest](../local_artifacts/canonical-ppp-h100/manifest.json)",
        f"- [Source/tool/model admission](../local_artifacts/canonical-ppp-h100/admission.json)",
        f"- [Plan](../.plan/{Path(target['plan']).name})",
        f"- [Tracker](../.plan/tracker/{Path(target['tracker']).name})","",
        "Reparse with `PYTHONPATH=/home/cliu/wk/amora <execution-env>/bin/python -m amora.benchmarking.canonical_ppp.campaign report --root /home/cliu/wk/gpu-perf/local_artifacts/canonical-ppp-h100`. No simulator is invoked by report generation.","",
        "## Revision History",""]
    history=root/"report-history.json"
    revisions=json.loads(history.read_text()) if history.exists() else []
    key=(data["native_completed"],sum(r[l].get("complete",False) for r in rows for l in lanes),state)
    if not revisions or revisions[-1]["key"] != list(key):
        revisions.append(dict(timestamp=data["timestamp"],key=list(key)))
        write(history,revisions)
    for revision in revisions:
        lines.append(f"- {revision['timestamp']}: Reparsed retained canonical evidence; native completed {revision['key'][0]}, complete simulator runs {revision['key'][1]}, state {revision['key'][2]}.")
        if revision.get('note'):lines.append('  '+revision['note'])
    report.write_text("\n".join(lines)+"\n")
    return data
