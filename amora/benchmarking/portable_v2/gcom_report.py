"""Recompute descriptive portable GCoM coverage and correlation from receipts."""
from __future__ import annotations
from collections import Counter,defaultdict
from datetime import datetime
import csv
import json
from pathlib import Path
import statistics
import re

from amora.backends.gcom_cuda.portable_v2 import utc,write

PROMPTS=(
    "persist a plan to run the same validation/correlation experiments for portable_core v2 (both the op gates and H3 gates) with GCoM ~/wk/modern-gpu-simulator-micro-2025/ with same H100 target.",
    "use as any many GPUs as possible as long as it's dedicated (no loads).",
    "and use as many CPU as possible for GCoM's data points.",
    "execute the plan",
    "can the gcom report use the layout of reports/assets/portable-v2-stock-aligned-ape-by-operator.svg for its figures?",
)


def scores(rows):
    import numpy as np
    matched=[r for r in rows if r.get("timing_qualified") and r.get("prediction_ms",0)>0]
    value=dict(total_cases=len(rows),eligible_cases=len(matched),status="not_evaluable")
    if not matched:return value
    native=np.array([r["native_ms"] for r in matched])
    predicted=np.array([r["prediction_ms"] for r in matched])
    signed=100*(predicted/native-1);errors=np.abs(signed)
    value.update(status="descriptive",mdape=float(np.median(errors)),p95_ape=float(np.percentile(errors,95)),
        signed_median_bias=float(np.median(signed)))
    def correlation(a,b):
        if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
        return float(np.corrcoef(a,b)[0,1])
    def ranks(a):
        result=np.zeros(len(a))
        for item in np.unique(a):
            indexes=np.flatnonzero(a==item)
            result[indexes]=((a<item).sum()+1+(a<=item).sum())/2
        return result
    value.update(pearson=correlation(native,predicted),spearman=correlation(ranks(native),ranks(predicted)),
        log_duration_correlation=correlation(np.log(native),np.log(predicted)))
    return value


def build_ledger(campaign):
    points=campaign.saved_points()
    native_only={}
    scopes=[campaign.root/name for name in ("pilot","witnesses","full")]+list(campaign.root.glob("pilot-attempt-*"))
    for scope in scopes:
        for path in scope.rglob("native.json"):
            record=json.loads(path.read_text())
            if record.get("binary_sha256")!=campaign.admission["native_binary"]["binary_sha256"]:continue
            if len(record["native"])!=3 or any(not r["receipt"]["dedicated_interval"] or r["receipt"]["exit_code"]!=0 for r in record["native"]):continue
            indexes={r["index"] for r in record["native"][0]["rows"]}
            for index in indexes:
                selected=[next((r for r in run["rows"] if r["index"]==index),None) for run in record["native"]]
                if all(r and r["passed"] and r["case_sha256"]==campaign.manifest["cases"][index]["case_sha256"] for r in selected):
                    native_only[index]=record["gpu"]
    rows=[]
    for case in campaign.manifest["cases"]:
        point=points.get(case["index"],{})
        quality=point.get("native_quality",{})
        native_device=native_only.get(case["index"])
        native_correct=point.get("native_correct")
        if native_correct is None and native_device and (not point.get("gpu") or point["gpu"]["uuid"]==native_device["uuid"]):
            native_correct=True
        consumers=Counter(ref[0] for node in case["program"]["nodes"] for ref in node["inputs"])
        node_ids={n["id"] for n in case["program"]["nodes"]}
        topology="fork" if any(consumers[i]>1 for i in node_ids) else "chain" if len(node_ids)>1 else "single"
        layout="offset" if any(i["offset"] for i in case["program"]["inputs"]) else "strided" if any(i["row_padding"] for i in case["program"]["inputs"]) else "dense"
        row=dict(index=case["index"],case_id=case["id"],case_sha256=case["case_sha256"],
            program_sha256=case["program_sha256"],aliases=case["aliases"],family=case["family"],
            shape=f"{case['shape']['M']} × {case['shape']['N']}",shape_elements=case["shape"]["M"]*case["shape"]["N"],
            operators=sorted({n["op"] for n in case["program"]["nodes"]}),
            nodes=len(case["program"]["nodes"]),state_edges=len(case["program"]["state_edges"]),unroll=case["program"]["unroll"],
            topology=topology,layout=layout,dtypes=sorted({i["tensor"]["dtype"] for i in case["program"]["inputs"]}),
            status=point.get("state","native_complete_waiting_trace" if case["index"] in native_only else "queued_hardware"),
            native_correct=native_correct,
            trace_complete=point.get("trace_complete",False),replay_admitted=point.get("replay_admitted",False),
            timing_qualified=point.get("timing_qualified",False),native_ms=quality.get("median_ms"),
            prediction_ms=point.get("modeled_target_ms"),gpu_uuid=point.get("gpu",{}).get("uuid"),
            physical_gpu=point.get("gpu",{}).get("index"),result_path=point.get("result_path"),
            gate_status="gate_not_evaluated",body_status=point.get("body_comparison_status","pending_runtime"))
        if row["timing_qualified"] and row["prediction_ms"] and row["native_ms"]:
            row["ape_percent"]=100*abs(row["prediction_ms"]/row["native_ms"]-1)
        rows.append(row)
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:
        groups["shape"][row["shape"]].append(row)
        groups["family"][row["family"]].append(row)
        groups["graph_size"][str(row["nodes"])].append(row)
        groups["state_edges"][str(row["state_edges"])].append(row)
        groups["topology"][row["topology"]].append(row)
        groups["layout"][row["layout"]].append(row)
        groups["unroll"][str(row["unroll"])].append(row)
        for dtype in row["dtypes"]:groups["dtype"][dtype].append(row)
        groups["gpu"][row["gpu_uuid"] or "unassigned"].append(row)
        for op in row["operators"]:groups["operator"][op].append(row)
        for suite,profile in row["aliases"]:groups["suite_profile"][suite+"/"+profile].append(row)
    summary={kind:{name:scores(items) for name,items in values.items()} for kind,values in groups.items()}
    summary["overall"]=scores(rows)
    batch_rows=[]
    for path in (campaign.root/"h3-batch").rglob("case-*.result.json"):
        point=json.loads(path.read_text())
        quality=point.get("native_quality",{})
        batch_rows.append(dict(index=point["case_index"],case_id=point["case_id"],gpu_uuid=point["gpu"]["uuid"],
            native_ms=quality.get("median_ms"),prediction_ms=point.get("modeled_target_ms"),
            timing_qualified=point.get("timing_qualified",False),status=point["state"],
            repetitions=64,native_binary_sha256=point.get("native_binary_sha256")))
    summary["h3_batch"]=scores(batch_rows)
    return dict(timestamp=utc(),rows=rows,summary=summary,status_counts=dict(Counter(r["status"] for r in rows)),
        h3_batch_rows=batch_rows,cross_device_anchors=anchor_comparison(campaign),
        exact_h3_references="blocked_missing_source",formal_gates="gate_not_evaluated")


def anchor_comparison(campaign):
    anchors=defaultdict(dict)
    for scope in ("pilot","witnesses","full"):
        for path in sorted((campaign.root/scope).rglob("native.json"),key=lambda p:p.stat().st_mtime):
            record=json.loads(path.read_text())
            for index in campaign.anchor_indices:
                values=[next((r["samples_ms"] for r in run["rows"] if r["index"]==index),[]) for run in record["native"]]
                if len(values)==3 and all(len(v)==30 for v in values):
                    anchors[str(index)][record["gpu"]["uuid"]]=statistics.median(statistics.median(v) for v in values)
    heterogeneous=False
    for values in anchors.values():
        if len(values)>1:
            median=statistics.median(values.values())
            heterogeneous |= any(abs(v/median-1)>.05 for v in values.values())
    return dict(native_anchor_medians_ms=dict(anchors),threshold_percent=5,
        pooled_score_status="descriptive; use device strata as primary" if heterogeneous else "matching protocol; heterogeneity not detected",
        devices_comparable=len({gpu for values in anchors.values() for gpu in values})>1)


def ape_table_figure(rows, *, title, subtitle, note, group_label):
    """Use the stock/aligned report's striped rows, markers and value columns."""
    import math
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import PercentFormatter
    peak=max((score.get(metric,0) for _,score in rows
              for metric in ("mdape","p95_ape")),default=0)
    limit=max(100,math.ceil((peak+5)/20)*20)
    width,top,bottom,row_height=14.5,1.30,.80,.29
    height=top+bottom+row_height*len(rows)
    left,right=3.20,9.00
    fig=plt.figure(figsize=(width,height),facecolor="white")
    ax=fig.add_axes([left/width,bottom/height,(right-left)/width,row_height*len(rows)/height])
    def text_at(x,y,text,**kwargs):
        ax.text((x-left)/(right-left),y,text,transform=ax.get_yaxis_transform(),
                va="center",clip_on=False,fontsize=9,**kwargs)
    for index,(name,score) in enumerate(rows):
        if index%2==0:
            ax.axhspan(index-.5,index+.5,color="#edf3f8" if not index else "#f5f7fa",zorder=0)
        weight="bold" if not index else "normal"
        text_at(.20,index,name,weight=weight)
        text_at(14.22,index,f"{score['eligible_cases']:,} / {score['total_cases']:,}",ha="right",weight=weight)
        if not score["eligible_cases"]:
            ax.text(2,index,"No qualified pairs",va="center",fontsize=9,color="#637489",style="italic")
            for x in (10.65,12.10):text_at(x,index,"—",ha="right",color="#637489")
            continue
        for metric,marker,offset in (("mdape","o",-.10),("p95_ape","D",.10)):
            ax.scatter(score[metric],index+offset,marker=marker,color="#195c96",s=24,zorder=4)
        text_at(10.65,index,f"{score['mdape']:.1f}",ha="right",color="#195c96",weight=weight)
        text_at(12.10,index,f"{score['p95_ape']:.1f}",ha="right",color="#195c96",weight=weight)
    for x,label in ((.20,group_label),(10.65,"MdAPE (%)"),(12.10,"P95 APE (%)"),(14.22,"Pairs / cases")):
        text_at(x,-.87,label,weight="bold",ha="left" if x==.20 else "right")
    ax.set(xlim=(0,limit),ylim=(len(rows)-.5,-.5),yticks=[],xticks=range(0,limit+1,20))
    ax.xaxis.set_major_formatter(PercentFormatter(100,decimals=0))
    ax.grid(axis="x",color="#d9e3ed",lw=.7);ax.set_axisbelow(True)
    ax.tick_params(axis="x",length=0,labelsize=9,pad=7)
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_xlabel("Program-span absolute percentage error",fontsize=10,labelpad=8)
    fig.text(.016,1-.14/height,title,fontsize=19,weight="bold",va="top")
    fig.text(.016,1-.54/height,subtitle,fontsize=10.5,va="top",color="#526578")
    fig.text(.016,1-.82/height,note,fontsize=9,va="top",color="#526578")
    fig.legend(handles=[Line2D([],[],marker="o",ls="none",color="#195c96",label="MdAPE"),
                        Line2D([],[],marker="D",ls="none",color="#195c96",label="P95 APE")],
               loc="upper right",bbox_to_anchor=(.983,1-.70/height),ncol=2,fontsize=9,frameon=False)
    fig.text(.016,.11/height,
        "APE = 100 × |GCoM service / native span − 1|. Equal case weights; linear percentiles. Missing estimates are not zero error.",
        fontsize=9,color="#526578")
    return fig


def figures(ledger,assets,prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({"svg.fonttype":"none","svg.hashsalt":"gcom-portable-v2-ape"})
    assets.mkdir(parents=True,exist_ok=True)
    paths=[]
    matched=[r for r in ledger["rows"] if r.get("ape_percent") is not None]
    if not matched:return paths
    for grouping in ("operator","shape"):
        items=sorted(ledger["summary"][grouping].items())
        if grouping=="shape":
            def shape_key(item):
                m,n=map(int,item[0].split(" × "))
                return m*n,m,n
            items.sort(key=shape_key)
        rows=[("All cases",ledger["summary"]["overall"]),*items]
        note=("Operators group whole programs and overlap. Only qualified device-matched pairs are plotted."
              if grouping=="operator" else "Concrete M × N shapes ordered by element count. Only qualified device-matched pairs are plotted.")
        fig=ape_table_figure(rows,title=f"Portable core v2: GCoM error by {grouping}",
            subtitle="Device-matched native H100 references · GCoM SASS service timeline at 1.8 GHz",
            note=note,group_label="Operator" if grouping=="operator" else "Shape (M × N)")
        name=f"{prefix}-gcom-v2-{grouping}-ape"
        fig.savefig(assets/(name+".svg"));fig.savefig(assets/(name+".png"),dpi=170);plt.close(fig)
        paths.append(assets/(name+".svg"))
    fig,ax=plt.subplots(figsize=(9,7),facecolor="white")
    for gpu in sorted({r["physical_gpu"] for r in matched}):
        subset=[r for r in matched if r["physical_gpu"]==gpu]
        ax.scatter([r["native_ms"] for r in subset],[r["prediction_ms"] for r in subset],s=15,alpha=.6,label=f"GPU {gpu}")
    values=[r[k] for r in matched for k in ("native_ms","prediction_ms")]
    ax.plot([min(values),max(values)],[min(values),max(values)],color="#637489",ls="--",lw=1,label="equal duration")
    ax.set(xscale="log",yscale="log",xlabel="Device-matched native program span (ms)",
        ylabel="GCoM sequential target service (ms)")
    ax.legend(frameon=False);ax.grid(color="#d9e3ed",alpha=.7,lw=.7)
    for spine in ("top","right"):ax.spines[spine].set_visible(False)
    fig.subplots_adjust(left=.12,right=.97,bottom=.12,top=.83)
    fig.text(.016,.97,"Portable core v2: GCoM vs. H100 durations",fontsize=18,weight="bold",va="top")
    fig.text(.016,.90,"Qualified device-matched pairs · GCoM at 1.8 GHz · modeled host submission gaps are zero",fontsize=10,color="#526578",va="top")
    fig.text(.016,.025,"Each point is one unique case. Native stream spans and modeled service timelines retain their declared timing policy.",fontsize=9,color="#526578")
    name=f"{prefix}-gcom-v2-duration-scatter"
    fig.savefig(assets/(name+".svg"));fig.savefig(assets/(name+".png"),dpi=170);plt.close(fig)
    paths.append(assets/(name+".svg"))
    return paths


def report(campaign):
    ledger=build_ledger(campaign)
    write(campaign.root/"case-ledger.json",ledger)
    fields=["index","case_id","status","physical_gpu","gpu_uuid","native_correct","trace_complete", "replay_admitted",
        "timing_qualified","native_ms","prediction_ms","ape_percent","shape","nodes","state_edges","program_sha256"]
    with (campaign.root/"case-ledger.csv").open("w") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction="ignore");writer.writeheader();writer.writerows(ledger["rows"])
    with (campaign.root/"launch-ledger.csv").open("w") as stream:
        fields=["case_index","gpu_uuid","trace_id","node_id","operator","phase","graph_ordinal","stream","grid","block"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for index,point in campaign.saved_points().items():
            nodes=campaign.manifest["cases"][index]["program"]["nodes"]
            batch=point.get("batch_repetitions",1)
            for ordinal,kernel in enumerate(point.get("trace_info",{}).get("kernels",[])):
                graph=ordinal//len(nodes);node=nodes[ordinal%len(nodes)]
                phase="validation" if graph==0 else "warmup" if graph<=3*batch else "target"
                writer.writerow(dict(case_index=index,gpu_uuid=point["gpu"]["uuid"],trace_id=kernel["trace_id"],
                    node_id=node["id"],operator=node["op"],phase=phase,graph_ordinal=graph,stream=kernel["stream"],
                    grid=json.dumps(kernel["grid"]),block=json.dumps(kernel["block"])))
    write(campaign.root/"gate-scope-ledger.json",dict(timestamp=utc(),formal_gate_status="gate_not_evaluated",
        exact_h3_references=campaign.manifest.get("blocked",{}),suite_profiles=ledger["summary"]["suite_profile"]))
    target=campaign.root/"report-target.json"
    if target.exists():identity=json.loads(target.read_text())
    else:
        prefix=datetime.now().astimezone().strftime("%y%m%d-%H%M")
        identity=dict(prefix=prefix,path=str(campaign.root.parent.parent/"reports"/(prefix+"-gcom-portable-v2-h100-report.md")))
        write(target,identity)
    destination=Path(identity["path"]);destination.parent.mkdir(parents=True,exist_ok=True)
    paths=figures(ledger,destination.parent/"assets",identity["prefix"])
    overall=ledger["summary"]["overall"]
    counts={key:sum(bool(r[key]) for r in ledger["rows"]) for key in ("native_correct","trace_complete","replay_admitted","timing_qualified")}
    history=[]
    if destination.exists() and "## Revision History\n" in destination.read_text():
        history=destination.read_text().split("## Revision History\n",1)[1].strip().splitlines()
    history.append(f"- {datetime.now().astimezone().isoformat(timespec='seconds')}: Recomputed coverage and descriptive scores from saved receipts; no simulator rerun.")
    def number(s,key):return f"{s[key]:.2f}" if s.get(key) is not None else "not evaluable"
    text=["# GCoM portable_core v2 against H100", "",f"Updated: {ledger['timestamp']}","",
        f"**{counts['replay_admitted']} / 1,155 cases have admitted replay; {overall['eligible_cases']} cases qualify for timing comparison.**",
        "", "## User requests",""]
    for prompt in PROMPTS:text.extend(["```text",prompt,"```",""])
    text.extend(["## Coverage and current execution", "", "| Stage | Cases |", "| --- | ---: |"])
    for stage,count in counts.items():text.append(f"| {stage.replace('_',' ')} | {count} / 1,155 |")
    text.extend(["",f"Case states: `{json.dumps(ledger['status_counts'],sort_keys=True)}`.","",
        "The panel covers operation-gate-v2 and both H3 development/smoke families. Exact historical H3 references remain `blocked_missing_source`; formal gate decisions remain `gate_not_evaluated`.","",
        "## Model and measurement contract","",
        "The pinned GCoM source is `122f91afac2155b80fe9f5e5d46d2817d8f09571`. Its H100 80 GB config uses CC 9.0, 132 SMs, 2,048 threads, 64 warps and 32 CTAs per SM, and a 1.8 GHz core clock. Hardware timing parameters retain their shipped values.","",
        "The simulator and tracer were rebuilt in an isolated worktree. Capture uses official NVBit 1.8.2, the documented channel/thread API adapter and documented CUkernel-to-CUfunction resolution for CUDA 13.3 launches. CUDA injection alone is the qualified loader for the unchanged shared-runtime executable. The simulator model source is unchanged.","",
        "Five explicit compatibility overrides disable absent-feature sidecar consumers: `tma_real_base_addr_enable=0`, `tma_operand_addr_tiling_enable=0`, `wait_block_enable=0`, `wait_block_exact_generation_enable=0`, and its dependent observer `wait_block_late_arrival_instrument_enable=0`. Every admitted trace must have a complete static opcode inventory proving absence of applicable synchronization/TMA consumers. These flags are refused for a trace with such instructions. Prior configuration-admission aborts are retained under distinct immutable keys; no successful deterministic replay is repeated.","",
        "Each physical H100 requires 30 clean seconds, a per-UUID lease and continuous foreign-context monitoring. Each hardware child sees its assigned UUID as logical device 0. Three fresh native processes provide 30 samples each on the same card as the trace; historical GPU 7 timing is not transferred to other cards.","",
        "The scored window follows the validation graph and three warmup graphs. GCoM service cycles include its fixed configured CTA launch delay and omit host submission gaps. They are compared descriptively with the matching native stream program span under that declared model policy. CUPTI body timestamps are retained as diagnostics; service cycles are not relabeled kernel-body cycles.","",
        "Native timing eligibility requires a median of at least 10 microseconds and relative MAD at most 5% within and across the three repeat processes. Short/noisy rows stay inconclusive. The separate 64-repetition H3 protocol still requires its own runtime qualification.","",
        "CPU replay uses disjoint NUMA-local slots, excludes observed busy cores and their SMT siblings, and respects CPU quota, memory and disk limits. An 8 GiB disk reserve and verified lossless archives bound storage. Every immutable simulator/config/trace key runs once.","",
        "## Descriptive timing results","",
        f"MdAPE: **{number(overall,'mdape')}%**; P95 APE: **{number(overall,'p95_ape')}%**; eligible cases: **{overall['eligible_cases']} / 1,155**.","",
        "| Suite / profile | Eligible / total | MdAPE (%) | P95 APE (%) | Pearson | Spearman |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"])
    for name,s in sorted(ledger["summary"]["suite_profile"].items()):
        text.append(f"| {name} | {s['eligible_cases']} / {s['total_cases']} | {number(s,'mdape')} | {number(s,'p95_ape')} | {number(s,'pearson')} | {number(s,'spearman')} |")
    text.extend(["", "Unique cases carry one weight; suite aliases and repeated samples do not add weights. Operator-containing groups overlap and measure whole-program durations. P95 uses linear interpolation. Constant or undersized groups have no correlation score.","",
        f"Cross-device anchor policy: {ledger['cross_device_anchors']['pooled_score_status']}. Per-GPU scores and anchor distributions are retained in the case ledger.","",
        "## Short H3 batch lane","",
        f"The separate 64-repetition realization has {ledger['summary']['h3_batch']['eligible_cases']} eligible batch rows out of {ledger['summary']['h3_batch']['total_cases']} recorded outcomes. MdAPE: {number(ledger['summary']['h3_batch'],'mdape')}; P95 APE: {number(ledger['summary']['h3_batch'],'p95_ape')}. These rows have independent native qualification and do not change single-invocation eligibility.","",
        "## Figures",""])
    if paths:
        for path in paths:text.extend([f"![{path.stem}](assets/{path.name})",""])
    else:text.extend(["No eligible matched timing rows exist yet; MdAPE, P95 and scatter figures will be generated when qualified replay receipts arrive.",""])
    text.extend(["## Evidence and reproduction", "",
        "- [Execution admission](../local_artifacts/gcom-portable-v2/execution-admission.json)",
        "- [Case ledger JSON](../local_artifacts/gcom-portable-v2/case-ledger.json)",
        "- [Case ledger CSV](../local_artifacts/gcom-portable-v2/case-ledger.csv)",
        "- [Execution tracker](../.plan/tracker/261125-1652-gcom-portable-v2-h100.md)","",
        "From `/home/cliu/wk/amora`, reparse saved evidence with:","", "```bash",
        "PYTHONPATH=. /home/cliu/wk/gpu-perf/local_artifacts/gcom-portable-v2/execution-env/bin/python \\",
        "  -m amora.benchmarking.portable_v2.gcom_campaign report \\",
        "  --root /home/cliu/wk/gpu-perf/local_artifacts/gcom-portable-v2", "```", "",
        "## References","",
        "- Rodrigo Huerta, Mojtaba Abaie Shoushtary, José-Lorenzo Cruz and Antonio González. *Dissecting and Modeling the Architecture of Modern GPU Cores*. MICRO 2025.",
        "- Rodrigo Huerta and Antonio González. *GPU Simulation Acceleration via Parallelization*. ISPASS 2025; and *Parallelizing a modern GPU simulator*, [arXiv:2401.10082](https://arxiv.org/abs/2401.10082).",
        "- Mahmoud Khairy, Zhensheng Shen, Tor M. Aamodt and Timothy G. Rogers. *Accel-Sim: An Extensible Simulation Framework for Validated GPU Modeling*. ISCA 2020.",
        "- [GCoM source README and citation requirements](../../modern-gpu-simulator-micro-2025/README.md).","",
        "## Revision History","",*history,""])
    destination.write_text("\n".join(text))
    tracker=campaign.root.parent.parent/".plan/tracker/261125-1652-gcom-portable-v2-h100.md"
    if tracker.exists():
        body=tracker.read_text()
        state_path=campaign.root/"execution-status.json"
        state=json.loads(state_path.read_text()) if state_path.exists() else {}
        state_name=state.get("state","runtime pilot queued")
        status=f"- Status: {state_name}; {counts['replay_admitted']} / 1,155 replay-admitted; {counts['timing_qualified']} eligible timing rows."
        body=re.sub(r"- Status:.*?(?=\n\n##)",status,body,count=1,flags=re.S)
        live="\n".join(["## Live execution","",f"Updated: {ledger['timestamp']}","",
            f"- Controller state: `{state_name}`; PID `{state.get('pid','not recorded')}`.",
            f"- Native correct / trace complete / replay admitted / timing eligible: `{counts['native_correct']} / {counts['trace_complete']} / {counts['replay_admitted']} / {counts['timing_qualified']}` of 1,155 unique cases.",
            "- [Runtime state](../../local_artifacts/gcom-portable-v2/execution-status.json), [case ledger](../../local_artifacts/gcom-portable-v2/case-ledger.json) and [launch ledger](../../local_artifacts/gcom-portable-v2/launch-ledger.csv).",
            f"- [Current report](../../reports/{destination.name}).","",
            ""])
        if "## Live execution\n" in body:body=re.sub(r"## Live execution\n.*?(?=## Revision History)",live,body,flags=re.S)
        else:body=body.replace("## Revision History\n",live+"## Revision History\n")
        body=body.rstrip()+f"\n- {datetime.now().astimezone().isoformat(timespec='seconds')}: Refreshed live execution and report counts from saved receipts; controller state {state_name}; no identical replay retry.\n"
        tracker.write_text(body)
    print(destination,flush=True)
    return ledger
