"""Campaign scheduling and case-level accounting for portable v2 CUDA/FlashGPU validation."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import argparse
import json
from pathlib import Path
import statistics

from .corpus import prepare_corpus, verify_corpus, write_json
from amora.backends.nvidia.portable_core_v2 import build_driver, pilot_indices, run_group

def build(root, cuda_root):
    result = build_driver((1<<31)-1, build_root=root/"build", evidence_root=root,
                          cuda_root=cuda_root)
    path = root/"builds.json"
    if path.exists():
        old = json.loads(path.read_text())
        if old["all"]["realization_sha256"] != result["realization_sha256"]:
            raise ValueError("existing campaign uses a different executable realization")
        return old
    value = {"all": result}
    write_json(path,value)
    return value

def execute(root, *, mode, scope="full", repeats=1, samples=1, warmup=0,
            sim_root, cuda_root, gpu="7", workers=8, chunk_size=32, timeout=1200,
            resume=False, case_indices=None, run_tag="", omp_threads=2):
    manifest = verify_corpus(root/"corpus")
    builds = build(root,cuda_root)
    selected = pilot_indices(manifest) if scope == "pilot" else list(range(len(manifest["cases"])))
    if case_indices is not None:
        if len(case_indices)!=len(set(case_indices)) or not set(case_indices)<=set(selected):
            raise ValueError("invalid restricted case selection")
        selected=list(case_indices)
    if mode != "native":
        successful = set()
        for path in (root/"native").glob("*.validation.json"):
            for row in json.loads(path.read_text())["rows"]:
                if row["passed"] and row["executable_sha256"] == builds["all"]["binary_sha256"]:
                    successful.add(row["index"])
        # Manual initial pilot is also a valid correctness qualification.
        path=root/"native-pilot.validation.json"
        if path.exists():
            for row in json.loads(path.read_text())["rows"]:
                if row["passed"] and row["executable_sha256"] == builds["all"]["binary_sha256"]:
                    successful.add(row["index"])
        if not set(selected) <= successful:
            raise ValueError("simulator selection lacks accepted native correctness")
    output=root/("native" if mode=="native" else "simulation")
    output.mkdir(exist_ok=True)
    tasks=[]
    for repeat in range(repeats):
        pieces = [selected] if mode=="native" else [selected[i:i+chunk_size] for i in range(0,len(selected),chunk_size)]
        for number,indices in enumerate(pieces):
            label=f"{mode}-{scope}-"+(run_tag+"-" if run_tag else "")+f"r{repeat}-s{number:04d}"
            tasks.append((label,indices))
    results=[]
    def run(label,indices):
        path=output/(label+".validation.json")
        if path.exists() and resume:
            return json.loads(path.read_text())
        return run_group(indices, manifest=manifest, builds=builds, corpus_root=root/"corpus",
                         evidence_root=output, sim_root=sim_root, cuda_root=cuda_root,
                         mode=mode,label=label,gpu=gpu,samples=samples,warmup=warmup,
                         timeout=timeout,omp_threads=omp_threads)
    with ThreadPoolExecutor(max_workers=1 if mode=="native" else workers) as pool:
        pending={pool.submit(run,label,indices):label for label,indices in tasks}
        for future in as_completed(pending):
            result=future.result();results.append(result)
            print(result["label"],"completed",sum(r["passed"] for r in result["rows"]),"/",len(result["rows"]),flush=True)
    # A crash cannot silently hide later cases: retry each uncompleted case alone.
    if mode!="native":
        interrupted={r["index"] for x in results for r in x["rows"] if r.get("status")=="not_completed"}
        if interrupted:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                pending={pool.submit(run,f"{mode}-{scope}-"+(run_tag+"-" if run_tag else "")+f"isolated-{i:04d}",[i]):i for i in sorted(interrupted)}
                for future in as_completed(pending):
                    result=future.result();results.append(result)
                    print(result["label"],"isolated",sum(r["passed"] for r in result["rows"]),flush=True)
    summary=dict(mode=mode,scope=scope,selected=len(selected),
                 run_labels=[x["label"] for x in results])
    write_json(output/(f"{mode}-{scope}"+("-"+run_tag if run_tag else "")+"-summary.json"),summary)
    return summary

def report(root):
    manifest=verify_corpus(root/"corpus")
    native={}
    for path in sorted((root/"native").glob("*.validation.json")):
        if "pilot" in path.name:
            continue
        for row in json.loads(path.read_text())["rows"]:
            native.setdefault(row["index"],[]).append(dict(row, source_validation=str(path.relative_to(root))))
    simulation={}
    observations=defaultdict(list)
    for path in sorted((root/"simulation").glob("*.validation.json")):
        document=json.loads(path.read_text())
        for row in document["rows"]:
            key=row["mode"],row["index"]
            candidate=dict(row, source_validation=str(path.relative_to(root)),
                           started_utc=document.get("receipt",{}).get("started_utc",""))
            observations[key].append(candidate)
            # Completion outranks interruption, even when the retry is newer.
            # Among equally complete observations use a deterministic date/path order.
            def priority(value):
                return (value.get("status")!="not_completed", value["started_utc"], value["source_validation"])
            if key not in simulation or priority(candidate)>priority(simulation[key]):
                simulation[key]=candidate
    rows=[]
    for case in manifest["cases"]:
        index=case["index"];runs=native.get(index,[])
        item=dict(index=index,case_id=case["id"],family=case["family"],shape=case["shape"],
                  aliases=case["aliases"],case_sha256=case["case_sha256"],
                  program_sha256=case["program_sha256"],
                  operators=sorted({n["op"] for n in case["program"]["nodes"]}),
                  node_count=len(case["program"]["nodes"]),
                  state_edge_count=len(case["program"]["state_edges"]),
                  unroll=case["program"]["unroll"],
                  native_observations=runs,
                  native_runs=len(runs),native_passed=bool(runs) and all(r["passed"] for r in runs))
        medians=[];quality=[]
        for run in runs:
            values=run.get("samples_ms",[])
            if not values:
                quality.append(False);continue
            median=statistics.median(values)
            mad=statistics.median(abs(v-median) for v in values)
            quality.append(len(values)>=30 and median>=0.01 and mad/median<=0.05)
            medians.append(median)
        if medians:
            median=statistics.median(medians);mad=statistics.median(abs(x-median) for x in medians)
            item.update(native_median_ms=median,native_repeat_medians_ms=medians,
                        native_timing_qualified=item["native_passed"] and len(runs)>=3 and all(quality)
                        and mad/median<=0.05)
        for mode in ("ptx","ptx-functional","sass-timing"):
            sim=simulation.get((mode,index))
            if sim:
                samples=[] if mode=="ptx-functional" else sim.get("samples_ms",[])
                item[mode]=dict(passed=sim["passed"],status=sim.get("status","completed"),
                               samples_ms=samples,source_validation=sim["source_validation"],
                               observations=observations[mode,index])
                if mode!="ptx-functional" and sim["passed"] and samples and min(samples)>0 and item.get("native_timing_qualified"):
                    simulated=statistics.median(samples)
                    item[mode].update(simulated_span_ms=simulated,
                                      difference_percent=(simulated/item["native_median_ms"]-1)*100)
        rows.append(item)
    value=dict(schema="portable-v2-flashgpu-comparison-v1",cases=len(rows),
               blocked_reference_profiles=manifest["blocked"],
               clock_policy="Hardware free-running; stock model frequency unchanged. Program spans include stream submission gaps.",
               timing_quality="3 independent native runs, 30 samples each, median >=10us, relative MAD <=5% within and across repeats",
               rows=rows)
    value["summary"]=summarize(rows)
    path=root/"comparison.json"
    path.write_text(json.dumps(value,indent=2)+"\n")
    return value

def summarize(rows):
    """Describe coverage and program-span differences without fitting the model."""
    def percentile(values, fraction):
        values=sorted(values)
        position=(len(values)-1)*fraction
        low=int(position);high=min(low+1,len(values)-1)
        return values[low]+(values[high]-values[low])*(position-low)
    def panel(items):
        result=dict(cases=len(items),native_passed=sum(r["native_passed"] for r in items),
                    native_timing_qualified=sum(bool(r.get("native_timing_qualified")) for r in items),
                    ptx_passed=sum(bool(r.get("ptx",{}).get("passed")) for r in items),
                    functional_passed=sum(bool(r.get("ptx-functional",{}).get("passed")) for r in items))
        paired=[r for r in items if "difference_percent" in r.get("ptx",{})]
        result["eligible_comparisons"]=len(paired)
        if paired:
            ratios=[r["ptx"]["simulated_span_ms"]/r["native_median_ms"] for r in paired]
            errors=[(x-1)*100 for x in ratios]
            absolute=[abs(x) for x in errors]
            result.update(median_ratio=statistics.median(ratios),
                          p05_ratio=percentile(ratios,.05),p95_ratio=percentile(ratios,.95),
                          min_ratio=min(ratios),max_ratio=max(ratios),
                          median_signed_difference_percent=statistics.median(errors),
                          median_absolute_difference_percent=statistics.median(absolute),
                          p95_absolute_difference_percent=percentile(absolute,.95),
                          mean_absolute_difference_percent=statistics.mean(absolute),
                          within_25_percent=sum(abs(x)<=25 for x in errors),
                          within_factor_two=sum(.5<=x<=2 for x in ratios),
                          simulated_slower=sum(x>1 for x in ratios),
                          median_native_us=statistics.median(r["native_median_ms"]*1000 for r in paired),
                          median_simulated_us=statistics.median(r["ptx"]["simulated_span_ms"]*1000 for r in paired))
        return result
    groups={name:defaultdict(list) for name in ("shape","operator","family","node_count","state_edges","unroll")}
    for row in rows:
        groups["shape"][json.dumps(row["shape"],sort_keys=True)].append(row)
        groups["family"][row["family"]].append(row)
        groups["node_count"][str(row["node_count"])].append(row)
        groups["state_edges"][str(row["state_edge_count"])].append(row)
        groups["unroll"][str(row["unroll"])].append(row)
        for op in row["operators"]:
            groups["operator"][op].append(row)
    result=dict(overall=panel(rows),operator_grouping="Cases containing each operator; groups overlap.",
                percentile_method="Linear interpolation on sorted case values; cases equally weighted.")
    for name,values in groups.items():
        result["by_"+name]={key:panel(items) for key,items in sorted(values.items())}
    return result

def configure_parser(parser):
    parser.add_argument("action",choices=("prepare","build","run","report"))
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--sim-root",type=Path,default=Path("../FlashGPU-Sim"))
    parser.add_argument("--cuda-root",type=Path,default=Path("/usr/local/cuda-13.3"))
    parser.add_argument("--mode",choices=("native","ptx","ptx-functional","sass-timing"),default="ptx")
    parser.add_argument("--scope",choices=("pilot","full"),default="full")
    parser.add_argument("--gpu",default="7")
    parser.add_argument("--repeats",type=int,default=1)
    parser.add_argument("--samples",type=int,default=1)
    parser.add_argument("--warmup",type=int,default=0)
    parser.add_argument("--workers",type=int,default=8)
    parser.add_argument("--chunk-size",type=int,default=32)
    parser.add_argument("--timeout",type=int,default=1200)
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--selection-indices",type=Path)
    parser.add_argument("--run-tag",default="")
    parser.add_argument("--omp-threads",type=int,default=2)

def dispatch(args):
    root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.action=="prepare":
        value=prepare_corpus(root/"corpus")
        print(json.dumps(dict(cases=len(value["cases"]),blocked=value["blocked"]),indent=2))
    elif args.action=="build":
        print(json.dumps(build(root,args.cuda_root),indent=2))
    elif args.action=="report":
        value=report(root);print(json.dumps(dict(cases=value["cases"],native_passed=sum(r["native_passed"] for r in value["rows"])),indent=2))
    else:
        value=execute(root,mode=args.mode,scope=args.scope,repeats=args.repeats,
                      samples=args.samples,warmup=args.warmup,sim_root=args.sim_root,
                      cuda_root=args.cuda_root,gpu=args.gpu,workers=args.workers,
                      chunk_size=args.chunk_size,timeout=args.timeout,resume=args.resume,
                      case_indices=json.loads(args.selection_indices.read_text()) if args.selection_indices else None,
                      run_tag=args.run_tag,omp_threads=args.omp_threads)
        print(json.dumps(value,indent=2))
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    return dispatch(parser.parse_args(argv))

if __name__=="__main__":
    raise SystemExit(main())

