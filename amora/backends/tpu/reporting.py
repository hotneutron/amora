"""Layered TPU results and collision-free generation/instance/runtime reports."""
from __future__ import annotations

import json
import os
from pathlib import Path

from amora.reports.json_report import render_report
from amora.schemas.evidence import EvidenceTier, FitStatus, UncertaintyCategory
from amora.schemas.results import (BackendInterpretation, LaunchDescriptor,
    NormalizedMeasurement, ProbeIdentity, ProbeResult, RawObservation, ToolContext)

from .artifacts import digest, file_hash, identifier, validate_manifest, write_json


def render_results(rows, caps, recipe, compiler_artifacts):
    results = []
    for row in rows:
        key = row["probe_id"]
        if row["status"] in ("failed", "unsupported"):
            result = ProbeResult.unsupported(key, row.get("reason", "execution_failed"),
                backend="tpu_jax", tool_context=ToolContext(tools=caps, device=caps["selected_device"]),
                raw_values={"case_id": row["case_id"], "status": row["status"]})
        else:
            metadata = row.get("metadata", {})
            timing = row.get("timing")
            is_metadata = key == "topology.device_attributes"
            value = (caps["target"] if is_metadata else
                     timing["median_ns_per_invocation"] if timing else row.get("correctness"))
            result = ProbeResult(
                identity=ProbeIdentity(key, backend="tpu_jax", source_hash=metadata.get("case_sha256", metadata.get("probe_source_sha256"))),
                tool_context=ToolContext(device=caps["selected_device"], tools=caps),
                launch=LaunchDescriptor(mode=metadata.get("mode", "metadata"), extras={
                    **metadata, "compiler_artifacts": compiler_artifacts.get(row["case_id"], {})}),
                raw_observation=RawObservation(
                    EvidenceTier.DIRECT_METADATA if is_metadata else EvidenceTier.TIMING_DIRECT if timing
                    else EvidenceTier.DIRECT_METADATA,
                    values=row, units={"dispatch_to_completion_ns": "ns"} if timing else {},
                    source="PJRT metadata" if is_metadata else "synchronized host timing" if timing
                    else "compiled correctness"),
                normalized_measurement=NormalizedMeasurement(
                    name=key if is_metadata else "dispatch_to_completion_ns" if timing else "correctness",
                    value=value, unit="ns" if timing else None,
                    fit_status=FitStatus.DIRECT if row["status"] == "passed" else FitStatus.BEHAVIORAL_ONLY,
                    uncertainty=UncertaintyCategory.STABLE_SCALAR if timing and timing["stable"] else
                    UncertaintyCategory.BEHAVIORAL_CLASS,
                    variance=timing.get("summary", {}) if timing else {},
                    assumptions=[] if is_metadata else ["includes host dispatch and completion overhead",
                        "IR/correctness do not prove ISA instruction counts or physical memory traffic"]),
                backend_interpretation=BackendInterpretation(key, interpretation={
                    "generation": caps["architecture"]["generation"], "case_id": row["case_id"],
                    "status": row["status"], "time_domain": "synchronized_host" if timing else None}),
            )
        results.append(result)
    return render_report(results, metadata={"backend_capabilities": caps, "recipe": recipe,
        "report_identity": {"vendor": "google_tpu", "family": caps["architecture"]["generation"],
            "sku": caps["architecture"]["generation"],
            "target_config_digest": caps["target"]["target_config_digest"],
            "runtime_digest": caps["runtime"]["runtime_digest"]}})


def write_tpu_report(report, out_dir="reports", *, family=None, sku=None, run_id=None, source_report=None):
    from amora.reports.markdown_report import _build_probes_doc, _build_family_environment
    qualification = report["metadata"].get("qualification")
    if qualification and qualification["status"] != "passed":
        raise ValueError("cannot publish an unqualified run to curated TPU reports")
    identity = report["metadata"]["report_identity"]
    if family and identity["family"] != "unknown" and family != identity["family"]:
        raise ValueError("profile_conflict: report family override")
    family = identifier(family or identity["family"])
    sku = identifier(sku or identity["sku"])
    target = identifier(identity["target_config_digest"])
    runtime = identifier(identity["runtime_digest"])
    if source_report is not None:
        source_report=Path(source_report)
        source_manifest=source_report.parent/"manifest.json"
        manifest=validate_manifest(source_manifest)
        if manifest["status"]!="passed":raise ValueError("source run is not qualified")
        if manifest["artifacts"].get(source_report.name)!=file_hash(source_report):
            raise ValueError("source report is not bound by the run manifest")
        report={**report,"metadata":{**report["metadata"],"source_evidence":{
            "root":os.path.relpath(source_report.parent,Path.cwd()),
            "manifest_sha256":file_hash(source_manifest),"run_digest":manifest["run_digest"],
            "report_sha256":file_hash(source_report),
            "artifact_paths_relative_to":"root for launch descriptors and run manifest references"}}}
    run_id = identifier(run_id or digest(report)[:20])
    if family != identity["family"] or sku != identity["sku"]:
        report = {**report, "metadata": {**report["metadata"],
            "report_label_overrides": {"family":family,"sku":sku,"source":"explicit CLI labels"}}}
    root = Path(out_dir) / "google_tpu" / family
    run_dir = root / target / runtime / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "report.json", report)
    destination = run_dir / f"probes-{sku}.md"
    destination.write_text(_build_probes_doc(report, vendor="google_tpu", family=family,
                                            sku=sku, generated_at=report["metadata"].get("generated_at", run_id)))
    if source_report is not None:
        with destination.open("a") as handle:
            handle.write(f"\n## Source evidence\n\n- Run digest: `{manifest['run_digest']}`\n"
                         f"- [Verified source manifest]({os.path.relpath(source_manifest,run_dir)})\n"
                         "- Compiler references in launch descriptors and the run manifest are relative to the source evidence root.\n")
    (run_dir / "environment.md").write_text(_build_family_environment(
        {"skus": {sku: {"capabilities": report["metadata"]["backend_capabilities"]}}},
        vendor="google_tpu", family=family))
    (run_dir/"README.md").write_text(
        f"# TPU {family} qualification run `{run_id}`\n\n"
        f"- [Family run index](../../../README.md)\n- [Probe report]({destination.name})\n"
        "- [Environment](environment.md)\n- [JSON report](report.json)\n")
    index_path = root / "manifest.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"schema_version": 1, "runs": {}}
    composite = "/".join((target, runtime, run_id))
    index["runs"][composite] = {**identity, "report": str(destination.relative_to(root))}
    index_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
    lines = [f"# TPU {family} qualification reports", "", "| target / runtime / run | report |",
             "|---|---|"]
    lines.extend(f"| `{key}` | [probes]({value['report']}) |" for key,value in sorted(index["runs"].items()))
    (root / "README.md").write_text("\n".join(lines) + "\n")
    return destination
