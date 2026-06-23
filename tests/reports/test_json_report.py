from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.arithmetic_latency.dependent_chain import run as run_dependent_chain
from amora.reports.json_report import render_report


def test_render_report_replaces_duplicate_tool_context_with_ref():
    caps = NvidiaCapabilities(cuda_available=False, gpu_available=False)
    results = run_dependent_chain(caps)
    metadata = {"backend_capabilities": caps.to_dict()}

    report = render_report(results, metadata=metadata)

    assert report["schema_version"] == 1
    assert report["metadata"]["backend_capabilities"]["backend"] == "nvidia_cuda"
    assert len(report["results"]) == 1
    tool_context = report["results"][0]["tool_context"]
    assert tool_context["tools"] == {"$ref": "metadata.backend_capabilities"}


def test_render_report_keeps_distinct_tool_context_inline():
    # When a probe emits a tool_context that does not match the metadata snapshot,
    # the report should leave it inline rather than collapsing to a $ref.
    caps = NvidiaCapabilities(cuda_available=False, gpu_available=False)
    results = run_dependent_chain(caps)
    metadata = {"backend_capabilities": {"backend": "different_backend"}}

    report = render_report(results, metadata=metadata)

    tool_context = report["results"][0]["tool_context"]
    assert tool_context["tools"] != {"$ref": "metadata.backend_capabilities"}
