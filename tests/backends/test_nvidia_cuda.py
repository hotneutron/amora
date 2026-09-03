import subprocess

from amora.backends.nvidia import cuda
from amora.backends.nvidia.cuda import NvidiaCapabilities, NvidiaDevice, ToolStatus
from amora.backends.nvidia.metrics import MetricResolver
from amora.backends.nvidia.stall_metrics import (
    resolve_stall_metric_family,
    resolve_stall_cycle_metrics,
    select_stall_launch_row,
)


def test_capabilities_to_dict_contains_tools_and_devices():
    caps = NvidiaCapabilities(
        cuda_available=True,
        gpu_available=True,
        tools={"nvcc": ToolStatus("nvcc", "/usr/bin/nvcc", True, "version")},
        devices=[NvidiaDevice(index=0, name="GPU", uuid="uuid", driver_version="driver")],
    )

    data = caps.to_dict()

    assert data["cuda_available"] is True
    assert data["gpu_available"] is True
    assert data["tools"]["nvcc"]["available"] is True
    assert data["devices"][0]["name"] == "GPU"


def test_discover_tool_prefers_specific_multiline_version(monkeypatch):
    monkeypatch.setattr(cuda.shutil, "which", lambda name: f"/opt/{name}")
    monkeypatch.setattr(
        cuda,
        "_run",
        lambda args: subprocess.CompletedProcess(
            args,
            0,
            (
                "NVIDIA (R) Nsight Compute Command Line Profiler\n"
                "Copyright (c) NVIDIA Corporation\n"
                "Version 2026.2.1.0 (build 38283040) (public-release)\n"
            ),
            "",
        ),
    )

    status = cuda.discover_tool("ncu", ["--version"])

    assert status.version == (
        "Version 2026.2.1.0 (build 38283040) (public-release)"
    )


def test_metric_resolver_selects_first_supported_candidate():
    resolver = MetricResolver(
        frozenset({"smsp__inst_executed.sum", "sm__cycles_active.avg"})
    )

    resolution = resolver.resolve("inst_executed")

    assert resolution.available is True
    assert resolution.selected_name == "smsp__inst_executed.sum"


def test_metric_resolver_reports_missing_metric():
    resolver = MetricResolver(frozenset())

    resolution = resolver.resolve("shared_conflicts")

    assert resolution.available is False
    assert resolution.reason == "no candidate metric supported"


def test_stall_metric_selection_prefers_one_family_and_aliases_no_instruction():
    supported = frozenset(
        {
            "smsp__warp_issue_stalled_wait_per_warp_active",
            "smsp__warp_issue_stalled_no_instruction_per_warp_active",
            "smsp__average_warps_issue_stalled_selected_per_issue_active",
        }
    )

    selection = resolve_stall_metric_family(supported)

    assert selection.family == "warp_issue_stalled_per_warp_active"
    assert selection.unit == "pct"
    assert selection.reason_to_metric == {
        "wait": "smsp__warp_issue_stalled_wait_per_warp_active.pct",
        "no_instructions": "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    }
    assert "selected" in selection.missing_reasons


def test_stall_metric_selection_maps_hopper_gmma_to_warpgroup_arrive():
    selection = resolve_stall_metric_family(
        frozenset(
            {
                "smsp__warp_issue_stalled_wait_per_warp_active",
                "smsp__warp_issue_stalled_gmma_per_warp_active",
            }
        )
    )

    assert selection.reason_to_metric["warpgroup_arrive"] == (
        "smsp__warp_issue_stalled_gmma_per_warp_active.pct"
    )
    assert "warpgroup_arrive" not in selection.missing_reasons


def test_stall_cycle_resolution_never_uses_per_issue_active_family():
    selection = resolve_stall_cycle_metrics(
        frozenset(
            {
                "smsp__warps_active",
                "smsp__cycles_active",
                "smsp__warps_eligible",
                "smsp__warp_issue_stalled_wait_per_warp_active",
                "smsp__average_warps_issue_stalled_barrier_per_issue_active",
            }
        )
    )

    assert selection.available is True
    assert selection.reason_to_metric == {
        "wait": "smsp__warp_issue_stalled_wait_per_warp_active.ratio"
    }
    assert "barrier" in selection.missing_reasons


def test_select_stall_launch_row_uses_whole_median_vector():
    reason_to_metric = {
        "wait": "stall_wait",
        "selected": "stall_selected",
    }
    rows = [
        {"stall_wait": "1", "stall_selected": "1"},
        {"stall_wait": "100", "stall_selected": "0"},
        {"stall_wait": "4", "stall_selected": "2"},
    ]

    index, stalls, total = select_stall_launch_row(rows, reason_to_metric)

    assert index == 2
    assert stalls == {"wait": 4.0, "selected": 2.0}
    assert total == 6.0


def test_discover_ncu_metrics_falls_back_to_list_metrics_and_preserves_query_error(monkeypatch):
    calls = []

    def fake_run(args, *, timeout=10):
        calls.append(args)
        if args[1] == "--query-metrics":
            return subprocess.CompletedProcess(
                args,
                returncode=1,
                stdout="",
                stderr="==ERROR== ERR_NVGPUCTRPERM - no permission\n",
            )
        return subprocess.CompletedProcess(
            args,
            returncode=0,
            stdout=(
                "sm__cycles_active.avg\n"
                " -sm__cycles_active.avg\n"
                " -arch:40:70:gpu__time_duration.sum\n"
                "breakdown:sm__throughput.avg.pct_of_peak_sustained_elapsed\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(cuda, "_run", fake_run)

    metrics, error = cuda._discover_ncu_metrics(ToolStatus("ncu", "/usr/bin/ncu", True))

    assert calls == [["/usr/bin/ncu", "--query-metrics"], ["/usr/bin/ncu", "--list-metrics"]]
    assert error == "==ERROR== ERR_NVGPUCTRPERM - no permission"
    assert "sm__cycles_active.avg" in metrics
    assert "gpu__time_duration.sum" in metrics
    assert "sm__throughput.avg.pct_of_peak_sustained_elapsed" in metrics


def test_discover_ncu_metrics_uses_offline_chip_query_for_v100(monkeypatch):
    calls = []

    def fake_run(args, *, timeout=10):
        calls.append(args)
        return subprocess.CompletedProcess(
            args,
            returncode=0,
            stdout=(
                "Chip gv100\n"
                "Metric Name  Metric Type\n"
                "sm__inst_executed  Counter\n"
                "sm__cycles_elapsed  Counter\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(cuda, "_run", fake_run)

    metrics, error = cuda._discover_ncu_metrics(
        ToolStatus("ncu", "/usr/bin/ncu", True),
        [NvidiaDevice(index=0, name="Tesla V100-SXM2-32GB")],
    )

    assert calls == [
        ["/usr/bin/ncu", "--query-metrics", "--chips", "gv100", "--query-metrics-mode", "base"]
    ]
    assert error is None
    assert "sm__inst_executed" in metrics
    assert "sm__cycles_elapsed" in metrics
