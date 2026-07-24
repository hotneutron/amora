import subprocess

from amora.backends.nvidia import cuda
from amora.backends.nvidia.cuda import NvidiaCapabilities, NvidiaDevice, ToolStatus
from amora.backends.nvidia.metrics import MetricResolver


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
