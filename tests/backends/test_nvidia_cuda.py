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
