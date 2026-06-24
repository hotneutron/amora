from pathlib import Path

from amora.backends.nvidia.build import CudaBuildConfig, default_output_for


def test_cuda_build_config_constructs_nvcc_command():
    config = CudaBuildConfig(
        source=Path("probe.cu"),
        output=Path("probe.cubin"),
        arch="sm_90",
        extra_flags=("--ptxas-options=-v",),
    )

    assert config.argv() == [
        "nvcc",
        "-arch",
        "sm_90",
        "-cubin",
        "probe.cu",
        "-o",
        "probe.cubin",
        "--ptxas-options=-v",
    ]


def test_default_output_for_uses_build_root():
    assert default_output_for("amora/probes/foo.cu") == Path("out/build/nvidia/baseline/foo.cubin")
