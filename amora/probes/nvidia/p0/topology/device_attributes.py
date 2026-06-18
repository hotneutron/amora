"""CUDA device-attribute metadata probe."""

from __future__ import annotations

from amora.backends.nvidia import CudaToolchain
from amora.schemas.results import EvidenceTier, ParameterEstimate, ProbeResult


ATTRIBUTE_TO_PARAMETER = {
    "multiprocessor_count": "gpgpu_sim_config::num_shader()",
    "warp_size": "shader_core_config::warp_size",
    "max_threads_per_multiprocessor": "shader_core_config::n_thread_per_shader",
    "max_blocks_per_multiprocessor": "shader_core_config::max_cta_per_core",
    "regs_per_multiprocessor": "shader_core_config::gpgpu_shader_registers",
    "shared_memory_per_multiprocessor": "shader_core_config::gpgpu_shmem_size",
    "shared_memory_per_block": "shader_core_config::gpgpu_shmem_per_block",
}


def estimates_from_attributes(attributes: dict[str, object]) -> list[ParameterEstimate]:
    """Map CUDA-visible attributes onto simulator parameter names."""

    estimates: list[ParameterEstimate] = []
    for attr_name, parameter_name in ATTRIBUTE_TO_PARAMETER.items():
        if attr_name not in attributes:
            continue
        estimates.append(
            ParameterEstimate(
                name=parameter_name,
                value=attributes[attr_name],
                evidence=EvidenceTier.DIRECT_METADATA,
                confidence=0.95,
                risk="low",
                notes=(f"Derived from CUDA attribute {attr_name}.",),
            )
        )
    return estimates


def device_attribute_probe(
    attributes: dict[str, object] | None = None,
    *,
    toolchain: CudaToolchain | None = None,
) -> ProbeResult:
    """Create a device-attribute result or a dry-run capability result."""

    tools = (toolchain or CudaToolchain()).target_summary()
    if attributes is None:
        return ProbeResult(
            name="topology/device_attributes.py",
            tier="P0",
            status="dry_run",
            measurements={"toolchain": tools},
            warnings=[
                "Runtime CUDA attribute collection is not available in this "
                "pure-Python path; use the generated CUDA probe binary on a "
                "CUDA host."
            ],
        )

    return ProbeResult(
        name="topology/device_attributes.py",
        tier="P0",
        status="ok",
        measurements={"attributes": attributes, "toolchain": tools},
        estimates=estimates_from_attributes(attributes),
    )
