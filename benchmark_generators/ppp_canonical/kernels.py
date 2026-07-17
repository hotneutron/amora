"""Canonical PPP kernel metadata used by the AMORA benchmark generator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KernelSpec:
    """One canonical kernel family and its execution contract."""

    kernel_id: str
    primary_resource: str
    shape_dimensions: tuple[str, ...]
    kernel_revision: int = 1
    measurement_semantics: str = "kernel_cycles"
    kernel_name_hw: str | None = None


CANONICAL_KERNELS: tuple[KernelSpec, ...] = (
    KernelSpec(
        "aligned_gemm_fp16",
        "compute_matrix",
        ("M", "N", "K"),
        kernel_name_hw="gemm",
    ),
    KernelSpec(
        "embedding",
        "transfer_off_chip_to_on_chip_small",
        ("B", "S", "D", "V", "U"),
        kernel_name_hw="embedding_lookup",
    ),
    KernelSpec(
        "flash_attention_fwd",
        "compute_matrix",
        ("B", "H", "S", "D"),
        measurement_semantics="bounded_sampled_attention",
        kernel_name_hw="flash_attention_fwd_kernel",
    ),
    KernelSpec(
        "flashmla_dense_decode",
        "compute_matrix",
        ("B", "S", "H"),
        measurement_semantics="bounded_sampled_attention",
        kernel_name_hw="flashmla_dense_decode_kernel",
    ),
    KernelSpec(
        "gelu",
        "compute_transcendental",
        ("N",),
        kernel_name_hw="gelu_kernel",
    ),
    KernelSpec(
        "gelu_gemm_fp16",
        "compute_matrix",
        ("M", "N", "K"),
        measurement_semantics="fused_component_aggregate",
    ),
    KernelSpec(
        "megamoe_fp8",
        "compute_matrix",
        ("B", "S", "D", "E"),
        measurement_semantics="bounded_sampled_moe",
        kernel_name_hw="megamoe_fp8_kernel",
    ),
    KernelSpec(
        "rmsnorm",
        "compute_vector",
        ("B", "N", "D"),
        kernel_name_hw="rmsnorm_kernel",
    ),
    KernelSpec(
        "rmsnorm_gemm_fp16",
        "compute_matrix",
        ("M", "N", "K"),
        measurement_semantics="fused_component_aggregate",
    ),
)

KERNELS_BY_ID = {spec.kernel_id: spec for spec in CANONICAL_KERNELS}
