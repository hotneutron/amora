"""Local hardware replay contracts for canonical PPP benchmark cases."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from amora.benchmarking.schema import BenchmarkCase


TEMPLATE_ROOT = Path(__file__).resolve().parent / "templates"


@dataclass(frozen=True)
class ReplayContract:
    """How one benchmark case is built and selected under NCU."""

    source: Path
    args: tuple[str, ...]
    launch_skip: int
    launch_count: int = 1
    link_flags: tuple[str, ...] = ()


def contract_for_case(case: BenchmarkCase) -> ReplayContract:
    """Return the AMORA-owned replay contract for a canonical PPP case."""

    shape = case.shape
    kernel_id = case.kernel_id
    if kernel_id == "aligned_gemm_fp16":
        return ReplayContract(
            TEMPLATE_ROOT / "gemm.cu",
            tuple(str(shape[name]) for name in ("M", "N", "K")),
            launch_skip=1,
            link_flags=("-lcublas",),
        )
    if kernel_id == "embedding":
        return ReplayContract(
            TEMPLATE_ROOT / "embedding.cu",
            tuple(str(shape[name]) for name in ("B", "S", "D", "V", "U")) + ("0",),
            launch_skip=2,
        )
    if kernel_id == "flash_attention_fwd":
        return ReplayContract(
            TEMPLATE_ROOT / "flash_attention.cu",
            tuple(str(shape[name]) for name in ("B", "H", "S", "D")) + ("4",),
            launch_skip=1,
        )
    if kernel_id == "flashmla_dense_decode":
        return ReplayContract(
            TEMPLATE_ROOT / "flashmla.cu",
            tuple(str(shape[name]) for name in ("B", "S", "H")) + ("2",),
            launch_skip=1,
        )
    if kernel_id == "gelu":
        return ReplayContract(
            TEMPLATE_ROOT / "gelu.cu",
            (str(shape["N"]),),
            launch_skip=1,
        )
    if kernel_id == "gelu_gemm_fp16":
        return ReplayContract(
            TEMPLATE_ROOT / "gelu_gemm.cu",
            tuple(str(shape[name]) for name in ("M", "N", "K")),
            launch_skip=3,
            link_flags=("-lcublas",),
        )
    if kernel_id == "megamoe_fp8":
        return ReplayContract(
            TEMPLATE_ROOT / "megamoe.cu",
            tuple(str(shape[name]) for name in ("B", "S", "D", "E")) + ("8", "1024"),
            launch_skip=1,
        )
    if kernel_id == "rmsnorm":
        return ReplayContract(
            TEMPLATE_ROOT / "rmsnorm.cu",
            tuple(str(shape[name]) for name in ("B", "N", "D")),
            launch_skip=1,
        )
    if kernel_id == "rmsnorm_gemm_fp16":
        return ReplayContract(
            TEMPLATE_ROOT / "rmsnorm_gemm.cu",
            tuple(str(shape[name]) for name in ("M", "N", "K")),
            launch_skip=3,
            link_flags=("-lcublas",),
        )
    raise KeyError(f"no replay contract for {kernel_id}")
