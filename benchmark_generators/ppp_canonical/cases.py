"""Deterministic shape generation for canonical PPP kernels."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import product
from math import gcd

from benchmark_generators.ppp_canonical.kernels import KernelSpec


_GEMM_MN = (256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288)
_GEMM_K = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192)
_BATCH = (1, 2, 4, 8)
_SEQUENCE = (384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768)
_HEADS = (1, 2, 4, 8, 12, 16)
_HIDDEN = (64, 96, 128, 192, 256, 384, 512, 768, 1024, 2048, 4096)


def _round_robin(values: Iterable[tuple[int, ...]], *, stride: int) -> list[tuple[int, ...]]:
    """Return a deterministic full-cycle ordering without random-state coupling."""

    ordered = list(values)
    if not ordered:
        return []
    step = stride
    while gcd(step, len(ordered)) != 1:
        step += 1
    index = 0
    result = []
    for _ in range(len(ordered)):
        result.append(ordered[index])
        index = (index + step) % len(ordered)
    return result


def _generated_item(
    shape: dict[str, int],
    *,
    tags: tuple[str, ...],
    axis_tags: tuple[str, ...],
) -> dict:
    return {
        "shape": shape,
        "shape_class": "sweep",
        "tags": list(tags),
        "axis_tags": list(axis_tags),
        "regime_tags": list(tags),
    }


def _dedupe_shapes(items: Iterable[dict]) -> list[dict]:
    """Keep the first deterministic candidate for every canonical shape."""

    seen: set[tuple[tuple[str, int], ...]] = set()
    unique: list[dict] = []
    for item in items:
        key = tuple(sorted(item["shape"].items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _gemm_candidates(tag: str = "generated_gemm") -> list[dict]:
    return [
        _generated_item(
            {"M": m, "N": n, "K": k},
            tags=(tag,),
            axis_tags=("M", "N", "K"),
        )
        for m, n, k in _round_robin(product(_GEMM_MN, _GEMM_MN, _GEMM_K), stride=137)
    ]


def _flash_attention_candidates() -> list[dict]:
    combos = [
        (batch, heads, sequence, hidden)
        for batch, heads, sequence, hidden in product(_BATCH, _HEADS, _SEQUENCE, (64, 96, 128))
        if batch * heads * sequence * hidden <= 4 * 16 * 32768 * 128
    ]
    return [
        _generated_item(
            {"B": batch, "H": heads, "S": sequence, "D": hidden},
            tags=("generated_attention",),
            axis_tags=("B", "H", "S", "D"),
        )
        for batch, heads, sequence, hidden in _round_robin(combos, stride=149)
    ]


def _flashmla_candidates() -> list[dict]:
    combos = [
        (batch, sequence, heads)
        for batch, sequence, heads in product(_BATCH, _SEQUENCE + (49152, 65536), _HEADS)
        if batch * heads * sequence * 576 <= 4 * 16 * 32768 * 576
    ]
    return [
        _generated_item(
            {"B": batch, "S": sequence, "H": heads},
            tags=("generated_flashmla",),
            axis_tags=("B", "S", "H"),
        )
        for batch, sequence, heads in _round_robin(combos, stride=113)
    ]


def _megamoe_candidates() -> list[dict]:
    combos = [
        (batch, sequence, hidden, experts)
        for batch, sequence, hidden, experts in product(
            _BATCH,
            (2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536),
            (1024, 1536, 2048, 3072, 4096, 5120, 6144, 7168),
            (32, 64, 96, 128, 192, 256),
        )
        if batch * sequence * hidden * 3 <= 8 * 32768 * 7168 * 3
    ]
    return [
        _generated_item(
            {"B": batch, "S": sequence, "D": hidden, "E": experts},
            tags=("generated_moe",),
            axis_tags=("B", "S", "D", "E"),
        )
        for batch, sequence, hidden, experts in _round_robin(combos, stride=257)
    ]


def _rmsnorm_candidates() -> list[dict]:
    combos = [
        (batch, rows, hidden)
        for batch, rows, hidden in product(
            (1, 2, 4, 8, 16),
            (32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192),
            (128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536),
        )
        if batch * rows * hidden <= 16 * 4096 * 32768
    ]
    return [
        _generated_item(
            {"B": batch, "N": rows, "D": hidden},
            tags=("generated_rmsnorm",),
            axis_tags=("B", "N", "D"),
        )
        for batch, rows, hidden in _round_robin(combos, stride=191)
    ]


def _gelu_candidates() -> list[dict]:
    values: set[int] = set()
    for step in (8192, 12288, 16384, 24576, 32768, 49152, 65536, 98304):
        values.update(range(step, 268435456 + 1, step))
    for base in (32768, 49152, 65536, 98304):
        value = base
        while value <= 268435456:
            values.add(value)
            value = int(value * 1.17) + 1024
    return [
        _generated_item({"N": value}, tags=("generated_gelu",), axis_tags=("N",))
        for value in sorted(values)
    ]


def _embedding_candidates() -> list[dict]:
    combos = []
    for batch, sequence, hidden in product(
        _BATCH,
        (512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768, 49152, 65536),
        _HIDDEN,
    ):
        rows = batch * sequence
        for ratio in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.01):
            unique = max(1, min(50000, int(rows * ratio)))
            bytes_estimate = 50000 * hidden * 2 + rows * hidden * 2
            if bytes_estimate <= 3_000_000_000:
                combos.append((batch, sequence, hidden, unique))
    return [
        _generated_item(
            {"B": batch, "S": sequence, "D": hidden, "V": 50000, "U": unique},
            tags=("generated_embedding",),
            axis_tags=("B", "S", "D", "U"),
        )
        for batch, sequence, hidden, unique in _round_robin(combos, stride=173)
    ]


def candidate_items(spec: KernelSpec) -> list[dict]:
    """Return deterministic, duplicate-free candidate shapes for one kernel."""

    if spec.kernel_id == "aligned_gemm_fp16":
        candidates = _gemm_candidates()
    elif spec.kernel_id == "flash_attention_fwd":
        candidates = _flash_attention_candidates()
    elif spec.kernel_id == "flashmla_dense_decode":
        candidates = _flashmla_candidates()
    elif spec.kernel_id == "megamoe_fp8":
        candidates = _megamoe_candidates()
    elif spec.kernel_id == "rmsnorm":
        candidates = _rmsnorm_candidates()
    elif spec.kernel_id == "gelu":
        candidates = _gelu_candidates()
    elif spec.kernel_id == "embedding":
        candidates = _embedding_candidates()
    elif spec.kernel_id == "gelu_gemm_fp16":
        candidates = _gemm_candidates("generated_gelu_gemm")
    elif spec.kernel_id == "rmsnorm_gemm_fp16":
        candidates = _gemm_candidates("generated_rmsnorm_gemm")
    else:
        raise KeyError(f"no candidate generator for {spec.kernel_id}")
    return _dedupe_shapes(candidates)
