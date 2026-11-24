# SPDX-License-Identifier: Apache-2.0
"""Operation contracts; import and shape inference require no array library."""
from __future__ import annotations

from .spec import Tensor

GROUPED = frozenset({"weighted_embedding_bag_mean", "multi_hash_embedding_sum", "indexed_group_min",
                     "indexed_group_l2", "weighted_embedding_bag_sum", "embedding_bag_mean",
                     "dual_hash_gather_sum", "indexed_group_max"})
BINARY_GROUPED = frozenset({"weighted_embedding_bag_mean", "multi_hash_embedding_sum", "weighted_embedding_bag_sum"})
SAME_ONE = frozenset({"relu", "row_roll", "half_scale_contract", "row_logsumexp", "cumulative_max_scan",
                      "bilinear_resample", "grouped_threshold_count", "weighted_stencil", "scaled_round_clip",
                      "causal_masked_softmax", "row_variance_normalize"})
SAME_TWO = frozenset({"add", "mul", "rotary_pair_transform", "dequantize_bias_gelu", "affine_state_scan",
                      "silu_gated_product"})
DIAGNOSTIC = frozenset({"causal_masked_softmax", "row_variance_normalize", "row_topk_select"})
DISCRETE = frozenset({"grouped_threshold_count"})
MODULAR = frozenset({"indexed_embedding_sum", "indexed_scatter_add"})


def infer(op, operands, attrs):
    """Check exact operand slots, shape, dtype and the permitted parameter domain."""
    allowed = {"row_stat_broadcast": {"stat", "temporary_dtype"}, "row_topk_select": {"k", "ties"},
               "cast": {"dtype"}}
    known = GROUPED | SAME_ONE | SAME_TWO | MODULAR | {"cast", "segment_sum4", "row_topk_select", "row_stat_broadcast"}
    if op not in known or set(attrs) != allowed.get(op, set()):
        raise ValueError(f"unsupported operation or attributes: {op}")
    n = 2 if op in SAME_TWO | BINARY_GROUPED | MODULAR else 1
    if len(operands) != n:
        raise ValueError(f"{op} requires {n} operands")
    x = operands[0]
    if op == "cast":
        if x.dtype not in ("fp16", "fp32") or attrs["dtype"] not in ("fp16", "fp32"):
            raise ValueError("unsupported cast")
        return (Tensor(x.shape, attrs["dtype"]),)
    if len(x.shape) != 2 or x.dtype not in ("fp16", "fp32"):
        raise ValueError("expected a rank-two floating tensor")
    m, d = x.shape
    if n == 2 and op not in MODULAR and x != operands[1]:
        raise ValueError("binary operand shape or dtype mismatch")
    if op in GROUPED and m % 4:
        raise ValueError("grouped indirect operations require rows divisible by four")
    if op == "rotary_pair_transform" and d % 2:
        raise ValueError("rotary pairs require even width")
    if op in ("grouped_threshold_count", "segment_sum4") and d % 4:
        raise ValueError("fixed groups require columns divisible by four")
    if op == "row_stat_broadcast" and (attrs["stat"] not in ("mean", "max") or
                                       attrs["temporary_dtype"] not in ("fp16", "fp32")):
        raise ValueError("unsupported reduction or temporary dtype")
    if op == "segment_sum4":
        return (Tensor((m, d // 4), x.dtype),)
    if op in MODULAR:
        index = operands[1]
        if index.dtype != "int32" or len(index.shape) != (2 if op == "indexed_embedding_sum" else 1):
            raise ValueError("invalid index dtype or rank")
        if op == "indexed_embedding_sum":
            if index.shape[1] != 4:
                raise ValueError("the indexed bag control uses fan-in four")
            return (Tensor((index.shape[0], d), x.dtype),)
        if index.shape != (m,):
            raise ValueError("scatter requires one destination per row")
    if op == "row_topk_select":
        k = attrs["k"]
        if type(k) is not int or not 1 <= k <= d or attrs["ties"] != "lowest_index":
            raise ValueError("unsupported top-k count or tie policy")
        return Tensor((m, k), x.dtype), Tensor((m, k), "int32")
    if op == "half_scale_contract" and m * d > 2**24:
        raise ValueError("contraction exceeds the portable slice budget")
    return (x,)
