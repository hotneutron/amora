# SPDX-License-Identifier: Apache-2.0
# Adapted from Accorde: package namespace relocation only (2026-10-22).
"""Semantic-only expansion of the immutable benchmark_generators.portable_core.base backends.

The four new opcodes are scan, segment_sum, scatter_add, and stencil. Their
arithmetic and outputs are FP32; offsets and scatter indices must be INT32.
Stencil is a same-size, zero-padded, per-channel 3x3 cross-correlation (weights
are signed and are not flipped). No API call establishes native instruction,
atomic, carry, compiler-lowering, materialization, or timing qualification.

Segment offsets are a declared Python list in attrs['offsets'], not values read
from a device during execution. Call validate_inputs before a prepared body to
check the input offsets against that list and check scatter index ranges; use
its returned attributes in execute. The returned validation flags are trusted
prepared-input proofs, not permission to change/reuse unvalidated inputs.
Validation may copy index/offset values to the host, but execute never does so
for the new Torch/JAX opcodes. Keep validation outside any timed body.

Actual user prompts are persisted by the main task in its plan artifacts.

## Revision History
- 2026-10-21T23:57:43Z: Add isolated expansion wrapper, static-input validation,
  portable FP32 implementations, and unqualified mechanism-realization metadata.
"""

from copy import deepcopy
from numbers import Integral
from typing import Any, Dict, List, Optional

import numpy as np

from benchmark_generators.portable_core.base.backends import BackendUnavailable
from benchmark_generators.portable_core.base.backends import create_backend as _create_backend


_ARITY = {"scan": 1, "segment_sum": 2, "scatter_add": 2, "stencil": 2}
_INT32_MAX = int(np.iinfo(np.int32).max)


def _attributes(op: str, args: List[Any], attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(args, (list, tuple)) or len(args) != _ARITY[op]:
        raise ValueError("{} requires {} operands".format(op, _ARITY[op]))
    if attrs is None:
        return {}
    if not isinstance(attrs, dict):
        raise ValueError("operation attributes must be a dictionary")
    return attrs


def _integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("{} must be an integer".format(label))
    return int(value)


def _axis(attrs: Dict[str, Any], ndim: int) -> int:
    axis = _integer(attrs.get("axis"), "scan axis")
    if not -ndim <= axis < ndim:
        raise ValueError("scan axis is outside the operand rank")
    return axis


def _offsets(attrs: Dict[str, Any], rows: int) -> List[int]:
    declared = attrs.get("offsets")
    if not isinstance(declared, list) or not declared:
        raise ValueError("segment_sum offsets must be a nonempty static declared list")
    offsets = [_integer(value, "segment offset") for value in declared]
    if offsets[0] != 0 or offsets[-1] != rows:
        raise ValueError("segment offsets must start at zero and end at the data row count")
    if any(value < 0 or value > _INT32_MAX for value in offsets):
        raise ValueError("segment offsets must be nonnegative INT32 values")
    if any(left > right for left, right in zip(offsets, offsets[1:])):
        raise ValueError("segment offsets must be nondecreasing")
    return offsets


def _rows(attrs: Dict[str, Any]) -> int:
    rows = _integer(attrs.get("rows"), "scatter_add rows")
    if rows < 0:
        raise ValueError("scatter_add rows must be nonnegative")
    return rows


def _layout(op: str, values: List[Any], attrs: Dict[str, Any],
            fp32: Any, int32: Any) -> None:
    """Validate only dtype/shape/static metadata, without reading device values."""
    value = values[0]
    if value.dtype != fp32:
        raise ValueError("{} data must have FP32 dtype".format(op))
    if op == "scan":
        _axis(attrs, value.ndim)
        return
    if op in ("segment_sum", "scatter_add"):
        if value.ndim != 2:
            raise ValueError("{} data must have shape [N, D]".format(op))
        indices = values[1]
        if indices.dtype != int32 or indices.ndim != 1:
            raise ValueError("{} offsets/indices must be a one-dimensional INT32 array".format(op))
        if op == "segment_sum":
            if indices.shape[0] != len(_offsets(attrs, value.shape[0])):
                raise ValueError("segment offset input length must match the static list")
        else:
            _rows(attrs)
            if indices.shape[0] != value.shape[0]:
                raise ValueError("scatter indices must have one entry per data row")
        return
    if value.ndim != 3:
        raise ValueError("stencil data must have shape [H, W, C]")
    weights = values[1]
    if weights.dtype != fp32 or tuple(weights.shape) != (value.shape[2], 3, 3):
        raise ValueError("stencil weights must have FP32 dtype and shape [C, 3, 3]")


def _prepared(op: str, attrs: Dict[str, Any]) -> None:
    proof = {"segment_sum": "offsets_validated", "scatter_add": "indices_validated"}.get(op)
    if proof is not None and attrs.get(proof) is not True:
        raise ValueError("{} requires validate_inputs outside the executor ({}=True)".format(op, proof))


def _numpy_execute(op: str, values: List[Any], attrs: Dict[str, Any]) -> np.ndarray:
    value = values[0]
    if op == "scan":
        return np.cumsum(value, axis=_axis(attrs, value.ndim), dtype=np.float32)
    if op == "segment_sum":
        offsets = _offsets(attrs, value.shape[0])
        output = np.zeros((len(offsets) - 1, value.shape[1]), dtype=np.float32)
        for group, (start, end) in enumerate(zip(offsets, offsets[1:])):
            output[group] = np.sum(value[start:end], axis=0, dtype=np.float32)
        return output
    if op == "scatter_add":
        output = np.zeros((_rows(attrs), value.shape[1]), dtype=np.float32)
        np.add.at(output, values[1], value)
        return output
    height, width, channels = value.shape
    padded = np.zeros((height + 2, width + 2, channels), dtype=np.float32)
    padded[1:height + 1, 1:width + 1] = value
    output = np.zeros_like(value)
    for i in range(3):
        for j in range(3):
            output += padded[i:i + height, j:j + width] * values[1][:, i, j]
    return output


class ExpandedBackend:
    """Delegate the old API and add four real, device-preserving semantic ops."""

    def __init__(self, delegate: Any) -> None:
        self._delegate = delegate
        self._provider = delegate.describe()["backend"]
        self.device = delegate.device

    def array(self, value: Any, dtype: str = "fp32") -> Any:
        return self._delegate.array(value, dtype)

    def numpy(self, value: Any) -> np.ndarray:
        return self._delegate.numpy(value)

    def validate_inputs(self, op: str, args: List[Any],
                        attrs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Check actual static offsets/index ranges BEFORE execution/timing.

        Returns an independent attribute dictionary with trusted validation flags.
        This is also suitable for host arrays before device upload. Revalidate
        whenever the input offsets/indices or their static metadata change.
        """
        if op not in _ARITY:
            if attrs is not None and not isinstance(attrs, dict):
                raise ValueError("operation attributes must be a dictionary")
            return deepcopy(attrs or {})
        checked = deepcopy(_attributes(op, args, attrs))
        # Host conversion is deliberately restricted to this preparation method.
        values = [self.numpy(arg) for arg in args]
        _layout(op, values, checked, np.float32, np.int32)
        if op == "segment_sum":
            if not np.array_equal(values[1], np.asarray(checked["offsets"], dtype=np.int32)):
                raise ValueError("segment offset input does not match its static declared list")
            checked["offsets_validated"] = True
        elif op == "scatter_add":
            indices = values[1]
            if np.any(indices < 0) or np.any(indices >= _rows(checked)):
                raise ValueError("scatter indices are outside the output rows")
            checked["indices_validated"] = True
        return checked

    def execute(self, op: str, args: List[Any],
                attrs: Optional[Dict[str, Any]] = None) -> Any:
        if op not in _ARITY:
            return self._delegate.execute(op, args, attrs)
        attrs = _attributes(op, args, attrs)
        _prepared(op, attrs)
        if self._provider == "numpy":
            values = [np.asarray(arg) for arg in args]
            _layout(op, values, attrs, np.float32, np.int32)
            return _numpy_execute(op, values, attrs)
        if self._provider == "torch":
            # No shared code depends on Torch internals, CUDA, or a host copy.
            torch = self._delegate._torch
            values = [torch.as_tensor(arg, device=self._delegate._device) for arg in args]
            _layout(op, values, attrs, torch.float32, torch.int32)
            value = values[0]
            if op == "scan":
                return torch.cumsum(value, dim=_axis(attrs, value.ndim), dtype=torch.float32)
            if op == "segment_sum":
                offsets = _offsets(attrs, value.shape[0])
                sums = [torch.sum(value[start:end], dim=0, dtype=torch.float32)
                        for start, end in zip(offsets, offsets[1:])]
                return (torch.stack(sums, dim=0) if sums else
                        torch.zeros((0, value.shape[1]), dtype=torch.float32, device=value.device))
            if op == "scatter_add":
                output = torch.zeros((_rows(attrs), value.shape[1]),
                                     dtype=torch.float32, device=value.device)
                # Keep INT32 indices and the actual provider API; do not claim
                # that this proves atomic instructions on any native device.
                return output.index_add_(0, values[1], value)
            height, width, channels = value.shape
            padded = torch.zeros((height + 2, width + 2, channels),
                                 dtype=torch.float32, device=value.device)
            padded[1:height + 1, 1:width + 1] = value
            output = torch.zeros_like(value)
            for i in range(3):
                for j in range(3):
                    output += padded[i:i + height, j:j + width] * values[1][:, i, j]
            return output
        if self._provider == "jax":
            jax = self._delegate._jax
            jnp = self._delegate._jnp
            with jax.default_device(self._delegate._device):
                values = [jax.device_put(jnp.asarray(arg), self._delegate._device) for arg in args]
                _layout(op, values, attrs, jnp.float32, jnp.int32)
                value = values[0]
                if op == "scan":
                    return jnp.cumsum(value, axis=_axis(attrs, value.ndim), dtype=jnp.float32)
                if op == "segment_sum":
                    offsets = _offsets(attrs, value.shape[0])
                    sums = [jnp.sum(value[start:end], axis=0, dtype=jnp.float32)
                            for start, end in zip(offsets, offsets[1:])]
                    return (jnp.stack(sums, axis=0) if sums else
                            jnp.zeros((0, value.shape[1]), dtype=jnp.float32))
                if op == "scatter_add":
                    output = jnp.zeros((_rows(attrs), value.shape[1]), dtype=jnp.float32)
                    return output.at[values[1]].add(value)
                height, width, _ = value.shape
                padded = jnp.pad(value, ((1, 1), (1, 1), (0, 0)),
                                 mode="constant", constant_values=0)
                output = jnp.zeros_like(value)
                for i in range(3):
                    for j in range(3):
                        output = output + padded[i:i + height, j:j + width] * values[1][:, i, j]
                return output
        raise BackendUnavailable("unsupported expansion provider {!r}".format(self._provider))

    def describe(self) -> Dict[str, Any]:
        description = deepcopy(self._delegate.describe())
        semantic = description["frontend_semantic_capability"]
        semantic["opcodes"] = list(dict.fromkeys(semantic["opcodes"] + list(_ARITY)))
        semantic["expansion_contracts"] = {
            "scan": {"input_dtype": "fp32", "output_dtype": "fp32", "inclusive": True,
                     "axis": "declared_integer"},
            "segment_sum": {"input_dtype": "fp32", "output_dtype": "fp32",
                            "offsets_dtype": "int32", "offsets": "validated_static_declared_list",
                            "empty_segments": "zero", "output_shape": "[G,D]"},
            "scatter_add": {"input_dtype": "fp32", "output_dtype": "fp32",
                            "indices_dtype": "int32", "indices": "validated_in_range",
                            "output_shape": "[rows,D]", "initial_output": "zero"},
            "stencil": {"input_dtype": "fp32", "output_dtype": "fp32",
                        "input_shape": "[H,W,C]", "weights_shape": "[C,3,3]",
                        "padding": "spatial_zero", "weights": "signed_per_channel",
                        "orientation": "cross_correlation_no_weight_flip"},
        }
        scan = {"numpy": "numpy.cumsum", "torch": "torch.cumsum", "jax": "jax.numpy.cumsum"}
        scatter = {"numpy": "numpy.add.at", "torch": "torch.Tensor.index_add_",
                   "jax": "jax.Array.at[indices].add"}
        realization = {
            "scan": {"implementation": scan[self._provider], "accumulator_contract": "fp32",
                     "inclusive": True, "native_carry_instruction_verified": False},
            "segment_sum": {"implementation": "static_owner_slices_fp32_sum",
                            "offsets_from": "validated_static_attributes_not_device_or_timing",
                            "empty_segments": "zero", "atomics_used": False},
            "scatter_add": {"implementation": scatter[self._provider],
                            "output_zeroing_included": True, "indices_dtype": "int32",
                            "atomic_realization": "provider_dependent_unverified",
                            "native_atomic_instruction_verified": False},
            "stencil": {"implementation": "explicit_zero_padding_slice_multiply_add",
                        "weights": "signed_per_channel", "compiler_fusion_verified": False},
        }
        for mechanism in realization.values():
            mechanism.update({"output_dtype": "fp32", "native_instruction_verified": False,
                              "compiler_lowering_verified": False,
                              "native_materialization_verified": False})
        description["mechanism_realization"] = realization
        description["realized_execution"].update(deepcopy(realization))
        return description


def create_backend(name: str, device: str = "cpu") -> ExpandedBackend:
    """Wrap the original real-device factory without modifying old adapters."""
    return ExpandedBackend(_create_backend(name, device))
