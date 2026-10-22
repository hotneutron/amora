# SPDX-License-Identifier: Apache-2.0
# Adapted from Accorde: package namespace relocation only (2026-10-22).
"""Portable, semantic-only adapters for the bounded cross-ISA campaign.

Floating-point arithmetic has FP32 results. BF16 casts round once, and matmul
always quantizes its two logical operands to BF16 before FP32 accumulation and
output. Rearrangements preserve dtype. These are frontend semantics, not claims
about native kernel lowering, instruction selection, or hardware timing.

Broadcast axis denotes vector alignment: axis 0 aligns an R-vector with the
first dimension through [:, None]; axis 1 aligns a C-vector with the second
through [None, :]. axis=None broadcasts already-reshaped operands normally.

The private gather attribute indices_validated=True trusts ranges validated by
the prepared-program caller, avoiding host synchronization in PyTorch/JAX.
Standalone calls still validate ranges; both paths validate axis/rank/INT32.

## Revision History
- 2026-10-21T22:43:11Z: Support general pre-reshaped broadcasts with axis=None.
- 2026-10-21T22:50:15Z: Align vector axes with the manifest; add prepared gather
  without host checks or hidden INT64 conversion. Remove internal delegation
  prompts; actual user prompts are persisted by main in campaign artifacts.
"""

import importlib
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BackendUnavailable(RuntimeError):
    """The requested framework or actual device cannot be used."""


_DTYPES = ("fp32", "bf16", "int32")
_ARITY = {
    "cast": 1, "add": 2, "sub": 2, "mul": 2, "div": 2,
    "square": 1, "relu": 1, "exp": 1, "rsqrt": 1, "silu": 1,
    "sum": 1, "max": 1, "broadcast": 1, "transpose": 1,
    "gather": 2, "matmul": 2,
}


def quantize_bf16_numpy(array: Any) -> np.ndarray:
    """Return FP32 storage containing BF16 round-to-nearest-even values.

    NaNs remain NaNs (including payloads too small to survive truncation),
    infinities and signed zero are preserved. Input storage is never mutated.
    """
    bits = np.asarray(array, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1)))
    rounded = rounded & np.uint32(0xFFFF0000)
    special = (bits & np.uint32(0x7F800000)) == np.uint32(0x7F800000)
    nan = special & ((bits & np.uint32(0x007FFFFF)) != 0)
    # A quiet BF16 NaN prevents a truncated payload from becoming infinity.
    special_bits = (bits & np.uint32(0xFFFF0000)) | np.where(
        nan, np.uint32(0x00400000), np.uint32(0)
    )
    result = np.where(special, special_bits, rounded).astype(np.uint32)
    return result.view(np.float32)


def _dtype(dtype: str) -> str:
    if dtype not in _DTYPES:
        raise ValueError("unsupported dtype {!r}; expected {}".format(dtype, _DTYPES))
    return dtype


def _operation(op: str, args: List[Any], attrs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if op not in _ARITY:
        raise ValueError("unsupported opcode {!r}".format(op))
    if not isinstance(args, (list, tuple)) or len(args) != _ARITY[op]:
        raise ValueError("{} requires {} operands".format(op, _ARITY[op]))
    if attrs is None:
        return {}
    if not isinstance(attrs, dict):
        raise ValueError("operation attributes must be a dictionary")
    return attrs


def _axis(attrs: Dict[str, Any], ndim: int) -> int:
    axis = attrs.get("axis")
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise ValueError("axis must be an integer")
    if not -ndim <= axis < ndim:
        raise ValueError("axis is outside the operand rank")
    return axis


def _broadcast_shape(value: Any, attrs: Dict[str, Any]) -> Tuple[int, ...]:
    shape = attrs.get("shape")
    if not isinstance(shape, (list, tuple)) or not shape or any(
        isinstance(dim, bool) or not isinstance(dim, int) or dim < 0 for dim in shape
    ):
        raise ValueError("broadcast shape must contain nonnegative integers")
    if attrs.get("axis") is None:
        # Already-reshaped operands use ordinary trailing-dimension broadcasting,
        # e.g. [group, L, 1] -> [group, L, D]. Never shrink their existing rank.
        try:
            compatible = np.broadcast_shapes(tuple(value.shape), tuple(shape)) == tuple(shape)
        except ValueError:
            compatible = False
        if not compatible:
            raise ValueError("source shape cannot broadcast to the requested shape")
        return tuple(shape)
    if len(shape) != 2:
        raise ValueError("axis-based vector broadcast requires a two-dimensional shape")
    axis = _axis(attrs, 2)
    if axis not in (0, 1):
        raise ValueError("broadcast axis must be 0 (R-vector) or 1 (C-vector)")
    if value.ndim != 1 or value.shape[0] != shape[axis]:
        raise ValueError("broadcast source must be a vector matching the selected axis")
    return tuple(shape)


def _matrix(value: Any, label: str) -> None:
    if value.ndim != 2:
        raise ValueError("{} requires a two-dimensional matrix".format(label))


def _matrices(left: Any, right: Any) -> None:
    _matrix(left, "matmul")
    _matrix(right, "matmul")
    if left.shape[1] != right.shape[0]:
        raise ValueError("matmul contraction dimensions do not match")


def _gather_layout(source: Any, indices: Any, attrs: Dict[str, Any], int32_dtype: Any) -> None:
    """Check shape/dtype metadata without transferring index values to the host."""
    _matrix(source, "gather")
    axis = attrs.get("axis", 0)
    if isinstance(axis, bool) or not isinstance(axis, int) or axis != 0:
        raise ValueError("gather supports only row axis 0")
    if indices.dtype != int32_dtype or indices.ndim != 1:
        raise ValueError("gather indices must be a one-dimensional INT32 array")


def _gather_indices(source: Any, indices: np.ndarray, attrs: Dict[str, Any]) -> None:
    _gather_layout(source, indices, attrs, np.int32)
    # Check explicitly: frameworks otherwise disagree on negative/OOB indices.
    if np.any(indices < 0) or np.any(indices >= source.shape[0]):
        raise ValueError("gather indices are outside the source rows")


def _device_spec(device: str, allowed: Tuple[str, ...]) -> Tuple[str, int]:
    if not isinstance(device, str):
        raise BackendUnavailable("device must be a string")
    match = re.fullmatch(r"([a-z]+)(?::([0-9]+))?", device)
    if match is None or match.group(1) not in allowed:
        raise BackendUnavailable("unsupported device {!r}; expected {}".format(device, allowed))
    return match.group(1), int(match.group(2) or 0)


def _description(name: str, device: str, version: str, storage: str,
                 implementation: str, physical_operands: str, materialization: str) -> Dict[str, Any]:
    return {
        "backend": name,
        "device": device,
        "version": version,
        "frontend_semantic_capability": {
            "opcodes": list(_ARITY),
            "dtypes": list(_DTYPES),
            "arithmetic_output_dtype": "fp32",
            "bf16_rounding": "round_to_nearest_even",
            "matmul_logical_operand_dtype": "bf16",
            "matmul_accumulator_dtype": "fp32",
            "matmul_output_dtype": "fp32",
            "gather_indices_dtype": "int32",
        },
        "realized_execution": {
            "bf16_storage": storage,
            "matmul": {
                "implementation": implementation,
                "physical_operand_dtype": physical_operands,
                "accumulator_contract": "fp32",
                "output_dtype": "fp32",
                "native_instruction_verified": False,
            },
            "broadcast_and_transpose_materialization": materialization,
        },
        "native_qualification": {
            "hardware_timing": "unqualified",
            "compiler_lowering": "unqualified",
            "materialization": "unqualified",
        },
        "native_hardware_timing_qualified": False,
        "native_materialization_qualified": False,
    }


class NumpyBackend:
    """FP32 semantic reference; BF16 values are represented in FP32 storage."""

    def __init__(self, device: str = "cpu") -> None:
        kind, index = _device_spec(device, ("cpu",))
        if index != 0:
            raise BackendUnavailable("NumPy has only the CPU reference device")
        self.device = kind

    def array(self, value: Any, dtype: str = "fp32") -> np.ndarray:
        dtype = _dtype(dtype)
        if dtype == "bf16":
            return quantize_bf16_numpy(value)
        return np.array(value, dtype=np.float32 if dtype == "fp32" else np.int32, copy=True)

    def numpy(self, value: Any) -> np.ndarray:
        return np.array(value, copy=True)

    def execute(self, op: str, args: List[Any], attrs: Optional[Dict[str, Any]] = None) -> np.ndarray:
        attrs = _operation(op, args, attrs)
        values = [np.asarray(arg) for arg in args]
        value = values[0]
        if op == "cast":
            return self.array(value, _dtype(attrs.get("dtype")))
        if op == "broadcast":
            shape = _broadcast_shape(value, attrs)
            vector = value if attrs.get("axis") is None else (value[:, None] if attrs["axis"] == 0 else value[None, :])
            return np.broadcast_to(vector, shape).copy()
        if op == "transpose":
            _matrix(value, "transpose")
            return value.T.copy(order="C")
        if op == "gather":
            indices = values[1]
            _gather_indices(value, indices, attrs)
            return np.take(value, indices, axis=0)
        if op == "matmul":
            _matrices(value, values[1])
            return np.matmul(quantize_bf16_numpy(value), quantize_bf16_numpy(values[1]))
        floats = [arg.astype(np.float32, copy=False) for arg in values]
        value = floats[0]
        if op in ("add", "sub", "mul", "div"):
            functions = {"add": np.add, "sub": np.subtract, "mul": np.multiply, "div": np.divide}
            return functions[op](value, floats[1])
        if op == "square":
            return np.square(value)
        if op == "relu":
            return np.maximum(value, np.float32(0))
        if op == "exp":
            return np.exp(value)
        if op == "rsqrt":
            return np.reciprocal(np.sqrt(value))
        if op == "silu":
            return value / (np.float32(1) + np.exp(-value))
        if op == "sum":
            return np.sum(value, axis=_axis(attrs, value.ndim), dtype=np.float32)
        if op == "max":
            return np.max(value, axis=_axis(attrs, value.ndim))
        raise AssertionError("validated opcode was not implemented")

    def describe(self) -> Dict[str, Any]:
        return _description(
            "numpy", self.device, np.__version__, "bf16_values_in_fp32_storage",
            "bf16_quantized_fp32_numpy_matmul", "fp32", "explicit_c_contiguous_copies",
        )


class TorchBackend:
    """PyTorch adapter with explicit promoted-FP32 BF16 matmul realization."""

    def __init__(self, device: str = "cpu") -> None:
        kind, index = _device_spec(device, ("cpu", "cuda", "npu"))
        try:
            self._torch = importlib.import_module("torch")
            if kind == "cpu":
                if index != 0:
                    raise BackendUnavailable("PyTorch CPU indices other than zero are unsupported")
                self.device = "cpu"
            else:
                if kind == "npu":
                    importlib.import_module("torch_npu")
                provider = getattr(self._torch, kind, None)
                if provider is None or not provider.is_available():
                    raise BackendUnavailable("PyTorch {} is not available".format(kind))
                if index >= provider.device_count():
                    raise BackendUnavailable("PyTorch {} device {} does not exist".format(kind, index))
                self.device = "{}:{}".format(kind, index)
            self._device = self._torch.device(self.device)
            # Allocation checks real device initialization, not just a name or build flag.
            self._torch.empty(0, device=self._device)
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable("cannot initialize PyTorch device {!r}: {}".format(device, exc)) from exc
        self._dtypes = {"fp32": self._torch.float32, "bf16": self._torch.bfloat16, "int32": self._torch.int32}

    def array(self, value: Any, dtype: str = "fp32") -> Any:
        return self._torch.as_tensor(value, dtype=self._dtypes[_dtype(dtype)], device=self._device).clone()

    def numpy(self, value: Any) -> np.ndarray:
        value = self._torch.as_tensor(value).detach().cpu()
        if value.dtype == self._torch.bfloat16:
            value = value.to(self._torch.float32)
        return value.numpy().copy()

    def execute(self, op: str, args: List[Any], attrs: Optional[Dict[str, Any]] = None) -> Any:
        attrs = _operation(op, args, attrs)
        torch = self._torch
        values = [torch.as_tensor(arg, device=self._device) for arg in args]
        value = values[0]
        if op == "cast":
            return self.array(value, _dtype(attrs.get("dtype")))
        if op == "broadcast":
            shape = _broadcast_shape(value, attrs)
            vector = value if attrs.get("axis") is None else (value.unsqueeze(1) if attrs["axis"] == 0 else value.unsqueeze(0))
            return vector.expand(shape).clone(memory_format=torch.contiguous_format)
        if op == "transpose":
            _matrix(value, "transpose")
            return value.transpose(0, 1).clone(memory_format=torch.contiguous_format)
        if op == "gather":
            indices = values[1]
            if attrs.get("indices_validated") is True:
                # Private prepared-program path: ranges were checked before execution.
                _gather_layout(value, indices, attrs, torch.int32)
            else:
                _gather_indices(value, self.numpy(indices), attrs)
            return torch.index_select(value, 0, indices)
        if op == "matmul":
            _matrices(value, values[1])
            # Native BF16 mm returns BF16 on CPU (and many device versions).
            # Promoting *before* mm preserves FP32 accumulation/output; converting
            # an already-rounded BF16 result to FP32 would not satisfy the contract.
            left = value.to(torch.bfloat16).to(torch.float32)
            right = values[1].to(torch.bfloat16).to(torch.float32)
            return torch.mm(left, right)
        floats = [arg.to(torch.float32) for arg in values]
        value = floats[0]
        if op in ("add", "sub", "mul", "div"):
            functions = {"add": torch.add, "sub": torch.sub, "mul": torch.mul, "div": torch.div}
            return functions[op](value, floats[1])
        if op == "square":
            return torch.square(value)
        if op == "relu":
            return torch.clamp_min(value, 0)
        if op == "exp":
            return torch.exp(value)
        if op == "rsqrt":
            return torch.rsqrt(value)
        if op == "silu":
            return value / (1 + torch.exp(-value))
        if op == "sum":
            return torch.sum(value, dim=_axis(attrs, value.ndim), dtype=torch.float32)
        if op == "max":
            return torch.amax(value, dim=_axis(attrs, value.ndim))
        raise AssertionError("validated opcode was not implemented")

    def describe(self) -> Dict[str, Any]:
        return _description(
            "torch", self.device, self._torch.__version__, "native_bf16",
            "bf16_quantize_then_promote_fp32_torch_mm", "fp32", "explicit_contiguous_clones",
        )


class JaxBackend:
    """Optional JAX CPU/TPU semantics; fusion/materialization is not qualified."""

    def __init__(self, device: str = "cpu") -> None:
        kind, index = _device_spec(device, ("cpu", "tpu"))
        try:
            self._jax = importlib.import_module("jax")
            self._jnp = importlib.import_module("jax.numpy")
            devices = self._jax.devices(kind)
            if index >= len(devices):
                raise BackendUnavailable("JAX {} device {} does not exist".format(kind, index))
            self._device = devices[index]
            self.device = "{}:{}".format(kind, index)
            self._jax.device_put(np.zeros(0, dtype=np.float32), self._device).block_until_ready()
        except BackendUnavailable:
            raise
        except Exception as exc:
            raise BackendUnavailable("cannot initialize JAX device {!r}: {}".format(device, exc)) from exc
        self._dtypes = {"fp32": self._jnp.float32, "bf16": self._jnp.bfloat16, "int32": self._jnp.int32}

    def array(self, value: Any, dtype: str = "fp32") -> Any:
        with self._jax.default_device(self._device):
            return self._jax.device_put(self._jnp.array(value, dtype=self._dtypes[_dtype(dtype)], copy=True), self._device)

    def numpy(self, value: Any) -> np.ndarray:
        if getattr(value, "dtype", None) == self._jnp.bfloat16:
            value = value.astype(self._jnp.float32)
        return np.array(value, copy=True)

    def execute(self, op: str, args: List[Any], attrs: Optional[Dict[str, Any]] = None) -> Any:
        attrs = _operation(op, args, attrs)
        with self._jax.default_device(self._device):
            return self._execute(op, args, attrs)

    def _execute(self, op: str, args: List[Any], attrs: Dict[str, Any]) -> Any:
        jnp = self._jnp
        values = [self._jax.device_put(jnp.asarray(arg), self._device) for arg in args]
        value = values[0]
        if op == "cast":
            return self.array(value, _dtype(attrs.get("dtype")))
        if op == "broadcast":
            shape = _broadcast_shape(value, attrs)
            vector = value if attrs.get("axis") is None else (value[:, None] if attrs["axis"] == 0 else value[None, :])
            return jnp.array(jnp.broadcast_to(vector, shape), copy=True)
        if op == "transpose":
            _matrix(value, "transpose")
            return jnp.array(jnp.transpose(value), copy=True)
        if op == "gather":
            indices = values[1]
            if attrs.get("indices_validated") is True:
                # Avoid host synchronization inside an already-validated program.
                _gather_layout(value, indices, attrs, jnp.int32)
            else:
                _gather_indices(value, self.numpy(indices), attrs)
            return jnp.take(value, indices, axis=0)
        if op == "matmul":
            _matrices(value, values[1])
            return self._jax.lax.dot_general(
                value.astype(jnp.bfloat16), values[1].astype(jnp.bfloat16),
                dimension_numbers=(((1,), (0,)), ((), ())),
                precision=self._jax.lax.Precision.HIGHEST,
                preferred_element_type=jnp.float32,
            )
        floats = [arg.astype(jnp.float32) for arg in values]
        value = floats[0]
        if op in ("add", "sub", "mul", "div"):
            functions = {"add": jnp.add, "sub": jnp.subtract, "mul": jnp.multiply, "div": jnp.divide}
            return functions[op](value, floats[1])
        if op == "square":
            return jnp.square(value)
        if op == "relu":
            return jnp.maximum(value, jnp.float32(0))
        if op == "exp":
            return jnp.exp(value)
        if op == "rsqrt":
            return self._jax.lax.rsqrt(value)
        if op == "silu":
            return value / (jnp.float32(1) + jnp.exp(-value))
        if op == "sum":
            return jnp.sum(value, axis=_axis(attrs, value.ndim), dtype=jnp.float32)
        if op == "max":
            return jnp.max(value, axis=_axis(attrs, value.ndim))
        raise AssertionError("validated opcode was not implemented")

    def describe(self) -> Dict[str, Any]:
        return _description(
            "jax", self.device, self._jax.__version__, "native_bf16",
            "lax_dot_general_preferred_element_type_fp32", "bf16",
            "not_guaranteed_lazy_execution_and_compiler_fusion",
        )


def create_backend(name: str, device: str = "cpu") -> Any:
    """Create an adapter only if its requested real device can initialize.

    No unavailable accelerator is silently replaced with CPU execution.
    """
    factories = {"numpy": NumpyBackend, "torch": TorchBackend, "pytorch": TorchBackend, "jax": JaxBackend}
    if not isinstance(name, str) or name.lower() not in factories:
        raise BackendUnavailable("unknown backend {!r}".format(name))
    return factories[name.lower()](device)
