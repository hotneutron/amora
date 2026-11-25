"""Packed single-device Torch CUDA correctness adapter for Portable-core V2."""
from __future__ import annotations

from contextlib import contextmanager
import os
import socket
import subprocess

from amora.benchmarking.portable_v2.contracts import Blocked, device_index
from benchmark_generators.portable_core.expanded.v2.execution import Backend, Prepared, execute_prepared


class _DeviceArrayFunctions:
    def __init__(self, torch, device):
        self.torch = torch
        self.device = device

    def __getattr__(self, name):
        if name in {"arange", "eye"}:
            fn = getattr(self.torch, name)
            return lambda *args, **kwargs: fn(*args, device=self.device, **kwargs)
        return getattr(self.torch, name)


class _CudaMath(Backend):
    def __init__(self, torch, device):
        self.torch = torch
        self.target = device
        self.name, self.device = "torch", str(device)
        self.xp = _DeviceArrayFunctions(torch, device)
        self.dtypes = {"fp32": torch.float32, "fp16": torch.float16, "int32": torch.int32}

    def array(self, value):
        return self.torch.as_tensor(value, device=self.target)

    def numpy(self, value):
        return value.detach().cpu().numpy()


class Target:
    def __init__(self, device, *, expected_generation=None):
        index = device_index("cuda_torch", device)
        if expected_generation is not None:
            raise ValueError("expected generation applies to TPU only")
        try:
            import torch
        except ImportError as error:
            raise Blocked("runtime_missing", "a CUDA-enabled Torch runtime is required") from error
        if not torch.version.cuda or torch.version.hip or not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise Blocked("device_unavailable", "no selected NVIDIA CUDA device is available to Torch")
        self.torch = torch
        self.device = torch.device("cuda", index)
        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
        self.math = _CudaMath(torch, self.device)
        # Target objects are constructed only in the isolated worker process.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        torch.use_deterministic_algorithms(True)
        props = torch.cuda.get_device_properties(self.device)
        try:
            proc = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                  check=True, capture_output=True, text=True, timeout=5)
            drivers = sorted(set(proc.stdout.split()))
        except (OSError, subprocess.SubprocessError):
            drivers = []
        # MIG is asserted only when the local device metadata identifies it.
        uuid = getattr(props, "uuid", None)
        self.identity = {"platform": "cuda", "instance": socket.gethostname(), "device": str(self.device),
                         "uuid": str(uuid) if uuid is not None else None,
                         "name": props.name, "sm": f"{props.major}{props.minor}",
                         "memory_bytes": props.total_memory, "driver_versions": drivers,
                         "mig": "observed" if "MIG" in props.name or str(uuid).startswith("MIG-") else "not_reported",
                         "visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
                         "torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                         "stream_policy": "single dedicated Torch CUDA stream", "math": "fp32; tf32/autocast disabled"}
        self.evidence = {"timing_exclusivity": "not_checked"}

    @contextmanager
    def _stream(self):
        torch = self.torch
        with (torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.inference_mode(),
              torch.autocast(device_type="cuda", enabled=False)):
            yield

    def finish(self):
        return {}, True

    def prepare(self, case, types, raw, directory):
        # The input validator rejects strided/offset views before invoking this adapter.
        with self._stream():
            bindings = {name: (self.torch.from_numpy(array).to(self.device, non_blocking=False, copy=True),)
                        for name, array in raw.items()}
            for name, (tensor,) in bindings.items():
                if (tensor.device != self.device or not tensor.is_contiguous() or tensor.storage_offset() != 0
                        or tuple(tensor.stride()) != tuple(s // raw[name].itemsize for s in raw[name].strides)):
                    raise Blocked("device_mismatch", "CUDA binding did not preserve the packed contract")
        self.stream.synchronize()
        prepared = Prepared(case, self.math, types, bindings, "bound_by_controller")

        def host(value):
            if value.device != self.device:
                raise Blocked("device_mismatch", "CUDA output did not remain on the selected device")
            return self.math.numpy(value)

        def run():
            with self._stream():
                out = execute_prepared(prepared)
                _, trace = execute_prepared(prepared, capture=True)
                return tuple(map(host, out)), {n.id: tuple(map(host, trace[n.id])) for n in case.program.nodes}

        return run, {"lowering": "eager", "ir": "not_available", "native_executable_sha256": "not_available"}
