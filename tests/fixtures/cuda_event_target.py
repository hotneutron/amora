"""Small CUDA-event protocol target used for an optional live GPU smoke."""

from __future__ import annotations

import hashlib
import json
import os

import torch


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.cuda.current_device()
    value = torch.ones(1 << 20, device=device)
    output = torch.empty_like(value)

    def launch() -> None:
        torch.add(value, 1.0, out=output)

    for _ in range(5):
        launch()
    torch.cuda.synchronize(device)
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            launch()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0 / 10)

    properties = torch.cuda.get_device_properties(device)
    subject = b"torch.add(float32, 2**20 elements)"
    identity = hashlib.sha256(subject).hexdigest()
    kernel_name = os.environ.get(
        "AMORA_SMOKE_KERNEL_NAME",
        "void vectorized_elementwise_kernel<4, FillFunctor<float>, Array<char *, 1>>(int, T2, T3)",
    )
    measurement_axes = json.loads(
        os.environ.get("AMORA_SMOKE_MEASUREMENT_AXES", "{}")
    )
    measurement_context = json.loads(
        os.environ.get("AMORA_SMOKE_MEASUREMENT_CONTEXT", "{}")
    )
    payload = {
        "schema_version": 1,
        "kind": "cuda_event_timing",
        "timing": {
            "unit": "us_per_launch",
            "samples": samples,
            "warmup_launches": 5,
            "launches_per_sample": 10,
            "cache_protocol": "warm_reuse",
            "clock_policy": os.environ.get(
                "AMORA_SMOKE_CLOCK_POLICY", "uncontrolled"
            ),
        },
        "device": {
            "uuid": os.environ.get("AMORA_SMOKE_GPU_UUID", f"index:{device}"),
            "name": properties.name,
            "sm_clock_mhz": "uncontrolled",
        },
        "subject_identity": {
            "kernel_name": kernel_name,
            "ttgir_sha256": identity,
            "cubin_sha256": identity,
        },
        "subject_metadata": {
            "registers_per_thread": 0,
            "shared_memory_bytes": 0,
            "spill_count": 0,
            "total_warps": 4,
        },
        "measurement_axes": measurement_axes,
        "tool_versions": {
            "python": os.sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": str(torch.version.cuda),
        },
        "measurement_context": measurement_context,
    }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
