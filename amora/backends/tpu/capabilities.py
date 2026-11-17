"""Explicit target-instance discovery, isolated from callers' runtimes."""
from __future__ import annotations

import os
from pathlib import Path
import re
import socket

from .architecture import resolve_profile
from .artifacts import digest, file_hash
from .compatibility import runtime_identity


def device_index(device):
    match = re.fullmatch(r"tpu(?::([0-9]+))?", device)
    if not match:
        raise ValueError("device must be tpu or tpu:<local-index>")
    return int(match.group(1) or 0)


def discover_capabilities(device="tpu:0", profile=None, timeout=120):
    device_index(device)
    runtime = runtime_identity()
    if not runtime["packages"]["jax"] or not runtime["packages"]["libtpu"]:
        return {"backend": "tpu_jax", "status": "unsupported", "runtime_available": False,
                "devices": [], "runtime": runtime, "unsupported_reasons": ["runtime_missing"]}
    from .worker import run_worker
    result = run_worker("discover", {"device": device, "profile": profile}, timeout)
    if result.get("status") == "failed":
        return {"backend": "tpu_jax", "runtime_available": False, "devices": [],
                "runtime": runtime, "unsupported_reasons": [result["reason"]], **result}
    return result


def discover_in_process(device="tpu:0", profile=None):
    from .ownership import snapshot
    ownership = snapshot()
    if ownership["state"] != "exclusive":
        return {"backend": "tpu_jax", "status": "unsupported", "runtime_available": False,
                "devices": [], "runtime": runtime_identity(), "ownership": ownership,
                "unsupported_reasons": ["resource_busy" if ownership["state"] == "busy"
                                        else "ownership_unverified"]}
    import jax
    devices = jax.devices("tpu")
    local = jax.local_devices(backend="tpu")
    index = device_index(device)
    if index >= len(local):
        raise ValueError("device_not_addressable")
    selected = local[index]
    chip_version, hardware = None, {}
    try:
        from jax.experimental.pallas import tpu as pltpu
        info = pltpu.get_tpu_info()
        chip_version = str(info.chip_version)
        hardware["source"] = "JAX TPU registry resolved by runtime identity; architecture metadata"
        for key in ("chip_version", "num_cores", "num_mxus", "mxu_column_size", "mxu_row_size",
                    "vmem_capacity_bytes", "smem_capacity_bytes", "hbm_capacity_bytes", "num_lanes", "num_sublanes"):
            value = getattr(info, key, None)
            if value is not None:
                hardware[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    except (ImportError, AttributeError, NotImplementedError, ValueError, RuntimeError) as error:
        hardware["info_unavailable"] = str(error)
    architecture = resolve_profile(selected.device_kind, profile, chip_version)
    records = []
    for item in devices:
        record = {"id": item.id, "name": item.device_kind, "kind": item.device_kind,
                  "process_index": item.process_index,
                  "addressable": item in local}
        for key in ("coords", "core_on_chip"):
            value = getattr(item, key, None)
            if value is not None:
                record[key] = list(value) if isinstance(value, tuple) else value
        try:
            record["memory_stats"] = item.memory_stats()
        except (RuntimeError, NotImplementedError):
            record["memory_stats"] = None
        records.append(record)
    runtime = runtime_identity()
    target = {"instance": socket.gethostname(), "generation": architecture["generation"],
              "instance_config_sha256": file_hash(os.environ["AMORA_TPU_INSTANCE_CONFIG"])
              if os.environ.get("AMORA_TPU_INSTANCE_CONFIG") else None,
              "selected_device_id": selected.id, "local_index": index,
              "process_index": jax.process_index(), "process_count": jax.process_count(),
              "devices": records, "hardware_info": hardware,
              "visibility": {key: os.environ[key] for key in (
                  "TPU_VISIBLE_CHIPS", "TPU_PROCESS_BOUNDS", "TPU_CHIPS_PER_PROCESS_BOUNDS",
                  "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS", "TPU_TOPOLOGY",
                  "TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES", "TPU_SKIP_MDS_QUERY",
                  "TPU_TOPOLOGY_ALT", "TPU_TOPOLOGY_WRAP", "TPU_LOG_DIR")
                  if key in os.environ}}
    # Allocator usage is an observation, not part of stable target configuration identity.
    stable = {**target, "devices": [{k: v for k, v in d.items() if k != "memory_stats"}
                                    for d in records]}
    target["target_config_digest"] = digest(stable)
    return {"backend": "tpu_jax", "status": "ok", "runtime_available": True,
            "ownership": ownership,
            "devices": records, "selected_device": next(d for d in records if d["id"] == selected.id),
            "architecture": architecture, "target": target, "runtime": runtime,
            "features": {"runtime": {"state": "observed"}, "jax_jit": {"state": "observed"},
                         "pallas": {"state": "declared"}, "profiling": {"state": "unknown"},
                         "collectives": {"state": "declared" if len(local) > 1 else "unsupported"}},
            "unsupported_reasons": [] if architecture["generation"] != "unknown" else ["profile_unresolved"]}
