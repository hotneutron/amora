"""Static probe registry; imports no accelerator framework."""

PROBES = {
    "topology.device_attributes": {"feature": "runtime", "kind": "metadata"},
    "runtime.dispatch_latency": {"feature": "jax_jit", "kind": "dispatch"},
    "matrix.matmul_throughput": {"feature": "jax_jit", "kind": "matmul"},
    "matrix.dependent_chain": {"feature": "jax_jit", "kind": "matrix_chain"},
    "vector.elementwise_throughput": {"feature": "jax_jit", "kind": "vector"},
    "transfer.host_device": {"feature": "runtime", "kind": "transfer"},
    "global_memory.streaming": {"feature": "pallas", "kind": "pallas_stream"},
    "local_memory.transfer_sweep": {"feature": "pallas", "kind": "pallas_dma"},
    "interconnect.collectives": {"feature": "collectives", "kind": "collective"},
}


def list_probes():
    return [{"probe_id": name, "implemented": True, "status": "capability_gated",
             "required_feature": spec["feature"]} for name, spec in PROBES.items()]
