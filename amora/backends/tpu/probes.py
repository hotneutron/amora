"""Data-dependent JAX, Pallas and explicit collective probe implementations."""
from __future__ import annotations


def build_probe(probe_id, spec, device):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import hashlib
    from .artifacts import digest, file_hash
    from amora.probes.tpu.baseline import PROBES
    from benchmark_generators.portable_core.base.backends import quantize_bf16_numpy
    from .runner import drain
    kind = PROBES[probe_id]["kind"]
    rng = np.random.default_rng(spec.get("seed", 0))
    shape = tuple(spec.get("shape", [128, 128]))
    metadata = {"kind": kind, "shape": list(shape), "mode": "jax_jit"}
    if kind in ("matmul", "matrix_chain"):
        m, n, k = tuple(spec.get("shape", [256, 256, 256]))
        left = quantize_bf16_numpy(rng.uniform(-0.5, 0.5, (m, k)).astype(np.float32))
        right = quantize_bf16_numpy(rng.uniform(-0.5, 0.5, (k, n)).astype(np.float32))
        count = spec.get("iterations", 1) if kind == "matrix_chain" else 1
        if kind == "matrix_chain" and (m != n or n != k):
            raise ValueError("matrix chain requires square matrices")
        pattern=spec.get("data_pattern","random")
        if pattern=="dyadic_hadamard":
            if kind!="matrix_chain" or k & (k-1):
                raise ValueError("dyadic Hadamard fixture requires a power-of-two square chain")
            hadamard=np.ones((1,1),dtype=np.float32)
            while hadamard.shape[0]<k:
                hadamard=np.block([[hadamard,hadamard],[hadamard,-hadamard]])
            right=hadamard*np.float32(2.0**(-int(np.ceil(np.log2(np.sqrt(k))))))
            left=rng.integers(-128,129,(m,k)).astype(np.float32)/np.float32(256)
        def function(a, b):
            for _ in range(count):
                a = jax.lax.dot_general(a.astype(jnp.bfloat16), b,
                    (((1,), (0,)), ((), ())), precision=jax.lax.Precision.HIGHEST,
                    preferred_element_type=jnp.float32)
            return (a,)
        expected = left
        for _ in range(count):
            expected = (quantize_bf16_numpy(expected).astype(np.float64) @ right.astype(np.float64)).astype(np.float32)
        raw = (left, right)
        arguments = tuple(jax.device_put(x, device).astype(jnp.bfloat16) for x in raw)
        metadata.update(shape=[m,n,k], logical_flops=2*m*n*k*count, matrix=True,data_pattern=pattern)
    elif kind in ("dispatch", "vector", "pallas_stream", "pallas_dma", "transfer"):
        if kind == "dispatch": shape = (1,)
        raw = (rng.uniform(-0.5, 0.5, shape).astype(np.float32),)
        arguments = tuple(jax.device_put(x, device) for x in raw)
        expected = (raw[0] * np.float32(1.5)).astype(np.float32)
        function = lambda x: (x + x * jnp.float32(0.5),)
        metadata.update(shape=list(shape), logical_flops=2*raw[0].size,
                        requested_bytes=2*raw[0].nbytes, matrix=False)
        if kind == "transfer":
            expected = raw[0]
            if spec.get("direction", "host_to_device") == "host_to_device":
                function = lambda: (jax.device_put(raw[0], device, may_alias=False),)
            else:
                def function():
                    fresh = jax.device_put(arguments[0], device, may_alias=False)
                    return (np.asarray(jax.device_get(fresh)),)
            metadata.update(mode="transfer", requested_bytes=raw[0].nbytes,
                            direction=spec.get("direction", "host_to_device"),
                            host_cache_policy="fresh device buffer for every D2H invocation",
                            device_copy_included=spec.get("direction", "host_to_device") == "device_to_host")
        elif kind in ("pallas_stream", "pallas_dma"):
            from jax.experimental import pallas as pl
            from jax.experimental.pallas import tpu as pltpu
            if len(shape) != 2:
                raise ValueError("Pallas streaming requires a two-dimensional shape")
            tile = tuple(spec.get("tile", [128, 128]))
            if len(tile) != 2 or any(s % t for s, t in zip(shape, tile)):
                raise ValueError("Pallas streaming shape must be divisible by tile")
            def kernel(x, y):
                y[...] = x[...] + x[...] * jnp.float32(0.5)
            if kind == "pallas_dma":
                def dma_kernel(x, y, local, semaphore):
                    region=(pl.ds(pl.program_id(0)*tile[0],tile[0]),
                            pl.ds(pl.program_id(1)*tile[1],tile[1]))
                    pltpu.async_copy(x.at[region], local, semaphore).wait()
                    local[...] = local[...] + local[...] * jnp.float32(0.5)
                    pltpu.async_copy(local, y.at[region], semaphore).wait()
                call = pl.pallas_call(dma_kernel, out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
                    grid=tuple(s//t for s,t in zip(shape,tile)),
                    in_specs=[pl.BlockSpec(shape,lambda i,j:(0,0),memory_space=pltpu.HBM)],
                    out_specs=pl.BlockSpec(shape,lambda i,j:(0,0),memory_space=pltpu.HBM),
                    scratch_shapes=(pltpu.VMEM(tile,jnp.float32),pltpu.SemaphoreType.DMA))
            else:
                call = pl.pallas_call(kernel, out_shape=jax.ShapeDtypeStruct(shape, jnp.float32),
                    grid=tuple(s//t for s, t in zip(shape, tile)),
                    in_specs=[pl.BlockSpec(tile, lambda i, j: (i, j))],
                    out_specs=pl.BlockSpec(tile, lambda i, j: (i, j)))
            function = lambda x: (call(x),)
            metadata.update(mode="pallas", tile=list(tile),
                            memory_spaces="HBM input/output; VMEM tile refs",
                            dma_semaphores=kind == "pallas_dma",
                            interpretation="controlled tiled stream; physical HBM traffic unmeasured")
    elif kind == "collective":
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
        devices = jax.local_devices(backend=device.platform)
        count = spec.get("devices", len(devices))
        if count < 2 or count > len(devices):
            raise ValueError("collective requires 2..local-device-count participants")
        payload_shape = tuple(spec.get("shape", [128]))
        raw = (rng.uniform(-0.5, 0.5, (count, *payload_shape)).astype(np.float32),)
        mesh = Mesh(np.array(devices[:count]), ("device",))
        sharding = NamedSharding(mesh, P("device"))
        arguments = (jax.device_put(raw[0], sharding),)
        primitive = spec.get("primitive", "psum")
        routes = tuple((i, (i+1) % count) for i in range(count))
        def body(x):
            if primitive == "psum": return jax.lax.psum(x, "device")
            if primitive == "all_gather": return jax.lax.all_gather(x, "device", axis=0, tiled=True)
            if primitive == "ppermute": return jax.lax.ppermute(x, "device", routes)
            raise ValueError("unknown collective primitive")
        mapped = jax.shard_map(body, mesh=mesh, in_specs=P("device"),
                              out_specs=P("device") if primitive != "all_gather" else P(),
                              check_vma=False)
        function = lambda x: (mapped(x),)
        if primitive == "psum": expected = np.repeat(np.sum(raw[0], axis=0, keepdims=True), count, axis=0)
        elif primitive == "all_gather": expected = raw[0]
        else: expected = np.roll(raw[0], 1, axis=0)
        metadata.update(mode="shard_map", primitive=primitive, participants=count,
                        participant_device_ids=[item.id for item in devices[:count]],
                        participant_coordinates=[list(getattr(item,"coords",())) for item in devices[:count]],
                        routes=[list(x) for x in routes] if primitive == "ppermute" else None,
                        requested_bytes=raw[0].nbytes, matrix=False)
    else:
        raise ValueError(f"unsupported probe: {probe_id}")
    drain(arguments)
    metadata["input_digest"] = digest([{"shape": list(x.shape), "dtype": str(x.dtype),
        "sha256": hashlib.sha256(x.tobytes()).hexdigest()} for x in raw])
    metadata["probe_source_sha256"] = file_hash(__file__)
    return function, arguments, (expected,), metadata
