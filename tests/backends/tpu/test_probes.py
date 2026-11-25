import os
import subprocess
import sys

import pytest


def test_probe_jit_cpu_correctness():
    jax = pytest.importorskip("jax")
    from amora.backends.tpu.probes import build_probe
    from amora.backends.tpu.portable_core import check_outputs
    device = jax.devices("cpu")[0]
    for probe,spec in (
        ("runtime.dispatch_latency", {}),
        ("vector.elementwise_throughput", {"shape": [7,17]}),
        ("matrix.matmul_throughput", {"shape": [7,17,11]}),
        ("matrix.dependent_chain", {"shape": [8,8,8], "iterations": 3}),
        ("transfer.host_device", {"shape": [7,17]}),
        ("transfer.host_device", {"shape": [7,17], "direction": "device_to_host"}),
    ):
        function,args,expected,metadata = build_probe(probe,spec,device)
        output = function() if metadata["mode"] == "transfer" else jax.jit(function)(*args)
        assert check_outputs(output,expected,matrix=metadata["matrix"])["passed"]


def test_collective_primitives_on_two_cpu_devices():
    pytest.importorskip("jax")
    script = '''import jax
from amora.backends.tpu.probes import build_probe
from amora.backends.tpu.portable_core import check_outputs
for primitive in ("psum", "all_gather", "ppermute"):
 f,args,expected,metadata=build_probe("interconnect.collectives", {"shape":[8], "devices":2,"primitive":primitive},jax.devices("cpu")[0])
 assert check_outputs(jax.jit(f)(*args),expected)["passed"],primitive
'''
    env = dict(os.environ,JAX_PLATFORMS="cpu",XLA_FLAGS="--xla_force_host_platform_device_count=2")
    subprocess.run([sys.executable,"-c",script],env=env,check=True,timeout=60)


def test_d2h_does_not_reuse_the_staged_arrays_host_cache(monkeypatch):
    jax=pytest.importorskip("jax")
    from amora.backends.tpu.probes import build_probe
    function,args,expected,metadata=build_probe("transfer.host_device",
        {"shape":[7,17],"direction":"device_to_host"},jax.devices("cpu")[0])
    copies=[];original=jax.device_put
    def put(value,*a,**kwargs):
        if value is args[0]:copies.append(kwargs.get("may_alias"))
        return original(value,*a,**kwargs)
    monkeypatch.setattr(jax,"device_put",put)
    function();function()
    assert copies==[False,False]
    assert metadata["device_copy_included"]


def test_multitile_dma_selects_regions_inside_whole_hbm_refs():
    jax=pytest.importorskip("jax")
    from jax.experimental.pallas import tpu as pltpu
    from amora.backends.tpu.probes import build_probe
    from amora.backends.tpu.portable_core import check_outputs
    with pltpu.force_tpu_interpret_mode():
        function,args,expected,_=build_probe("local_memory.transfer_sweep",
            {"shape":[256,256],"tile":[128,128]},jax.devices("cpu")[0])
        assert check_outputs(jax.jit(function)(*args),expected)["passed"]


def test_bounded_dense_chain_fixture():
    jax=pytest.importorskip("jax")
    import numpy as np
    from amora.backends.tpu.probes import build_probe
    from amora.backends.tpu.portable_core import check_outputs
    for iterations in (1,2,4):
        function,args,expected,metadata=build_probe("matrix.dependent_chain",
            {"shape":[32,32,32],"iterations":iterations,"data_pattern":"dyadic_hadamard"},jax.devices("cpu")[0])
        assert np.linalg.norm(np.asarray(args[1],dtype=np.float32),2)<=1
        assert check_outputs(jax.jit(function)(*args),expected,matrix=True)["passed"]
