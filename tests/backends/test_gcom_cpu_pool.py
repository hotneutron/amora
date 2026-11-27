"""Dynamic point allocation stays inside the admitted CPU/NUMA budget."""
import pytest
from amora.backends.gcom_cuda.portable_v2_resources import ReplayCpuPool


def test_large_and_small_points_use_disjoint_sets_and_recycle_them():
    pool=ReplayCpuPool([dict(numa_node=0,cpus=list(range(40))),dict(numa_node=1,cpus=list(range(40,80)))])
    large_a=pool.acquire(32);large_b=pool.acquire(32);small_a=pool.acquire(8);small_b=pool.acquire(8)
    allocations=[large_a,large_b,small_a,small_b]
    cpus=[c for a in allocations for c in a["cpus"]]
    assert len(cpus)==len(set(cpus))==80
    assert large_a["omp_threads"]==large_b["omp_threads"]==32
    assert small_a["omp_threads"]==small_b["omp_threads"]==8
    pool.release(large_a)
    recycled=pool.acquire(32)
    assert recycled==large_a
    with pytest.raises(ValueError):
        pool.release(large_b);pool.release(large_b)


def test_small_remainder_fills_capacity_without_numa_crossing():
    pool=ReplayCpuPool([dict(numa_node=0,cpus=list(range(14)))])
    first=pool.acquire(8);second=pool.acquire(32)
    assert first["omp_threads"]==8 and second["omp_threads"]==6
    assert not set(first["cpus"]) & set(second["cpus"])


def test_duplicate_cpu_admission_is_rejected():
    with pytest.raises(ValueError):ReplayCpuPool([dict(numa_node=0,cpus=[0,1]),dict(numa_node=1,cpus=[1,2])])


def test_four_thread_policy_fills_48_independent_instances():
    pool=ReplayCpuPool([dict(numa_node=0,cpus=list(range(96))),dict(numa_node=1,cpus=list(range(96,192)))],fallback_threads=4)
    slots=[pool.acquire(4) for _ in range(48)]
    cpus=[cpu for slot in slots for cpu in slot["cpus"]]
    assert len(cpus)==len(set(cpus))==192
    assert all(slot["omp_threads"]==4 for slot in slots)
    for slot in slots:pool.release(slot)


def test_four_thread_policy_never_expands_a_small_remainder_to_eight():
    pool=ReplayCpuPool([dict(numa_node=0,cpus=list(range(10)))],fallback_threads=4)
    slots=[pool.acquire(4) for _ in range(3)]
    assert [s["omp_threads"] for s in slots]==[4,4,2]


def test_runtime_policy_validation(tmp_path):
    import json
    from amora.benchmarking.portable_v2.gcom_campaign import Campaign
    campaign=Campaign.__new__(Campaign);campaign.root=tmp_path
    assert campaign.resource_policy()["max_instances"]==24
    (tmp_path/"runtime-policy.json").write_text(json.dumps(dict(replay_threads=4,large_replay_threads=4,max_instances=48,wave_cases=64)))
    assert campaign.resource_policy()==dict(replay_threads=4,large_replay_threads=4,max_instances=48,wave_cases=64)
    (tmp_path/"runtime-policy.json").write_text('{"max_instances":0}')
    with pytest.raises(ValueError):campaign.resource_policy()
