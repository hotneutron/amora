from amora.backends.nvidia.cuda import NvidiaCapabilities
from amora.probes.nvidia.baseline.topology.occupancy import (
    DEFAULT_BLOCK_SIZES,
    DEFAULT_REGISTERS_PER_THREAD,
    DEFAULT_SHARED_MEMORY_BYTES,
    generate_occupancy_points,
    run,
)


def test_generate_occupancy_points_is_full_cross_product():
    points = generate_occupancy_points(
        block_sizes=(32, 64),
        register_counts=(16, 32),
        shared_memory_sizes=(0, 1024),
    )

    assert len(points) == 2 * 2 * 2
    assert {(p.threads_per_block, p.registers_per_thread, p.dynamic_shared_memory_bytes) for p in points} == {
        (32, 16, 0), (32, 16, 1024), (32, 32, 0), (32, 32, 1024),
        (64, 16, 0), (64, 16, 1024), (64, 32, 0), (64, 32, 1024),
    }


def test_generate_occupancy_points_warps_per_block_round_up():
    points = generate_occupancy_points(block_sizes=(33,), register_counts=(16,), shared_memory_sizes=(0,))
    assert points[0].warps_per_block == 2


def test_run_reports_planning_sweep_without_gpu():
    caps = NvidiaCapabilities(cuda_available=False, gpu_available=False)
    results = run(caps)

    assert len(results) == 1
    data = results[0].to_dict()
    assert data["raw_observation"]["evidence_tier"] == "direct_metadata"
    assert data["normalized_measurement"]["fit_status"] == "direct"
    expected_count = (
        len(DEFAULT_BLOCK_SIZES)
        * len(DEFAULT_REGISTERS_PER_THREAD)
        * len(DEFAULT_SHARED_MEMORY_BYTES)
    )
    assert data["raw_observation"]["values"]["point_count"] == expected_count
    assert data["launch"]["mode"] == "planning"
