from amora.backends.nvidia import CudaToolchain
from amora.probes.nvidia.p0.topology.device_attributes import (
    device_attribute_probe,
    estimates_from_attributes,
)
from amora.probes.nvidia.p0.topology.occupancy import (
    OccupancyPoint,
    fit_residency_limit,
    generate_occupancy_points,
)


def test_estimates_from_attributes_maps_cuda_names_to_simulator_names():
    estimates = estimates_from_attributes(
        {
            "multiprocessor_count": 120,
            "warp_size": 32,
            "max_threads_per_multiprocessor": 2048,
        }
    )

    by_name = {estimate.name: estimate for estimate in estimates}

    assert by_name["gpgpu_sim_config::num_shader()"].value == 120
    assert by_name["shader_core_config::warp_size"].value == 32
    assert by_name["shader_core_config::n_thread_per_shader"].value == 2048


def test_device_attribute_probe_dry_run_reports_toolchain_inventory():
    result = device_attribute_probe(toolchain=CudaToolchain(search_path="/missing"))

    assert result.status == "dry_run"
    assert result.measurements["toolchain"]["tools"]["nvcc"]["available"] is False
    assert result.warnings


def test_generate_occupancy_points_builds_cross_product():
    points = generate_occupancy_points(
        block_sizes=(32, 64),
        register_counts=(16,),
        shared_memory_sizes=(0, 1024),
    )

    assert len(points) == 4
    assert points[0].warps_per_block == 1
    assert points[-1].threads_per_block == 64


def test_fit_residency_limit_identifies_register_limit():
    fitted = fit_residency_limit(
        OccupancyPoint(
            threads_per_block=256,
            registers_per_thread=128,
            dynamic_shared_memory_bytes=0,
        ),
        {
            "max_threads_per_multiprocessor": 2048,
            "max_blocks_per_multiprocessor": 16,
            "regs_per_multiprocessor": 65536,
            "shared_memory_per_multiprocessor": 100000,
        },
    )

    assert fitted["resident_blocks"] == 2
    assert fitted["limiting_resource"] == "registers"
