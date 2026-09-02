"""CPU-only protocol target for barrier-topology orchestration tests."""

from __future__ import annotations

import argparse
import json
import os


def _axes(panel: str, topology: str) -> dict[str, object]:
    return {
        "panel": panel,
        "topology": topology,
        "M": 128,
        "N": 128,
        "K": 256,
        "tile_shape": "128x128x64",
        "grid": "132x1x1",
        "pipeline_depth": 3,
        "producer_asymmetry_level": "transition",
        "cache_protocol": "disjoint_rotation",
        "cta_concurrency": 132,
        "address_partition_mapping": "round_robin",
        "launch_batch_size": 8,
        "clock_policy": "base",
    }


def _common(panel: str, topology: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "device": {"uuid": "GPU-synthetic", "name": "synthetic-h100"},
        "subject_identity": {
            "kernel_name": "synthetic_barrier_topology",
            "source_sha256": f"source-{topology}",
            "ttgir_sha256": f"ttgir-{topology}",
            "ptx_sha256": f"ptx-{topology}",
            "sass_sha256": f"sass-{topology}",
            "cubin_sha256": f"cubin-{topology}",
        },
        "subject_metadata": {
            "registers_per_thread": 64,
            "shared_memory_bytes": 65536,
            "spill_count": 0,
            "total_warps": 8,
            "total_threads": 256,
            "actor_count": 2,
            "compiled_queue_capacity": 3,
            "producer_lead": 2,
            "request_count": 2,
            "request_bytes": 32768,
            "numerical_result_valid": True,
            "numerical_result_fingerprint": "correct-result",
            "useful_operation_fingerprint": {
                "tma": "2x32768-byte-load",
                "hgmma": "m128n128k64",
            },
            "synchronization_instruction_fingerprint": {
                "digest": f"sync-{topology}",
                "non_sync_digest": "same-non-sync",
                "difference_classes": [
                    "synchronization",
                    "predicate",
                    "branch",
                    "barrier_address_setup",
                ],
            },
        },
        "measurement_axes": _axes(panel, topology),
        "tool_versions": {"synthetic_target": "1"},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("BT-EQ", "BT-CAUSAL"), required=True)
    parser.add_argument("--topology", required=True)
    args = parser.parse_args()
    payload = _common(args.panel, args.topology)
    lane = os.environ.get("AMORA_MEASUREMENT_LANE", "cuda_event_timing")
    if lane == "device_intervals":
        payload["kind"] = "cuda_device_intervals"
        payload["clock"] = "synthetic_globaltimer_ns"
        if args.panel == "BT-CAUSAL":
            fast_issue = 90.0 if args.topology == "pairwise" else 180.0
            intervals = (
                ("fast_tma_issue_to_barrier_release", 0.0, 80.0),
                ("slow_tma_issue_to_barrier_release", 0.0, 160.0),
                ("fast_barrier_release_to_consumer_issue", 80.0, fast_issue),
                ("slow_barrier_release_to_consumer_issue", 160.0, 180.0),
                ("wgmma_issue_to_completion", 180.0, 280.0),
                ("stage_buffer_release", 280.0, 290.0),
            )
        else:
            intervals = (
                ("tma_issue_to_barrier_release", 0.0, 100.0),
                ("barrier_release_to_consumer_issue", 100.0, 110.0),
                ("wgmma_issue_to_completion", 110.0, 210.0),
                ("stage_buffer_release", 210.0, 220.0),
            )
        payload["intervals"] = [
            {
                "name": name,
                "start_ns": start,
                "end_ns": end,
                "duration_ns": end - start,
            }
            for name, start, end in intervals
        ]
        payload["instrumentation"] = {
            "unprofiled_event_overhead_percent": 1.0,
            "overhead_threshold_percent": 5.0,
            "sass_bracketing_verified": True,
            "instruction_sequence_unchanged": True,
        }
    else:
        payload["kind"] = "cuda_event_timing"
        payload["timing"] = {
            "unit": "us_per_launch",
            "samples": [20.0, 20.01, 19.99],
            "warmup_launches": 2,
            "launches_per_sample": 8,
            "cache_protocol": "disjoint_rotation",
            "clock_policy": "base",
        }
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
