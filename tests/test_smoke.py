import json
import subprocess
import sys

from amora.probes.nvidia.p0.arithmetic_latency.sources import arithmetic_source_probe


def test_smoke_import_and_cuda_source_discovery():
    result = arithmetic_source_probe()

    assert result.status == "planned"
    assert len(result.artifacts) == 2
    assert all(artifact.sha256 for artifact in result.artifacts)


def test_smoke_cli_dry_run_outputs_json_profile():
    completed = subprocess.run(
        [sys.executable, "-m", "amora", "--target", "nvidia", "--tier", "p0", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )

    profile = json.loads(completed.stdout)

    assert profile["target"]["vendor"] == "nvidia"
    assert profile["target"]["dry_run"] is True
    assert any(result["tier"] == "P0" for result in profile["raw_results"])
