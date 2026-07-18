import json

import pytest

from amora import cli
from amora.benchmarking.classification import ClassificationResult, build_classification_manifest
from amora.benchmarking.detailed import (
    build_detailed_comparison,
    write_detailed_comparison,
)
from amora.benchmarking.materialize import load_manifest


def test_cli_lists_nvidia_probes(capsys):
    code = cli.main(["nvidia", "list"])

    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert code == 0
    assert "probes" in data
    assert any(probe["probe_id"] == "topology.device_attributes" for probe in data["probes"])


def test_cli_run_all_writes_report(tmp_path):
    output = tmp_path / "nvidia-baseline.json"

    code = cli.main(["nvidia", "run", "--all", "--output", str(output)])

    assert code == 0
    data = json.loads(output.read_text())
    assert data["schema_version"] == 1
    assert data["results"]


def test_cli_lists_and_materializes_benchmarks(tmp_path, capsys):
    code = cli.main(["benchmarks", "list"])

    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["benchmarks"][0]["benchmark_id"] == "ppp_canonical"

    output = tmp_path / "manifest.json"
    code = cli.main(
        [
            "benchmarks",
            "materialize",
            "ppp_canonical",
            "--cases",
            "9",
            "--seed",
            "7",
            "--output",
            str(output),
        ]
    )

    response = json.loads(capsys.readouterr().out)
    manifest = json.loads(output.read_text())
    assert code == 0
    assert response["case_count_materialized"] == 9
    assert manifest["case_count_materialized"] == 9


def test_cli_partial_benchmark_classification_does_not_assign_ranks(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    classification_path = tmp_path / "classification.json"
    assert cli.main(
        [
            "benchmarks",
            "materialize",
            "ppp_canonical",
            "--cases",
            "9",
            "--seed",
            "7",
            "--output",
            str(manifest_path),
        ]
    ) == 0
    capsys.readouterr()

    code = cli.main(
        [
            "benchmarks",
            "classify",
            "ppp_canonical",
            "--manifest",
            str(manifest_path),
            "--limit",
            "2",
            "--output",
            str(classification_path),
        ]
    )

    response = json.loads(capsys.readouterr().out)
    classification = json.loads(classification_path.read_text())
    assert code == 0
    assert response["classification"] == str(classification_path)
    assert response["classification_digest"] == classification["classification_digest"]
    assert response["rank_counts"] == {}
    assert classification["case_coverage_complete"] is False
    assert classification["rank_assignments"] == {}


def test_cli_detail_rejects_incomplete_classification(tmp_path, capsys):
    manifest_path = tmp_path / "manifest.json"
    classification_path = tmp_path / "classification.json"
    assert cli.main(
        [
            "benchmarks",
            "materialize",
            "ppp_canonical",
            "--cases",
            "9",
            "--seed",
            "7",
            "--output",
            str(manifest_path),
        ]
    ) == 0
    capsys.readouterr()
    assert cli.main(
        [
            "benchmarks",
            "classify",
            "ppp_canonical",
            "--manifest",
            str(manifest_path),
            "--limit",
            "2",
            "--output",
            str(classification_path),
        ]
    ) == 0
    capsys.readouterr()

    with pytest.raises(ValueError, match="complete classification overlay"):
        cli.main(
            [
                "benchmarks",
                "detail",
                "ppp_canonical",
                "--manifest",
                str(manifest_path),
                "--classification",
                str(classification_path),
                "--size-rank",
                "small",
            ]
        )


def _complete_classification_for_cli(manifest):
    return build_classification_manifest(
        case_set_digest=manifest.case_set_digest,
        target=manifest.target,
        results=[
            ClassificationResult(
                case_key=case.case_key,
                status="classified",
                total_instructions=float(index + 1),
            )
            for index, case in enumerate(manifest.cases)
        ],
        expected_case_keys=[case.case_key for case in manifest.cases],
    )


def test_cli_review_writes_small_marker_and_detail_rejects_ungated_medium(
    tmp_path, capsys
):
    manifest_path = tmp_path / "manifest.json"
    assert cli.main(
        [
            "benchmarks",
            "materialize",
            "ppp_canonical",
            "--cases",
            "9",
            "--seed",
            "7",
            "--output",
            str(manifest_path),
        ]
    ) == 0
    capsys.readouterr()
    manifest = load_manifest(manifest_path)
    classification = _complete_classification_for_cli(manifest)
    classification_path = tmp_path / "classification.json"
    classification_path.write_text(
        json.dumps(classification.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    small = build_detailed_comparison(
        manifest=manifest,
        classification=classification,
        size_rank="small",
        hardware_results=[],
        simulation_results=[],
        run_id="small-cli-run",
    )
    small_comparison = write_detailed_comparison(small, tmp_path / "small-comparison")[
        "json"
    ]

    with pytest.raises(ValueError, match="requires an accepted small review marker"):
        cli.main(
            [
                "benchmarks",
                "detail",
                "ppp_canonical",
                "--manifest",
                str(manifest_path),
                "--classification",
                str(classification_path),
                "--size-rank",
                "medium",
            ]
        )

    review_path = tmp_path / "small-review.json"
    assert cli.main(
        [
            "benchmarks",
            "review",
            "ppp_canonical",
            "--manifest",
            str(manifest_path),
            "--classification",
            str(classification_path),
            "--comparison",
            str(small_comparison),
            "--reviewer",
            "test",
            "--reviewed-at",
            "2026-07-18T00:00:00+00:00",
            "--output",
            str(review_path),
        ]
    ) == 0
    response = json.loads(capsys.readouterr().out)
    review = json.loads(review_path.read_text())
    assert response["review_marker"] == str(review_path)
    assert response["accepted_run_ids"] == ["small-cli-run"]
    assert review["reviewed_rank"] == "small"
