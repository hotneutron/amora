import json

from amora import cli


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
