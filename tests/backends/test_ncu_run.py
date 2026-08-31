"""Unit tests for NCU CSV parsing and command building (no GPU required)."""

import subprocess

import pytest

from amora.backends.nvidia.cuda import NvidiaCapabilities, ToolStatus
from amora.backends.nvidia import ncu
from amora.backends.nvidia.ncu import (
    NcuCommand,
    detect_sampling_interval_option,
    parse_sampling_interval_option,
)
from amora.backends.nvidia import ncu_run
from amora.backends.nvidia.ncu_run import (
    NcuUnavailable,
    parse_ncu_csv,
    parse_ncu_source_pc_sampling_csv,
    parse_ncu_source_pc_sampling_csv_with_metadata,
    run_command_pc_sampling,
    run_command_profiled,
    run_kernel_profiled,
)


# Representative `ncu --csv --page raw` output: wide format (one column per
# metric), preceded by the driver's own JSON and an NCU units row.
NCU_CSV = '''{"device_name":"NVIDIA H100","sweep":[]}
"ID","Process ID","Kernel Name","l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum","smsp__inst_executed.sum"
"","","","",""
"0","12345","amora_baseline_shared_bank_stride","1,234,567","8,192"
'''

NCU_CSV_ZERO = '''"ID","Kernel Name","l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"
"0","k","0"
'''


def test_parse_ncu_csv_extracts_numeric_metrics():
    metrics, rows = parse_ncu_csv(NCU_CSV)
    assert metrics["l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"] == 1234567.0
    assert metrics["smsp__inst_executed.sum"] == 8192.0
    # Units row (blank Kernel Name) is skipped; only the real kernel row counts.
    assert len(rows) == 1


def test_parse_ncu_csv_handles_zero_and_missing_header():
    metrics, _ = parse_ncu_csv(NCU_CSV_ZERO)
    assert metrics["l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum"] == 0.0
    # No CSV header at all -> empty.
    assert parse_ncu_csv("just some log lines\nno csv here") == ({}, [])


def test_ncu_command_argv_csv_raw():
    cmd = NcuCommand(
        executable="ncu",
        metrics=("a.sum", "b.sum"),
        target=("./driver", "--x"),
        csv=True,
        page="raw",
        launch_count=1,
        kernel_name="amora_kernel",
    )
    argv = cmd.argv()
    assert argv[0] == "ncu"
    assert "--csv" in argv
    assert "raw" in argv
    assert "--launch-count" in argv and "1" in argv
    assert "--kernel-name" in argv and "amora_kernel" in argv
    assert "--metrics" in argv and "a.sum,b.sum" in argv
    # target comes last
    assert argv[-2:] == ["./driver", "--x"]


def test_ncu_command_argv_supports_source_report_import_and_sampling():
    profile = NcuCommand(
        executable="ncu",
        metrics=(),
        target=("./driver",),
        section="SourceCounters",
        sampling_interval="7",
        output="/tmp/a.ncu-rep",
        force_overwrite=True,
    ).argv()
    assert profile[:3] == ["ncu", "--target-processes", "all"]
    assert ["--section", "SourceCounters"] == profile[
        profile.index("--section"):profile.index("--section") + 2
    ]
    assert ["--sampling-interval", "7"] == profile[
        profile.index("--sampling-interval"):profile.index("--sampling-interval") + 2
    ]
    assert "--force-overwrite" in profile

    imported = NcuCommand(
        executable="ncu",
        metrics=(),
        target=(),
        csv=True,
        page="source",
        print_source="sass",
        import_report="/tmp/a.ncu-rep",
    ).argv()
    assert imported == [
        "ncu",
        "--csv",
        "--import",
        "/tmp/a.ncu-rep",
        "--page",
        "source",
        "--print-source",
        "sass",
    ]


def test_ncu_command_argv_supports_2026_warp_sampling_option():
    argv = NcuCommand(
        executable="ncu",
        metrics=(),
        target=("python", "fixture.py"),
        section="SourceCounters",
        sampling_interval="5",
        sampling_interval_option="--warp-sampling-interval",
    ).argv()

    assert argv[-2:] == ["python", "fixture.py"]
    assert argv[argv.index("--warp-sampling-interval") :][:2] == [
        "--warp-sampling-interval",
        "5",
    ]
    assert "--sampling-interval" not in argv


def test_sampling_interval_option_detection_prefers_new_spelling(monkeypatch):
    assert (
        parse_sampling_interval_option("--sampling-interval <cycles>")
        == "--sampling-interval"
    )
    assert (
        parse_sampling_interval_option(
            "--sampling-interval X\n--warp-sampling-interval X"
        )
        == "--warp-sampling-interval"
    )

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            returncode=0,
            stdout="  --warp-sampling-interval <cycles>",
            stderr="",
        )

    monkeypatch.setattr(ncu.subprocess, "run", fake_run)
    option, error = detect_sampling_interval_option("/opt/ncu")

    assert option == "--warp-sampling-interval"
    assert error is None


def test_parse_ncu_source_pc_sampling_csv_extracts_reason_counts():
    text = '''==PROF== report
"PC","Function Name","SASS","smsp__pcsamp_warps_issue_stalled_wait","smsp__pcsamp_warps_issue_stalled_no_instruction"
"/*0060*/","amora_kernel","FFMA R0, R1, R2, R3 ;","12","3"
"0x70","amora_kernel","LDG.E R2, [R4] ;","1,024",""
'''

    samples = parse_ncu_source_pc_sampling_csv(text)

    assert len(samples) == 2
    assert samples[0].pc_offset == 0x60
    assert samples[0].function == "amora_kernel"
    assert samples[0].opcode == "FFMA R0, R1, R2, R3 ;"
    assert samples[0].stalls == {"wait": 12.0, "no_instructions": 3.0}
    assert samples[0].samples == 15.0
    assert samples[1].pc_offset == 0x70
    assert samples[1].stalls == {"wait": 1024.0}
    assert samples[1].raw_address == 0x70
    assert samples[1].address_base == 0


def test_parse_2026_source_page_aliases_and_normalizes_absolute_pcs():
    text = '''"Kernel Name","jit_kernel",
"Address","Source","stall_long_sb","stall_wait","stall_long_sb (Not Issued)","stall_wait (Not Issued)"
"0x7f000010","      LDG.E R2, [R4]","99","88","12","0"
"0x7f000020","      WARPGROUP.DEPBAR.LE gsb0, 0x0","77","66","0","3"
'''

    samples, metadata = parse_ncu_source_pc_sampling_csv_with_metadata(text)

    assert [sample.pc_offset for sample in samples] == [0, 0x10]
    assert [sample.raw_address for sample in samples] == [0x7F000010, 0x7F000020]
    assert all(sample.address_base == 0x7F000010 for sample in samples)
    assert samples[0].function == "jit_kernel"
    assert samples[0].opcode == "LDG.E R2, [R4]"
    assert samples[0].stalls == {"long_scoreboard": 12.0}
    assert samples[1].stalls == {"wait": 3.0}
    assert metadata["address_mode"] == "absolute"
    assert metadata["address_base"] == 0x7F000010
    assert metadata["column_mode"] == "not_issued"


def test_parse_absolute_pcs_normalizes_each_function_independently():
    text = '''"Address","Function Name","Source","stall_wait (Not Issued)"
"0x7f000010","kernel_a","NOP","1"
"0x7f000020","kernel_a","EXIT","2"
"0x7f100080","kernel_b","NOP","3"
'''

    samples, metadata = parse_ncu_source_pc_sampling_csv_with_metadata(text)

    assert [sample.pc_offset for sample in samples] == [0, 0x10, 0]
    assert metadata["address_base"] is None
    assert metadata["address_bases"] == {
        "kernel_a": 0x7F000010,
        "kernel_b": 0x7F100080,
    }


def test_run_command_profiled_profiles_python_target_without_build(monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        return subprocess.CompletedProcess(args, 0, NCU_CSV, "")

    monkeypatch.setattr(ncu_run.subprocess, "run", fake_run)
    caps = NvidiaCapabilities(
        tools={"ncu": ToolStatus("ncu", "/opt/ncu", True)},
    )

    result = run_command_profiled(
        ("/venv/bin/python", "fixture.py", "--depth", "3"),
        capabilities=caps,
        metrics=("smsp__inst_executed.sum",),
        kernel_name="regex:jit_kernel",
    )

    assert calls[0][-4:] == (
        "/venv/bin/python",
        "fixture.py",
        "--depth",
        "3",
    )
    assert result.target_command == calls[0][-4:]
    assert result.source_path is None
    assert result.metrics["smsp__inst_executed.sum"] == 8192.0


def test_run_command_pc_sampling_detects_option_and_records_target(monkeypatch, tmp_path):
    source_csv = '''"Kernel Name","jit_kernel",
"Address","Source","stall_wait (Not Issued)"
"0x7f000010","      NOP","4"
'''
    calls = []

    def fake_run(args, **kwargs):
        calls.append(tuple(args))
        if args[1] == "--help":
            return subprocess.CompletedProcess(
                args, 0, "--warp-sampling-interval <cycles>", ""
            )
        if "--import" in args:
            return subprocess.CompletedProcess(args, 0, source_csv, "")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(ncu.subprocess, "run", fake_run)
    caps = NvidiaCapabilities(
        tools={"ncu": ToolStatus("ncu", "/opt/ncu", True)},
    )
    target = ("/venv/bin/python", "fixture.py")

    result = run_command_pc_sampling(
        target,
        capabilities=caps,
        report_path=tmp_path / "jit.ncu-rep",
        sampling_interval="5",
    )

    profile = calls[1]
    assert profile[profile.index("--warp-sampling-interval") + 1] == "5"
    assert profile[-2:] == target
    assert result.target_command == target
    assert result.sampling_interval_option == "--warp-sampling-interval"
    assert result.parser_metadata["column_mode"] == "not_issued"
    assert result.samples[0].pc_offset == 0


def test_run_kernel_profiled_reports_ncu_stdout_when_stderr_is_empty(monkeypatch, tmp_path):
    binary = tmp_path / "driver"
    binary.write_text("binary")
    source = tmp_path / "driver.cu"
    source.write_text("source")

    monkeypatch.setattr(ncu_run, "build_executable", lambda *args, **kwargs: (binary, "source-sha"))

    def fake_run(args, **kwargs):
        return subprocess.CompletedProcess(
            args,
            returncode=255,
            stdout="==ERROR== ERR_NVGPUCTRPERM - no permission\n",
            stderr="",
        )

    monkeypatch.setattr(ncu_run.subprocess, "run", fake_run)

    caps = NvidiaCapabilities(
        tools={"ncu": ToolStatus("ncu", "/usr/bin/ncu", True)},
    )
    with pytest.raises(NcuUnavailable, match="ERR_NVGPUCTRPERM"):
        run_kernel_profiled(
            source,
            capabilities=caps,
            metrics=("sm__inst_executed.sum",),
            build_root=tmp_path,
        )
