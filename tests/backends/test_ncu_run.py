"""Unit tests for NCU CSV parsing and command building (no GPU required)."""

from amora.backends.nvidia.ncu import NcuCommand
from amora.backends.nvidia.ncu_run import parse_ncu_csv


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
