"""Unit tests for SASS validation parsing and gating (no GPU required)."""

from amora.backends.nvidia.sass import (
    SassExpectation,
    extract_kernel_section,
    gate_decision,
    parse_sass_opcodes,
    validate_sass,
)


# A trimmed, representative cuobjdump -sass dump for an FP32 dependent chain.
FFMA_SASS = """
	code for sm_90
		Function : amora_baseline_fp32_dependent_chain
	.headerflags    @"EF_CUDA_SM90"
        /*0000*/                   MOV R1, c[0x0][0x28] ;
        /*0010*/                   CS2R R2, SRZ ;
        /*0020*/                   FFMA R0, R0, R4, R5 ;
        /*0030*/                   FFMA R6, R0, R4, R5 ;
        /*0040*/                   FFMA R7, R6, R4, R5 ;
        /*0050*/                   FFMA R8, R7, R4, R5 ;
        /*0060*/                   EXIT ;
		Function : some_other_kernel
        /*0000*/                   LDG.E R2, [R4] ;
        /*0010*/                   EXIT ;
"""

# A dump where the compiler turned the chain into global loads (should reject).
SPILLED_SASS = """
		Function : amora_baseline_fp32_dependent_chain
        /*0000*/                   LDG.E R0, [R4] ;
        /*0010*/                   STL [R1], R0 ;
        /*0020*/                   EXIT ;
"""


def test_parse_sass_opcodes_histogram_and_order():
    hist, ordered = parse_sass_opcodes(FFMA_SASS)
    assert hist.get("FFMA") == 4
    assert hist.get("LDG") == 1  # from the other kernel section
    # Ordered list preserves register operands.
    ffmas = [regs for fam, regs in ordered if fam == "FFMA"]
    assert ffmas[1][0] == 6  # dst of 2nd FFMA is R6


def test_extract_kernel_section_scopes_to_symbol():
    section = extract_kernel_section(FFMA_SASS, "amora_baseline_fp32_dependent_chain")
    hist, _ = parse_sass_opcodes(section)
    assert hist.get("FFMA") == 4
    assert "LDG" not in hist  # the other kernel is excluded


def test_validate_pass_for_clean_dependent_chain():
    exp = SassExpectation(
        kernel_symbol="amora_baseline_fp32_dependent_chain",
        required_opcodes={"FFMA": 4},
        forbidden_opcodes=("LDG", "STG", "LDL", "STL"),
        require_dependency=True,
        dependency_opcode="FFMA",
    )
    v = validate_sass(FFMA_SASS, exp, disassembly_hash="abc")
    assert v.validated is True
    assert gate_decision(v, exp) == "pass"
    assert v.dependency_confirmed is True


def test_validate_reject_on_forbidden_and_missing():
    exp = SassExpectation(
        kernel_symbol="amora_baseline_fp32_dependent_chain",
        required_opcodes={"FFMA": 4},
        forbidden_opcodes=("LDG", "STG", "LDL", "STL"),
    )
    v = validate_sass(SPILLED_SASS, exp)
    assert v.validated is False
    assert gate_decision(v, exp) == "reject"
    assert any("forbidden" in viol for viol in v.violations)
    assert any("missing required FFMA" in viol for viol in v.violations)


def test_validate_downgrade_on_low_count():
    exp = SassExpectation(
        kernel_symbol="amora_baseline_fp32_dependent_chain",
        required_opcodes={"FFMA": 100},  # more than the 4 present
    )
    v = validate_sass(FFMA_SASS, exp)
    assert v.validated is True  # present, just low
    assert gate_decision(v, exp) == "downgrade"
    assert any(viol.startswith("low FFMA") for viol in v.violations)


def test_count_distinct_registers_via_expectation():
    exp = SassExpectation(
        kernel_symbol="amora_baseline_fp32_dependent_chain",
        required_opcodes={"FFMA": 4},
        count_registers_opcode="FFMA",
    )
    v = validate_sass(FFMA_SASS, exp)
    # FFMA operands in the fixture use registers R0,R4,R5,R6,R7,R8 -> 6 distinct.
    assert v.register_count == 6
