"""SASS validation for NVIDIA kernel probes.

A kernel-bound probe trusts that the source it timed compiled to the intended
instruction stream. This module verifies that by disassembling the probe's
device code (cubin) and checking an opcode-level :class:`SassExpectation`:

- required opcodes appear with at least a minimum count,
- forbidden opcodes do not appear in the timed kernel,
- (best effort) the destination of one target op feeds the next (dependency).

The parsing and gating logic here are pure functions over disassembly *text*,
so they can be unit-tested against captured ``cuobjdump -sass`` fixtures without
a GPU. Actual cubin building / disassembly lives in the runner.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping


# A SASS line looks roughly like:
#   /*0010*/   FFMA R0, R0, R4, R0 ;
# or with predication:
#   /*0020*/ @!P0 BRA `(.L_x_1) ;
_SASS_LINE = re.compile(
    r"/\*[0-9a-fA-F]+\*/\s*(?:@!?P\d+\s+)?(?P<op>[A-Z][A-Z0-9_.]*)\b(?P<args>[^;]*);?"
)
# Opcode "family" is the mnemonic before the first '.', uppercased
# (e.g. LDG.E.SYS -> LDG, FFMA -> FFMA).
_OPCODE_FAMILY = re.compile(r"^([A-Z][A-Z0-9]*)")
# Register operands like R0, R12, R255 (not RZ which is the zero register).
_REG = re.compile(r"\bR(\d+)\b")


@dataclass(frozen=True)
class SassExpectation:
    """What a probe's timed kernel must (and must not) contain."""

    kernel_symbol: str
    required_opcodes: Mapping[str, int] = field(default_factory=dict)
    forbidden_opcodes: tuple[str, ...] = ()
    require_dependency: bool = False
    # Opcode family whose register dataflow defines the dependency chain.
    dependency_opcode: str | None = None


@dataclass(frozen=True)
class SassValidation:
    """Structured outcome of checking a kernel's SASS against expectations."""

    validated: bool
    disassembly_hash: str | None
    opcode_histogram: dict[str, int]
    satisfied: list[str]
    violations: list[str]
    dependency_confirmed: bool | None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "validated": self.validated,
            "disassembly_hash": self.disassembly_hash,
            "opcode_histogram": dict(self.opcode_histogram),
            "satisfied": list(self.satisfied),
            "violations": list(self.violations),
            "dependency_confirmed": self.dependency_confirmed,
            "reason": self.reason,
        }


def _opcode_family(op: str) -> str:
    match = _OPCODE_FAMILY.match(op)
    return match.group(1) if match else op


def extract_kernel_section(sass_text: str, kernel_symbol: str) -> str:
    """Return the disassembly lines belonging to ``kernel_symbol``.

    ``cuobjdump -sass`` prints a ``Function : <name>`` header per kernel. If the
    symbol cannot be located the whole text is returned (callers still get a
    histogram; scoping just becomes coarser).
    """

    lines = sass_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if kernel_symbol in line and ("Function" in line or ".text." in line or "Function :" in line):
            start = i
            break
    if start is None:
        return sass_text
    # End at the next function header.
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if "Function :" in lines[j] or lines[j].startswith(".text."):
            end = j
            break
    return "\n".join(lines[start:end])


def parse_sass_opcodes(sass_text: str) -> tuple[dict[str, int], list[tuple[str, list[int]]]]:
    """Return (opcode_family_histogram, ordered [(family, [dst, src...]) ...]).

    The second element preserves instruction order with the register numbers in
    operand order, enabling a best-effort dependency check.
    """

    histogram: dict[str, int] = {}
    ordered: list[tuple[str, list[int]]] = []
    for match in _SASS_LINE.finditer(sass_text):
        op = match.group("op")
        family = _opcode_family(op)
        # Skip obvious non-instruction tokens.
        if family in {"Function", "PROGRAM", "SCHI"}:
            continue
        histogram[family] = histogram.get(family, 0) + 1
        regs = [int(r) for r in _REG.findall(match.group("args") or "")]
        ordered.append((family, regs))
    return histogram, ordered


def _dependency_confirmed(ordered: list[tuple[str, list[int]]], opcode: str) -> bool | None:
    """Best-effort: do consecutive ``opcode`` instructions chain dst -> src?

    Returns True/False, or None when there are too few target ops to judge.
    """

    targets = [regs for fam, regs in ordered if fam == opcode and regs]
    if len(targets) < 2:
        return None
    chained = 0
    comparisons = 0
    for prev, cur in zip(targets, targets[1:]):
        dst = prev[0]
        srcs = cur[1:] if len(cur) > 1 else cur
        comparisons += 1
        if dst in srcs:
            chained += 1
    if comparisons == 0:
        return None
    # Allow unrolling/renaming slack: a majority chaining counts as confirmed.
    return chained >= max(1, comparisons // 2)


def validate_sass(
    sass_text: str,
    expectation: SassExpectation,
    *,
    disassembly_hash: str | None = None,
) -> SassValidation:
    """Check disassembly text against ``expectation`` and return the outcome."""

    section = extract_kernel_section(sass_text, expectation.kernel_symbol)
    histogram, ordered = parse_sass_opcodes(section)

    satisfied: list[str] = []
    violations: list[str] = []

    for op, min_count in expectation.required_opcodes.items():
        family = _opcode_family(op)
        count = histogram.get(family, 0)
        if count >= min_count:
            satisfied.append(f"{family}>={min_count} ({count})")
        elif count == 0:
            violations.append(f"missing required {family} (expected >={min_count})")
        else:
            violations.append(f"low {family}: {count} < {min_count}")

    for op in expectation.forbidden_opcodes:
        family = _opcode_family(op)
        if histogram.get(family, 0) > 0:
            violations.append(f"forbidden {family} present ({histogram[family]})")

    dependency_confirmed: bool | None = None
    if expectation.require_dependency:
        dep_op = _opcode_family(expectation.dependency_opcode or "")
        dependency_confirmed = _dependency_confirmed(ordered, dep_op) if dep_op else None

    # "validated" means no hard violations. Low-count / unconfirmed-dependency are
    # downgrade signals handled by the caller, not hard failures here.
    hard_violation = any(
        v.startswith("missing required ") or v.startswith("forbidden ") for v in violations
    )
    validated = not hard_violation
    reason = "; ".join(violations) if violations else None

    return SassValidation(
        validated=validated,
        disassembly_hash=disassembly_hash,
        opcode_histogram=histogram,
        satisfied=satisfied,
        violations=violations,
        dependency_confirmed=dependency_confirmed,
        reason=reason,
    )


def gate_decision(validation: SassValidation, expectation: SassExpectation) -> str:
    """Map a validation outcome to one of: 'pass', 'downgrade', 'reject'.

    - reject: a required opcode is missing or a forbidden opcode is present;
      the timed region does not measure what the probe claims.
    - downgrade: required ops present but below count, or a requested dependency
      could not be confirmed.
    - pass: all good.
    """

    if not validation.validated:
        return "reject"
    low = any(v.startswith("low ") for v in validation.violations)
    dep_unconfirmed = (
        expectation.require_dependency and validation.dependency_confirmed is False
    )
    if low or dep_unconfirmed:
        return "downgrade"
    return "pass"
