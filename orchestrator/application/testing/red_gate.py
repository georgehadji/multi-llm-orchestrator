"""RED-gate decision logic — pure functions (E-2).

The RED-gate proves a suite is non-vacuous before the implementation is
generated: a suite that passes against a ``NotImplementedError`` stub
asserts nothing. This module holds the *decision* rules only (pure, no
I/O); execution lives in the generator (it needs a materialized workspace
and the runner port).

Mode ladder (``ORCH_RED_GATE``):
* ``off``      — gate skipped entirely
* ``warn``     — (default, one release) vacuity measured and logged, never
                 acted on
* ``enforce``  — vacuous tests discarded; >50% vacuity triggers one
                 corrective regeneration, then task failure
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RedGateResult:
    """Outcome of a RED-gate pass."""

    assertion_ok: bool
    total: int = 0
    vacuous_node_ids: tuple[str, ...] = ()
    decision: str = "proceed"  # proceed | regenerate | fail
    mode: str = "warn"
    diagnosis: str = ""
    discard_node_ids: tuple[str, ...] = ()
    assertion_errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def vacuous_ratio(self) -> float:
        """Fraction of executed tests that passed against the stub."""
        if self.total == 0:
            return 0.0
        return len(self.vacuous_node_ids) / self.total

    @property
    def executed(self) -> int:
        return self.total


def decide_red_gate(
    *,
    assertion_ok: bool,
    assertion_errors: tuple[str, ...],
    total: int,
    vacuous_node_ids: tuple[str, ...],
    mode: str,
    retry_used: bool = False,
) -> RedGateResult:
    """Pure decision: what should the pipeline do with this suite?

    Args:
        assertion_ok: Whether the AST assertion floor passed.
        assertion_errors: Floor violations (diagnostics).
        total: Number of tests executed against the stub.
        vacuous_node_ids: Node ids that PASSED against the stub.
        mode: ``off`` | ``warn`` | ``enforce``.
        retry_used: Whether a corrective regeneration already happened.

    Returns:
        A RedGateResult whose ``decision`` is one of proceed/regenerate/fail.
    """
    if mode == "off":
        return RedGateResult(
            assertion_ok=True,
            total=total,
            vacuous_node_ids=vacuous_node_ids,
            decision="proceed",
            mode=mode,
        )

    if not assertion_ok:
        diagnosis = "Assertion floor failed: " + "; ".join(assertion_errors[:3])
        if mode == "enforce":
            decision = "regenerate" if not retry_used else "fail"
        else:
            decision = "proceed"  # warn: log only
        return RedGateResult(
            assertion_ok=False,
            total=total,
            vacuous_node_ids=vacuous_node_ids,
            decision=decision,
            mode=mode,
            diagnosis=diagnosis,
            assertion_errors=assertion_errors,
        )

    if total == 0:
        # No tests executed against the stub (collection error) — not a
        # vacuity verdict; let the normal pipeline handle it.
        return RedGateResult(
            assertion_ok=True,
            total=0,
            decision="proceed",
            mode=mode,
        )

    vacuous_ratio = len(vacuous_node_ids) / total
    diagnosis = (
        f"{len(vacuous_node_ids)}/{total} tests pass against a "
        f"NotImplementedError stub (vacuous ratio {vacuous_ratio:.0%})"
    )

    if mode != "enforce":
        return RedGateResult(
            assertion_ok=True,
            total=total,
            vacuous_node_ids=vacuous_node_ids,
            decision="proceed",
            mode=mode,
            diagnosis=diagnosis,
        )

    if vacuous_ratio > 0.5:
        decision = "regenerate" if not retry_used else "fail"
        return RedGateResult(
            assertion_ok=True,
            total=total,
            vacuous_node_ids=vacuous_node_ids,
            decision=decision,
            mode=mode,
            diagnosis=diagnosis,
        )

    # <= 50% vacuous: discard the vacuous tests and proceed.
    return RedGateResult(
        assertion_ok=True,
        total=total,
        vacuous_node_ids=vacuous_node_ids,
        decision="proceed",
        mode=mode,
        diagnosis=diagnosis,
        discard_node_ids=vacuous_node_ids,
    )


def discard_vacuous_tests(test_code: str, node_ids: set[str]) -> str:
    """Remove the named test functions from the suite (AST-based).

    Args:
        test_code: Test module source.
        node_ids: ``<file>::<test_name>`` node ids to remove.

    Returns:
        The suite without the vacuous test functions (re-serialized).
    """
    import ast as _ast

    names = {nid.split("::")[-1] for nid in node_ids if "::" in nid}
    if not names:
        return test_code
    tree = _ast.parse(test_code)
    new_body: list[_ast.stmt] = []
    removed = 0
    for stmt in tree.body:
        if isinstance(stmt, (_ast.FunctionDef, _ast.AsyncFunctionDef)) and stmt.name in names:
            removed += 1
            continue
        new_body.append(stmt)
    if removed == 0:
        return test_code
    tree.body = new_body
    _ast.fix_missing_locations(tree)
    return _ast.unparse(tree)
