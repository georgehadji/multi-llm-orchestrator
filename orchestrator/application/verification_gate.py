"""
VerificationGate — deterministic test/lint floor for the evaluator.

ENH-1 (Loop Engineering §V.C): "The evaluator should act, not just read."

Composes pluggable async checks (tests, lint, type-check) into a boolean
verdict that serves as a hard floor beneath the LLM quality score:
  - gate passes  → LLM score determines quality
  - gate fails   → score capped at FAIL_SCORE_FLOOR regardless of LLM opinion

This prevents the nodding loop: an LLM praising its own broken output can no
longer gate-crash completion.

All checks run via the existing bash-safety guardrails; no new shell exposure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Awaitable, Callable

logger = logging.getLogger("orchestrator.application.verification_gate")


# ── Types ─────────────────────────────────────────────────────────────────────

CheckFn = Callable[[str], Awaitable[tuple[bool, str]]]
"""Async function (artifact: str) -> (passed: bool, reason: str)."""


@dataclass
class VerificationCheck:
    """A single named check in the gate."""

    name: str
    run: CheckFn


@dataclass
class GateResult:
    """Outcome of running all checks in a VerificationGate.

    Attributes:
        checks:  name → passed
        reasons: name → failure reason (only failures populated)
        score:   floor-capped score; callers use this when gate failed
    """

    checks: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    score: float = 1.0

    @property
    def passed(self) -> bool:
        return all(self.checks.values()) if self.checks else True


# ── Gate ─────────────────────────────────────────────────────────────────────


class VerificationGate:
    """Chain-of-responsibility: runs all checks, collects failures, sets floor.

    Usage:
        gate = VerificationGate(checks=[lint_check, test_check])
        result = await gate.run(artifact_text)
        if not result.passed:
            # skip LLM evaluation — artifact is demonstrably broken
            return capped_report(result.score)
    """

    FAIL_SCORE_FLOOR: float = 0.15
    """Score cap for any artifact that fails a deterministic check.
    Must be below any reasonable acceptance_threshold (typically 0.7+)."""

    def __init__(self, checks: list[VerificationCheck] | None = None) -> None:
        self._checks = checks or []

    async def run(self, artifact: str) -> GateResult:
        """Run all checks against *artifact*; return aggregated GateResult.

        All checks run regardless of early failures so the caller sees every
        problem in one pass.
        """
        passed_map: dict[str, bool] = {}
        reasons: dict[str, str] = {}

        for check in self._checks:
            try:
                ok, reason = await check.run(artifact)
                passed_map[check.name] = ok
                if not ok:
                    reasons[check.name] = reason
                    logger.warning(
                        "VerificationGate: check '%s' FAILED — %s", check.name, reason
                    )
                else:
                    logger.debug("VerificationGate: check '%s' passed", check.name)
            except Exception as exc:
                passed_map[check.name] = False
                reasons[check.name] = f"check raised: {exc}"
                logger.error(
                    "VerificationGate: check '%s' raised unexpectedly: %s",
                    check.name,
                    exc,
                    exc_info=True,
                )

        all_passed = all(passed_map.values()) if passed_map else True
        score = 1.0 if all_passed else self.FAIL_SCORE_FLOOR

        return GateResult(checks=passed_map, reasons=reasons, score=score)

    @classmethod
    def default(cls) -> "VerificationGate":
        """Return a gate with no checks (safe default, opt-in checks via ORCH_VERIFY_ACTS).

        Checks are added by callers that have access to the sandbox/bash-guard layer.
        The gate itself is always present; it is the checks that are optional.
        """
        return cls(checks=[])
