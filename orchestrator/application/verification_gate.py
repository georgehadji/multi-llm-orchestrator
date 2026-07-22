"""
VerificationGate — deterministic test/lint floor for the evaluator.

ENH-1 (Loop Engineering §V.C): "The evaluator should act, not just read."
WBS-1: structured outcomes, artifact hashes, execution receipts, policy support.

Composes pluggable async checks (tests, lint, type-check) into a boolean
verdict that serves as a hard floor beneath the LLM quality score:
  - gate passes  → LLM score determines quality
  - gate fails   → score capped at FAIL_SCORE_FLOOR regardless of LLM opinion

This prevents the nodding loop: an LLM praising its own broken output can no
longer gate-crash completion.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from ..domain.verification import CheckOutcome, ExecutionReceipt, VerificationPolicy

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
        checks:        name → passed (bool) — backward compatible.
        reasons:       name → failure reason (only failures populated).
        score:         floor-capped score; callers use this when gate failed.
        receipts:      structured execution receipts (WBS-1).
        artifact_hash: SHA-256 hex digest of the artifact (WBS-1).
        policy:        The policy used (if any) for this run.
    """

    checks: dict[str, bool] = field(default_factory=dict)
    reasons: dict[str, str] = field(default_factory=dict)
    score: float = 1.0
    receipts: list[ExecutionReceipt] = field(default_factory=list)
    artifact_hash: str | None = None
    policy: VerificationPolicy | None = None

    @property
    def passed(self) -> bool:
        """All checks passed (backward-compatible property)."""
        return all(self.checks.values()) if self.checks else True

    @property
    def failure_summary(self) -> dict[str, str]:
        """Return {check_name: reason} for every failed or blocked check."""
        return {
            r.check_name: r.reason or ""
            for r in self.receipts
            if r.outcome in (CheckOutcome.FAILED, CheckOutcome.BLOCKED)
        }

    @property
    def status_summary(self) -> str:
        """Human-readable one-liner of gate outcomes."""
        parts: list[str] = []
        for r in self.receipts:
            tag = {
                CheckOutcome.PASSED: "[OK]",
                CheckOutcome.FAILED: "[FAIL]",
                CheckOutcome.BLOCKED: "[X]",
                CheckOutcome.NOT_RUN: "[-]",
            }.get(r.outcome, "[?]")
            parts.append(f"{tag} {r.check_name}: {r.outcome.value}")
        return ", ".join(parts) if parts else "(no checks)"


# ── Helpers ────────────────────────────────────────────────────────────────────


def compute_artifact_hash(artifact: str) -> str:
    """Return SHA-256 hex digest of artifact content."""
    return hashlib.sha256(artifact.encode("utf-8")).hexdigest()


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

    def __init__(
        self,
        checks: list[VerificationCheck] | None = None,
        policy: VerificationPolicy | None = None,
    ) -> None:
        self._checks = checks or []
        self._policy = policy

    async def run(
        self,
        artifact: str,
        policy: VerificationPolicy | None = None,
    ) -> GateResult:
        """Run all checks against *artifact*; return aggregated GateResult.

        All checks run regardless of early failures so the caller sees every
        problem in one pass.

        Args:
            artifact: The text/code to verify.
            policy:   Optional override policy. If set, gates without matching
                      checks will still record NOT_RUN receipts for mandated
                      checks that have no checker registered.

        Returns:
            GateResult with backward-compat checks/reasons/score plus
            structured receipts.
        """
        effective_policy = policy or self._policy
        artifact_hash = compute_artifact_hash(artifact)

        passed_map: dict[str, bool] = {}
        reasons: dict[str, str] = {}
        receipts: list[ExecutionReceipt] = []

        # Build a set of check names we have registered
        registered_names = {c.name for c in self._checks}

        # First, run all registered checks
        for check in self._checks:
            start = time.monotonic()
            duration: float | None = None
            try:
                ok, reason = await check.run(artifact)
                duration = (time.monotonic() - start) * 1000
                passed_map[check.name] = ok
                if not ok:
                    reasons[check.name] = reason
                    logger.warning("VerificationGate: check '%s' FAILED — %s", check.name, reason)
                else:
                    logger.debug("VerificationGate: check '%s' passed", check.name)

                outcome = CheckOutcome.PASSED if ok else CheckOutcome.FAILED
                receipts.append(
                    ExecutionReceipt(
                        check_name=check.name,
                        outcome=outcome,
                        reason=reason or None,
                        duration_ms=duration,
                        artifact_hash=artifact_hash,
                    )
                )
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                passed_map[check.name] = False
                reasons[check.name] = f"check raised: {exc}"
                logger.error(
                    "VerificationGate: check '%s' raised unexpectedly: %s",
                    check.name,
                    exc,
                    exc_info=True,
                )
                receipts.append(
                    ExecutionReceipt(
                        check_name=check.name,
                        outcome=CheckOutcome.BLOCKED,
                        reason=str(exc),
                        duration_ms=duration,
                        artifact_hash=artifact_hash,
                    )
                )

        # Add NOT_RUN receipts for policy-mandated checks with no registered checker
        if effective_policy:
            for check_name in effective_policy.required_checks:
                if check_name not in registered_names:
                    passed_map[check_name] = False
                    reasons.setdefault(check_name, "no checker registered")
                    receipts.append(
                        ExecutionReceipt(
                            check_name=check_name,
                            outcome=CheckOutcome.NOT_RUN,
                            reason="no checker registered for required check",
                            artifact_hash=artifact_hash,
                        )
                    )

        all_passed = all(passed_map.values()) if passed_map else True
        score = 1.0 if all_passed else self.FAIL_SCORE_FLOOR

        return GateResult(
            checks=passed_map,
            reasons=reasons,
            score=score,
            receipts=receipts,
            artifact_hash=artifact_hash,
            policy=effective_policy,
        )

    @classmethod
    def default(cls) -> "VerificationGate":
        """Return a gate with no checks (safe default, opt-in checks via ORCH_VERIFY_ACTS).

        Checks are added by callers that have access to the sandbox/bash-guard layer.
        The gate itself is always present; it is the checks that are optional.
        """
        return cls(checks=[])
