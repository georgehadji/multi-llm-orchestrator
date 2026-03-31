"""
REPLVerifier — Code verification with self-healing loop scaffold
================================================================
Provides progressive verification levels (NONE/SYNTAX/EXECUTION/FULL)
for LLM-generated code. The self_healing_loop scaffold handles retry
logic; LLM re-generation is delegated to the engine layer.

Pattern: Chain of Responsibility (verification levels)
Async: No — subprocess calls are sync/bounded
Layer: L2 Verification

Usage:
    from orchestrator.verification import REPLVerifier, VerificationLevel

    verifier = REPLVerifier(level=VerificationLevel.SYNTAX)
    result = verifier.verify("print('hello')")
"""

from __future__ import annotations

import ast
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("orchestrator.verification")


class VerificationLevel(Enum):
    """Progressive verification levels ordered from least to most strict."""

    NONE = "none"
    SYNTAX = "syntax"
    EXECUTION = "execution"
    FULL = "full"


@dataclass
class VerificationResult:
    """Result of a verification attempt."""

    passed: bool
    level: VerificationLevel
    error_message: str | None = field(default=None)
    output: str | None = field(default=None)


class REPLVerifier:
    """Verifies LLM-generated code at a configured level.

    Implements a Chain of Responsibility: each level subsumes the checks
    of all levels below it (FULL ⊃ EXECUTION ⊃ SYNTAX ⊃ NONE).
    """

    def __init__(self, level: VerificationLevel = VerificationLevel.SYNTAX) -> None:
        self.level = level

    def verify(self, code: str) -> VerificationResult:
        """Verify *code* at the configured level.

        Args:
            code: Python source code to verify.

        Returns:
            VerificationResult indicating pass/fail and optional details.
        """
        if self.level == VerificationLevel.NONE:
            return self._verify_none(code)
        if self.level == VerificationLevel.SYNTAX:
            return self._verify_syntax(code)
        if self.level == VerificationLevel.EXECUTION:
            return self._verify_execution(code)
        # FULL
        return self._verify_full(code)

    # ------------------------------------------------------------------
    # Private verification implementations
    # ------------------------------------------------------------------

    def _verify_none(self, code: str) -> VerificationResult:
        """NONE level — always passes regardless of content."""
        logger.debug("NONE verification: auto-pass")
        return VerificationResult(passed=True, level=VerificationLevel.NONE)

    def _verify_syntax(self, code: str) -> VerificationResult:
        """SYNTAX level — parse with ast.parse; fail on SyntaxError."""
        try:
            ast.parse(code)
            logger.debug("SYNTAX verification: passed")
            return VerificationResult(passed=True, level=VerificationLevel.SYNTAX)
        except SyntaxError as exc:
            msg = f"SyntaxError at line {exc.lineno}: {exc.msg}"
            logger.debug("SYNTAX verification failed: %s", msg)
            return VerificationResult(
                passed=False,
                level=VerificationLevel.SYNTAX,
                error_message=msg,
            )

    def _verify_execution(self, code: str) -> VerificationResult:
        """EXECUTION level — run in subprocess; fail on non-zero exit code."""
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if proc.returncode == 0:
                logger.debug("EXECUTION verification: passed (rc=0)")
                return VerificationResult(
                    passed=True,
                    level=VerificationLevel.EXECUTION,
                    output=proc.stdout,
                )
            error_detail = proc.stderr.strip() or f"exit code {proc.returncode}"
            logger.debug("EXECUTION verification failed: %s", error_detail)
            return VerificationResult(
                passed=False,
                level=VerificationLevel.EXECUTION,
                error_message=error_detail,
                output=proc.stdout,
            )
        except subprocess.TimeoutExpired:
            msg = "Execution timed out after 10 seconds"
            logger.warning("EXECUTION verification timeout")
            return VerificationResult(
                passed=False,
                level=VerificationLevel.EXECUTION,
                error_message=msg,
            )

    def _verify_full(self, code: str) -> VerificationResult:
        """FULL level — execution + non-empty stdout without 'error' keyword."""
        exec_result = self._verify_execution(code)
        if not exec_result.passed:
            # Propagate execution failure, re-tag level as FULL
            return VerificationResult(
                passed=False,
                level=VerificationLevel.FULL,
                error_message=exec_result.error_message,
                output=exec_result.output,
            )
        stdout = exec_result.output or ""
        if not stdout.strip():
            msg = "FULL verification failed: stdout is empty"
            logger.debug(msg)
            return VerificationResult(
                passed=False,
                level=VerificationLevel.FULL,
                error_message=msg,
                output=stdout,
            )
        if "error" in stdout.lower():
            msg = "FULL verification failed: stdout contains 'error'"
            logger.debug(msg)
            return VerificationResult(
                passed=False,
                level=VerificationLevel.FULL,
                error_message=msg,
                output=stdout,
            )
        logger.debug("FULL verification: passed")
        return VerificationResult(passed=True, level=VerificationLevel.FULL, output=stdout)


def self_healing_loop(
    code: str,
    verifier: REPLVerifier,
    max_attempts: int = 3,
) -> tuple[str, VerificationResult]:
    """Scaffold for the self-healing verification loop.

    Tries to verify *code* up to *max_attempts* times. On the first passing
    result the function returns immediately. On failure the function returns
    the original code and the last failing result — actual LLM re-generation
    is the responsibility of the engine layer.

    Args:
        code: Python source code to verify.
        verifier: Configured REPLVerifier instance.
        max_attempts: Maximum number of verification attempts.

    Returns:
        Tuple of (code, VerificationResult) where VerificationResult reflects
        the final verification outcome.
    """
    last_result: VerificationResult | None = None
    for attempt in range(1, max_attempts + 1):
        result = verifier.verify(code)
        logger.debug(
            "self_healing_loop attempt %d/%d: passed=%s", attempt, max_attempts, result.passed
        )
        if result.passed:
            return code, result
        last_result = result
    # All attempts exhausted without a pass — return code + last failure
    assert last_result is not None  # max_attempts >= 1 guaranteed by caller convention
    return code, last_result
