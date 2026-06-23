"""
Tool Call Guardrails — Runtime Safety Checks
==============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Per-turn guardrail controller that detects:
1. Repeated identical generations (agent stuck in a loop)
2. Destructive patterns in generated code (rm -rf, os.system("format"), etc.)
3. Output size regression (sudden shrinking suggests model degradation)

Adapted from Hermes Agent's ToolCallGuardrailController pattern.

Integration: Called from engine.py._execute_task() BEFORE deterministic
validators. BLOCK decisions short-circuit to a synthetic failure result;
WARN decisions log and continue.

Invariants:
- reset_for_turn() is called at each iteration start
- Never raises — always returns GuardrailResult
- Thread-safe (no mutable shared state across calls)
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar

logger = logging.getLogger("orchestrator.safety.guardrails")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


class GuardrailDecision(Enum):
    ALLOW = "allow"
    WARN = "warn"
    BLOCK = "block"


@dataclass
class GuardrailResult:
    """Result of a guardrail check.

    Attributes:
        decision: ALLOW → proceed; WARN → log and continue; BLOCK → short-circuit
        reason: Human-readable explanation of the decision
        synthetic_output: When BLOCKed, an optional short-circuit result string
    """

    decision: GuardrailDecision
    reason: str = ""
    synthetic_output: str | None = None

    @property
    def is_allowed(self) -> bool:
        return self.decision == GuardrailDecision.ALLOW

    @property
    def is_blocked(self) -> bool:
        return self.decision == GuardrailDecision.BLOCK


# ─────────────────────────────────────────────────────────────────────────────
# Destructive pattern detection
# ─────────────────────────────────────────────────────────────────────────────

# Patterns that indicate generated code may be destructive.
# Each tuple: (regex_pattern, description)
_DESTRUCTIVE_PATTERNS: ClassVar[list[tuple[str, str]]] = [
    # Shell: rm -rf, del /f, format
    (r"\brm\s+[-]?rf?\b", "recursive force delete (rm -rf)"),
    (r"\bdel\s+/[fqs]/?\b", "force delete (del /f)"),
    (r"\bformat\s+\w:", "disk format"),
    (r"\bmkfs\.\w+", "filesystem creation"),
    (r"\bdd\s+if=", "raw disk write (dd)"),
    (r"\bchmod\s+777\b", "world-writable permissions"),
    # Python: dangerous stdlib calls
    (r'\bos\.system\s*\(\s*["\'](?:rm|del|format|mkfs|dd)', "os.system with destructive command"),
    (r"\bshutil\.rmtree\s*\(", "recursive directory delete (shutil.rmtree)"),
    (r"\bpathlib\.Path\.unlink\b", "file deletion via pathlib"),
    # SQL: dangerous operations (without WHERE)
    (r"\bDROP\s+TABLE\b", "SQL DROP TABLE"),
    (r"\bDELETE\s+FROM\b(?!.*\bWHERE\b)", "SQL DELETE without WHERE"),
    # Subprocess: running destructive commands
    (r'\bsubprocess\.\w+\s*\(.*["\'](?:rm|del /f|format)', "subprocess with destructive command"),
]

# Output size regression threshold: if output is less than this fraction of
# the previous iteration's output, flag as regression.
_OUTPUT_REGRESSION_RATIO: float = 0.10

# Minimum output length (chars) before regression check kicks in.
_OUTPUT_REGRESSION_MIN_BASELINE: int = 200


def _check_destructive(output: str) -> str | None:
    """Check output for destructive patterns.

    Returns a description string if a destructive pattern is found,
    None if the output is clean.
    """
    for pattern, description in _DESTRUCTIVE_PATTERNS:
        if re.search(pattern, output, re.IGNORECASE | re.MULTILINE):
            return description
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Guardrail controller
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ToolCallGuardrailController:
    """Per-turn guardrail controller.

    Tracks generation hashes, output sizes, and halt decisions across
    iterations within a single task turn.  Must be reset between tasks.

    Usage:
        guard = ToolCallGuardrailController()
        for iteration in range(max_iterations):
            guard.reset_for_turn()
            output = generate(...)
            result = guard.check(output, task_type="code_generation")
            if result.is_blocked:
                break
    """

    _generation_hashes: set[str] = field(default_factory=set)
    _last_output_size: int = 0
    _halt_decision: GuardrailResult | None = None

    def reset_for_turn(self) -> None:
        """Reset per-iteration state. Call at the start of each iteration."""
        self._generation_hashes.clear()
        self._halt_decision = None
        self._last_output_size = 0

    def check(
        self,
        output: str,
        task_type: str | None = None,
    ) -> GuardrailResult:
        """Check generated output against guardrails.

        Three checks in order:
        1. Repeat detection — same output[0:500] hash as a prior iteration
        2. Destructive pattern detection — only for code_generation tasks
        3. Output size regression — new output significantly smaller than prior

        Args:
            output: The generated output text to check.
            task_type: Optional task type string; destructive checks only
                       run for ``code_generation``.

        Returns:
            A GuardrailResult.  Callers should check ``is_blocked`` before
            proceeding to validation.
        """
        # ── Check 1: Repeat detection ────────────────────────────────────────
        h = hashlib.sha256(output[:500].encode()).hexdigest()
        if h in self._generation_hashes:
            logger.warning(
                "Guardrail REPEAT: Output[0:500] hash identical to previous "
                "generation — possible loop"
            )
            return GuardrailResult(
                GuardrailDecision.WARN,
                "Output identical to previous generation — possible loop",
            )
        self._generation_hashes.add(h)

        # ── Check 2: Destructive pattern detection ───────────────────────────
        if task_type and "code_generation" in task_type:
            destructive = _check_destructive(output)
            if destructive is not None:
                logger.warning("Guardrail BLOCK: Destructive pattern detected — %s", destructive)
                return GuardrailResult(
                    GuardrailDecision.BLOCK,
                    f"Destructive pattern detected: {destructive}",
                    synthetic_output=(
                        f"# [GUARDRAIL] Code blocked: {destructive}\n"
                        f"# The generation produced a destructive pattern and has been "
                        f"intercepted. No code was executed or written."
                    ),
                )

        # ── Check 3: Output size regression ──────────────────────────────────
        current_size = len(output.strip())
        if (
            self._last_output_size >= _OUTPUT_REGRESSION_MIN_BASELINE
            and current_size < self._last_output_size * _OUTPUT_REGRESSION_RATIO
        ):
            logger.warning(
                "Guardrail REGRESSION: Output size dropped from %d to %d "
                "chars (< %.0f%% of previous)",
                self._last_output_size,
                current_size,
                _OUTPUT_REGRESSION_RATIO * 100,
            )
            # WARN but don't BLOCK — regression may be legitimate (e.g. a
            # short "looks good" response after a long critique).
            return GuardrailResult(
                GuardrailDecision.WARN,
                f"Output ({current_size} chars) < 10% of previous "
                f"({self._last_output_size} chars)",
            )
        self._last_output_size = current_size

        return GuardrailResult(GuardrailDecision.ALLOW, "All guardrails passed")
