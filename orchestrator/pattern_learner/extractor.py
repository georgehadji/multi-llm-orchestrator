"""
Pattern Extractor — Analyze TaskResult History for Reusable Patterns
======================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Extracts reusable patterns from successful task executions. A "successful
pattern" is defined by:
- Quality score >= 0.85 (above acceptance threshold)
- All deterministic validators passed
- At least 2 iterations (critique -> revise cycle was productive)
- Not a duplicate of an existing pattern (by code hash)

Patterns are identified by prompt fingerprint and code hash. The
Extractor feeds into PatternStore for persistence and PatternInjector
for reuse in future generations.

Integration: Called from engine.py._execute_task() after a task
completes successfully. Ephemeral — patterns are stored but not
injected unless PatternInjector is enabled.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import TaskResult, TaskType

logger = logging.getLogger("orchestrator.pattern_learner.extractor")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExtractedPattern:
    """A reusable pattern extracted from a successful task execution.

    Attributes:
        pattern_id: Deterministic hash-based ID.
        task_type: The TaskType this pattern applies to.
        prompt_fingerprint: Hash of the normalized generation prompt.
        generated_code_hash: Hash of the output (for deduplication).
        quality_score: The TaskResult.score at extraction time.
        model_used: Model identifier that produced this pattern.
        validation_results: Dict of validator_name -> bool.
        created_at: Unix timestamp of extraction.
        provenance: "agent" for auto-extracted, "bundled" for shipped.
    """

    pattern_id: str
    task_type: str
    prompt_fingerprint: str
    generated_code_hash: str
    quality_score: float
    model_used: str
    validation_results: dict[str, bool] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    provenance: str = "agent"


# Thresholds for extraction
_MIN_QUALITY_SCORE: float = 0.85
_MIN_ITERATIONS: int = 2


# ─────────────────────────────────────────────────────────────────────────────
# PatternExtractor
# ─────────────────────────────────────────────────────────────────────────────


class PatternExtractor:
    """Analyzes TaskResult history to extract successful patterns.

    Usage:
        extractor = PatternExtractor()
        pattern = await extractor.extract(task_type, result, prompt)
        if pattern:
            await pattern_store.insert(pattern, prompt, result.output)
    """

    def __init__(
        self,
        min_quality: float = _MIN_QUALITY_SCORE,
        min_iterations: int = _MIN_ITERATIONS,
    ) -> None:
        """Initialize extractor with configurable thresholds.

        Args:
            min_quality: Minimum quality score for extraction (default 0.85).
            min_iterations: Minimum iterations for extraction (default 2).
        """
        self._min_quality = min_quality
        self._min_iterations = min_iterations

    # ── Public API ──────────────────────────────────────────────────────────

    async def extract(
        self,
        task_type: str,
        result: "TaskResult",
        prompt: str,
    ) -> ExtractedPattern | None:
        """Extract a pattern from a successful task result.

        Returns None when:
        - Quality score is below threshold
        - Deterministic validators failed
        - Iterations count is below minimum
        - Output is empty

        Args:
            task_type: String task type (e.g. ``"code_generation"``).
            result: The completed TaskResult.
            prompt: The original generation prompt.

        Returns:
            An ExtractedPattern if all thresholds are met, None otherwise.
        """
        # Threshold checks
        if result.score < self._min_quality:
            logger.debug(
                "Pattern extraction skipped: score %.3f < %.2f",
                result.score,
                self._min_quality,
            )
            return None

        if not result.deterministic_check_passed:
            logger.debug(
                "Pattern extraction skipped: deterministic validators failed " "for %s",
                result.task_id,
            )
            return None

        if result.iterations < self._min_iterations:
            logger.debug(
                "Pattern extraction skipped: %d iterations < %d",
                result.iterations,
                self._min_iterations,
            )
            return None

        if not result.output or not result.output.strip():
            logger.debug(
                "Pattern extraction skipped: empty output for %s",
                result.task_id,
            )
            return None

        # Compute hashes
        prompt_fp = self._fingerprint(prompt)
        code_hash = hashlib.sha256(result.output.encode()).hexdigest()
        pattern_id = f"pat_{task_type[:8]}_{prompt_fp[:12]}"

        # Build validation results summary
        validation_results: dict[str, bool] = {}
        for record in result.attempt_history:
            if record.validators_failed:
                for vf in record.validators_failed:
                    validation_results[vf] = False

        pattern = ExtractedPattern(
            pattern_id=pattern_id,
            task_type=task_type,
            prompt_fingerprint=prompt_fp,
            generated_code_hash=code_hash,
            quality_score=result.score,
            model_used=result.model_used.value if result.model_used else "unknown",
            validation_results=validation_results,
            provenance="agent",
        )

        logger.info(
            "Extracted pattern %s for %s (score=%.3f, %d iters)",
            pattern_id,
            task_type,
            result.score,
            result.iterations,
        )

        return pattern

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _fingerprint(text: str) -> str:
        """Normalize a prompt for fingerprinting.

        Normalization: lowercase, strip whitespace, sort lines, hash.
        This ensures that semantically identical prompts with cosmetic
        differences (whitespace, variable names) produce the same fingerprint.
        """
        normalized = text.lower().strip()
        # Sort non-empty lines to minimize order-dependence
        lines = sorted(line.strip() for line in normalized.splitlines() if line.strip())
        normalized = "\n".join(lines)
        return hashlib.sha256(normalized.encode()).hexdigest()
