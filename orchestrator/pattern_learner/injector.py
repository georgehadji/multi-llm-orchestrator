"""
Pattern Injector — Ephemeral Few-Shot Injection from Past Patterns
=====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

At task generation time, injects relevant past patterns into the prompt
context. The injection is ephemeral — it is sent to the LLM but NOT
persisted to TaskResult or project state. This preserves prompt caching
and ensures zero side effects.

Opt-in: controlled by ORCH_PATTERN_INJECTION=true (default off).
When disabled, inject() returns the prompt unchanged.

Integration: Called from engine.py._execute_task() after dependency
context is built but BEFORE the LLM API call. The injected reference
patterns appear as a "## Reference" section in the prompt.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pattern_store import PatternStore

logger = logging.getLogger("orchestrator.pattern_learner.injector")


# ─────────────────────────────────────────────────────────────────────────────
# PatternInjector
# ─────────────────────────────────────────────────────────────────────────────


class PatternInjector:
    """Injects relevant past patterns into generation prompts.

    The injection is a "## Reference: Successful Patterns from Prior Runs"
    section appended to the prompt. Each reference includes quality score
    and reuse count so the LLM can gauge reliability.

    Usage:
        injector = PatternInjector(store, enabled=True)
        enriched_prompt = await injector.inject("code_generation", prompt)

    The enriched prompt is ephemeral — not persisted to project state.
    """

    def __init__(
        self,
        store: "PatternStore | None" = None,
        enabled: bool = False,
        max_patterns: int = 2,
    ) -> None:
        """Initialize pattern injector.

        Args:
            store: PatternStore instance. Required when enabled.
            enabled: When False (default), returns prompt unchanged.
            max_patterns: Maximum number of reference patterns to include.
                Default 2.
        """
        self._store = store
        self._enabled = enabled
        self._max_patterns = max_patterns

    # ── Public API ──────────────────────────────────────────────────────────

    async def inject(
        self,
        task_type: str,
        prompt: str,
    ) -> tuple[str, bool]:
        """Inject relevant past patterns into the prompt.

        Args:
            task_type: Task type string (e.g. ``"code_generation"``).
            prompt: The current generation prompt.

        Returns:
            Tuple of ``(enriched_prompt, injection_added)`` where
            ``injection_added`` is True when patterns were injected.
        """
        if not self._enabled or self._store is None:
            return prompt, False

        patterns = await self._store.find_similar(
            task_type=task_type,
            prompt=prompt,
            limit=self._max_patterns,
        )

        if not patterns:
            return prompt, False

        injection = "\n\n## Reference: Successful Patterns from Prior Runs\n\n"
        for i, pat in enumerate(patterns, 1):
            reuse_info = (
                f" (reused {pat['reuse_count']}×, " f"avg score: {pat['avg_score_on_reuse']:.2f})"
                if pat.get("reuse_count", 0) > 0
                else " (new pattern)"
            )
            injection += (
                f"### Pattern {i} — {pat['task_type']}{reuse_info}\n"
                f"Quality score at extraction: {pat['quality_score']:.2f}\n"
                f"Model: {pat['model_used']}\n"
                f"```\n{pat['generated_code'][:1500]}\n```\n\n"
            )

        enriched = prompt + injection

        logger.info(
            "PatternInjector: injected %d pattern(s) into %s",
            len(patterns),
            task_type,
        )

        return enriched, True

    @property
    def is_enabled(self) -> bool:
        return self._enabled
