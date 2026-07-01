"""
Pattern Curator — Lifecycle Management for Agent-Created Patterns
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Periodic review of agent-created patterns. Follows Hermes Agent's
curator invariants:
- Only touches provenance="agent" patterns (bundled/hub-installed are off-limits)
- Never deletes — archives to status='archived'
- Pinned patterns (future: status='pinned') are exempt from auto-transitions
- Runs after every N projects (configurable)

Invariants:
- Only touches provenance="agent"
- Never deletes
- Pinned exempt
- Runs every N projects (configured externally via Engine)

Integration: Called from MemoryManager.maybe_consolidate() or directly
from Orchestrator after project completion. Delegates to PatternStore
for all data operations — Curator is logic-only.

UnifiedClient wiring (Phase 5): The curator receives a UnifiedClient
reference at construction and uses it for LLM-driven duplicate detection.
When ``client`` is None, duplicate detection is skipped.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from ..models import Model

if TYPE_CHECKING:
    from ..api_clients import UnifiedClient
    from .pattern_store import PatternStore

logger = logging.getLogger("orchestrator.pattern_learner.curator")

# Cheapest model for curator review calls — same tier as context compressor
_DEFAULT_REVIEW_MODEL: Model = Model.ZHIPU_GLM_5_2

# Minimum group size to trigger duplicate detection (avoids 1-pattern no-ops)
_MIN_GROUP_SIZE_FOR_DEDUP: int = 3


# ─────────────────────────────────────────────────────────────────────────────
# PatternCurator
# ─────────────────────────────────────────────────────────────────────────────


class PatternCurator:
    """Lifecycle manager for agent-created patterns.

    Usage:
        curator = PatternCurator(store, client=unified_client)
        report = await curator.review()
        print(report.summary)
    """

    def __init__(
        self,
        store: "PatternStore",
        client: "UnifiedClient | None" = None,
        stale_days: int = 60,
        review_model: Model | None = None,
    ) -> None:
        """Initialize curator.

        Args:
            store: PatternStore instance.
            client: UnifiedClient for LLM-powered review calls (Phase 5).
                When None, duplicate detection is skipped.
            stale_days: Days of inactivity before a pattern is archived.
                Default 60.
            review_model: Model to use for review LLM calls. Defaults to
                ZHIPU_GLM_5_2 (canonical GLM model).
        """
        self._store = store
        self._client = client
        self._stale_days = stale_days
        self._review_model = review_model or _DEFAULT_REVIEW_MODEL

    # ── Public API ──────────────────────────────────────────────────────────

    async def review(self) -> "CuratorReport":
        """Run one review cycle.

        Steps:
        1. Archive stale patterns (status='active' with no reuse for N days)
        2. Detect near-duplicates via LLM and archive lower-quality copies
        3. Compute and return summary statistics.

        Returns:
            A CuratorReport with actions taken.
        """
        actions: list[str] = []
        dup_archived = 0

        # 1. Archive stale agent-created patterns
        archived = await self._store.archive_stale(days=self._stale_days)
        if archived:
            actions.append(f"Archived {archived} stale pattern(s)")

        # 2. LLM-powered duplicate detection (Phase 5: UnifiedClient wire)
        try:
            dup_archived = await self._merge_duplicates()
            if dup_archived:
                actions.append(f"Archived {dup_archived} near-duplicate pattern(s)")
        except Exception as exc:
            logger.warning("Curator duplicate detection failed: %s", exc)

        # 3. Collect stats
        stats = await self._store.get_stats()

        report = CuratorReport(
            actions=actions,
            active_patterns=stats.get("active_patterns", 0),
            archived_this_cycle=archived + dup_archived,
            total_reuses=stats.get("total_reuses", 0),
        )

        if actions:
            logger.info("Curator review: %s", "; ".join(actions))
        else:
            logger.debug("Curator review: no actions needed")

        return report

    # ── LLM-Powered Duplicate Detection ─────────────────────────────────────

    async def _merge_duplicates(self) -> int:
        """Detect near-duplicate patterns via LLM and archive lower-quality copies.

        Groups active patterns by task_type, then for each group with
        >= _MIN_GROUP_SIZE_FOR_DEDUP patterns, asks the review model to
        identify semantically near-duplicate pairs. Archives the lower-quality
        pattern from each pair.

        Returns:
            Number of patterns archived as duplicates.
        """
        if self._client is None:
            logger.debug("Curator: no UnifiedClient — skipping duplicate detection")
            return 0

        # Fetch active patterns grouped by task_type
        groups = await self._store.get_active_patterns_by_type()
        if not groups:
            return 0

        total_archived = 0

        for task_type, patterns in groups.items():
            if len(patterns) < _MIN_GROUP_SIZE_FOR_DEDUP:
                continue

            duplicates = await self._detect_duplicates_in_group(task_type, patterns)
            for lower_quality_id in duplicates:
                await self._store.archive_single(lower_quality_id)
                total_archived += 1

        if total_archived:
            logger.info(
                "Curator duplicate merge: archived %d pattern(s) across %d groups",
                total_archived,
                len(groups),
            )

        return total_archived

    async def _detect_duplicates_in_group(
        self,
        task_type: str,
        patterns: list[dict[str, Any]],
    ) -> list[str]:
        """Ask the review LLM to identify near-duplicate patterns in a group.

        Args:
            task_type: The task type group being reviewed.
            patterns: List of pattern dicts with keys: pattern_id,
                quality_score, prompt_text (first 500 chars).

        Returns:
            List of pattern_ids to archive (the lower-quality copy of each pair).
        """
        # Build a compact representation for the LLM
        entries: list[dict[str, Any]] = []
        for p in patterns:
            # Truncate prompt for token efficiency
            prompt_snippet = (p.get("prompt_text") or "")[:500]
            entries.append(
                {
                    "id": p["pattern_id"],
                    "score": p.get("quality_score", 0),
                    "prompt": prompt_snippet,
                }
            )

        system = (
            "You are a code pattern curator. Given a list of patterns of the "
            "same task type, identify pairs that are near-duplicates (the same "
            "underlying task described with slightly different wording). "
            "For each duplicate pair, select the LOWER quality_score pattern "
            "to archive. Return ONLY a JSON array of pattern_ids to archive. "
            "If no duplicates found, return an empty array []."
        )

        prompt = (
            f"Task type: {task_type}\n\n"
            f"Patterns:\n{json.dumps(entries, indent=2)}\n\n"
            "Return the JSON array of pattern_ids to archive (lower-quality "
            "from each duplicate pair):"
        )

        try:
            response = await self._client.call(
                model=self._review_model,
                system=system,
                prompt=prompt,
                max_tokens=512,
                temperature=0.0,
            )

            result = json.loads(response.text.strip())
            if isinstance(result, list):
                valid_ids = {p["pattern_id"] for p in patterns}
                to_archive = [pid for pid in result if pid in valid_ids]
                if to_archive:
                    logger.debug(
                        "Curator: %d duplicates detected in %s group",
                        len(to_archive),
                        task_type,
                    )
                return to_archive
            return []

        except (json.JSONDecodeError, Exception) as exc:
            logger.debug("Curator duplicate detection failed for %s: %s", task_type, exc)
            return []


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


class CuratorReport:
    """Report of a single curator review cycle.

    Attributes:
        actions: Human-readable descriptions of actions taken.
        active_patterns: Number of active (non-archived) patterns after review.
        archived_this_cycle: Number of patterns archived in this cycle.
        total_reuses: Cumulative count of pattern reuses across all history.
    """

    def __init__(
        self,
        actions: list[str] | None = None,
        active_patterns: int = 0,
        archived_this_cycle: int = 0,
        total_reuses: int = 0,
    ) -> None:
        self.actions = actions or []
        self.active_patterns = active_patterns
        self.archived_this_cycle = archived_this_cycle
        self.total_reuses = total_reuses

    @property
    def summary(self) -> str:
        """One-line summary of the review cycle."""
        if not self.actions:
            return "Curator review: no actions needed"
        return "Curator review: " + "; ".join(self.actions)

    def __repr__(self) -> str:
        return (
            f"CuratorReport(actions={len(self.actions)}, "
            f"active={self.active_patterns}, "
            f"archived={self.archived_this_cycle}, "
            f"reuses={self.total_reuses})"
        )
