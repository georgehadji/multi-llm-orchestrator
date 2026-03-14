"""
SessionLifecycleManager — Automatic HOT/WARM/COLD Session Lifecycle
===================================================================
Wraps MemoryTierManager and adds:
1. LLM-based summarization when entries transition HOT→WARM
2. A periodic asyncio background scheduler that runs migrations automatically

Usage:
    manager = SessionLifecycleManager(memory_tier_manager=mem_mgr)
    await manager.start()          # begin automatic migrations
    await manager.stop()           # cancel background task
    await manager.run_migration()  # run one cycle manually
"""
from __future__ import annotations

import asyncio
from typing import Dict, Optional

from .log_config import get_logger
from .memory_tier import MemoryTierManager

logger = get_logger(__name__)

_DEFAULT_INTERVAL_SECONDS: int = 3600  # 1 hour
_SUMMARY_PROMPT = (
    "Summarize the following memory entry in 2-3 sentences, capturing the key facts "
    "and context that would be most useful for future retrieval:\n\n{content}"
)


class SessionLifecycleManager:
    """
    Manages automatic session lifecycle transitions for MemoryTierManager.

    Adds LLM summarization before HOT→WARM migrations and optionally
    runs those migrations on a periodic background schedule.
    """

    def __init__(
        self,
        memory_tier_manager: MemoryTierManager,
        migration_interval_seconds: int = _DEFAULT_INTERVAL_SECONDS,
        llm_model: str = "deepseek-chat",
    ) -> None:
        self._mem = memory_tier_manager
        self._interval = migration_interval_seconds
        self._model = llm_model
        self._task: Optional[asyncio.Task] = None

    # ── Public API ────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the background migration scheduler."""
        if self._task and not self._task.done():
            logger.debug("SessionLifecycleManager scheduler already running")
            return
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info(
            "SessionLifecycleManager started (interval=%ds)", self._interval
        )

    async def stop(self) -> None:
        """Cancel the background scheduler and wait for it to finish."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("SessionLifecycleManager stopped")

    async def run_migration(self) -> Dict[str, int]:
        """
        Run one lifecycle migration cycle.

        Summarizes HOT entries that are due for WARM migration, then
        delegates the actual tier move to MemoryTierManager.migrate_tiers().

        Returns migration counts from migrate_tiers().
        """
        await self._summarize_due_entries()
        counts = await self._mem.migrate_tiers()
        if sum(counts.values()):
            logger.info("Lifecycle migration: %s", counts)
        return counts

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _scheduler_loop(self) -> None:
        """Periodic background task — runs run_migration() every interval."""
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.run_migration()
            except Exception:
                logger.exception("SessionLifecycleManager: migration error (continuing)")

    async def _summarize_due_entries(self) -> None:
        """
        For each HOT entry old enough to migrate, attempt LLM summarization.
        Sets entry.summary in-place before migrate_tiers() moves it.
        Failures are logged and silently skipped (fail-open).
        """
        for entry in list(self._mem._hot_index.values()):
            if entry.age_days < self._mem.hot_ttl_days:
                continue  # Not yet due for migration
            if entry.summary:
                continue  # Already has a summary

            try:
                summary = await self._summarize_content(entry.content)
                entry.summary = summary
                self._mem._save_memory(entry)
                logger.debug("Summarized entry %s before HOT→WARM migration", entry.id)
            except Exception as exc:
                logger.warning(
                    "Could not summarize entry %s (will migrate without summary): %s",
                    entry.id,
                    exc,
                )

    async def _summarize_content(self, content: str) -> str:
        """
        Call LLM to produce a compact summary of a memory entry's content.

        Raises on failure — caller handles fail-open logic.
        """
        from .api_clients import UnifiedClient

        client = UnifiedClient()
        prompt = _SUMMARY_PROMPT.format(content=content[:3000])  # cap input tokens
        response = await client.chat_completion(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
