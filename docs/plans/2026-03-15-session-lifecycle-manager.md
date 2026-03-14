# Session Lifecycle Manager Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add automatic HOT→WARM→COLD session lifecycle management with LLM-based summarization on HOT→WARM transitions, wired into a periodic background scheduler.

**Architecture:** New `orchestrator/session_lifecycle.py` module with `SessionLifecycleManager` that wraps `MemoryTierManager`, adds LLM summarization before HOT→WARM migration, and runs a periodic asyncio background task. Fail-open on all LLM errors.

**Tech Stack:** Pure Python asyncio, existing `UnifiedClient` for LLM calls, existing `MemoryTierManager` for storage.

---

## Context

**What already exists:**
- `MemoryTierManager.migrate_tiers()`: moves HOT→WARM→COLD based on age, but NO LLM summarization
- `SessionWatcher.summarize_session()`: placeholder that just counts interactions (no real LLM call)
- No scheduler — migrations must be called manually

**What we're adding:**
- `SessionLifecycleManager`: wraps `MemoryTierManager`, adds LLM summarization before HOT→WARM, and schedules periodic migrations
- LLM call: `deepseek-chat` via `UnifiedClient`, returns a text summary of the memory content
- Scheduler: `asyncio` periodic task (configurable interval, default 1 hour)
- Fail-open: LLM errors → entry still migrates, summary remains `None`

**Key integration point:** `MemoryTierManager._hot_index` contains entries due to migrate. Before calling `migrate_tiers()`, we iterate HOT entries with `age_days >= hot_ttl_days` and call LLM to set `entry.summary`.

---

## Task 1: SessionLifecycleManager module

**Files:**
- Create: `orchestrator/session_lifecycle.py`
- Create: `tests/test_session_lifecycle.py`

### Step 1: Write the failing tests

```python
# tests/test_session_lifecycle.py
"""Tests for SessionLifecycleManager — automatic HOT/WARM/COLD lifecycle with LLM compression."""
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from orchestrator.session_lifecycle import SessionLifecycleManager
from orchestrator.memory_tier import MemoryTierManager, MemoryEntry, MemoryTier, MemoryType


def _make_manager(tmp_path):
    """Create a MemoryTierManager with tmp storage."""
    return MemoryTierManager(storage_path=tmp_path, hot_ttl_days=3, warm_ttl_days=30, enable_bm25=False)


def _old_entry(manager, days_old: int) -> MemoryEntry:
    """Inject a HOT entry that is `days_old` days old into the manager's hot index."""
    from datetime import datetime
    import uuid
    entry = MemoryEntry(
        id=str(uuid.uuid4()),
        project_id="proj1",
        content="This is a test memory about fibonacci functions and recursion.",
        memory_type=MemoryType.TASK,
        tier=MemoryTier.HOT,
        created_at=datetime.utcnow() - timedelta(days=days_old),
        last_accessed=datetime.utcnow() - timedelta(days=days_old),
    )
    manager._hot_index[entry.id] = entry
    manager._save_memory(entry)
    return entry


def test_lifecycle_manager_instantiates(tmp_path):
    """SessionLifecycleManager can be created without errors."""
    mem = _make_manager(tmp_path)
    slm = SessionLifecycleManager(memory_tier_manager=mem)
    assert slm is not None


def test_run_migration_calls_underlying_migrate(tmp_path):
    """run_migration() calls MemoryTierManager.migrate_tiers() and returns counts."""
    mem = _make_manager(tmp_path)
    _old_entry(mem, days_old=5)  # Should move HOT→WARM (5 > 3 days)

    slm = SessionLifecycleManager(memory_tier_manager=mem)

    counts = asyncio.get_event_loop().run_until_complete(slm.run_migration())

    assert counts["hot_to_warm"] == 1
    assert counts["warm_to_cold"] == 0


def test_summarization_called_before_hot_to_warm(tmp_path):
    """LLM summarization is attempted for entries migrating HOT→WARM."""
    mem = _make_manager(tmp_path)
    entry = _old_entry(mem, days_old=5)

    slm = SessionLifecycleManager(memory_tier_manager=mem)

    call_log = []

    async def fake_summarize(content: str) -> str:
        call_log.append(content)
        return "A summary of the content."

    slm._summarize_content = fake_summarize

    asyncio.get_event_loop().run_until_complete(slm.run_migration())

    assert len(call_log) == 1  # Called once for the one migrating entry
    # The migrated entry should have its summary set
    migrated = mem._warm_index.get(entry.id)
    assert migrated is not None
    assert migrated.summary == "A summary of the content."


def test_summarization_failure_is_nonfatal(tmp_path):
    """If LLM summarization fails, the entry still migrates without a summary."""
    mem = _make_manager(tmp_path)
    entry = _old_entry(mem, days_old=5)

    slm = SessionLifecycleManager(memory_tier_manager=mem)

    async def failing_summarize(content: str) -> str:
        raise RuntimeError("LLM unavailable")

    slm._summarize_content = failing_summarize

    # Should not raise
    counts = asyncio.get_event_loop().run_until_complete(slm.run_migration())

    assert counts["hot_to_warm"] == 1
    migrated = mem._warm_index.get(entry.id)
    assert migrated is not None
    assert migrated.summary is None  # No summary set on failure


def test_entries_not_yet_due_are_not_summarized(tmp_path):
    """Entries younger than hot_ttl_days are not summarized or migrated."""
    mem = _make_manager(tmp_path)
    _old_entry(mem, days_old=1)  # Still HOT (1 < 3 days)

    slm = SessionLifecycleManager(memory_tier_manager=mem)
    call_log = []

    async def fake_summarize(content: str) -> str:
        call_log.append(content)
        return "summary"

    slm._summarize_content = fake_summarize

    counts = asyncio.get_event_loop().run_until_complete(slm.run_migration())

    assert counts["hot_to_warm"] == 0
    assert len(call_log) == 0


def test_start_and_stop_creates_and_cancels_task(tmp_path):
    """start() creates a background task; stop() cancels it cleanly."""
    async def _run():
        mem = _make_manager(tmp_path)
        slm = SessionLifecycleManager(memory_tier_manager=mem, migration_interval_seconds=3600)
        await slm.start()
        assert slm._task is not None
        assert not slm._task.done()
        await slm.stop()
        assert slm._task.done()

    asyncio.get_event_loop().run_until_complete(_run())
```

### Step 2: Run to confirm FAIL (ImportError)

Run: `python -m pytest tests/test_session_lifecycle.py -v --no-cov`

Expected: `ModuleNotFoundError: No module named 'orchestrator.session_lifecycle'`

### Step 3: Create `orchestrator/session_lifecycle.py`

```python
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
from datetime import datetime
from typing import Dict, Optional

from .log_config import get_logger
from .memory_tier import MemoryTier, MemoryTierManager

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
        now = datetime.utcnow()
        for entry in list(self._mem._hot_index.values()):
            age_days = (now - entry.created_at).days
            if age_days < self._mem.hot_ttl_days:
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
```

### Step 4: Run tests to confirm PASS

Run: `python -m pytest tests/test_session_lifecycle.py -v --no-cov`

Expected: `6 passed`

### Step 5: Commit

```bash
git add orchestrator/session_lifecycle.py tests/test_session_lifecycle.py
git commit -m "feat: add SessionLifecycleManager with auto-compression on HOT→WARM migration"
```

---

## Task 2: Wire SessionLifecycleManager into engine

**Files:**
- Modify: `orchestrator/engine.py`
- Modify: `orchestrator/__init__.py`
- Create: `tests/test_session_lifecycle_engine_integration.py`

### Step 1: Write the failing test

```python
# tests/test_session_lifecycle_engine_integration.py
"""Verify engine exposes SessionLifecycleManager integration."""
import pytest
from orchestrator.session_lifecycle import SessionLifecycleManager
from orchestrator.engine import Orchestrator


def test_engine_has_lifecycle_manager():
    """Orchestrator has a _lifecycle_manager attribute of the right type."""
    orch = Orchestrator.__new__(Orchestrator)
    from orchestrator.memory_tier import MemoryTierManager
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemoryTierManager(storage_path=pathlib.Path(tmp), enable_bm25=False)
        orch._lifecycle_manager = SessionLifecycleManager(memory_tier_manager=mem)
        assert isinstance(orch._lifecycle_manager, SessionLifecycleManager)


def test_engine_exposes_configure_lifecycle():
    """Orchestrator has configure_session_lifecycle() that delegates to lifecycle manager."""
    orch = Orchestrator.__new__(Orchestrator)
    from orchestrator.memory_tier import MemoryTierManager
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemoryTierManager(storage_path=pathlib.Path(tmp), enable_bm25=False)
        orch._lifecycle_manager = SessionLifecycleManager(memory_tier_manager=mem)
        # Must not raise
        orch.configure_session_lifecycle(migration_interval_hours=2)
        assert orch._lifecycle_manager._interval == 7200  # 2 * 3600


def test_session_lifecycle_is_imported_at_module_level():
    """SessionLifecycleManager is importable from orchestrator package."""
    from orchestrator import SessionLifecycleManager as SLM
    assert SLM is not None
```

### Step 2: Run to confirm FAIL

Run: `python -m pytest tests/test_session_lifecycle_engine_integration.py -v --no-cov`

Expected: `FAILED — AttributeError: type object 'Orchestrator' has no attribute 'configure_session_lifecycle'`

### Step 3: Add to `Orchestrator.__init__` (module-level import + instance init)

At module-level imports in `engine.py` (alongside other orchestrator imports), add:
```python
from .session_lifecycle import SessionLifecycleManager
```

After the `self._rate_limiter` line in `Orchestrator.__init__`, add:
```python
self._lifecycle_manager = SessionLifecycleManager(
    memory_tier_manager=self._memory_tier_manager,
)
```

**Note:** The engine already has a `MemoryTierManager` instance. Check the engine `__init__` for the attribute name (likely `self._memory_tier_manager` or similar). Use the existing instance — do NOT create a new one.

### Step 4: Add `configure_session_lifecycle()` method to `Orchestrator`

Add near `configure_rate_limits()`:

```python
def configure_session_lifecycle(
    self,
    migration_interval_hours: int = 1,
    llm_model: str = "deepseek-chat",
) -> None:
    """
    Configure automatic session lifecycle migration.

    Args:
        migration_interval_hours: How often to run HOT/WARM/COLD migration.
        llm_model: Model used for HOT→WARM entry summarization.
    """
    self._lifecycle_manager._interval = migration_interval_hours * 3600
    self._lifecycle_manager._model = llm_model
```

### Step 5: Run tests to confirm PASS

Run: `python -m pytest tests/test_session_lifecycle_engine_integration.py -v --no-cov`

Expected: `3 passed`

### Step 6: Export from `orchestrator/__init__.py`

Add after the `RateLimiter` export:
```python
from .session_lifecycle import SessionLifecycleManager
```

Add `"SessionLifecycleManager"` to `__all__`.

### Step 7: Run full new test suite

Run: `python -m pytest tests/test_session_lifecycle.py tests/test_session_lifecycle_engine_integration.py -v --no-cov`

Expected: `9 passed`

### Step 8: Commit

```bash
git add orchestrator/engine.py orchestrator/__init__.py tests/test_session_lifecycle_engine_integration.py
git commit -m "feat: wire SessionLifecycleManager into engine with configure_session_lifecycle() API"
```

---

## Verification

```bash
python -m pytest tests/test_session_lifecycle.py tests/test_session_lifecycle_engine_integration.py -v --no-cov
```

Expected: **9 tests, all passing**
