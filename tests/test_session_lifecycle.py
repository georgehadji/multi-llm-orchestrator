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
