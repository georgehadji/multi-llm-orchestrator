"""
Phase 7 — Port interface tests.

Verifies:
  - CachePort, StatePort, EventPort Protocol definitions are correct
  - Concrete adapters satisfy the protocols (structural subtyping)
  - NullCache, NullState, NullEventBus implement the ports correctly
  - engine.py accepts NullAdapters in place of real infrastructure
"""

from __future__ import annotations

import pytest

from orchestrator.ports import (
    CachePort,
    EventPort,
    NullCache,
    NullEventBus,
    NullState,
    StatePort,
)


# ─────────────────────────────────────────────────────────────────────────────
# NullCache satisfies CachePort
# ─────────────────────────────────────────────────────────────────────────────


def test_null_cache_satisfies_port():
    assert isinstance(NullCache(), CachePort)


@pytest.mark.asyncio
async def test_null_cache_get_returns_none():
    cache = NullCache()
    result = await cache.get("gpt-4o", "hello", 100, None, 0.7)
    assert result is None


@pytest.mark.asyncio
async def test_null_cache_put_is_noop():
    cache = NullCache()
    await cache.put("gpt-4o", "hello", 100, "hi", 10, 5, None, 0.7)
    # No assertion — must not raise


@pytest.mark.asyncio
async def test_null_cache_close_is_noop():
    cache = NullCache()
    await cache.close()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# NullState satisfies StatePort
# ─────────────────────────────────────────────────────────────────────────────


def test_null_state_satisfies_port():
    assert isinstance(NullState(), StatePort)


@pytest.mark.asyncio
async def test_null_state_load_unknown_returns_none():
    state = NullState()
    assert await state.load_project("nonexistent") is None


@pytest.mark.asyncio
async def test_null_state_save_and_load_roundtrip():
    from unittest.mock import MagicMock
    from orchestrator.models import ProjectState, ProjectStatus

    state = NullState()
    mock_ps = MagicMock(spec=ProjectState)
    await state.save_project("proj-1", mock_ps)
    loaded = await state.load_project("proj-1")
    assert loaded is mock_ps


@pytest.mark.asyncio
async def test_null_state_checkpoint_noop():
    from unittest.mock import MagicMock
    from orchestrator.models import ProjectState

    state = NullState()
    mock_ps = MagicMock(spec=ProjectState)
    await state.save_checkpoint("proj-1", "task-1", mock_ps)
    # No assertion — must not raise


@pytest.mark.asyncio
async def test_null_state_close_is_noop():
    state = NullState()
    await state.close()  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# NullEventBus satisfies EventPort
# ─────────────────────────────────────────────────────────────────────────────


def test_null_event_bus_satisfies_port():
    assert isinstance(NullEventBus(), EventPort)


@pytest.mark.asyncio
async def test_null_event_bus_publish_discards_events():
    bus = NullEventBus()
    await bus.publish({"type": "task_started", "task_id": "t1"})
    # No assertion — must not raise, no side effects


# ─────────────────────────────────────────────────────────────────────────────
# Concrete adapters satisfy protocols (runtime_checkable)
# ─────────────────────────────────────────────────────────────────────────────


def test_disk_cache_satisfies_cache_port():
    """DiskCache must satisfy CachePort without any changes."""
    from orchestrator.cache import DiskCache
    assert isinstance(DiskCache(), CachePort)


def test_state_manager_satisfies_state_port():
    """StateManager must satisfy StatePort without any changes."""
    from orchestrator.state import StateManager
    assert isinstance(StateManager(), StatePort)


# ─────────────────────────────────────────────────────────────────────────────
# Engine accepts NullAdapters (dependency injection works)
# ─────────────────────────────────────────────────────────────────────────────


def test_engine_accepts_null_cache():
    """Engine __init__ must accept a NullCache without raising."""
    from orchestrator.engine import Orchestrator
    orch = Orchestrator(cache=NullCache())
    assert orch.cache is not None


def test_engine_accepts_null_state():
    """Engine __init__ must accept a NullState without raising."""
    from orchestrator.engine import Orchestrator
    orch = Orchestrator(state_manager=NullState())
    assert orch.state_mgr is not None


def test_engine_accepts_both_null_adapters():
    """Engine __init__ must accept both NullCache and NullState together."""
    from orchestrator.engine import Orchestrator
    orch = Orchestrator(cache=NullCache(), state_manager=NullState())
    assert orch.cache is not None
    assert orch.state_mgr is not None
