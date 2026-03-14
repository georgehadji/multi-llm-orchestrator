"""Verify engine exposes SessionLifecycleManager integration."""
import pathlib
import tempfile
import pytest
from orchestrator.session_lifecycle import SessionLifecycleManager
from orchestrator.engine import Orchestrator


def _make_wired_orch(tmp):
    """Return a bare Orchestrator-shell with _lifecycle_manager wired."""
    from orchestrator.memory_tier import MemoryTierManager
    orch = Orchestrator.__new__(Orchestrator)
    mem = MemoryTierManager(storage_path=pathlib.Path(tmp), enable_bm25=False)
    orch._lifecycle_manager = SessionLifecycleManager(memory_tier_manager=mem)
    return orch


def test_engine_has_lifecycle_manager():
    """Orchestrator has a _lifecycle_manager attribute of the right type."""
    with tempfile.TemporaryDirectory() as tmp:
        orch = _make_wired_orch(tmp)
        assert isinstance(orch._lifecycle_manager, SessionLifecycleManager)


def test_engine_exposes_configure_lifecycle():
    """configure_session_lifecycle() sets interval and model on lifecycle manager."""
    with tempfile.TemporaryDirectory() as tmp:
        orch = _make_wired_orch(tmp)
        orch.configure_session_lifecycle(migration_interval_hours=2)
        assert orch._lifecycle_manager._interval == 7200  # 2 * 3600


def test_configure_lifecycle_raises_if_scheduler_running():
    """configure_session_lifecycle() raises RuntimeError if scheduler already running."""
    async def _run():
        with tempfile.TemporaryDirectory() as tmp:
            orch = _make_wired_orch(tmp)
            await orch._lifecycle_manager.start()
            try:
                with pytest.raises(RuntimeError, match="before starting the scheduler"):
                    orch.configure_session_lifecycle(migration_interval_hours=1)
            finally:
                await orch._lifecycle_manager.stop()

    import asyncio
    asyncio.get_event_loop().run_until_complete(_run())


def test_session_lifecycle_is_imported_at_module_level():
    """SessionLifecycleManager is importable from orchestrator package."""
    from orchestrator import SessionLifecycleManager as SLM
    assert SLM is not None
