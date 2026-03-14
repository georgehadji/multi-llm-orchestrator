"""Verify engine exposes SessionLifecycleManager integration."""
import pathlib
import tempfile
import pytest
from orchestrator.session_lifecycle import SessionLifecycleManager
from orchestrator.engine import Orchestrator


def test_engine_has_lifecycle_manager():
    """Orchestrator has a _lifecycle_manager attribute of the right type."""
    orch = Orchestrator.__new__(Orchestrator)
    from orchestrator.memory_tier import MemoryTierManager
    with tempfile.TemporaryDirectory() as tmp:
        mem = MemoryTierManager(storage_path=pathlib.Path(tmp), enable_bm25=False)
        orch._lifecycle_manager = SessionLifecycleManager(memory_tier_manager=mem)
        assert isinstance(orch._lifecycle_manager, SessionLifecycleManager)


def test_engine_exposes_configure_lifecycle():
    """Orchestrator has configure_session_lifecycle() that delegates to lifecycle manager."""
    orch = Orchestrator.__new__(Orchestrator)
    from orchestrator.memory_tier import MemoryTierManager
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
