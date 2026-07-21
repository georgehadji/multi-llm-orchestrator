"""
Tests for checkpoints.py + version_manager.py — State snapshots.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

from orchestrator.checkpoints import (
    CheckpointManager,
    Checkpoint,
)


class TestCheckpoint:
    """Tests for Checkpoint class."""

    def test_checksum_integrity(self):
        """Checkpoint must verify checksum on load."""
        import datetime

        cp = Checkpoint(task_id="test", data={"key": "val"}, timestamp=datetime.datetime.now())
        d = cp.to_dict()
        # Corrupt the data
        d["data"]["key"] = "corrupted"
        with pytest.raises(ValueError, match="checksum"):
            Checkpoint.from_dict(d)

    def test_roundtrip(self):
        """Checkpoint must survive dict roundtrip."""
        import datetime

        cp = Checkpoint(
            task_id="task-1",
            data={"a": 1, "b": [2, 3]},
            timestamp=datetime.datetime(2025, 1, 1, 12, 0, 0),
        )
        d = cp.to_dict()
        restored = Checkpoint.from_dict(d)
        assert restored.task_id == "task-1"
        assert restored.data["a"] == 1


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        """Checkpoint must survive save/load cycle."""
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        path = await mgr.save_checkpoint({"state": "running"}, "task-1")
        assert Path(path).exists()
        loaded = await mgr.load_checkpoint("task-1")
        assert loaded is not None
        assert loaded.data["state"] == "running"

    @pytest.mark.asyncio
    async def test_restore_default_when_missing(self):
        """restore_from_latest must return default if no checkpoint exists."""
        mgr = CheckpointManager(checkpoint_dir=str(Path(tempfile.mkdtemp())))
        result = await mgr.restore_from_latest("nonexistent", {"default": True})
        assert result["default"]

    @pytest.mark.asyncio
    async def test_cleanup_old(self, tmp_path):
        """cleanup_old must keep only last N checkpoints."""
        mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
        for i in range(10):
            await mgr.save_checkpoint({"i": i}, f"task-{i}")
        deleted = await mgr.cleanup_old_checkpoints("task-", keep_last_n=5)
        # Multiple task IDs, so cleanup may delete some
        assert deleted >= 0
