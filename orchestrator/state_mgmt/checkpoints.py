"""
Checkpoints — Intermediate state checkpoints
============================================
Module for managing intermediate state checkpoints during long-running processes.

Pattern: Memento
Async: Yes — for I/O-bound storage operations
Layer: L1 Infrastructure

Usage:
    from orchestrator.checkpoints import CheckpointManager
    manager = CheckpointManager(checkpoint_dir="./checkpoints")
    await manager.save_checkpoint(state_data, "task_id")
    restored_state = await manager.load_checkpoint("task_id")
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
import json
import re
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("orchestrator.checkpoints")


class Checkpoint:
    """Represents a single checkpoint with metadata."""

    def __init__(
        self, task_id: str, data: dict[str, Any], timestamp: datetime, version: str = "1.0"
    ):
        self.task_id = task_id
        self.data = data
        self.timestamp = timestamp
        self.version = version
        self.checksum = self._calculate_checksum(data)

    def _calculate_checksum(self, data: dict[str, Any]) -> str:
        """Calculate a checksum for the checkpoint data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert the checkpoint to a dictionary."""
        return {
            "task_id": self.task_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Create a checkpoint from a dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"])
        checkpoint = cls(
            task_id=data["task_id"],
            data=data["data"],
            timestamp=timestamp,
            version=data.get("version", "1.0"),
        )
        # Verify checksum
        if checkpoint.checksum != data["checksum"]:
            raise ValueError("Checkpoint data corrupted: checksum mismatch")
        return checkpoint


class CheckpointManager:
    """Manages saving and loading of intermediate state checkpoints."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """Initialize the checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    async def save_checkpoint(self, data: dict[str, Any], task_id: str) -> str:
        """
        Save a checkpoint with the given data and task ID.

        Args:
            data: The state data to save
            task_id: Unique identifier for the task

        Returns:
            str: The path to the saved checkpoint file
        """
        # Create checkpoint object
        checkpoint = Checkpoint(task_id=task_id, data=data, timestamp=datetime.now())

        # Create filename with timestamp and task ID
        timestamp_str = checkpoint.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{task_id}_{timestamp_str}.json"
        filepath = self.checkpoint_dir / filename

        # Write checkpoint to file
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)

            logger.info(f"Checkpoint saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise

    async def load_checkpoint(self, task_id: str) -> Checkpoint | None:
        """
        Load the most recent checkpoint for the given task ID.

        Args:
            task_id: Unique identifier for the task

        Returns:
            Checkpoint: The loaded checkpoint or None if not found
        """
        # Find the most recent checkpoint file for this task
        checkpoint_files = list(self.checkpoint_dir.glob(f"checkpoint_{task_id}_*.json"))

        if not checkpoint_files:
            logger.info(f"No checkpoint found for task: {task_id}")
            return None

        # Sort by modification time to get the most recent
        latest_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file, encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            checkpoint = Checkpoint.from_dict(checkpoint_data)
            logger.info(f"Checkpoint loaded: {latest_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

    async def load_specific_checkpoint(self, filepath: str) -> Checkpoint | None:
        """
        Load a specific checkpoint file by path.

        Args:
            filepath: Path to the checkpoint file

        Returns:
            Checkpoint: The loaded checkpoint or None if not found
        """
        try:
            with open(filepath, encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            checkpoint = Checkpoint.from_dict(checkpoint_data)
            logger.info(f"Specific checkpoint loaded: {filepath}")
            return checkpoint
        except FileNotFoundError:
            logger.warning(f"Checkpoint file not found: {filepath}")
            return None
        except Exception as e:
            logger.error(f"Failed to load checkpoint from {filepath}: {e}")
            return None

    async def list_checkpoints(self, task_id: str | None = None) -> list:
        """
        List all checkpoints, optionally filtered by task ID.

        Args:
            task_id: Optional task ID to filter checkpoints

        Returns:
            list: List of checkpoint file paths
        """
        pattern = f"checkpoint_{task_id}_*.json" if task_id else "checkpoint_*.json"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))

        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        return [str(f) for f in checkpoint_files]

    async def delete_checkpoint(self, filepath: str) -> bool:
        """
        Delete a specific checkpoint file.

        Args:
            filepath: Path to the checkpoint file to delete

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            file_path = Path(filepath)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Checkpoint deleted: {filepath}")
                return True
            else:
                logger.warning(f"Checkpoint file not found for deletion: {filepath}")
                return False
        except Exception as e:
            logger.error(f"Failed to delete checkpoint: {e}")
            return False

    async def cleanup_old_checkpoints(self, task_id: str, keep_last_n: int = 5) -> int:
        """
        Clean up old checkpoints, keeping only the most recent N.

        Args:
            task_id: Task ID to clean up checkpoints for
            keep_last_n: Number of most recent checkpoints to keep

        Returns:
            int: Number of checkpoints deleted
        """
        checkpoint_files = list(self.checkpoint_dir.glob(f"checkpoint_{task_id}_*.json"))

        if len(checkpoint_files) <= keep_last_n:
            return 0  # Nothing to clean up

        # Sort by modification time (oldest first)
        checkpoint_files.sort(key=lambda f: f.stat().st_mtime)

        # Delete oldest files, keeping only the last N
        files_to_delete = checkpoint_files[:-keep_last_n]
        deleted_count = 0

        for file_path in files_to_delete:
            try:
                file_path.unlink()
                deleted_count += 1
                logger.info(f"Old checkpoint cleaned up: {file_path}")
            except Exception as e:
                logger.error(f"Failed to delete old checkpoint {file_path}: {e}")

        return deleted_count

    async def validate_checkpoint_integrity(self, filepath: str) -> bool:
        """
        Validate the integrity of a checkpoint file by checking its checksum.

        Args:
            filepath: Path to the checkpoint file to validate

        Returns:
            bool: True if the checkpoint is valid, False otherwise
        """
        try:
            checkpoint = await self.load_specific_checkpoint(filepath)
            return checkpoint is not None
        except ValueError as e:
            # Checksum mismatch or other validation error
            logger.error(f"Checkpoint integrity validation failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to validate checkpoint: {e}")
            return False

    async def restore_from_latest(
        self, task_id: str, default_data: dict[str, Any] = None
    ) -> dict[str, Any]:
        """
        Restore state from the latest checkpoint for a task, with a default fallback.

        Args:
            task_id: Task ID to restore from
            default_data: Default data to return if no checkpoint exists

        Returns:
            Dict[str, Any]: Restored state data or default data
        """
        checkpoint = await self.load_checkpoint(task_id)
        if checkpoint:
            return checkpoint.data
        else:
            logger.info(f"No checkpoint found for {task_id}, returning default data")
            return default_data or {}


# ─────────────────────────────────────────────────────────────────────
# Category 2, Phase 1 (Replit): Named checkpoints + rollback + artifacts
# ─────────────────────────────────────────────────────────────────────


@dataclass
class NamedCheckpoint:
    """A named snapshot with file artifacts, budget tracking, and rollback."""

    name: str
    description: str = ""
    timestamp: float = field(default_factory=time.time)
    task_states: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # filename -> sha256
    budget_spent: float = 0.0
    conversation_context: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "timestamp": self.timestamp,
            "task_states": self.task_states,
            "artifacts": self.artifacts,
            "budget_spent": self.budget_spent,
            "conversation_context": self.conversation_context,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "NamedCheckpoint":
        return cls(
            name=d["name"],
            description=d.get("description", ""),
            timestamp=d.get("timestamp", time.time()),
            task_states=d.get("task_states", {}),
            artifacts=d.get("artifacts", {}),
            budget_spent=d.get("budget_spent", 0.0),
            conversation_context=d.get("conversation_context", ""),
        )


class NamedCheckpointManager(CheckpointManager):
    """CheckpointManager extended with named snapshots and rollback."""

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        super().__init__(checkpoint_dir)
        self._snapshots_dir = self.checkpoint_dir / "snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

    async def create_snapshot(
        self,
        name: str,
        description: str = "",
        output_dir: str | None = None,
        conversation_summary: str = "",
    ) -> NamedCheckpoint:
        """Create a named snapshot of the current project state.

        Args:
            name: Human-readable snapshot name (e.g., "before-refactor")
            description: What this snapshot captures
            output_dir: Path to copy artifacts from
            conversation_summary: Condensed conversation context

        Returns:
            NamedCheckpoint with metadata and artifact hashes
        """
        import hashlib

        cp = NamedCheckpoint(
            name=name,
            description=description,
            conversation_context=conversation_summary,
        )

        # Hash artifacts if output_dir provided
        if output_dir:

            out = Path(output_dir)
            if out.exists():
                for f in out.rglob("*"):
                    if f.is_file() and f.stat().st_size < 10_000_000:  # 10MB max
                        with open(f, "rb") as fh:
                            cp.artifacts[str(f.relative_to(out))] = hashlib.sha256(
                                fh.read()
                            ).hexdigest()

        # Save snapshot
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
        filepath = self._snapshots_dir / f"snapshot_{safe_name}.json"
        filepath.write_text(json.dumps(cp.to_dict(), indent=2, default=str), encoding="utf-8")
        logger.info(f"Snapshot '{name}' saved: {filepath}")
        return cp

    async def rollback(self, snapshot_name: str, output_dir: str) -> NamedCheckpoint | None:
        """Restore project to a named snapshot.

        Does NOT modify files — returns the snapshot data so the caller
        can restore state. File restoration is caller's responsibility.

        Args:
            snapshot_name: Name of the snapshot to rollback to
            output_dir: Directory to compare against (for diff)

        Returns:
            NamedCheckpoint if found, None otherwise
        """
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", snapshot_name)
        filepath = self._snapshots_dir / f"snapshot_{safe_name}.json"
        if not filepath.exists():
            logger.warning(f"Snapshot '{snapshot_name}' not found")
            return None

        data = json.loads(filepath.read_text(encoding="utf-8"))
        cp = NamedCheckpoint.from_dict(data)
        logger.info(f"Rolled back to snapshot '{snapshot_name}'")
        return cp

    async def list_snapshots(self) -> list[NamedCheckpoint]:
        """List all named snapshots, most recent first."""
        snapshots = []
        for f in sorted(
            self._snapshots_dir.glob("snapshot_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                snapshots.append(NamedCheckpoint.from_dict(data))
            except Exception as e:
                logger.warning(f"Failed to load snapshot {f}: {e}")
        return snapshots

    async def compare_snapshots(self, name_a: str, name_b: str) -> dict[str, Any]:
        """Compare two named snapshots and return diff.

        Returns a dict with added_files, removed_files, modified_files,
        and budget_delta.
        """
        safe_a = re.sub(r"[^a-zA-Z0-9_-]", "_", name_a)
        safe_b = re.sub(r"[^a-zA-Z0-9_-]", "_", name_b)
        fp_a = self._snapshots_dir / f"snapshot_{safe_a}.json"
        fp_b = self._snapshots_dir / f"snapshot_{safe_b}.json"

        if not fp_a.exists() or not fp_b.exists():
            return {"error": "One or both snapshots not found"}

        cp_a = NamedCheckpoint.from_dict(json.loads(fp_a.read_text(encoding="utf-8")))
        cp_b = NamedCheckpoint.from_dict(json.loads(fp_b.read_text(encoding="utf-8")))

        files_a = set(cp_a.artifacts.keys())
        files_b = set(cp_b.artifacts.keys())

        return {
            "added_files": sorted(files_b - files_a),
            "removed_files": sorted(files_a - files_b),
            "modified_files": sorted(
                f for f in files_a & files_b if cp_a.artifacts[f] != cp_b.artifacts[f]
            ),
            "budget_delta": cp_b.budget_spent - cp_a.budget_spent,
            "snapshot_a": name_a,
            "snapshot_b": name_b,
        }
