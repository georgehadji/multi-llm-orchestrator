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

import asyncio
import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("orchestrator.checkpoints")


class Checkpoint:
    """Represents a single checkpoint with metadata."""
    
    def __init__(self, task_id: str, data: Dict[str, Any], timestamp: datetime, version: str = "1.0"):
        self.task_id = task_id
        self.data = data
        self.timestamp = timestamp
        self.version = version
        self.checksum = self._calculate_checksum(data)
    
    def _calculate_checksum(self, data: Dict[str, Any]) -> str:
        """Calculate a checksum for the checkpoint data."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the checkpoint to a dictionary."""
        return {
            "task_id": self.task_id,
            "data": self.data,
            "timestamp": self.timestamp.isoformat(),
            "version": self.version,
            "checksum": self.checksum
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Checkpoint:
        """Create a checkpoint from a dictionary."""
        timestamp = datetime.fromisoformat(data["timestamp"])
        checkpoint = cls(
            task_id=data["task_id"],
            data=data["data"],
            timestamp=timestamp,
            version=data.get("version", "1.0")
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
    
    async def save_checkpoint(self, data: Dict[str, Any], task_id: str) -> str:
        """
        Save a checkpoint with the given data and task ID.
        
        Args:
            data: The state data to save
            task_id: Unique identifier for the task
            
        Returns:
            str: The path to the saved checkpoint file
        """
        # Create checkpoint object
        checkpoint = Checkpoint(
            task_id=task_id,
            data=data,
            timestamp=datetime.now()
        )
        
        # Create filename with timestamp and task ID
        timestamp_str = checkpoint.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{task_id}_{timestamp_str}.json"
        filepath = self.checkpoint_dir / filename
        
        # Write checkpoint to file
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(checkpoint.to_dict(), f, indent=2, default=str)
            
            logger.info(f"Checkpoint saved: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            raise
    
    async def load_checkpoint(self, task_id: str) -> Optional[Checkpoint]:
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
            with open(latest_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            
            checkpoint = Checkpoint.from_dict(checkpoint_data)
            logger.info(f"Checkpoint loaded: {latest_file}")
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    async def load_specific_checkpoint(self, filepath: str) -> Optional[Checkpoint]:
        """
        Load a specific checkpoint file by path.
        
        Args:
            filepath: Path to the checkpoint file
            
        Returns:
            Checkpoint: The loaded checkpoint or None if not found
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
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
    
    async def list_checkpoints(self, task_id: Optional[str] = None) -> list:
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
    
    async def restore_from_latest(self, task_id: str, default_data: Dict[str, Any] = None) -> Dict[str, Any]:
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