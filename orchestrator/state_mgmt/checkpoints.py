"""
Checkpoints — Backward-compatibility shim
==========================================
The canonical implementation lives in orchestrator/checkpoints.py, which also
has ContentCheckpointManager (content-preserving rollback via a SnapshotPort)
that this module used to lack — the two had silently diverged. See
docs/hunts/t3-persistence/inventory.md C1.

Usage:
    from orchestrator.state_mgmt.checkpoints import CheckpointManager
    manager = CheckpointManager(checkpoint_dir="./checkpoints")
    await manager.save_checkpoint(state_data, "task_id")
    restored_state = await manager.load_checkpoint("task_id")
"""

from ..checkpoints import *  # noqa: F401, F403
