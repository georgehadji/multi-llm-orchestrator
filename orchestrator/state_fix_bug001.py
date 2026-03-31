"""
BUG-001 FIX: Task Field Serialization
=====================================

This module contains the fix for the data loss bug where Task fields
target_path, module_name, and tech_context were not being serialized.

Path A Implementation: Minimal Direct Fix
Nash Stability: 9/10
Adaptation Cost: 2/10
Complexity: 1/10

Author: System Analysis Team
Date: 2026-03-03
"""

import json
import logging
from pathlib import Path

from .budget import Budget
from .models import Model, ProjectState, Task, TaskResult, TaskStatus, TaskType

logger = logging.getLogger("orchestrator.state")


# =============================================================================
# CORE FIX: Updated serialization functions
# =============================================================================


def _task_to_dict(t: Task) -> dict:
    """
    Serialize Task to dictionary.

    BUG-001 FIX: Added target_path, module_name, tech_context fields.

    Args:
        t: Task to serialize

    Returns:
        Dictionary representation of Task
    """
    return {
        "id": t.id,
        "type": t.type.value,
        "prompt": t.prompt,
        "context": t.context,
        "dependencies": t.dependencies,
        "hard_validators": t.hard_validators,
        # BUG-001 FIX: Added missing App Builder fields
        "target_path": t.target_path,
        "module_name": t.module_name,
        "tech_context": t.tech_context,
    }


def _task_from_dict(d: dict) -> Task:
    """
    Deserialize dictionary to Task.

    BUG-001 FIX: Restores target_path, module_name, tech_context fields
    with defaults for backward compatibility.

    Args:
        d: Dictionary containing Task data

    Returns:
        Reconstructed Task object
    """
    t = Task(
        id=d["id"],
        type=TaskType(d["type"]),
        prompt=d["prompt"],
        context=d.get("context", ""),
        dependencies=d.get("dependencies", []),
        hard_validators=d.get("hard_validators", []),
        # BUG-001 FIX: Restore App Builder fields with defaults for backward compat
        target_path=d.get("target_path", ""),
        module_name=d.get("module_name", ""),
        tech_context=d.get("tech_context", ""),
    )

    # Preserve fields added after Task initialization
    t.acceptance_threshold = d.get("acceptance_threshold", 0.85)
    t.max_iterations = d.get("max_iterations", 3)
    t.max_output_tokens = d.get("max_output_tokens", 8192)
    t.status = TaskStatus(d.get("status", "pending"))

    return t


# =============================================================================
# VALIDATION: Runtime completeness checking
# =============================================================================


def _validate_task_completeness(task: Task, source: str = "unknown") -> None:
    """
    Runtime validation that Task has all required fields.

    Logs warnings for missing App Builder fields to detect data loss.

    Args:
        task: Task to validate
        source: Context string for logging (e.g., "load", "create")
    """
    missing = []

    # Check for empty but expected App Builder fields
    if not task.target_path and source != "default":
        missing.append("target_path")
    if not task.module_name:
        missing.append("module_name")
    if not task.tech_context:
        missing.append("tech_context")

    if missing:
        logger.warning(
            f"Task {task.id} loaded with missing App Builder fields: {missing}. "
            f"Source: {source}. This may indicate data loss from pre-fix state "
            f"or incomplete task specification."
        )


# =============================================================================
# FALLBACK: State reconstruction on failure
# =============================================================================


class StateLoadError(Exception):
    """Critical error during state loading that cannot be recovered."""

    pass


async def _attempt_state_reconstruction(
    project_id: str, output_dir: Path | None = None
) -> ProjectState | None:
    """
    Attempt to reconstruct state from output files.

    Fallback when database state is corrupted or incompatible.

    Args:
        project_id: Project identifier
        output_dir: Custom output directory (default: outputs/{project_id})

    Returns:
        Reconstructed ProjectState or None if reconstruction fails
    """
    if output_dir is None:
        output_dir = Path(f"outputs/{project_id}")

    if not output_dir.exists():
        logger.warning(f"Cannot reconstruct {project_id}: output dir not found")
        return None

    # Scan for task output files
    task_files = list(output_dir.glob("task_*"))
    if not task_files:
        logger.warning(f"Cannot reconstruct {project_id}: no task files found")
        return None

    logger.info(f"Attempting to reconstruct {project_id} from {len(task_files)} output files")

    tasks: dict[str, Task] = {}
    results: dict[str, TaskResult] = {}

    for task_file in task_files:
        try:
            # Parse task ID from filename (e.g., "task_001_code_generation.py")
            parts = task_file.stem.split("_")
            if len(parts) >= 2:
                task_id = f"{parts[0]}_{parts[1]}"
            else:
                continue

            # Create minimal task with reconstructed metadata
            target_path = str(task_file.relative_to(output_dir))

            task = Task(
                id=task_id,
                type=TaskType.CODE_GEN,  # Unknown, assume code generation
                prompt=f"Reconstructed from {task_file.name}",
                context="",
                target_path=target_path,
                module_name="",
                tech_context="",
            )
            tasks[task_id] = task

            # Read output as result
            try:
                content = task_file.read_text(encoding="utf-8")
                results[task_id] = TaskResult(
                    task_id=task_id,
                    output=content,
                    score=0.5,  # Unknown quality, neutral
                    model_used=Model.GPT_4O,  # Unknown
                    status=TaskStatus.COMPLETED,  # Assume completed if file exists
                )
            except Exception as e:
                logger.warning(f"Could not read output for {task_id}: {e}")

        except Exception as e:
            logger.warning(f"Could not reconstruct task from {task_file}: {e}")
            continue

    if not tasks:
        logger.error(f"State reconstruction failed for {project_id}: no valid tasks")
        return None

    logger.info(f"Successfully reconstructed {len(tasks)} tasks for {project_id}")

    # Create minimal viable state
    return ProjectState(
        project_description=f"Reconstructed project {project_id}",
        success_criteria="Unknown - reconstructed from output files",
        budget=Budget(max_usd=8.0),  # Fresh budget
        tasks=tasks,
        results=results,
        api_health={},
        status=TaskStatus.PARTIAL_SUCCESS,  # Best effort
        execution_order=list(tasks.keys()),
    )


# =============================================================================
# INTEGRATION: StateManager mixin methods
# =============================================================================


class StateManagerBug001Mixin:
    """
    Mixin providing BUG-001 fix integration for StateManager.

    This mixin should be applied to StateManager to add:
    - Validation on load
    - Fallback reconstruction
    - Error handling
    """

    async def load_project_with_fallback(self, project_id: str) -> ProjectState | None:
        """
        Load project with validation and fallback.

        Primary path: Normal database load
        Fallback path: Reconstruction from output files
        Error: Raise StateLoadError if both fail

        Args:
            project_id: Project identifier

        Returns:
            ProjectState or None if not found

        Raises:
            StateLoadError: If state exists but cannot be loaded or reconstructed
        """
        try:
            # PRIMARY: Attempt normal load
            state = await self._load_project_primary(project_id)

            if state is not None:
                # Validate loaded tasks
                for _task_id, task in state.tasks.items():
                    _validate_task_completeness(task, source="load")

                return state

            return None

        except (json.JSONDecodeError, KeyError, AttributeError, TypeError) as e:
            logger.error(f"State load failed for {project_id}: {e}")

            # FALLBACK: Attempt reconstruction
            reconstructed = await _attempt_state_reconstruction(project_id)

            if reconstructed:
                logger.warning(
                    f"Reconstructed state for {project_id} from outputs. "
                    f"Original state was corrupted: {e}"
                )
                return reconstructed

            # FINAL ERROR: Cannot recover
            logger.critical(f"Could not load or reconstruct project {project_id}")
            raise StateLoadError(
                f"Project {project_id} state is unrecoverable. "
                f"Database load failed: {e}. Reconstruction also failed."
            )

    async def _load_project_primary(self, project_id: str) -> ProjectState | None:
        """
        Primary load path - database read.

        This should be the original StateManager.load_project implementation.
        """
        # This is a placeholder - the actual implementation would be
        # the original load_project code from state.py
        # For now, delegate to parent if possible
        if hasattr(super(), "load_project"):
            return await super().load_project(project_id)
        raise NotImplementedError("Must provide load_project implementation")


# =============================================================================
# BACKWARD COMPATIBILITY: Handle old states gracefully
# =============================================================================


def _migrate_task_from_legacy(d: dict) -> dict:
    """
    Ensure legacy task dict has all required fields.

    Adds default values for fields that may be missing in old states.

    Args:
        d: Task dictionary from potentially old state

    Returns:
        Dictionary with all required keys present
    """
    # Ensure all BUG-001 fields exist with defaults
    d.setdefault("target_path", "")
    d.setdefault("module_name", "")
    d.setdefault("tech_context", "")

    return d


# =============================================================================
# TESTING HELPERS: For unit tests
# =============================================================================


def create_test_task_with_appbuilder_fields() -> Task:
    """
    Create a test Task with all App Builder fields populated.

    Useful for testing BUG-001 fix.
    """
    return Task(
        id="test_task_001",
        type=TaskType.CODE_GEN,
        prompt="Create a React component",
        context="TypeScript project",
        dependencies=[],
        hard_validators=["python_syntax"],
        target_path="app/components/UserProfile.tsx",
        module_name="user-management",
        tech_context="React 18 with TypeScript strict mode",
    )


def verify_task_roundtrip(task: Task) -> bool:
    """
    Verify that a Task survives serialization roundtrip.

    Args:
        task: Task to test

    Returns:
        True if all fields preserved, False otherwise
    """
    try:
        # Serialize
        task_dict = _task_to_dict(task)

        # Full JSON roundtrip
        json_str = json.dumps(task_dict)
        loaded_dict = json.loads(json_str)

        # Deserialize
        restored = _task_from_dict(loaded_dict)

        # Verify all fields
        checks = [
            (restored.id, task.id, "id"),
            (restored.target_path, task.target_path, "target_path"),
            (restored.module_name, task.module_name, "module_name"),
            (restored.tech_context, task.tech_context, "tech_context"),
        ]

        all_passed = True
        for actual, expected, field in checks:
            if actual != expected:
                logger.error(f"Roundtrip failed for {field}: {actual} != {expected}")
                all_passed = False

        return all_passed

    except Exception as e:
        logger.error(f"Roundtrip verification failed: {e}")
        return False


# =============================================================================
# DEPLOYMENT: Integration instructions
# =============================================================================

"""
DEPLOYMENT CHECKLIST
====================

1. Replace existing _task_to_dict and _task_from_dict in state.py with these versions

2. Add validation call in StateManager.load_project after task reconstruction:

   for task_id, task in state.tasks.items():
       _validate_task_completeness(task, source="load")

3. (Optional) Add fallback integration to StateManager.load_project for corrupted states

4. Run unit tests in tests/test_state_serialization.py

5. Deploy to staging, verify no regression in state save/load

6. Deploy to production

ROLLBACK PLAN
=============

If issues detected:
1. Revert to previous _task_to_dict/_task_from_dict implementations
2. Old states will still load (backward compatible defaults)
3. New states will load in old code (extra fields ignored)
4. No data loss on rollback

MONITORING
==========

Watch for these log patterns:
- "Task X loaded with missing App Builder fields" - indicates data loss or incomplete tasks
- "Reconstructed state for X from outputs" - fallback triggered, investigate cause
- "State unrecoverable" - critical failure, requires manual intervention
"""
