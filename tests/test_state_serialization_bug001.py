"""
FALSIFYING UNIT TESTS for BUG-001: Task Field Serialization
===========================================================

These tests are designed to FAIL if BUG-001 ever returns (regression).
Each test has a clear falsification condition.

Test Philosophy:
- If BUG-001 is fixed, all tests pass
- If BUG-001 returns, at least one test will fail
- Tests are self-documenting about what they falsify

Author: System Analysis Team
Date: 2026-03-03
"""

import json
import inspect
import pytest
from dataclasses import fields

# Import the fixed functions
from orchestrator.state_fix_bug001 import (
    _task_to_dict,
    _task_from_dict,
    _validate_task_completeness,
    verify_task_roundtrip,
    create_test_task_with_appbuilder_fields,
    _migrate_task_from_legacy,
)
from orchestrator.models import Task, TaskType, ProjectState, Budget, TaskResult, TaskStatus, Model

# =============================================================================
# FALSIFICATION TEST SUITE
# =============================================================================


class TestBUG001TaskFieldSerialization:
    """
    PRIMARY FALSIFICATION TESTS for BUG-001.

    If these tests fail, BUG-001 (Task field data loss) has returned.
    """

    def test_falsify_task_fields_serialized(self):
        """
        FALSIFIES: The claim that 'target_path', 'module_name', 'tech_context'
        are properly serialized in _task_to_dict.

        If this fails: BUG-001 exists - fields not in serialization output.
        """
        task = Task(
            id="task_falsify_001",
            type=TaskType.CODE_GEN,
            prompt="Build a React component",
            context="Use TypeScript",
            dependencies=["task_000"],
            hard_validators=["python_syntax"],
            target_path="app/components/UserProfile.tsx",
            module_name="user-management",
            tech_context="React 18 with TypeScript strict mode",
        )

        task_dict = _task_to_dict(task)

        # FALSIFICATION CHECKS
        # If any of these fail, the fields are not being serialized
        assert (
            "target_path" in task_dict
        ), "BUG-001 REGRESSION: 'target_path' not in serialized dict"
        assert (
            "module_name" in task_dict
        ), "BUG-001 REGRESSION: 'module_name' not in serialized dict"
        assert (
            "tech_context" in task_dict
        ), "BUG-001 REGRESSION: 'tech_context' not in serialized dict"

        # Verify values are preserved
        assert task_dict["target_path"] == "app/components/UserProfile.tsx"
        assert task_dict["module_name"] == "user-management"
        assert task_dict["tech_context"] == "React 18 with TypeScript strict mode"

    def test_falsify_task_fields_deserialized(self):
        """
        FALSIFIES: The claim that 'target_path', 'module_name', 'tech_context'
        are properly restored in _task_from_dict.

        If this fails: BUG-001 exists - fields not restored from dict.
        """
        task_dict = {
            "id": "task_falsify_002",
            "type": "code_generation",
            "prompt": "Test task",
            "context": "",
            "dependencies": [],
            "hard_validators": [],
            "target_path": "app/dashboard/Metrics.tsx",
            "module_name": "",  # Empty string - should be preserved
            "tech_context": "Next.js 14 with App Router",
        }

        restored = _task_from_dict(task_dict)

        # FALSIFICATION CHECKS
        assert (
            restored.target_path == "app/dashboard/Metrics.tsx"
        ), f"BUG-001 REGRESSION: target_path not restored: got '{restored.target_path}'"
        assert (
            restored.module_name == ""
        ), f"BUG-001 REGRESSION: module_name not restored: got '{restored.module_name}'"
        assert (
            restored.tech_context == "Next.js 14 with App Router"
        ), f"BUG-001 REGRESSION: tech_context not restored: got '{restored.tech_context}'"

    def test_falsify_task_complete_roundtrip(self):
        """
        FALSIFIES: The claim that Task fields survive full save/load cycle.

        If this fails: BUG-001 exists - data lost during roundtrip.
        """
        original = create_test_task_with_appbuilder_fields()

        # Full roundtrip: Task -> dict -> JSON -> dict -> Task
        task_dict = _task_to_dict(original)
        json_str = json.dumps(task_dict)
        loaded_dict = json.loads(json_str)
        restored = _task_from_dict(loaded_dict)

        # FALSIFICATION: Compare all App Builder fields
        assert restored.target_path == original.target_path, (
            f"BUG-001 REGRESSION: target_path mismatch after roundtrip\n"
            f"  Original: {original.target_path}\n"
            f"  Restored: {restored.target_path}"
        )

        assert (
            restored.module_name == original.module_name
        ), f"BUG-001 REGRESSION: module_name mismatch after roundtrip"

        assert (
            restored.tech_context == original.tech_context
        ), f"BUG-001 REGRESSION: tech_context mismatch after roundtrip"

    def test_falsify_all_task_fields_covered(self):
        """
        FALSIFIES: The claim that serialization handles ALL Task dataclass fields.

        If this fails: New fields were added to Task but not to serialization.
        This would be a recurrence of the BUG-001 pattern.
        """
        # Get all field names from Task dataclass
        task_fields = {f.name for f in fields(Task)}

        # Create a sample task and serialize it
        sample_task = Task(id="test_coverage")
        serialized = _task_to_dict(sample_task)
        serialized_keys = set(serialized.keys())

        # Fields that are NOT required to be serialized (computed or internal)
        optional_fields = {
            "acceptance_threshold",  # Has default, often not set
            "max_iterations",  # Has default
            "max_output_tokens",  # Has default
            "status",  # Runtime state, has default
        }

        required_fields = task_fields - optional_fields
        missing = required_fields - serialized_keys

        # FALSIFICATION: If any required field is missing, BUG exists
        assert not missing, (
            f"BUG-001 PATTERN DETECTED: Task fields not serialized: {missing}\n"
            f"This is the same bug pattern as BUG-001 - fields added to dataclass "
            f"but not to serialization. Add these fields to _task_to_dict."
        )


class TestBUG001BackwardCompatibility:
    """
    Tests for backward compatibility with old states.

    These ensure the fix doesn't break loading of pre-existing states.
    """

    def test_load_legacy_state_missing_fields(self):
        """
        Test: Old state format (missing App Builder fields) loads without error.

        This simulates states created before BUG-001 was fixed.
        """
        # Simulate old state format (pre-fix)
        legacy_task_dict = {
            "id": "task_legacy",
            "type": "code_generation",
            "prompt": "Legacy task from before fix",
            "context": "",
            "dependencies": [],
            "hard_validators": [],
            # INTENTIONALLY MISSING: target_path, module_name, tech_context
        }

        # Should not raise
        restored = _task_from_dict(legacy_task_dict)

        # Should have defaults
        assert restored.target_path == ""
        assert restored.module_name == ""
        assert restored.tech_context == ""

    def test_migrate_legacy_dict(self):
        """
        Test: Migration helper adds missing fields to legacy dicts.
        """
        legacy = {
            "id": "task_to_migrate",
            "type": "code_generation",
            # Missing App Builder fields
        }

        migrated = _migrate_task_from_legacy(legacy)

        # Should now have all fields
        assert "target_path" in migrated
        assert "module_name" in migrated
        assert "tech_context" in migrated

        # Should have default values
        assert migrated["target_path"] == ""
        assert migrated["module_name"] == ""
        assert migrated["tech_context"] == ""


class TestBUG001EdgeCases:
    """
    Edge case tests to ensure robustness.
    """

    def test_unicode_preservation(self):
        """
        Test: Unicode characters in App Builder fields are preserved.
        """
        task = Task(
            id="task_unicode",
            target_path="app/用户/页面.tsx",  # Chinese
            module_name="用户仪表板",
            tech_context="支持中文和日本語",
        )

        # Full JSON roundtrip
        task_dict = _task_to_dict(task)
        json_str = json.dumps(task_dict)
        loaded_dict = json.loads(json_str)
        restored = _task_from_dict(loaded_dict)

        # Unicode should be preserved exactly
        assert restored.target_path == "app/用户/页面.tsx"
        assert restored.module_name == "用户仪表板"
        assert restored.tech_context == "支持中文和日本語"

    def test_special_characters(self):
        """
        Test: Special characters in paths are preserved.
        """
        task = Task(
            id="task_special",
            target_path="app/path with spaces/file.tsx",
            module_name="module-with-dashes",
            tech_context="Uses 'quotes' and \"double quotes\" and \\ backslash",
        )

        task_dict = _task_to_dict(task)
        json_str = json.dumps(task_dict)
        loaded_dict = json.loads(json_str)
        restored = _task_from_dict(loaded_dict)

        assert restored.target_path == "app/path with spaces/file.tsx"
        assert restored.module_name == "module-with-dashes"
        assert "quotes" in restored.tech_context

    def test_empty_strings_preserved(self):
        """
        Test: Empty strings are preserved (not converted to None or defaults).
        """
        task = Task(
            id="task_empty",
            target_path="",  # Explicitly empty
            module_name="",
            tech_context="",
        )

        task_dict = _task_to_dict(task)
        restored = _task_from_dict(task_dict)

        # Empty strings should remain empty, not become None or other defaults
        assert restored.target_path == ""
        assert restored.module_name == ""
        assert restored.tech_context == ""
        assert isinstance(restored.target_path, str)


class TestBUG001Validation:
    """
    Tests for validation and detection of missing fields.
    """

    def test_validation_detects_missing_fields(self, caplog):
        """
        Test: Validation logs warning for tasks with missing App Builder fields.
        """
        import logging

        # Create task with some missing fields
        task = Task(
            id="task_incomplete",
            target_path="app/page.tsx",  # Has this
            module_name="",  # Missing (empty)
            tech_context="",  # Missing (empty)
        )

        # Validate should log warning
        with caplog.at_level(logging.WARNING):
            _validate_task_completeness(task, source="test")

        # Check warning was logged
        assert "task_incomplete" in caplog.text
        assert "missing App Builder fields" in caplog.text
        assert "module_name" in caplog.text
        assert "tech_context" in caplog.text


class TestBUG001RegressionMetaTests:
    """
    Meta-tests that inspect the source code to prevent regression.

    These tests verify that the fix is present in the source code.
    """

    def test_source_code_contains_serialization_fields(self):
        """
        FALSIFIES: The claim that the source code includes App Builder field serialization.

        This test inspects the actual source code to ensure the fix is present.
        If this fails, the fix has been removed or reverted.
        """
        from orchestrator import state_fix_bug001

        source = inspect.getsource(state_fix_bug001._task_to_dict)

        # FALSIFICATION: If these strings are not in source, fix is missing
        assert "target_path" in source, (
            "SOURCE REGRESSION: 'target_path' not found in _task_to_dict source. "
            "BUG-001 fix has been removed."
        )

        assert "module_name" in source, (
            "SOURCE REGRESSION: 'module_name' not found in _task_to_dict source. "
            "BUG-001 fix has been removed."
        )

        assert "tech_context" in source, (
            "SOURCE REGRESSION: 'tech_context' not found in _task_to_dict source. "
            "BUG-001 fix has been removed."
        )

    def test_source_code_contains_deserialization_fields(self):
        """
        FALSIFIES: The claim that the source code includes App Builder field deserialization.
        """
        from orchestrator import state_fix_bug001

        source = inspect.getsource(state_fix_bug001._task_from_dict)

        assert (
            "target_path" in source
        ), "SOURCE REGRESSION: 'target_path' not found in _task_from_dict source."
        assert (
            "module_name" in source
        ), "SOURCE REGRESSION: 'module_name' not found in _task_from_dict source."
        assert (
            "tech_context" in source
        ), "SOURCE REGRESSION: 'tech_context' not found in _task_from_dict source."


class TestBUG001Integration:
    """
    Integration tests with full ProjectState.
    """

    def test_full_state_roundtrip(self):
        """
        Test: Full ProjectState with Tasks survives roundtrip.
        """
        from orchestrator.state_fix_bug001 import _task_to_dict, _task_from_dict

        # Create a realistic project state
        original_state = ProjectState(
            project_description="Test project with App Builder",
            success_criteria="All components generated",
            budget=Budget(max_usd=10.0, spent_usd=2.5),
            tasks={
                "task_001": Task(
                    id="task_001",
                    type=TaskType.CODE_GEN,
                    prompt="Create user profile page",
                    target_path="app/user/profile.tsx",
                    module_name="user-profile",
                    tech_context="Next.js 14, React Server Components",
                ),
                "task_002": Task(
                    id="task_002",
                    type=TaskType.CODE_GEN,
                    prompt="Create dashboard layout",
                    target_path="app/dashboard/layout.tsx",
                    module_name="dashboard",
                    tech_context="Next.js App Router, parallel routes",
                ),
            },
            results={
                "task_001": TaskResult(
                    task_id="task_001",
                    output="export default function Profile() {}",
                    score=0.92,
                    model_used=Model.GPT_4O,
                    status=TaskStatus.COMPLETED,
                ),
            },
            api_health={Model.GPT_4O: True},
            status=TaskStatus.PARTIAL_SUCCESS,
            execution_order=["task_001", "task_002"],
        )

        # Serialize all tasks
        tasks_dict = {k: _task_to_dict(v) for k, v in original_state.tasks.items()}

        # Full JSON roundtrip
        json_str = json.dumps(tasks_dict)
        loaded_dict = json.loads(json_str)

        # Deserialize
        restored_tasks = {k: _task_from_dict(v) for k, v in loaded_dict.items()}

        # Verify App Builder fields preserved
        assert restored_tasks["task_001"].target_path == "app/user/profile.tsx"
        assert restored_tasks["task_001"].module_name == "user-profile"
        assert restored_tasks["task_002"].target_path == "app/dashboard/layout.tsx"
        assert restored_tasks["task_002"].module_name == "dashboard"


# =============================================================================
# TEST UTILITIES
# =============================================================================


def test_create_test_task_helper():
    """Test the test helper creates valid tasks."""
    task = create_test_task_with_appbuilder_fields()

    assert task.id == "test_task_001"
    assert task.target_path == "app/components/UserProfile.tsx"
    assert task.module_name == "user-management"
    assert "TypeScript" in task.tech_context


def test_verify_roundtrip_helper():
    """Test the roundtrip verification helper."""
    task = create_test_task_with_appbuilder_fields()

    # Should pass for properly serialized task
    assert verify_task_roundtrip(task) is True


# =============================================================================
# PYTEST CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    # Run with: pytest tests/test_state_serialization_bug001.py -v
    pytest.main([__file__, "-v"])
