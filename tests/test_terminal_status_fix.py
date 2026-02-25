"""
Tests for Fix #1: Terminal COMPLETED_DEGRADED status to prevent infinite resume loops

Currently, PARTIAL_SUCCESS is used for both:
1. Truly-incomplete runs (interrupted mid-execution)
2. Runs that completed but with degraded quality (some tasks failed validation)

This causes projects to be re-entered via _resume_project() on every subsequent
invocation with PARTIAL_SUCCESS, leading to:
- Unbounded re-execution of expensive tasks
- Output data destruction (overwritten on each pass)
- Loss of parallelism (sequential execution only, 3-5x slower)

The fix introduces COMPLETED_DEGRADED (terminal status, not resumable) to distinguish:
- PARTIAL_SUCCESS: genuinely interrupted mid-run (resumable)
- COMPLETED_DEGRADED: all tasks executed but some failed validation (terminal)
"""

import pytest
from orchestrator.models import ProjectStatus


def test_completed_degraded_status_exists():
    """
    FAILING TEST: Verifies that COMPLETED_DEGRADED status exists in ProjectStatus enum.

    Currently this will fail because COMPLETED_DEGRADED does not exist yet.

    The fix must add COMPLETED_DEGRADED as a new ProjectStatus enum value and update
    orchestrator.engine._determine_final_status() to return it when:
    - All tasks have executed (all have results)
    - All tasks have status in {COMPLETED, DEGRADED, FAILED}
    - But some failed deterministic validation (det_ok = False)
    """
    assert hasattr(ProjectStatus, 'COMPLETED_DEGRADED'), \
        "COMPLETED_DEGRADED status must be added to ProjectStatus enum to prevent infinite resume loops"


def test_partial_success_vs_completed_degraded_distinction():
    """
    Documents the desired distinction between PARTIAL_SUCCESS and COMPLETED_DEGRADED:

    PARTIAL_SUCCESS: Some execution results are missing (interrupted mid-run)
    COMPLETED_DEGRADED: All execution results exist but some failed validation

    This distinction prevents resume loops by ensuring only truly-incomplete runs
    (missing results) can be resumed.
    """
    # This test documents the semantic difference after the fix
    # It will pass once COMPLETED_DEGRADED exists and logic is updated
    project_statuses = [
        attr for attr in dir(ProjectStatus)
        if not attr.startswith('_') and attr.isupper()
    ]

    # Both statuses should exist after fix
    should_exist = ['PARTIAL_SUCCESS', 'COMPLETED_DEGRADED']
    for status in should_exist:
        assert status in project_statuses, f"{status} must exist in ProjectStatus"
