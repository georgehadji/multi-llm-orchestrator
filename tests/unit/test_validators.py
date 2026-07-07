"""Unit tests for engine input validators (F35)."""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.application.validators import (
    MAX_BUDGET_USD,
    MAX_MAX_PARALLEL_TASKS,
    MAX_PROJECT_DESCRIPTION_LEN,
    MAX_PROJECT_ID_LEN,
    MAX_SUCCESS_CRITERIA_LEN,
    MIN_BUDGET_USD,
    ValidationError,
    validate_job_spec,
    validate_project_args,
)
from orchestrator.budget import Budget


class TestValidateProjectArgs:
    def test_valid_args_pass(self):
        validate_project_args("Build a todo app", "CRUD works", "proj-1", Path("/tmp/output"))

    def test_minimal_valid_args_pass(self):
        validate_project_args("X", "Y")

    def test_empty_description_raises(self):
        with pytest.raises(ValidationError, match="project_description"):
            validate_project_args("", "Works")

    def test_whitespace_description_raises(self):
        with pytest.raises(ValidationError, match="project_description"):
            validate_project_args("   ", "Works")

    def test_non_string_description_raises(self):
        with pytest.raises(ValidationError, match="project_description"):
            validate_project_args(None, "Works")

    def test_oversized_description_raises(self):
        huge = "x" * (MAX_PROJECT_DESCRIPTION_LEN + 1)
        with pytest.raises(ValidationError, match="project_description exceeds"):
            validate_project_args(huge, "Works")

    def test_empty_success_criteria_raises(self):
        with pytest.raises(ValidationError, match="success_criteria"):
            validate_project_args("Valid", "")

    def test_whitespace_success_criteria_raises(self):
        with pytest.raises(ValidationError, match="success_criteria"):
            validate_project_args("Valid", "   ")

    def test_non_string_success_criteria_raises(self):
        with pytest.raises(ValidationError, match="success_criteria"):
            validate_project_args("Valid", 123)

    def test_oversized_success_criteria_raises(self):
        huge = "x" * (MAX_SUCCESS_CRITERIA_LEN + 1)
        with pytest.raises(ValidationError, match="success_criteria exceeds"):
            validate_project_args("Valid", huge)

    def test_project_id_too_long_raises(self):
        long_id = "x" * (MAX_PROJECT_ID_LEN + 1)
        with pytest.raises(ValidationError, match="project_id"):
            validate_project_args("Valid", "Works", long_id)

    def test_project_id_none_allowed(self):
        validate_project_args("Valid", "Works", None)

    def test_invalid_output_dir_type_raises(self):
        with pytest.raises(ValidationError, match="output_dir must be a Path"):
            validate_project_args("Valid", "Works", output_dir="/tmp/output")

    def test_output_dir_none_allowed(self):
        validate_project_args("Valid", "Works", output_dir=None)


class FakeJobSpec:
    """Minimal stand-in for JobSpec variants."""

    def __init__(
        self,
        description: str = "Valid",
        criteria: str = "Works",
        max_parallel_tasks: int = 3,
        budget: Budget | None = None,
    ):
        self.project_description = description
        self.success_criteria = criteria
        self.max_parallel_tasks = max_parallel_tasks
        self.budget = budget


class TestValidateJobSpec:
    def test_valid_spec_passes(self):
        spec = FakeJobSpec(budget=Budget(max_usd=10.0, max_time_seconds=300))
        validate_job_spec(spec)

    def test_none_spec_raises(self):
        with pytest.raises(ValidationError, match="JobSpec must not be None"):
            validate_job_spec(None)

    def test_empty_description_raises(self):
        spec = FakeJobSpec(description="", budget=Budget(max_usd=10.0, max_time_seconds=300))
        with pytest.raises(ValidationError, match="JobSpec.project_description"):
            validate_job_spec(spec)

    def test_oversized_description_raises(self):
        spec = FakeJobSpec(
            description="x" * (MAX_PROJECT_DESCRIPTION_LEN + 1),
            budget=Budget(max_usd=10.0, max_time_seconds=300),
        )
        with pytest.raises(ValidationError, match="JobSpec.project_description exceeds"):
            validate_job_spec(spec)

    def test_empty_success_criteria_raises(self):
        spec = FakeJobSpec(criteria="", budget=Budget(max_usd=10.0, max_time_seconds=300))
        with pytest.raises(ValidationError, match="JobSpec.success_criteria"):
            validate_job_spec(spec)

    def test_oversized_success_criteria_raises(self):
        spec = FakeJobSpec(
            criteria="x" * (MAX_SUCCESS_CRITERIA_LEN + 1),
            budget=Budget(max_usd=10.0, max_time_seconds=300),
        )
        with pytest.raises(ValidationError, match="JobSpec.success_criteria exceeds"):
            validate_job_spec(spec)

    def test_max_parallel_tasks_too_low_raises(self):
        spec = FakeJobSpec(max_parallel_tasks=0, budget=Budget(max_usd=10.0, max_time_seconds=300))
        with pytest.raises(ValidationError, match="max_parallel_tasks"):
            validate_job_spec(spec)

    def test_max_parallel_tasks_too_high_raises(self):
        spec = FakeJobSpec(
            max_parallel_tasks=MAX_MAX_PARALLEL_TASKS + 1,
            budget=Budget(max_usd=10.0, max_time_seconds=300),
        )
        with pytest.raises(ValidationError, match="max_parallel_tasks"):
            validate_job_spec(spec)

    def test_budget_max_usd_too_low_raises(self):
        spec = FakeJobSpec(budget=Budget(max_usd=-0.01, max_time_seconds=300))
        with pytest.raises(ValidationError, match="budget.max_usd"):
            validate_job_spec(spec)

    def test_budget_max_usd_too_high_raises(self):
        spec = FakeJobSpec(budget=Budget(max_usd=MAX_BUDGET_USD + 0.01, max_time_seconds=300))
        with pytest.raises(ValidationError, match="budget.max_usd"):
            validate_job_spec(spec)

    def test_budget_none_allowed(self):
        spec = FakeJobSpec(budget=None)
        validate_job_spec(spec)

    def test_boundary_budget_values_pass(self):
        spec_min = FakeJobSpec(budget=Budget(max_usd=MIN_BUDGET_USD, max_time_seconds=300))
        spec_max = FakeJobSpec(budget=Budget(max_usd=MAX_BUDGET_USD, max_time_seconds=300))
        validate_job_spec(spec_min)
        validate_job_spec(spec_max)
