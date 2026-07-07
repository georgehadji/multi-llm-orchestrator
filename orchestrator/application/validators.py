"""Input validation helpers for engine entry points."""

from __future__ import annotations

from pathlib import Path


class ValidationError(ValueError):
    """Raised when engine inputs fail validation."""


MAX_PROJECT_DESCRIPTION_LEN = 10_000
MAX_SUCCESS_CRITERIA_LEN = 10_000
MAX_PROJECT_ID_LEN = 256
MIN_BUDGET_USD = 0.0
MAX_BUDGET_USD = 10_000.0
MAX_MAX_PARALLEL_TASKS = 100


def validate_project_args(
    project_description: object,
    success_criteria: object,
    project_id: object = "",
    output_dir: object | None = None,
) -> None:
    """Validate arguments passed to Orchestrator.run_project().

    Raises:
        ValidationError: if any argument is malformed or out of bounds.
    """
    if not isinstance(project_description, str) or not project_description.strip():
        raise ValidationError("project_description must be a non-empty string")
    if len(project_description) > MAX_PROJECT_DESCRIPTION_LEN:
        raise ValidationError(
            f"project_description exceeds {MAX_PROJECT_DESCRIPTION_LEN} characters"
        )

    if not isinstance(success_criteria, str) or not success_criteria.strip():
        raise ValidationError("success_criteria must be a non-empty string")
    if len(success_criteria) > MAX_SUCCESS_CRITERIA_LEN:
        raise ValidationError(f"success_criteria exceeds {MAX_SUCCESS_CRITERIA_LEN} characters")

    if project_id is not None and (
        not isinstance(project_id, str) or len(project_id) > MAX_PROJECT_ID_LEN
    ):
        raise ValidationError(
            f"project_id must be a string with at most {MAX_PROJECT_ID_LEN} characters"
        )

    if output_dir is not None and not isinstance(output_dir, Path):
        raise ValidationError("output_dir must be a Path or None")


def validate_job_spec(spec: object) -> None:
    """Validate a JobSpec before execution.

    This is a lightweight guard; the JobSpec dataclass itself enforces types.
    Raises:
        ValidationError: if the spec contains out-of-bounds values.
    """
    if spec is None:
        raise ValidationError("JobSpec must not be None")

    # Use getattr to remain compatible with both JobSpec variants.
    description = getattr(spec, "project_description", None) or getattr(spec, "description", "")
    criteria = getattr(spec, "success_criteria", None) or ""
    max_parallel_tasks = getattr(spec, "max_parallel_tasks", 3)
    budget = getattr(spec, "budget", None)

    if not isinstance(description, str) or not description.strip():
        raise ValidationError("JobSpec.project_description must be a non-empty string")
    if len(description) > MAX_PROJECT_DESCRIPTION_LEN:
        raise ValidationError(
            f"JobSpec.project_description exceeds {MAX_PROJECT_DESCRIPTION_LEN} characters"
        )

    if not isinstance(criteria, str) or not criteria.strip():
        raise ValidationError("JobSpec.success_criteria must be a non-empty string")
    if len(criteria) > MAX_SUCCESS_CRITERIA_LEN:
        raise ValidationError(
            f"JobSpec.success_criteria exceeds {MAX_SUCCESS_CRITERIA_LEN} characters"
        )

    if (
        not isinstance(max_parallel_tasks, int)
        or not 1 <= max_parallel_tasks <= MAX_MAX_PARALLEL_TASKS
    ):
        raise ValidationError(
            f"JobSpec.max_parallel_tasks must be an integer between 1 and {MAX_MAX_PARALLEL_TASKS}"
        )

    if budget is not None:
        max_usd = getattr(budget, "max_usd", None)
        if max_usd is not None and (
            not isinstance(max_usd, (int, float))
            or max_usd < MIN_BUDGET_USD
            or max_usd > MAX_BUDGET_USD
        ):
            raise ValidationError(
                f"JobSpec.budget.max_usd must be between {MIN_BUDGET_USD} and {MAX_BUDGET_USD}"
            )
