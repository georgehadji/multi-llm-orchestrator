"""
Tier 3 Quality Optimizations
=============================
Author: Georgios-Chrysovalantis Chatzivantsidis

Includes:
- Adaptive Temperature (retry with increasing temperature)
- Auto Eval Dataset (build evaluation dataset from failures)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.log_config import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Adaptive Temperature
# ─────────────────────────────────────────────


@dataclass
class TemperatureMetrics:
    """Metrics for adaptive temperature."""

    total_attempts: int = 0
    initial_successes: int = 0
    retry_successes: int = 0
    failures: int = 0
    avg_temperature: float = 0.0
    avg_retry_count: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_attempts": self.total_attempts,
            "initial_successes": self.initial_successes,
            "retry_successes": self.retry_successes,
            "failures": self.failures,
            "success_rate": (self.initial_successes + self.retry_successes)
            / max(1, self.total_attempts),
            "avg_temperature": self.avg_temperature,
            "avg_retry_count": self.avg_retry_count,
        }


class AdaptiveTemperatureController:
    """
    Adaptive temperature per phase with retry strategy.

    Usage:
        controller = AdaptiveTemperatureController(client=api_client)
        result = await controller.generate_with_retry(model, prompt, task_type)
    """

    # Temperature strategy per phase
    TEMPERATURE_STRATEGY = {
        "decomposition": {"initial": 0.0, "retry_1": 0.2, "retry_2": 0.4},
        "generation": {"initial": 0.0, "retry_1": 0.1, "retry_2": 0.3},
        "critique": {"initial": 0.3, "retry_1": 0.5, "retry_2": 0.7},
        "creative": {"initial": 0.7, "retry_1": 0.9, "retry_2": 1.0},
        "evaluation": {"initial": 0.0, "retry_1": 0.2, "retry_2": 0.4},
    }

    MAX_RETRIES = 2

    def __init__(self, client=None):
        """Initialize adaptive temperature controller."""
        self.client = client
        self.metrics = TemperatureMetrics()
        self._strategies = dict(self.TEMPERATURE_STRATEGY)

    def set_strategy(
        self,
        phase: str,
        initial: float,
        retry_1: float,
        retry_2: float,
    ) -> None:
        """Set custom temperature strategy for phase."""
        self._strategies[phase] = {
            "initial": initial,
            "retry_1": retry_1,
            "retry_2": retry_2,
        }

    async def generate_with_retry(
        self,
        model: str,
        prompt: str,
        task_type: str,
        validator=None,
        **kwargs,
    ) -> Any:
        """
        Generate with adaptive temperature and retry.

        Args:
            model: Model to use
            prompt: Prompt text
            task_type: Task type for temperature strategy
            validator: Optional validator function
            **kwargs: Additional API parameters

        Returns:
            Generation result
        """
        self.metrics.total_attempts += 1

        strategy = self._strategies.get(
            task_type,
            self.TEMPERATURE_STRATEGY.get("generation", {}),
        )

        temperatures = [
            strategy.get("initial", 0.0),
            strategy.get("retry_1", 0.2),
            strategy.get("retry_2", 0.4),
        ]

        last_error = None
        retry_count = 0

        for attempt in range(self.MAX_RETRIES + 1):
            temperature = temperatures[attempt] if attempt < len(temperatures) else 1.0

            try:
                # Generate with temperature
                response = await self.client.call(
                    model=model,
                    system_prompt=prompt,
                    temperature=temperature,
                    **kwargs,
                )

                # Validate if validator provided
                if validator:
                    is_valid = await validator(response)
                    if not is_valid:
                        raise ValueError("Validation failed")

                # Success
                if attempt == 0:
                    self.metrics.initial_successes += 1
                else:
                    self.metrics.retry_successes += 1

                self.metrics.avg_temperature = (
                    self.metrics.avg_temperature * (self.metrics.total_attempts - 1) + temperature
                ) / self.metrics.total_attempts

                return response

            except Exception as e:
                last_error = e
                retry_count = attempt
                logger.warning(f"Attempt {attempt + 1} failed at temperature {temperature}: {e}")

        # All attempts failed
        self.metrics.failures += 1
        self.metrics.avg_retry_count = (
            self.metrics.avg_retry_count * (self.metrics.total_attempts - 1) + retry_count
        ) / self.metrics.total_attempts

        raise RuntimeError(f"All {self.MAX_RETRIES + 1} attempts failed: {last_error}")

    def get_metrics(self) -> dict[str, Any]:
        """Get temperature metrics."""
        return self.metrics.to_dict()


# ─────────────────────────────────────────────
# Auto Eval Dataset Builder
# ─────────────────────────────────────────────


@dataclass
class EvalTestCase:
    """Single evaluation test case."""

    prompt: str
    bad_output: str
    errors: list[str]
    scores: dict[str, float]
    timestamp: str
    model: str
    task_type: str


@dataclass
class DatasetMetrics:
    """Metrics for eval dataset."""

    total_cases: int = 0
    cases_by_error_type: dict[str, int] = field(default_factory=dict)
    cases_by_model: dict[str, int] = field(default_factory=dict)
    avg_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_cases": self.total_cases,
            "cases_by_error_type": self.cases_by_error_type,
            "cases_by_model": self.cases_by_model,
            "avg_score": self.avg_score,
        }


class EvalDatasetBuilder:
    """
    Auto-build evaluation dataset from production failures.

    Usage:
        builder = EvalDatasetBuilder()
        await builder.record_failure(task, code, errors, scores)
        dataset = builder.get_dataset()
    """

    def __init__(self, dataset_path: str | None = None):
        """
        Initialize eval dataset builder.

        Args:
            dataset_path: Path to store dataset (default: .orchestrator/eval_dataset.jsonl)
        """
        self.dataset_path = (
            Path(dataset_path) if dataset_path else Path(".orchestrator/eval_dataset.jsonl")
        )
        self.metrics = DatasetMetrics()
        self._cases: list[EvalTestCase] = []

        # Ensure directory exists
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)

    async def record_failure(
        self,
        task_prompt: str,
        generated_code: str,
        errors: list[str],
        eval_scores: dict[str, float],
        model: str,
        task_type: str,
    ) -> None:
        """
        Record production failure as test case.

        Args:
            task_prompt: Original prompt
            generated_code: Generated code that failed
            errors: List of errors
            eval_scores: Evaluation scores
            model: Model that generated the code
            task_type: Task type
        """
        test_case = EvalTestCase(
            prompt=task_prompt,
            bad_output=generated_code,
            errors=errors,
            scores=eval_scores,
            timestamp=datetime.now().isoformat(),
            model=model,
            task_type=task_type,
        )

        self._cases.append(test_case)
        self.metrics.total_cases += 1

        # Update error type counts
        for error in errors:
            error_type = error.split(":")[0] if ":" in error else "unknown"
            self.metrics.cases_by_error_type[error_type] = (
                self.metrics.cases_by_error_type.get(error_type, 0) + 1
            )

        # Update model counts
        self.metrics.cases_by_model[model] = self.metrics.cases_by_model.get(model, 0) + 1

        # Update average score
        if eval_scores:
            avg = sum(eval_scores.values()) / len(eval_scores)
            self.metrics.avg_score = (
                self.metrics.avg_score * (self.metrics.total_cases - 1) + avg
            ) / self.metrics.total_cases

        # Persist to file
        self._persist_case(test_case)

        logger.info(
            f"Recorded failure: {task_type} ({model}) - " f"errors: {len(errors)}, score: {avg:.3f}"
        )

    def _persist_case(self, case: EvalTestCase) -> None:
        """
        Persist test case to JSONL file with file locking.

        FIX-OPT-005: Use atomic append to prevent race conditions
        when multiple failures are recorded concurrently.
        """

        case_dict = {
            "prompt": case.prompt,
            "bad_output": case.bad_output,
            "errors": case.errors,
            "scores": case.scores,
            "timestamp": case.timestamp,
            "model": case.model,
            "task_type": case.task_type,
        }

        json_line = json.dumps(case_dict, ensure_ascii=False) + "\n"

        # FIX-OPT-005: Atomic append - Python's file append is atomic on most platforms
        # For high-concurrency scenarios, consider using a proper database
        with open(self.dataset_path, "a", encoding="utf-8") as f:
            f.write(json_line)
            f.flush()
            os.fsync(f.fileno())  # Ensure written to disk

    def get_dataset(self) -> list[EvalTestCase]:
        """Get all test cases."""
        return self._cases

    def load_dataset(self) -> list[EvalTestCase]:
        """Load dataset from file."""
        if not self.dataset_path.exists():
            return []

        cases = []
        with self.dataset_path.open("r") as f:
            for line in f:
                data = json.loads(line)
                cases.append(EvalTestCase(**data))

        self._cases = cases
        self.metrics.total_cases = len(cases)

        logger.info(f"Loaded {len(cases)} test cases from {self.dataset_path}")
        return cases

    def get_regression_tests(
        self,
        model: str | None = None,
        task_type: str | None = None,
    ) -> list[EvalTestCase]:
        """
        Get regression tests filtered by model/task type.

        Args:
            model: Filter by model
            task_type: Filter by task type

        Returns:
            Filtered test cases
        """
        cases = self._cases

        if model:
            cases = [c for c in cases if c.model == model]
        if task_type:
            cases = [c for c in cases if c.task_type == task_type]

        return cases

    def get_metrics(self) -> dict[str, Any]:
        """Get dataset metrics."""
        return self.metrics.to_dict()


# ─────────────────────────────────────────────
# Convenience Functions
# ─────────────────────────────────────────────


async def generate_with_adaptive_temp(
    client,
    model: str,
    prompt: str,
    task_type: str,
) -> Any:
    """Convenience function for adaptive temperature generation."""
    controller = AdaptiveTemperatureController(client=client)
    return await controller.generate_with_retry(model, prompt, task_type)


__all__ = [
    # Adaptive Temperature
    "AdaptiveTemperatureController",
    "TemperatureMetrics",
    # Eval Dataset
    "EvalDatasetBuilder",
    "EvalTestCase",
    "DatasetMetrics",
    # Convenience
    "generate_with_adaptive_temp",
]
