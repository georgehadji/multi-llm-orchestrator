"""
ExperienceBuffer — Cross-task learning and strategy adaptation
=================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 6 of the Agentic System Implementation Plan.
Records what worked and what didn't across tasks, and adapts
execution strategy based on experience.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from ..models import Model

logger = logging.getLogger("orchestrator.learning.experience_buffer")


@dataclass
class SuccessPattern:
    """Record of a successful execution pattern."""
    pattern_hash: str
    task_type: str
    method: str
    model: str
    score: float
    count: int = 1


class ExperienceBuffer:
    """Remembers what worked and what didn't across tasks."""

    def __init__(self) -> None:
        self.successes: list[SuccessPattern] = []
        self.failures: list[SuccessPattern] = []
        self.model_scores: dict[str, list[float]] = defaultdict(list)
        self.method_scores: dict[str, list[float]] = defaultdict(list)

    def record_success(self, task_type: str, method: str, model: str, score: float) -> None:
        """Record a successful execution."""
        pattern = self._hash_pattern(task_type, method, model)
        self.successes.append(SuccessPattern(
            pattern_hash=pattern, task_type=task_type,
            method=method, model=model, score=score,
        ))
        self.model_scores[f"{model}:{task_type}"].append(score)
        self.method_scores[f"{method}:{task_type}"].append(score)

    def record_failure(self, task_type: str, method: str, model: str, score: float = 0.0) -> None:
        """Record a failed execution."""
        pattern = self._hash_pattern(task_type, method, model)
        self.failures.append(SuccessPattern(
            pattern_hash=pattern, task_type=task_type,
            method=method, model=model, score=score,
        ))

    def best_method_for(self, task_type: str) -> str | None:
        """Get the best-performing method for a task type."""
        scores = self.method_scores
        best_method = None
        best_score = 0.0
        for key, vals in scores.items():
            method_name, ttype = key.split(":", 1)
            if ttype == task_type and vals:
                avg = sum(vals) / len(vals)
                if avg > best_score:
                    best_score = avg
                    best_method = method_name
        return best_method

    def best_model_for(self, task_type: str) -> str | None:
        """Get the best-performing model for a task type."""
        best_model = None
        best_score = 0.0
        for key, vals in self.model_scores.items():
            model_name, ttype = key.split(":", 1)
            if ttype == task_type and vals:
                avg = sum(vals) / len(vals)
                if avg > best_score:
                    best_score = avg
                    best_model = model_name
        return best_model

    def _hash_pattern(self, task_type: str, method: str, model: str) -> str:
        return hashlib.md5(f"{task_type}:{method}:{model}".encode()).hexdigest()[:12]
