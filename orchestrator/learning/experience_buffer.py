"""
ExperienceBuffer — Cross-task learning and strategy adaptation
=================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 6 of the Agentic System Implementation Plan.
Records what worked and what didn't across tasks, and adapts
execution strategy based on experience.
"""

from __future__ import annotations
import os

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

    # Maximum entries to retain in the successes/failures audit lists.
    # model_scores and method_scores (dicts of lists) are bounded naturally
    # by the finite set of (model × task_type) and (method × task_type) keys.
    # The raw audit lists have no such natural bound — cap them explicitly.
    # Architecture spec (CLAUDE.md) documents 200; mindmap documents 50.
    # 200 chosen to retain enough recent history without unbounded growth.
    _MAX_AUDIT_SIZE: int = 200

    def record_success(self, task_type: str, method: str, model: str, score: float) -> None:
        """Record a successful execution."""
        pattern = self._hash_pattern(task_type, method, model)
        self.successes.append(SuccessPattern(
            pattern_hash=pattern, task_type=task_type,
            method=method, model=model, score=score,
        ))
        # BUG-008 FIX: cap the audit list to prevent unbounded memory/disk growth.
        # AgentMemory.record() applies the same pattern with [-50:].
        if len(self.successes) > self._MAX_AUDIT_SIZE:
            self.successes = self.successes[-self._MAX_AUDIT_SIZE:]
        self.model_scores[f"{model}:{task_type}"].append(score)
        self.method_scores[f"{method}:{task_type}"].append(score)

    def record_failure(self, task_type: str, method: str, model: str, score: float = 0.0) -> None:
        """Record a failed execution."""
        pattern = self._hash_pattern(task_type, method, model)
        self.failures.append(SuccessPattern(
            pattern_hash=pattern, task_type=task_type,
            method=method, model=model, score=score,
        ))
        # BUG-008 FIX: same cap as record_success.
        if len(self.failures) > self._MAX_AUDIT_SIZE:
            self.failures = self.failures[-self._MAX_AUDIT_SIZE:]

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


    def save(self, path: str | None = None) -> None:
        import json
        p = path or os.path.join(os.path.expanduser("~"), ".orchestrator", "experience.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        data = {
            "successes": [s.__dict__ for s in self.successes],
            "failures": [f.__dict__ for f in self.failures],
            "model_scores": dict(self.model_scores),
            "method_scores": dict(self.method_scores),
        }
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: str | None = None) -> "ExperienceBuffer":
        import json, os
        p = path or os.path.join(os.path.expanduser("~"), ".orchestrator", "experience.json")
        if not os.path.exists(p):
            return cls()
        try:
            with open(p, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            buf = cls()
            for s in data.get("successes", []):
                buf.successes.append(SuccessPattern(**s))
            for f_ in data.get("failures", []):
                buf.failures.append(SuccessPattern(**f_))
            from collections import defaultdict
            buf.model_scores = defaultdict(list, data.get("model_scores", {}))
            buf.method_scores = defaultdict(list, data.get("method_scores", {}))
            return buf
        except (json.JSONDecodeError, KeyError, TypeError):
            return cls()
