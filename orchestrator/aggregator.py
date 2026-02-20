"""
Cross-run profile aggregator.

Records (model, task_type, score, cost, latency) from completed runs
and computes aggregated statistics to guide future model routing.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
from .models import Model, TaskType


@dataclass
class RunRecord:
    project_id: str
    task_type: TaskType
    model: Model
    score: float
    cost_usd: float
    latency_ms: float


class ProfileAggregator:
    """
    In-memory cross-run aggregator.
    """

    def __init__(self) -> None:
        self._records: dict[tuple[Model, TaskType], list[RunRecord]] = defaultdict(list)

    def record(self, run: RunRecord) -> None:
        self._records[(run.model, run.task_type)].append(run)

    def stats_for(self, model: Model, task_type: TaskType) -> dict:
        records = self._records.get((model, task_type), [])
        if not records:
            return {"count": 0, "avg_score": 0.0, "avg_cost": 0.0, "avg_latency": 0.0}
        n = len(records)
        return {
            "count":       n,
            "avg_score":   sum(r.score      for r in records) / n,
            "avg_cost":    sum(r.cost_usd   for r in records) / n,
            "avg_latency": sum(r.latency_ms for r in records) / n,
        }

    def best_model(self, task_type: TaskType) -> Optional[Model]:
        """Model with the highest average score for this task type."""
        candidates = {model for (model, tt) in self._records if tt == task_type}
        if not candidates:
            return None
        return max(candidates, key=lambda m: self.stats_for(m, task_type)["avg_score"])

    def cost_efficiency_ranking(
        self, task_type: TaskType
    ) -> list[tuple[Model, float]]:
        """(model, efficiency) sorted by score/cost descending."""
        candidates = {model for (model, tt) in self._records if tt == task_type}
        results = []
        for model in candidates:
            s = self.stats_for(model, task_type)
            efficiency = s["avg_score"] / s["avg_cost"] if s["avg_cost"] > 0 else s["avg_score"]
            results.append((model, efficiency))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def summary_table(self) -> dict[TaskType, list[dict]]:
        """Dict of task_type -> list of model stats, sorted by avg_score desc."""
        by_type: dict[TaskType, list[dict]] = defaultdict(list)
        for (model, task_type), records in self._records.items():
            s = self.stats_for(model, task_type)
            by_type[task_type].append({"model": model, **s})
        for task_type in by_type:
            by_type[task_type].sort(key=lambda x: x["avg_score"], reverse=True)
        return dict(by_type)
