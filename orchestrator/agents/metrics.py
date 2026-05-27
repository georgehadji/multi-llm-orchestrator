"""
AgentMetrics — Per-agent observability tracking
==================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Optimization E-12: Tracks per-agent counters for tasks completed,
avg score, avg latency, cost, and error rate.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.agents.metrics")


@dataclass
class AgentMetricsSnapshot:
    """Snapshot of agent metrics at a point in time."""

    agent_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    avg_score: float = 0.0
    avg_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    error_rate: float = 0.0


class AgentMetrics:
    """Track per-agent performance metrics."""

    def __init__(self) -> None:
        self._start_time = time.time()
        self.tasks_completed: int = 0
        self.tasks_failed: int = 0
        self.scores: list[float] = []
        self.latencies: list[float] = []
        self.total_cost: float = 0.0

    def record_task(
        self, success: bool, score: float, latency_ms: float, cost_usd: float = 0.0
    ) -> None:
        """Record a task execution."""
        if success:
            self.tasks_completed += 1
        else:
            self.tasks_failed += 1
        self.scores.append(score)
        self.latencies.append(latency_ms)
        self.total_cost += cost_usd

    @property
    def avg_score(self) -> float:
        if not self.scores:
            return 0.0
        return sum(self.scores) / len(self.scores)

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 0.0
        return sum(self.latencies) / len(self.latencies)

    @property
    def error_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        return self.tasks_failed / total

    def snapshot(self, agent_id: str) -> AgentMetricsSnapshot:
        """Get a metrics snapshot."""
        return AgentMetricsSnapshot(
            agent_id=agent_id,
            tasks_completed=self.tasks_completed,
            tasks_failed=self.tasks_failed,
            avg_score=round(self.avg_score, 4),
            avg_latency_ms=round(self.avg_latency_ms, 2),
            total_cost_usd=round(self.total_cost, 6),
            error_rate=round(self.error_rate, 4),
        )


class MetricsRegistry:
    """Global registry of per-agent metrics."""

    _instance: "MetricsRegistry | None" = None

    def __new__(cls) -> "MetricsRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._metrics: dict[str, AgentMetrics] = {}
        return cls._instance

    def get(self, agent_id: str) -> AgentMetrics:
        if agent_id not in self._metrics:
            self._metrics[agent_id] = AgentMetrics()
        return self._metrics[agent_id]

    def all_snapshots(self) -> list[AgentMetricsSnapshot]:
        return [m.snapshot(aid) for aid, m in self._metrics.items()]
