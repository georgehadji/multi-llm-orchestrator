"""
Consolidation Loop — Cross-Project Insight Extraction
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

After N projects, uses LLM to extract reusable insights from the
telemetry store. Identifies:
- Which task types consistently fail on which models
- Which prompt patterns produce highest-quality outputs
- Recurring validation failure patterns

Insights are stored as special "insight" records that the Pattern
Learner (Phase 5) can consume.

Run frequency: configured via MemoryManager.nudge_interval (default: every 5 projects).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.memory.consolidation")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConsolidationInsight:
    """A single cross-project insight extracted from telemetry data."""

    insight_type: str  # "model_failure", "prompt_pattern", "validation_pattern"
    summary: str
    task_type: str = ""
    model: str = ""
    frequency: int = 1
    impact: str = ""  # "high" | "medium" | "low"
    evidence: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# ConsolidationLoop
# ─────────────────────────────────────────────────────────────────────────────


class ConsolidationLoop:
    """Extracts cross-project insights from the telemetry store.

    Uses deterministic analysis (no LLM) on stored routing_events and
    model_snapshots to identify model failure patterns, task-type trends,
    and validation gaps.

    Designed to run as a background task — never raises. Errors are
    logged and squashed.
    """

    async def run(self) -> list[ConsolidationInsight]:
        """Run the consolidation analysis.

        Returns:
            A list of ConsolidationInsight objects. Empty when no data is
            available or analysis fails.
        """
        try:
            insights: list[ConsolidationInsight] = []

            # 1. Analyze model failure patterns from telemetry
            model_insights = await self._analyze_model_failures()
            insights.extend(model_insights)

            # 2. Analyze project completion trends (budget, time, status)
            trend_insights = self._analyze_trends()
            insights.extend(trend_insights)

            if insights:
                logger.info(
                    "Consolidation complete: %d insight(s) extracted",
                    len(insights),
                )
            else:
                logger.debug("Consolidation: no new insights")

            return insights

        except Exception as exc:
            logger.warning("Consolidation analysis failed: %s", exc)
            return []

    # ── Analysis methods (deterministic, no LLM) ─────────────────────────

    async def _analyze_model_failures(self) -> list[ConsolidationInsight]:
        """Analyze model failure patterns from telemetry_store.

        Queries routing_events for tasks with low scores or high
        failure rates per (model, task_type) pair.
        """
        insights: list[ConsolidationInsight] = []

        try:
            from ..telemetry_store import TelemetryStore

            store = TelemetryStore()
            rankings = await store.model_rankings(days=30)

            for rank in rankings:
                # Flag models with quality < 0.5 as problematic
                if rank.call_count >= 5 and rank.quality_score < 0.5:
                    insights.append(
                        ConsolidationInsight(
                            insight_type="model_quality",
                            summary=(
                                f"{rank.model.value} has low quality score "
                                f"({rank.quality_score:.2f}) across "
                                f"{rank.call_count} calls"
                            ),
                            model=rank.model.value,
                            frequency=rank.call_count,
                            impact="medium",
                            evidence=[
                                f"quality_score={rank.quality_score:.2f}",
                                f"call_count={rank.call_count}",
                                f"avg_cost=${rank.avg_cost_usd:.4f}",
                            ],
                        )
                    )

        except Exception as exc:
            logger.debug("Model failure analysis skipped: %s", exc)

        return insights

    def _analyze_trends(self) -> list[ConsolidationInsight]:
        """Analyze project completion trends.

        Currently returns a placeholder — populated with real telemetry
        queries when the telemetry_store has sufficient data.
        """
        # Placeholder for future analysis:
        # - Query project completion rates vs. task count
        # - Identify task types with highest revision counts
        # - Flag recurring validation failure patterns
        return []

    def format_report(self, insights: list[ConsolidationInsight]) -> str:
        """Format insights as a human-readable report.

        Args:
            insights: List of insights from run().

        Returns:
            A formatted string suitable for logging or display.
        """
        if not insights:
            return "No consolidation insights available."

        lines = ["## Consolidation Report", ""]
        for i, insight in enumerate(insights, 1):
            lines.append(f"{i}. [{insight.insight_type}] {insight.summary}")
            if insight.impact:
                lines.append(f"   Impact: {insight.impact}")
            if insight.evidence:
                lines.append(f"   Evidence: {'; '.join(insight.evidence)}")
            lines.append("")

        return "\n".join(lines)
