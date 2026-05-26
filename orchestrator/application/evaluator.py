"""
EvaluatorService — LLM-based task output scoring.
==================================================
Fully extracted from engine.py._evaluate() and engine.py._parse_score().

Part of Application Layer (Phase 4) — Canonical location.

Responsibilities:
  - Run 2-pass self-consistency evaluation (delta <= 0.05 guard)
  - Normalise raw LLM score text -> float in [0.0, 1.0]
  - Charge evaluation cost to budget

Dependencies injected at construction; no reference back to Orchestrator.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Callable

from ..api_clients import UnifiedClient
from ..budget import Budget
from ..feedback import CritiqueItem, CritiqueReport, CritiqueSeverity
from ..models import Model, Task, TaskType
from ..resilience import ResiliencePolicy as _ResiliencePolicy
from ..telemetry import TelemetryCollector
from ..tracing import Tracer

logger = logging.getLogger("orchestrator.services.evaluator")


class EvaluatorService:
    """
    Stateless LLM evaluator.

    Args:
        client:         Unified LLM client (already circuit-broken in api_clients).
        budget:         Per-run budget; evaluation calls are charged here.
        get_models_fn:  Callable (TaskType) -> list[Model] — returns ordered
                        model list for a given task type. Provided by engine's
                        _get_available_models.
        consistency_runs:   Number of independent scoring runs (default 2).
        consistency_delta:  Max allowed delta between runs before using the lower
                            score (default 0.05).
    """

    def __init__(
        self,
        client: UnifiedClient,
        budget: Budget,
        get_models_fn: Callable[[TaskType], list[Model]],
        consistency_runs: int = 2,
        consistency_delta: float = 0.05,
        tracer: Tracer | None = None,
        telemetry: TelemetryCollector | None = None,
    ) -> None:
        self._client = client
        self._budget = budget
        self._get_models = get_models_fn
        self._consistency_runs = consistency_runs
        self._consistency_delta = consistency_delta
        self._tracer = tracer
        self._telemetry = telemetry

    # -- Public interface ----------------------------------------------------

    async def evaluate(self, task: Task, output: str, policy: _ResiliencePolicy | None = None) -> CritiqueReport:
        """
        Score output against task using self-consistency evaluation.

        Returns a CritiqueReport with score (0.0-1.0) and structured critique items.
        Falls back to score 0.5 if no evaluation models are available or all runs fail.
        """
        if self._tracer is not None:
            with self._tracer.trace(
                "evaluator.evaluate",
                {"task_id": task.id, "task_type": task.type.value},
            ) as span:
                report = await self._evaluate_inner(task, output, policy)
                span.set_attribute("eval.score", report.score)
                return report
        return await self._evaluate_inner(task, output, policy)

    async def _evaluate_inner(self, task: Task, output: str, policy: _ResiliencePolicy | None = None) -> CritiqueReport:
        eval_models = self._get_models(TaskType.EVALUATE)
        if not eval_models:
            logger.debug("  %s: no eval models available, returning 0.5", task.id)
            return CritiqueReport(task_id=task.id, score=0.5)

        eval_model = eval_models[0]
        logger.debug("  %s: evaluating with %s", task.id, eval_model.value)

        eval_prompt = (
            f"Score this output on a scale of 0.0 to 1.0.\n"
            f"Evaluate: correctness, completeness, quality, adherence to task.\n"
            f"Identify specific issues as BLOCKER (must fix), MAJOR (should fix), "
            f"MINOR (nice to fix), or SUGGESTION (optional).\n\n"
            f"TASK: {task.prompt}\n"
            f"ACCEPTANCE THRESHOLD: {task.acceptance_threshold}\n\n"
            f"OUTPUT:\n{output}\n\n"
            f'Return ONLY JSON: {{"score": <float>, '
            f'"issues": [{{"severity": "blocker|major|minor|suggestion", '
            f'"category": "security|architecture|style|correctness|performance|completeness", '
            f'"description": "...", '
            f'"location": "<optional>", '
            f'"suggestion": "<optional>"}}]}}'
        )

        scores: list[float] = []
        total_cost = 0.0
        eval_start = time.monotonic()  # BUG-003 FIX: track wall time
        for run in range(self._consistency_runs):
            try:
                logger.debug("  %s: eval run %d/%d starting…", task.id, run + 1, self._consistency_runs)
                response = await self._client.call(
                    eval_model,
                    eval_prompt,
                    system="You are a precise evaluator. Score exactly, return only JSON.",
                    max_tokens=300,
                    temperature=0.1,
                    timeout=60,
                    policy=policy,
                )
                parsed = self.parse_score(response.text)
                logger.debug(
                    "  %s: eval run %d/%d complete, score=%.3f",
                    task.id, run + 1, self._consistency_runs, parsed,
                )
                await self._budget.charge(response.cost_usd, "evaluation")
                total_cost += response.cost_usd
                scores.append(parsed)
            except Exception as exc:
                logger.warning("Evaluation run %d/%d failed: %s", run + 1, self._consistency_runs, exc)
                scores.append(0.5)

        final_score = self._aggregate(scores, task.id)
        eval_latency_ms = (time.monotonic() - eval_start) * 1000  # BUG-003 FIX

        if self._telemetry is not None:
            try:
                self._telemetry.record_call(
                    model=eval_model,
                    latency_ms=eval_latency_ms,
                    cost_usd=total_cost,
                    success=True,
                    quality_score=final_score,
                )
            except Exception:
                logger.debug("Telemetry recording failed for eval %s", task.id, exc_info=True)

        # Build CritiqueReport with parsed items from the first run
        items: list[CritiqueItem] = []
        last_response = response if 'response' in dir() else None
        if scores and last_response:
            try:
                json_data = json.loads(last_response.text)
                for issue in json_data.get("issues", []):
                    try:
                        severity = CritiqueSeverity(issue.get("severity", "minor"))
                    except ValueError:
                        severity = CritiqueSeverity.MINOR
                    items.append(CritiqueItem(
                        severity=severity,
                        category=issue.get("category", "correctness"),
                        description=issue.get("description", ""),
                        location=issue.get("location"),
                        suggestion=issue.get("suggestion"),
                    ))
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        return CritiqueReport(
            task_id=task.id,
            score=final_score,
            items=items,
            model_used=eval_model.value if eval_model else None,
            tokens_used=last_response.input_tokens + last_response.output_tokens if last_response else 0,
        )

    # -- Helpers -------------------------------------------------------------

    def _aggregate(self, scores: list[float], task_id: str) -> float:
        """Apply self-consistency: if delta > threshold, take the lower score."""
        if len(scores) == 2:
            delta = abs(scores[0] - scores[1])
            if delta > self._consistency_delta:
                logger.warning(
                    "Evaluation inconsistency for %s: %.3f vs %.3f (delta=%.3f > %.2f). "
                    "Using lower score.",
                    task_id, scores[0], scores[1], delta, self._consistency_delta,
                )
                return min(scores)
            return sum(scores) / len(scores)
        return scores[0] if scores else 0.5

    @staticmethod
    def parse_score(text: str) -> float:
        """
        Normalise raw LLM evaluation text to a float in [0.0, 1.0].

        Tries (in order):
          1. Direct JSON / json5 parse (handles markdown fences)
          2. Regex patterns for common human-readable formats
          3. Any bare float between 0 and 1 in the text
          4. Returns 0.5 as a safe default fallback
        """
        text = text.strip()
        logger.debug("parse_score: input length=%d", len(text))

        # -- Try 1: JSON / json5 ----------------------------------------------
        try:
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
                text = text.strip()

            try:
                import json5  # type: ignore[import]
                data = json5.loads(text)
            except (ImportError, Exception):
                data = json.loads(text)

            if isinstance(data, dict):
                score = float(data.get("score", data.get("Score", 0.5)))
            elif isinstance(data, (int, float)):
                score = float(data)
            else:
                match = re.search(r"([0-9]*\.?[0-9]+)", str(data))
                score = float(match.group(1)) if match else 0.5

            logger.debug("parse_score: JSON score=%.3f", score)
            return max(0.0, min(1.0, score))

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.debug("parse_score: JSON parse failed: %s", exc)

        # -- Try 2: common human-readable patterns ----------------------------
        _PATTERNS = [
            r'"?score"?\s*[:=]\s*([0-9]*\.?[0-9]+)',
            r"\u8bc4\u5206\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"\u5f97\u5206\s*[:=]\s*([0-9]*\.?[0-9]+)",
            r"([0-9]\.[0-9]{1,2})\s*/\s*1",
            r"([0-9]{1,2})\s*%",
            r"([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b",
            r"([0-9]+(?:\.[0-9]+)?)\s*/\s*100\b",
            r"\bout\s+of\s+10[,.\s:]*([0-9]+(?:\.[0-9]+)?)",
            r"([0-9]+(?:\.[0-9]+)?)\s+out\s+of\s+10",
            r"rating\s*[:=]\s*([0-9]*\.?[0-9]+)",
        ]
        for pattern in _PATTERNS:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                score = float(m.group(1))
                if re.search(r"/\s*100\b", pattern) or score > 10:
                    score = score / 100.0
                elif re.search(r"/\s*10\b|out\s+of\s+10", pattern) or 1 < score <= 10:
                    score = score / 10.0
                if "%" in pattern and score > 1.0:
                    score = score / 100.0
                logger.debug("parse_score: regex pattern '%s' matched %.3f", pattern, score)
                return max(0.0, min(1.0, score))

        # -- Try 3: bare float ------------------------------------------------
        m = re.search(r"\b(0\.[0-9]+|1\.0+)\b", text)
        if m:
            score = float(m.group(1))
            logger.debug("parse_score: bare float=%.3f", score)
            return max(0.0, min(1.0, score))

        logger.warning("parse_score: could not parse from: %s\u2026", text[:150])
        return 0.5
