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
from typing import TYPE_CHECKING, Callable

from ..domain.ports import LLMClient, TelemetryPort, TracingPort
from ..budget import Budget
from ..operations.feedback import CritiqueItem, CritiqueReport, CritiqueSeverity
from ..models import Model, Task, TaskType
from ..resilience import ResiliencePolicy as _ResiliencePolicy

if TYPE_CHECKING:
    from .verification_gate import VerificationGate

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

    _SYSTEM_PROMPT = (
        "Adversarial code reviewer. "
        "ASSUME this output is BROKEN until proven otherwise. "
        "Do NOT praise. Find what fails. "
        "Score 0.0 for clearly broken code; reserve 0.8+ only when everything works correctly."
    )

    def __init__(
        self,
        client: LLMClient,
        budget: Budget,
        get_models_fn: Callable[[TaskType], list[Model]],
        consistency_runs: int = 2,
        consistency_delta: float = 0.05,
        tracer: TracingPort | None = None,
        telemetry: TelemetryPort | None = None,
        verification_gate: "VerificationGate | None" = None,
    ) -> None:
        self._client = client
        self._budget = budget
        self._get_models = get_models_fn
        self._consistency_runs = consistency_runs
        self._consistency_delta = consistency_delta
        self._tracer = tracer
        self._telemetry = telemetry
        self._gate = verification_gate

    # -- Public interface ----------------------------------------------------

    async def evaluate(
        self, task: Task, output: str, policy: _ResiliencePolicy | None = None
    ) -> CritiqueReport:
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

    async def _evaluate_inner(
        self, task: Task, output: str, policy: _ResiliencePolicy | None = None
    ) -> CritiqueReport:
        # Deterministic gate runs first — hard veto before any LLM opinion.
        gate_result = None
        if self._gate is not None:
            gate_result = await self._gate.run(output)
            if not gate_result.passed:
                from .verification_gate import VerificationGate

                logger.warning(
                    "  %s: VerificationGate FAILED — deterministic floor applied "
                    "(score=%.2f). Failures: %s",
                    task.id,
                    VerificationGate.FAIL_SCORE_FLOOR,
                    gate_result.reasons,
                )
                return CritiqueReport(
                    task_id=task.id,
                    score=VerificationGate.FAIL_SCORE_FLOOR,
                    passed_validators=False,
                    deterministic={
                        "passed": False,
                        "checks": gate_result.checks,
                        "reasons": gate_result.reasons,
                        "artifact_hash": gate_result.artifact_hash,
                        "failure_summary": gate_result.failure_summary,
                        "status_summary": gate_result.status_summary,
                    },
                )

        eval_models = self._get_models(TaskType.EVALUATE)
        if not eval_models:
            logger.debug("  %s: no eval models available, returning 0.5", task.id)
            return CritiqueReport(task_id=task.id, score=0.5, passed_validators=True)

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
                logger.debug(
                    "  %s: eval run %d/%d starting…", task.id, run + 1, self._consistency_runs
                )
                response = await self._client.call(  # type: ignore[no-untyped-call]
                    eval_model,
                    eval_prompt,
                    system=self._SYSTEM_PROMPT,
                    max_tokens=300,
                    temperature=0.1,
                    timeout=60,
                    policy=policy,
                )
                parsed = self.parse_score(response.text)
                logger.debug(
                    "  %s: eval run %d/%d complete, score=%.3f",
                    task.id,
                    run + 1,
                    self._consistency_runs,
                    parsed,
                )
                await self._budget.charge(response.cost_usd, "evaluation")
                total_cost += response.cost_usd
                scores.append(parsed)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(
                    "Evaluation run %d/%d failed: %s", run + 1, self._consistency_runs, exc
                )
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
        last_response = response if "response" in dir() else None
        if scores and last_response:
            try:
                json_data = json.loads(last_response.text)
                for issue in json_data.get("issues", []):
                    try:
                        severity = CritiqueSeverity(issue.get("severity", "minor"))
                    except ValueError:
                        severity = CritiqueSeverity.MINOR
                    items.append(
                        CritiqueItem(
                            severity=severity,
                            category=issue.get("category", "correctness"),
                            description=issue.get("description", ""),
                            location=issue.get("location"),
                            suggestion=issue.get("suggestion"),
                        )
                    )
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        # Build deterministic data for the report
        deterministic_data = None
        if gate_result is not None:
            deterministic_data = {
                "passed": gate_result.passed,
                "checks": gate_result.checks,
                "reasons": gate_result.reasons,
                "artifact_hash": gate_result.artifact_hash,
                "failure_summary": gate_result.failure_summary,
                "status_summary": gate_result.status_summary,
            }

        return CritiqueReport(
            task_id=task.id,
            score=final_score,
            items=items,
            passed_validators=True,
            model_used=eval_model.value if eval_model else None,
            tokens_used=(
                getattr(last_response, "input_tokens", 0)
                + getattr(last_response, "output_tokens", 0)
                if last_response
                else 0
            ),
            deterministic=deterministic_data,
        )

    # -- Helpers -------------------------------------------------------------

    def _aggregate(self, scores: list[float], task_id: str) -> float:
        """Apply self-consistency aggregation across N scoring runs.

        - 0 runs → 0.5 safe default.
        - 1 run  → that run.
        - 2 runs → mean, unless Δ > threshold (high disagreement) → lower score.
        - 3+ runs → median (robust to a single outlier run); still logs when the
          max-min spread exceeds the threshold.

        BUGFIX: the previous implementation handled only len == 2 and returned
        ``scores[0]`` for any other length, silently discarding runs 2..N when
        ``consistency_runs`` was configured above 2.
        """
        if not scores:
            return 0.5
        if len(scores) == 1:
            return scores[0]

        spread = max(scores) - min(scores)
        if len(scores) == 2:
            if spread > self._consistency_delta:
                logger.warning(
                    "Evaluation inconsistency for %s: %.3f vs %.3f (delta=%.3f > %.2f). "
                    "Using lower score.",
                    task_id,
                    scores[0],
                    scores[1],
                    spread,
                    self._consistency_delta,
                )
                return min(scores)
            return sum(scores) / len(scores)

        # 3+ runs: median is robust to one bad run; warn on high spread.
        ordered = sorted(scores)
        mid = len(ordered) // 2
        median = ordered[mid] if len(ordered) % 2 == 1 else (ordered[mid - 1] + ordered[mid]) / 2
        if spread > self._consistency_delta:
            logger.warning(
                "Evaluation inconsistency for %s across %d runs (spread=%.3f > %.2f). "
                "Using median=%.3f.",
                task_id,
                len(scores),
                spread,
                self._consistency_delta,
                median,
            )
        return median

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

        # Strip reasoning-model thinking so numbers inside the chain-of-thought
        # are never mistaken for the score (closed blocks, then truncated tail).
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL | re.IGNORECASE).strip()

        logger.debug("parse_score: input length=%d", len(text))

        # -- Try 1: JSON / json5 ----------------------------------------------
        try:
            if text.startswith("```"):
                text = re.sub(r"^```\w*\n?", "", text)
                text = re.sub(r"\n?```$", "", text)
                text = text.strip()

            try:
                import json5

                data = json5.loads(text)
            except (ImportError, Exception):
                data = json.loads(text)

            if isinstance(data, dict):
                score = float(data.get("score", data.get("Score", 0.5)))
                score = max(0.0, min(1.0, score))
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
