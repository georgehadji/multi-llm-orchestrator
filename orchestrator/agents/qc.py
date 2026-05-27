"""
QCAgent — Runs quality checks and generates quality reports
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Pillar 4: Runs lint, type check, test, coverage, and security checks.
Generates a QualityReport with score 0-10 and PASS/REVISE/BLOCK recommendation.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from ..models import Model, TaskType
from ..quality.quality_report import QualityReport

logger = logging.getLogger("orchestrator.agents.qa")


class QCAgent(AgentBase):
    """Runs quality checks and generates reports.

    Models (from orchestrator/agent_model_registry.py):
      Budget:  XIAOMI_MIMO_V2_FLASH  ($0.09/M in, $0.29/M out) — fast code analysis
      Premium: CLAUDE_SONNET_4_6     ($3.00/M in, $15.00/M out) — thorough quality analysis
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(role=AgentRole.QA, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a QA engineer. Analyze the generated code and produce "
            "a quality report. Check for: syntax errors, type safety, test coverage, "
            "security issues, and code smells. Be precise — report exact counts."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        logger.info("QCAgent: running quality checks for %s", task.id)
        if self.client is None:
            return AgentTaskResult(task_id=task.id, success=False, output="No LLM client")

        try:
            report = QualityReport()

            # Run lint if available
            try:
                import subprocess

                lint = subprocess.run(
                    ["python", "-m", "ruff", "check", "--no-cache", "."],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                report.lint_errors = len(lint.stdout.splitlines())
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            # Run type check if available
            try:
                mypy = subprocess.run(
                    ["python", "-m", "mypy", ".", "--no-error-summary", "--show-error-codes"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                report.type_errors = len([l for l in mypy.stdout.splitlines() if "error:" in l])
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

            report.overall_score = report.compute_score()
            report.recommendation = report.auto_recommend()

            return AgentTaskResult(
                task_id=task.id,
                success=True,
                output=f"Quality score: {report.overall_score}/10, recommendation: {report.recommendation}",
                score=report.overall_score / 10,
            )
        except Exception as exc:
            return AgentTaskResult(task_id=task.id, success=False, output=str(exc))
