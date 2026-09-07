"""
CodebaseInvestigatorAgent — wraps CodebaseAnalyzer behind the AgentBase interface.
===================================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Thin agent wrapper around the production-grade CodebaseAnalyzer CLI tool.
Enables other agents and the coordinator to request codebase understanding
dynamically during a workflow run, rather than relying on upfront context injection.

Typical dispatch triggers (handled by AgentOrchestrator._decompose_goal):
    "understand X", "trace X", "explore X", "investigate X",
    "how does X work", "map dependencies of X"

Architecture:
    Sits entirely in the application layer.
    Imports only from application-layer modules (agents/, analyzer.py).
    Satisfies the AgentBase interface — no infrastructure changes required.
"""

from __future__ import annotations

import logging
from pathlib import Path

from typing import Any

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult

logger = logging.getLogger("orchestrator.agents.investigator")

# Focus areas the investigator runs by default.
# Caller can override via task.context["focus"].
_DEFAULT_FOCUS = ["architecture", "quality"]

# Budget ceiling for a single investigation call.
_DEFAULT_BUDGET_USD = 1.0

# Module-level import so that tests can patch
# "orchestrator.agents.investigator.CodebaseAnalyzer" reliably.
# Graceful degradation: if analyzer.py is unavailable (optional dep),
# handle_task() returns a failure result rather than crashing at import time.
try:
    from ..analyzer import CodebaseAnalyzer
except Exception:  # ImportError or any transitive failure
    CodebaseAnalyzer = None  # type: ignore[assignment,misc]


class CodebaseInvestigatorAgent(AgentBase):
    """Investigates an existing codebase on demand.

    Wraps the production-grade CodebaseAnalyzer behind the AgentBase interface
    so that the coordinator and other agents can dispatch investigation requests
    during a workflow run.

    Accepted context keys in task.context (all optional):
        codebase_path (str | Path): Root directory to analyze. Defaults to ".".
        focus (list[str]):          Focus areas — subset of
                                    ["architecture", "quality", "security",
                                     "performance", "improvements"].
                                    Defaults to ["architecture", "quality"].
        budget_usd (float):         Max LLM spend. Defaults to 1.0.
    """

    def __init__(self, **kwargs: Any) -> None:
        # Every sibling agent binds its own role here; this class did not, so it
        # alone required the caller to pass role=AgentRole.INVESTIGATOR and could
        # silently be constructed under the wrong role.
        super().__init__(role=AgentRole.INVESTIGATOR, **kwargs)

    @property
    def system_prompt(self) -> str:
        return (
            "You are a code archaeologist. Given a codebase path and an objective, "
            "produce clear, structured findings: execution paths, module responsibilities, "
            "dependency relationships, and relevant patterns. "
            "Be precise — cite file paths and function names. "
            "Prioritize actionable observations over exhaustive listings."
        )

    async def handle_task(self, task: AgentTask) -> AgentTaskResult:
        """Run CodebaseAnalyzer on the target codebase with the given objective.

        Returns an AgentTaskResult whose ``output`` field is the full Markdown
        report produced by CodebaseAnalyzer.
        """
        if CodebaseAnalyzer is None:
            msg = "CodebaseAnalyzer not available (orchestrator.analyzer failed to import)."
            logger.error(msg)
            return AgentTaskResult(task_id=task.id, success=False, output=msg)

        ctx = task.context if isinstance(task.context, dict) else {}
        codebase_path = Path(ctx.get("codebase_path", ".")).resolve()
        focus: list[str] | None = ctx.get("focus", _DEFAULT_FOCUS)
        budget_usd: float = float(ctx.get("budget_usd", _DEFAULT_BUDGET_USD))

        logger.info(
            "CodebaseInvestigatorAgent: investigating '%s' | goal='%s' | focus=%s",
            codebase_path,
            task.goal[:80],
            focus,
        )

        try:
            analyzer = CodebaseAnalyzer()
            report = await analyzer.analyze(
                path=codebase_path,
                focus=focus,
                budget_usd=budget_usd,
            )
        except Exception as exc:
            logger.exception("Investigation failed for path=%s", codebase_path)
            return AgentTaskResult(task_id=task.id, success=False, output=str(exc))

        summary = (
            f"Investigation complete for: {task.goal[:60]} | "
            f"{report.files_analyzed} files analyzed | "
            f"${report.total_cost:.4f} spent"
        )
        logger.info(summary)

        return AgentTaskResult(
            task_id=task.id,
            success=True,
            output=report.markdown,
            score=1.0,
            messages=[summary],
        )
