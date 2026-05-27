"""
CIPipeline — Continuous integration for generated code
=========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Capability 9 of the Agentic System Implementation Plan.
Chains lint → type-check → test → build checks after code generation.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.ci.pipeline")


@dataclass
class CIStepResult:
    """Result of a single CI step."""

    name: str
    passed: bool
    output: str = ""
    is_critical: bool = False


class CIStep(ABC):
    """A single CI step (lint, test, build, etc.)."""

    name: str = ""

    @abstractmethod
    async def execute(self, workspace: Any = None) -> CIStepResult: ...


class LintStep(CIStep):
    name = "lint"

    async def execute(self, workspace=None) -> CIStepResult:
        try:
            import subprocess

            result = subprocess.run(
                ["python", "-m", "ruff", "check", ".", "--no-cache"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return CIStepResult(name="lint", passed=result.returncode == 0, output=result.stdout)
        except Exception as exc:
            return CIStepResult(name="lint", passed=False, output=str(exc))


@dataclass
class CIReport:
    """Report from running the CI pipeline."""

    passed: bool = False
    steps: list[CIStepResult] = field(default_factory=list)
    all_pass: bool = False


class CIPipeline:
    """Chain of quality checks that runs after code generation."""

    def __init__(self, steps: list[CIStep] | None = None) -> None:
        self.steps = steps or []

    async def run(self, workspace: Any = None) -> CIReport:
        """Run all CI steps in order."""
        results: list[CIStepResult] = []
        for step in self.steps:
            result = await step.execute(workspace)
            results.append(result)
            if not result.passed and result.is_critical:
                break
        all_pass = all(r.passed for r in results)
        return CIReport(passed=all_pass, steps=results, all_pass=all_pass)
