"""
ConstitutionGate — Pre-Generation Constitutional Enforcement
=============================================================
Author: Orchestrator core

Pipeline stage that runs **before** ``GenerateStage`` and rejects tasks that
would violate declared project constraints (protected paths, forbidden imports).

Design
------
- Uses the real ``ProjectConstitution.check_task()`` aggregator.
- Aborts via existing ``ctx.should_abort`` / ``ctx.abort_reason`` — no new
  pipeline plumbing needed.
- ``required_validators`` and ``require_tests`` are **not** enforced here;
  they are appended to the task's ``hard_validators`` at container wiring time
  so the existing ``ValidateStage`` handles them (DRY — reuse validation path).
- Zero generation tokens are spent on aborted tasks.
"""

from __future__ import annotations

import logging

from ...domain.constitution import ProjectConstitution
from ..pipeline import PipelineContext, PipelineStage

logger = logging.getLogger(__name__)


class ConstitutionGate:
    """Phase -1 gate: reject a task BEFORE spending generation tokens.

    The stage is a no-op when constitution is empty (all default values).
    """

    # Pipeline stage ordering — lower values run first
    priority: int = -100

    @classmethod
    def build_kwargs(cls, **deps):
        return {}

    def __init__(self, constitution: ProjectConstitution | None = None) -> None:
        self._c = constitution or ProjectConstitution()

    def set_constitution(self, constitution: ProjectConstitution) -> None:
        """Swap the active constitution at runtime.

        Needed for the --from-speckit path: the Spec-Kit constitution is
        parsed per-run from an external artifact, not loaded once from
        .orchestrator/constitution.json at container-build time.
        """
        self._c = constitution

    # Explicitly satisfy PipelineStage protocol
    @property
    def __constitution_gate_marker(self) -> bool:
        return True

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Check task against constitution; abort if violations found."""
        # Skip if constitution is empty (all defaults = no restrictions)
        if self._is_empty():
            return ctx

        violations = self._c.check_task(ctx.task)
        if violations:
            ctx.should_abort = True
            ctx.abort_reason = "constitution: " + "; ".join(violations)
            logger.warning(
                "Constitution gate ABORT for task %s: %s",
                getattr(ctx.task, "id", "unknown"),
                ctx.abort_reason,
            )
        else:
            logger.debug(
                "Constitution gate PASS for task %s",
                getattr(ctx.task, "id", "unknown"),
            )

        return ctx

    def _is_empty(self) -> bool:
        """Check if constitution has any active restrictions."""
        c = self._c
        return not (
            c.protect_paths
            or c.forbidden_imports
            or c.require_tests
            or c.required_validators
            or c.max_file_size_bytes > 0
        )
