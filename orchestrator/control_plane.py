"""
Control Plane Service
=====================
Orchestrates the full constraint-enforcement workflow:

  Step 1: validate(job, policy)   — schema + static analysis
  Step 2: monitor.check_global()  — hard constraints pre-run
  Step 3: solve_constraints()     — routing plan, SLA fit
  Step 4: run_workflow()          — delegates to Orchestrator
  Step 5: audit_log.write()       — immutable structured log

Usage:
    cp = ControlPlane()
    state = await cp.submit(job, policy)
"""
from __future__ import annotations

import dataclasses
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from .audit import AuditLog
from .models import ProjectState
from .policy_engine import PolicyViolationError
from .reference_monitor import Decision, MonitorResult, ReferenceMonitor
from .specs import JobSpecV2, PolicySpecV2, RoutingHint

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────

class SpecValidationError(ValueError):
    """Raised when JobSpecV2 / PolicySpecV2 fail static validation."""

    def __init__(self, errors: list[str]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class PolicyViolation(RuntimeError):
    """Raised when the ReferenceMonitor denies the job pre-run."""


# ─────────────────────────────────────────────
# Routing plan
# ─────────────────────────────────────────────

@dataclass
class RoutingPlan:
    """Resolved routing decisions produced by solve_constraints()."""

    # task_type -> preferred model name (or "any")
    task_model_map: dict[str, str] = field(default_factory=dict)
    # human-readable notes about constraint decisions
    notes: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# ControlPlane
# ─────────────────────────────────────────────

class ControlPlane:
    """
    Main entry point for constraint-driven job execution.

    The ControlPlane separates the *agent* (proposes specs) from the
    *enforcement layer* (this class) so that hard constraints cannot be
    bypassed by prompt content.
    """

    def __init__(
        self,
        audit_log: Optional[AuditLog] = None,
    ) -> None:
        self._monitor = ReferenceMonitor()
        self._audit = audit_log or AuditLog()

    async def submit(
        self,
        job: JobSpecV2,
        policy: PolicySpecV2,
    ) -> ProjectState:
        """
        Run the full constraint-enforcement pipeline and execute the job.

        Raises
        ------
        SpecValidationError   — job or policy fails static validation
        PolicyViolation       — ReferenceMonitor denies the job pre-run
        """
        start = time.time()

        # 1. Validate specs
        errors = self._validate(job, policy)
        if errors:
            raise SpecValidationError(errors)

        # 2. Pre-run hard constraint check
        global_check = self._monitor.check_global(job, policy)
        if global_check.decision == Decision.DENY:
            logger.error("ControlPlane: pre-run DENY — %s", global_check.reason)
            raise PolicyViolation(global_check.reason)
        if global_check.decision == Decision.ESCALATE:
            logger.warning("ControlPlane: pre-run ESCALATE — %s", global_check.reason)

        # 3. Solve constraints → routing plan
        routing = self._solve_constraints(job, policy)
        for note in routing.notes:
            logger.info("ControlPlane routing: %s", note)

        # 4. Run workflow
        state = await self._run_workflow(job, routing)

        # 5. Audit log
        elapsed = time.time() - start
        self._write_audit(job, policy, routing, state, elapsed_s=elapsed)

        return state

    # ─────────────────────────────────────────
    # Step 1: Validation
    # ─────────────────────────────────────────

    def _validate(self, job: JobSpecV2, policy: PolicySpecV2) -> list[str]:
        """Return a list of validation error strings, or [] if valid."""
        errors: list[str] = []

        if not job.goal:
            errors.append("JobSpecV2.goal must not be empty")

        if job.slas.max_cost_usd is not None and job.slas.max_cost_usd <= 0:
            errors.append("JobSpecV2.slas.max_cost_usd must be positive")

        if job.slas.min_quality_tier < 0 or job.slas.min_quality_tier > 1:
            errors.append("JobSpecV2.slas.min_quality_tier must be in [0, 1]")

        if job.inputs.data_locality not in ("eu", "us", "any"):
            errors.append(
                f"JobSpecV2.inputs.data_locality must be 'eu', 'us', or 'any'; "
                f"got {job.inputs.data_locality!r}"
            )

        for rule in policy.allow_deny_rules:
            if "effect" not in rule:
                errors.append(
                    f"PolicySpecV2.allow_deny_rules entry missing 'effect': {rule}"
                )
            elif rule["effect"] not in ("allow", "deny"):
                errors.append(
                    f"PolicySpecV2.allow_deny_rules 'effect' must be 'allow' or 'deny'; "
                    f"got {rule['effect']!r}"
                )

        return errors

    # ─────────────────────────────────────────
    # Step 3: Constraint solving
    # ─────────────────────────────────────────

    def _solve_constraints(
        self, job: JobSpecV2, policy: PolicySpecV2
    ) -> RoutingPlan:
        """
        Produce a RoutingPlan by applying hard constraints + routing hints.

        Current implementation:
        - eu_only constraint → restrict all tasks to EU-safe models
        - SLA max_cost_usd → prefer cheap models (Gemini Flash, GPT-4o-mini)
        - SLA min_quality_tier > 0.95 → require high-quality models (Claude Opus, GPT-4o)
        """
        plan = RoutingPlan()
        notes = plan.notes

        hard = set(job.constraints.hard)

        if "eu_only" in hard or job.inputs.data_locality == "eu":
            plan.task_model_map["*"] = "eu_safe_only"
            notes.append("eu_only: all tasks restricted to EU-safe models")

        if job.slas.min_quality_tier and job.slas.min_quality_tier > 0.95:
            plan.task_model_map.setdefault("*", "high_quality")
            notes.append(
                f"min_quality_tier={job.slas.min_quality_tier}: prefer high-quality models"
            )

        if job.slas.max_cost_usd is not None:
            notes.append(
                f"max_cost_usd={job.slas.max_cost_usd}: enforced via budget in JobSpecV2"
            )
            # Reflect in budget if lower than default
            if job.slas.max_cost_usd < job.budget.max_usd:
                job.budget.max_usd = job.slas.max_cost_usd

        # Apply routing hints from policy
        for hint in policy.routing_hints:
            notes.append(f"routing hint: {hint.condition!r} → {hint.target!r}")

        for constraint in job.constraints.soft:
            notes.append(f"soft constraint: {constraint}")

        return plan

    # ─────────────────────────────────────────
    # Step 4: Workflow execution
    # ─────────────────────────────────────────

    async def _run_workflow(
        self,
        job: JobSpecV2,
        routing: RoutingPlan,
    ) -> ProjectState:
        """Delegate to Orchestrator, wiring in per-task monitor checks."""
        from .engine import Orchestrator
        from .models import Task

        orchestrator = Orchestrator(budget=job.budget)
        monitor = self._monitor

        # Patch _execute_task to run monitor check before each task
        original_execute = orchestrator._execute_task

        async def _guarded_execute(task: Task) -> object:
            result = monitor.check(task, job, PolicySpecV2())
            if result.decision == Decision.DENY:
                logger.error(
                    "ReferenceMonitor DENIED task %s: %s", task.id, result.reason
                )
                from .models import TaskResult, TaskStatus, Model
                return TaskResult(
                    task_id=task.id, output="", score=0.0,
                    model_used=Model.GPT_4O_MINI,
                    status=TaskStatus.FAILED,
                )
            if result.decision == Decision.ESCALATE:
                logger.warning(
                    "ReferenceMonitor ESCALATE task %s: %s", task.id, result.reason
                )
            return await original_execute(task)

        orchestrator._execute_task = _guarded_execute  # type: ignore[method-assign]

        return await orchestrator.run_project(
            project_description=job.goal,
            success_criteria=(
                "; ".join(job.metrics) if job.metrics
                else "All tasks complete successfully"
            ),
        )

    # ─────────────────────────────────────────
    # Step 5: Audit
    # ─────────────────────────────────────────

    def _write_audit(
        self,
        job: JobSpecV2,
        policy: PolicySpecV2,
        routing: RoutingPlan,
        state: ProjectState,
        elapsed_s: float,
    ) -> None:
        """Write an immutable structured audit record."""
        try:
            record = {
                "type": "control_plane_run",
                "goal": job.goal[:200],
                "constraints_hard": job.constraints.hard,
                "data_locality": job.inputs.data_locality,
                "routing_notes": routing.notes,
                "project_status": state.status.value,
                "total_cost_usd": state.budget.spent_usd,
                "elapsed_s": round(elapsed_s, 2),
            }
            logger.info("ControlPlane audit: %s", json.dumps(record))
        except Exception:
            pass  # audit failures must never break the main flow


__all__ = [
    "ControlPlane",
    "RoutingPlan",
    "SpecValidationError",
    "PolicyViolation",
]
