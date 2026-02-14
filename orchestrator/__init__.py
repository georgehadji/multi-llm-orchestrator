"""
Multi-LLM Orchestrator
======================
Local multi-model orchestration for autonomous project completion.
Supports: OpenAI GPT, Google Gemini, Anthropic Claude, Kimi K2.5.

This system solves a joint optimization problem over cost, latency, compliance,
quality, and trust — at the level of semantic tasks, not just infrastructure
routing. It couples policy-as-code, constraint solving, and adaptive re-planning
in a single orchestration engine.

Basic usage (backwards-compatible):
    from orchestrator import Orchestrator, Budget

    budget = Budget(max_usd=8.0, max_time_seconds=5400)
    orch = Orchestrator(budget=budget)
    state = asyncio.run(orch.run_project(
        project_description="...",
        success_criteria="...",
    ))

Policy-driven usage (new):
    from orchestrator import Orchestrator, Budget, JobSpec, PolicySet, Policy

    spec = JobSpec(
        project_description="...",
        success_criteria="...",
        budget=Budget(max_usd=8.0),
        policy_set=PolicySet(global_policies=[
            Policy("gdpr", allow_training_on_output=False),
            Policy("eu_only", allowed_regions=["eu", "global"]),
        ]),
    )
    orch = Orchestrator()
    state = asyncio.run(orch.run_job(spec))
"""

from .models import (
    Budget, Model, Task, TaskResult, TaskType, TaskStatus,
    ProjectState, ProjectStatus, build_default_profiles,
)
from .engine import Orchestrator
from .cache import DiskCache
from .state import StateManager
from .validators import run_validators, VALIDATORS
from .policy import ModelProfile, Policy, PolicySet, JobSpec
from .policy_engine import PolicyEngine, PolicyViolationError
from .planner import ConstraintPlanner
from .telemetry import TelemetryCollector

__all__ = [
    # ── Existing public API (unchanged) ─────────────────────────────────────
    "Orchestrator", "Budget", "Model", "Task", "TaskResult",
    "TaskType", "TaskStatus", "ProjectState", "ProjectStatus",
    "DiskCache", "StateManager", "run_validators", "VALIDATORS",
    # ── New policy-driven API ────────────────────────────────────────────────
    "ModelProfile", "Policy", "PolicySet", "JobSpec",
    "PolicyEngine", "PolicyViolationError",
    "ConstraintPlanner", "TelemetryCollector",
    "build_default_profiles",
]
