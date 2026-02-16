"""
Multi-LLM Orchestrator
======================
Local multi-model orchestration for autonomous project completion.
Supports: OpenAI GPT, Google Gemini, Anthropic Claude.

Usage:
    from orchestrator import Orchestrator, Budget

    budget = Budget(max_usd=8.0, max_time_seconds=5400)
    orch = Orchestrator(budget=budget)
    state = asyncio.run(orch.run_project(
        project_description="...",
        success_criteria="...",
    ))
"""

from .models import (
    Budget, Model, Task, TaskResult, TaskType, TaskStatus,
    ProjectState, ProjectStatus, build_default_profiles,
)
from .engine import Orchestrator
from .cache import DiskCache
from .state import StateManager
from .validators import run_validators, async_run_validators, VALIDATORS
from .policy import ModelProfile, Policy, PolicySet, JobSpec
from .policy_engine import PolicyEngine, PolicyViolationError
from .planner import ConstraintPlanner
from .telemetry import TelemetryCollector

__all__ = [
    # Core
    "Orchestrator", "Budget", "Model", "Task", "TaskResult",
    "TaskType", "TaskStatus", "ProjectState", "ProjectStatus",
    "DiskCache", "StateManager",
    # Validators
    "run_validators", "async_run_validators", "VALIDATORS",
    # Policy / planner / telemetry
    "ModelProfile", "Policy", "PolicySet", "JobSpec",
    "PolicyEngine", "PolicyViolationError",
    "ConstraintPlanner", "TelemetryCollector",
    "build_default_profiles",
]
