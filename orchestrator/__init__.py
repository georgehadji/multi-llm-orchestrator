"""
Multi-LLM Orchestrator
======================
Local multi-model orchestration for autonomous project completion.
Supports: OpenAI GPT, Google Gemini, Anthropic Claude, Kimi K2.5.

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
from .policy import (
    ModelProfile, Policy, PolicySet, JobSpec,
    EnforcementMode, RateLimit, PolicyHierarchy,
)
from .policy_engine import PolicyEngine, PolicyViolationError
from .planner import ConstraintPlanner
from .telemetry import TelemetryCollector
from .optimization import OptimizationBackend, GreedyBackend, WeightedSumBackend, ParetoBackend
from .audit import AuditLog, AuditRecord
# Improvement 3: event hooks + metrics exporters
from .hooks   import HookRegistry, EventType
from .metrics import MetricsExporter, ConsoleExporter, JSONExporter, PrometheusExporter
# Improvement 4: Policy DSL
from .policy_dsl import load_policy_file, load_policy_dict, PolicyAnalyzer, AnalysisReport
# Improvement 6: economic / cost layer
from .cost    import BudgetHierarchy, CostPredictor, CostForecaster, ForecastReport, RiskLevel
# Improvement 5: advanced agents
from .agents  import AgentPool, TaskChannel

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
    # Governance — Improvement 2
    "EnforcementMode", "RateLimit", "PolicyHierarchy",
    # Optimization backends — Improvement 1
    "OptimizationBackend", "GreedyBackend", "WeightedSumBackend", "ParetoBackend",
    # Audit log
    "AuditLog", "AuditRecord",
    # Telemetry hooks — Improvement 3
    "HookRegistry", "EventType",
    # Metrics exporters — Improvement 3
    "MetricsExporter", "ConsoleExporter", "JSONExporter", "PrometheusExporter",
    # Policy DSL — Improvement 4
    "load_policy_file", "load_policy_dict", "PolicyAnalyzer", "AnalysisReport",
    # Cost layer — Improvement 6
    "BudgetHierarchy", "CostPredictor", "CostForecaster", "ForecastReport", "RiskLevel",
    # Advanced agents — Improvement 5
    "AgentPool", "TaskChannel",
]
