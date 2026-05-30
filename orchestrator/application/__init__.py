"""
Application Layer — Use Cases & Orchestration
==============================================
Contains the orchestration facade, task execution, and supporting services.
Depends on domain ports (Protocols) — never on infrastructure adapters directly.

Phase 4 canonical location for all application-layer services.
"""

# Core orchestration
from .critique_cycle import CritiqueCycle, CritiqueState
from .budget_enforcer import BudgetEnforcer
from .fallback_handler import FallbackHandler
from .task_executor import ExecutionContext, TaskExecutor
from .evaluator import EvaluatorService
from .executor import ExecutorMetrics, ExecutorResult, ExecutorService
from .decomposer import DecomposerMetrics, DecomposerResult, DecomposerService
from .observability import ModelSummary, ObservabilityService
from .dependency_resolver import DependencyResolver
from .context_compressor import ContextCompressor

__all__ = [
    # Engine core (now application-layer)
    "CritiqueCycle",
    "CritiqueState",
    "BudgetEnforcer",
    "FallbackHandler",
    "ExecutionContext",
    "TaskExecutor",
    # Services
    "EvaluatorService",
    "ExecutorMetrics",
    "ExecutorResult",
    "ExecutorService",
    "DecomposerMetrics",
    "DecomposerResult",
    "DecomposerService",
    "ModelSummary",
    "ObservabilityService",
    # Supporting
    "DependencyResolver",
    "ContextCompressor",
]
