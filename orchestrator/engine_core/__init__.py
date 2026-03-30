"""
Engine Core Package
===================
Author: Georgios-Chrysovalantis Chatzivantsidis

Decomposed engine modules following Single Responsibility Principle.

Modules:
- core: Main orchestration control loop (facade)
- task_executor: Task execution with validation
- critique_cycle: Generate→critique→revise pipeline
- fallback_handler: Circuit breaker & model health
- budget_enforcer: Budget monitoring & enforcement
- dependency_resolver: DAG resolution & topological sort
"""

from .budget_enforcer import BudgetEnforcer
from .core import OrchestratorCore
from .critique_cycle import CritiqueCycle, CritiqueState
from .dependency_resolver import DependencyResolver
from .fallback_handler import FallbackHandler
from .task_executor import TaskExecutor

__all__ = [
    'OrchestratorCore',
    'TaskExecutor',
    'CritiqueCycle',
    'CritiqueState',
    'FallbackHandler',
    'BudgetEnforcer',
    'DependencyResolver',
]
