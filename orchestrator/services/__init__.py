"""
Services Package — Backward-compatibility shim.
================================================
The canonical implementations now live in orchestrator/application/.
All imports are re-exported from there.
"""

from ..application.evaluator import EvaluatorService
from ..application.executor import ExecutorMetrics, ExecutorResult, ExecutorService
from ..application.decomposer import DecomposerMetrics, DecomposerResult, DecomposerService
from ..application.observability import ModelSummary, ObservabilityService

# Backward-compat alias: GeneratorService was renamed to DecomposerService
GeneratorService = DecomposerService
GeneratorResult = DecomposerResult
GeneratorMetrics = DecomposerMetrics

__all__ = [
    "EvaluatorService",
    "ExecutorMetrics",
    "ExecutorResult",
    "ExecutorService",
    "DecomposerMetrics",
    "DecomposerResult",
    "DecomposerService",
    "GeneratorService",  # backward-compat alias
    "GeneratorResult",   # backward-compat alias
    "GeneratorMetrics",  # backward-compat alias
    "ModelSummary",
    "ObservabilityService",
]
