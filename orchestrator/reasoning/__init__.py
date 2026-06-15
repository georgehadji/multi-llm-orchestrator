"""Reasoning package.

Imports are lazy to avoid circular dependencies between
method_selector, ara_integration, and ara_pipelines.
"""


def __getattr__(name):
    """Lazy import symbols from submodules."""
    if name in {"ARAExecutionStrategy", "ARAStrategyConfig"}:
        from .ara_execution_strategy import ARAExecutionStrategy, ARAStrategyConfig

        return locals()[name]
    if name in {"ARAPipelineIntegration", "create_ara_integration"}:
        from .ara_integration import ARAPipelineIntegration, create_ara_integration

        return locals()[name]
    if name in {"ReasoningMethod", "BasePipeline", "PipelineState"}:
        from .ara_pipelines import BasePipeline, PipelineState, ReasoningMethod

        return locals()[name]
    raise AttributeError(f"module 'orchestrator.reasoning' has no attribute '{name}'")
