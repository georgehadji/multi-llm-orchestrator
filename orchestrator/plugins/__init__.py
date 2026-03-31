"""
Plugin System Package
=====================
Author: Georgios-Chrysovalantis Chatzivantsidis

Optional plugin architecture for advanced features.

Usage:
    from orchestrator.plugins import get_plugin_registry, Plugin

    # Get registry
    registry = get_plugin_registry()

    # Register plugins
    from orchestrator.plugins.cost_optimization import CostOptimizationPlugin
    registry.register(CostOptimizationPlugin())

    # Initialize
    await registry.initialize_all()

    # Use in orchestrator
    await registry.execute_pre_task(task)
    # ... execute task ...
    await registry.execute_post_task(task, result)

    # Shutdown
    await registry.shutdown_all()
"""

from .base import (
    Plugin,
    PluginRegistry,
    PluginContext,
    PluginMetadata,
    PluginPhase,
    PluginPriority,
    get_plugin_registry,
    reset_plugin_registry,
)

from .cost_optimization import (
    CostOptimizationPlugin,
    CostOptimizationConfig,
    create_cost_optimization_plugin,
)

from .nash_stability import (
    NashStabilityPlugin,
)

__all__ = [
    # Base
    "Plugin",
    "PluginRegistry",
    "PluginContext",
    "PluginMetadata",
    "PluginPhase",
    "PluginPriority",
    "get_plugin_registry",
    "reset_plugin_registry",
    # Cost Optimization
    "CostOptimizationPlugin",
    "CostOptimizationConfig",
    "create_cost_optimization_plugin",
    # Nash Stability
    "NashStabilityPlugin",
]
