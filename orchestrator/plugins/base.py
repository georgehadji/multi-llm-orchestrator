"""
Plugin System — Base Classes & Registry
========================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Plugin architecture for optional features extraction.
Allows decoupling of advanced features from core orchestrator.

Part of Phase 5: Plugin Architecture
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .models import Task, TaskResult, ProjectState

logger = logging.getLogger(__name__)


class PluginPhase(Enum):
    """Lifecycle phases for plugin execution."""
    INIT = "init"
    PRE_PROJECT = "pre_project"
    PRE_TASK = "pre_task"
    POST_TASK = "post_task"
    POST_PROJECT = "post_project"
    SHUTDOWN = "shutdown"


class PluginPriority(Enum):
    """Plugin execution priority."""
    HIGH = 1
    NORMAL = 2
    LOW = 3


@dataclass
class PluginContext:
    """Context passed to plugin hooks."""
    project_state: Optional[ProjectState] = None
    task: Optional[Task] = None
    task_result: Optional[TaskResult] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PluginMetadata:
    """Metadata about a plugin."""
    name: str
    version: str
    description: str
    author: str
    priority: PluginPriority = PluginPriority.NORMAL
    dependencies: list[str] = field(default_factory=list)


class Plugin(ABC):
    """
    Base class for all orchestrator plugins.
    
    Plugins can hook into the orchestration lifecycle at various points:
    - Initialization
    - Before project starts
    - Before each task
    - After each task
    - After project completes
    - Shutdown
    
    Example:
        class MyPlugin(Plugin):
            async def pre_task(self, context: PluginContext) -> None:
                logger.info(f"About to execute task: {context.task.id}")
    """
    
    metadata: PluginMetadata
    
    def __init__(self) -> None:
        """Initialize plugin."""
        self._enabled = True
        self._initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize plugin resources.
        
        Called once when plugin is loaded.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> None:
        """
        Shutdown plugin resources.
        
        Called once when orchestrator shuts down.
        """
        pass
    
    async def on_pre_project(self, context: PluginContext) -> None:
        """
        Called before project execution starts.
        
        Args:
            context: Plugin context with project state
        """
        pass
    
    async def on_post_project(self, context: PluginContext) -> None:
        """
        Called after project execution completes.
        
        Args:
            context: Plugin context with project state and results
        """
        pass
    
    async def on_pre_task(self, context: PluginContext) -> None:
        """
        Called before each task execution.
        
        Args:
            context: Plugin context with task definition
        """
        pass
    
    async def on_post_task(self, context: PluginContext) -> None:
        """
        Called after each task execution.
        
        Args:
            context: Plugin context with task result
        """
        pass
    
    def enable(self) -> None:
        """Enable plugin."""
        self._enabled = True
        logger.info(f"Plugin {self.metadata.name} enabled")
    
    def disable(self) -> None:
        """Disable plugin."""
        self._enabled = False
        logger.info(f"Plugin {self.metadata.name} disabled")
    
    @property
    def is_enabled(self) -> bool:
        """Check if plugin is enabled."""
        return self._enabled
    
    @property
    def is_initialized(self) -> bool:
        """Check if plugin is initialized."""
        return self._initialized


class PluginRegistry:
    """
    Registry for managing plugin lifecycle.
    
    Responsibilities:
    1. Register/unregister plugins
    2. Manage plugin dependencies
    3. Execute plugin hooks in order
    4. Handle plugin errors gracefully
    
    Usage:
        registry = PluginRegistry()
        registry.register(MyPlugin())
        await registry.initialize_all()
        
        # Execute hooks
        await registry.execute_hook(PluginPhase.PRE_TASK, context)
        
        await registry.shutdown_all()
    """
    
    def __init__(self) -> None:
        """Initialize plugin registry."""
        self._plugins: dict[str, Plugin] = {}
        self._hooks: dict[PluginPhase, list[Plugin]] = {
            phase: [] for phase in PluginPhase
        }
        self._initialized = False
    
    def register(self, plugin: Plugin) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        name = plugin.metadata.name
        
        if name in self._plugins:
            logger.warning(f"Plugin {name} already registered, replacing")
            self.unregister(name)
        
        self._plugins[name] = plugin
        
        # Add to hook lists
        self._hooks[PluginPhase.INIT].append(plugin)
        self._hooks[PluginPhase.PRE_PROJECT].append(plugin)
        self._hooks[PluginPhase.PRE_TASK].append(plugin)
        self._hooks[PluginPhase.POST_TASK].append(plugin)
        self._hooks[PluginPhase.POST_PROJECT].append(plugin)
        self._hooks[PluginPhase.SHUTDOWN].append(plugin)
        
        logger.info(f"Plugin {name} registered")
    
    def unregister(self, name: str) -> None:
        """
        Unregister a plugin.
        
        Args:
            name: Plugin name to unregister
        """
        if name not in self._plugins:
            logger.warning(f"Plugin {name} not found")
            return
        
        plugin = self._plugins.pop(name)
        
        # Remove from hook lists
        for phase_list in self._hooks.values():
            if plugin in phase_list:
                phase_list.remove(plugin)
        
        logger.info(f"Plugin {name} unregistered")
    
    def get(self, name: str) -> Optional[Plugin]:
        """
        Get plugin by name.
        
        Args:
            name: Plugin name
            
        Returns:
            Plugin instance or None
        """
        return self._plugins.get(name)
    
    def list_plugins(self) -> list[str]:
        """
        List all registered plugin names.
        
        Returns:
            List of plugin names
        """
        return list(self._plugins.keys())
    
    async def initialize_all(self) -> None:
        """Initialize all registered plugins."""
        if self._initialized:
            logger.warning("Plugin registry already initialized")
            return
        
        logger.info(f"Initializing {len(self._plugins)} plugins")
        
        for plugin in self._plugins.values():
            try:
                await plugin.initialize()
                plugin._initialized = True
                logger.info(f"Plugin {plugin.metadata.name} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin.metadata.name}: {e}")
                plugin.disable()
        
        self._initialized = True
    
    async def shutdown_all(self) -> None:
        """Shutdown all registered plugins."""
        if not self._initialized:
            return
        
        logger.info("Shutting down all plugins")
        
        for plugin in reversed(list(self._plugins.values())):
            try:
                if plugin.is_initialized:
                    await plugin.shutdown()
                    plugin._initialized = False
                    logger.info(f"Plugin {plugin.metadata.name} shut down")
            except Exception as e:
                logger.error(f"Error shutting down plugin {plugin.metadata.name}: {e}")
        
        self._initialized = False
    
    async def execute_hook(
        self,
        phase: PluginPhase,
        context: PluginContext,
    ) -> None:
        """
        Execute hook for all plugins at given phase.
        
        Args:
            phase: Plugin phase
            context: Plugin context
        """
        if not self._initialized:
            logger.warning("Plugin registry not initialized")
            return
        
        plugins = self._hooks.get(phase, [])
        
        # Sort by priority
        plugins_sorted = sorted(
            plugins,
            key=lambda p: p.metadata.priority.value if p.is_enabled else 999
        )
        
        for plugin in plugins_sorted:
            if not plugin.is_enabled:
                continue
            
            try:
                if phase == PluginPhase.PRE_PROJECT:
                    await plugin.on_pre_project(context)
                elif phase == PluginPhase.POST_PROJECT:
                    await plugin.on_post_project(context)
                elif phase == PluginPhase.PRE_TASK:
                    await plugin.on_pre_task(context)
                elif phase == PluginPhase.POST_TASK:
                    await plugin.on_post_task(context)
            except Exception as e:
                logger.error(
                    f"Error in plugin {plugin.metadata.name} "
                    f"hook {phase.value}: {e}"
                )
                # Continue with other plugins
    
    async def execute_pre_project(self, project_state: ProjectState) -> None:
        """Execute pre-project hooks."""
        context = PluginContext(project_state=project_state)
        await self.execute_hook(PluginPhase.PRE_PROJECT, context)
    
    async def execute_post_project(self, project_state: ProjectState) -> None:
        """Execute post-project hooks."""
        context = PluginContext(project_state=project_state)
        await self.execute_hook(PluginPhase.POST_PROJECT, context)
    
    async def execute_pre_task(self, task: Task) -> None:
        """Execute pre-task hooks."""
        context = PluginContext(task=task)
        await self.execute_hook(PluginPhase.PRE_TASK, context)
    
    async def execute_post_task(self, task: Task, result: TaskResult) -> None:
        """Execute post-task hooks."""
        context = PluginContext(task=task, task_result=result)
        await self.execute_hook(PluginPhase.POST_TASK, context)


# Global plugin registry instance
_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
    return _registry


def reset_plugin_registry() -> None:
    """Reset global plugin registry (for testing)."""
    global _registry
    _registry = None
