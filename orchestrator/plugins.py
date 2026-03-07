"""
Plugin System for Multi-LLM Orchestrator
========================================

Lightweight plugin architecture for extensibility without core modifications.

Plugin Types:
- ValidatorPlugin: Custom code validators (e.g., for specific languages)
- IntegrationPlugin: External service integrations (Teams, Discord, etc.)
- RouterPlugin: Custom routing strategies
- FeedbackPlugin: Custom production feedback processors

Usage:
    from orchestrator.plugins import PluginRegistry, ValidatorPlugin
    
    class MyValidator(ValidatorPlugin):
        def validate(self, code: str, context: dict) -> ValidationResult:
            ...
    
    registry = PluginRegistry()
    registry.register(MyValidator())
"""

from __future__ import annotations

import abc
import importlib
import inspect
import logging
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Generic, List, Optional, Type, TypeVar, Protocol
from enum import Enum, auto

from .log_config import get_logger
from .models import Model, TaskType, Task

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Plugin Types & Interfaces
# ═══════════════════════════════════════════════════════════════════════════════

class PluginType(Enum):
    """Types of plugins supported by the system."""
    VALIDATOR = auto()      # Custom code validators
    INTEGRATION = auto()    # External service integrations
    ROUTER = auto()         # Custom routing strategies
    FEEDBACK = auto()       # Production feedback processors
    MONITORING = auto()     # Custom monitoring/alerting
    TRANSFORM = auto()      # Code transformation pipelines


@dataclass
class PluginMetadata:
    """Metadata for a plugin."""
    name: str
    version: str
    author: str
    description: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    config_schema: Optional[Dict[str, Any]] = None


@dataclass
class ValidationResult:
    """Result from a validator plugin."""
    passed: bool
    score: float  # 0.0 - 1.0
    issues: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class RoutingSuggestion:
    """Suggestion from a router plugin."""
    model: Model
    confidence: float  # 0.0 - 1.0
    reason: str
    estimated_cost: Optional[float] = None
    estimated_latency_ms: Optional[float] = None


@dataclass
class FeedbackPayload:
    """Payload for feedback plugins."""
    project_id: str
    deployment_id: str
    task_type: TaskType
    model_used: Model
    generated_code: str
    runtime_errors: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    user_rating: Optional[int] = None  # 1-5


# ═══════════════════════════════════════════════════════════════════════════════
# Plugin Base Classes
# ═══════════════════════════════════════════════════════════════════════════════

class Plugin(abc.ABC):
    """Base class for all plugins."""
    
    @property
    @abc.abstractmethod
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        pass
    
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the plugin with configuration."""
        pass
    
    def shutdown(self) -> None:
        """Cleanup when plugin is unloaded."""
        pass
    
    def health_check(self) -> tuple[bool, Optional[str]]:
        """Check plugin health. Returns (healthy, error_message)."""
        return True, None


class ValidatorPlugin(Plugin):
    """Plugin for custom code validation."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="base-validator",
            version="1.0.0",
            author="",
            description="Base validator plugin",
            plugin_type=PluginType.VALIDATOR,
        )
    
    @abc.abstractmethod
    def can_validate(self, file_path: str, language: str) -> bool:
        """Check if this validator can handle the given file."""
        pass
    
    @abc.abstractmethod
    def validate(self, code: str, context: Dict[str, Any]) -> ValidationResult:
        """Validate code and return result."""
        pass
    
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language identifiers."""
        return []


class IntegrationPlugin(Plugin):
    """Plugin for external service integrations."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="base-integration",
            version="1.0.0",
            author="",
            description="Base integration plugin",
            plugin_type=PluginType.INTEGRATION,
        )
    
    @abc.abstractmethod
    async def send_notification(self, event_type: str, payload: Dict[str, Any]) -> bool:
        """Send a notification."""
        pass
    
    @abc.abstractmethod
    def is_configured(self) -> bool:
        """Check if integration is properly configured."""
        pass


class RouterPlugin(Plugin):
    """Plugin for custom routing strategies."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="base-router",
            version="1.0.0",
            author="",
            description="Base router plugin",
            plugin_type=PluginType.ROUTER,
        )
    
    @abc.abstractmethod
    def suggest_models(
        self,
        task: Task,
        available_models: List[Model],
        context: Dict[str, Any],
    ) -> List[RoutingSuggestion]:
        """
        Suggest models for a task.
        
        Returns list of suggestions sorted by preference (best first).
        """
        pass
    
    def get_weight(self) -> float:
        """
        Get the weight of this router's suggestions.
        
        Higher weight = more influence on final decision.
        """
        return 1.0


class FeedbackPlugin(Plugin):
    """Plugin for processing production feedback."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="base-feedback",
            version="1.0.0",
            author="",
            description="Base feedback plugin",
            plugin_type=PluginType.FEEDBACK,
        )
    
    @abc.abstractmethod
    async def process_feedback(self, payload: FeedbackPayload) -> Dict[str, Any]:
        """
        Process production feedback.
        
        Returns extracted insights for knowledge base.
        """
        pass
    
    def should_process(self, payload: FeedbackPayload) -> bool:
        """Check if this plugin should process the feedback."""
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Plugin Registry
# ═══════════════════════════════════════════════════════════════════════════════

class PluginRegistry:
    """
    Central registry for all plugins.
    
    Manages plugin lifecycle, discovery, and access.
    """
    
    def __init__(self):
        self._plugins: Dict[PluginType, List[Plugin]] = {
            t: [] for t in PluginType
        }
        self._by_name: Dict[str, Plugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
    
    def register(self, plugin: Plugin) -> None:
        """Register a plugin."""
        meta = plugin.metadata
        
        if meta.name in self._by_name:
            logger.warning(f"Plugin {meta.name} already registered, skipping")
            return
        
        # Check dependencies
        for dep in meta.dependencies:
            if dep not in self._by_name:
                raise PluginDependencyError(
                    f"Plugin {meta.name} requires {dep} but it's not loaded"
                )
        
        self._plugins[meta.plugin_type].append(plugin)
        self._by_name[meta.name] = plugin
        
        logger.info(f"Registered plugin: {meta.name} v{meta.version} ({meta.plugin_type.name})")
    
    def unregister(self, name: str) -> None:
        """Unregister a plugin by name."""
        if name not in self._by_name:
            return
        
        plugin = self._by_name[name]
        plugin.shutdown()
        
        self._plugins[plugin.metadata.plugin_type].remove(plugin)
        del self._by_name[name]
        
        logger.info(f"Unregistered plugin: {name}")
    
    def get(self, name: str) -> Optional[Plugin]:
        """Get a plugin by name."""
        return self._by_name.get(name)
    
    def get_by_type(self, plugin_type: PluginType) -> List[Plugin]:
        """Get all plugins of a specific type."""
        return self._plugins[plugin_type].copy()
    
    def get_validators(self) -> List[ValidatorPlugin]:
        """Get all validator plugins."""
        return [p for p in self._plugins[PluginType.VALIDATOR] if isinstance(p, ValidatorPlugin)]
    
    def get_integrations(self) -> List[IntegrationPlugin]:
        """Get all integration plugins."""
        return [p for p in self._plugins[PluginType.INTEGRATION] if isinstance(p, IntegrationPlugin)]
    
    def get_routers(self) -> List[RouterPlugin]:
        """Get all router plugins."""
        return [p for p in self._plugins[PluginType.ROUTER] if isinstance(p, RouterPlugin)]
    
    def get_feedback_processors(self) -> List[FeedbackPlugin]:
        """Get all feedback plugins."""
        return [p for p in self._plugins[PluginType.FEEDBACK] if isinstance(p, FeedbackPlugin)]
    
    def discover(self, package_path: str) -> int:
        """
        Discover and load plugins from a package path.
        
        Returns number of plugins loaded.
        """
        count = 0
        try:
            package = importlib.import_module(package_path)
            for _, name, is_pkg in pkgutil.iter_modules(
                package.__path__ if hasattr(package, '__path__') else [],
                package.__name__ + "."
            ):
                try:
                    module = importlib.import_module(name)
                    for obj_name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Plugin) and 
                            obj is not Plugin and
                            not inspect.isabstract(obj)):
                            try:
                                instance = obj()
                                self.register(instance)
                                count += 1
                            except Exception as e:
                                logger.error(f"Failed to instantiate plugin {obj_name}: {e}")
                except Exception as e:
                    logger.error(f"Failed to load plugin module {name}: {e}")
        except ImportError:
            logger.warning(f"Plugin package {package_path} not found")
        
        return count
    
    def discover_directory(self, directory: Path) -> int:
        """Discover plugins from a directory."""
        count = 0
        if not directory.exists():
            return 0
        
        for file in directory.glob("*.py"):
            if file.name.startswith("_"):
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    f"_plugin_{file.stem}", file
                )
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    for obj_name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, Plugin) and 
                            obj is not Plugin and
                            not inspect.isabstract(obj)):
                            try:
                                instance = obj()
                                self.register(instance)
                                count += 1
                            except Exception as e:
                                logger.error(f"Failed to instantiate plugin {obj_name}: {e}")
            except Exception as e:
                logger.error(f"Failed to load plugin from {file}: {e}")
        
        return count
    
    def run_health_checks(self) -> Dict[str, tuple[bool, Optional[str]]]:
        """Run health checks on all plugins."""
        results = {}
        for name, plugin in self._by_name.items():
            results[name] = plugin.health_check()
        return results
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all registered plugin metadata."""
        return [p.metadata for p in self._by_name.values()]


# ═══════════════════════════════════════════════════════════════════════════════
# Built-in Plugins
# ═══════════════════════════════════════════════════════════════════════════════

class PythonTypeCheckerValidator(ValidatorPlugin):
    """Built-in validator using mypy."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="mypy-type-checker",
            version="1.0.0",
            author="orchestrator",
            description="Type checking using mypy",
            plugin_type=PluginType.VALIDATOR,
        )
    
    def can_validate(self, file_path: str, language: str) -> bool:
        return language == "python" or file_path.endswith(".py")
    
    def validate(self, code: str, context: Dict[str, Any]) -> ValidationResult:
        # Simplified - real implementation would use mypy API
        return ValidationResult(passed=True, score=1.0)
    
    def get_supported_languages(self) -> List[str]:
        return ["python"]


class TeamsIntegration(IntegrationPlugin):
    """Microsoft Teams integration plugin."""
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="microsoft-teams",
            version="1.0.0",
            author="orchestrator",
            description="Microsoft Teams notifications",
            plugin_type=PluginType.INTEGRATION,
        )
    
    def initialize(self, config: Dict[str, Any]) -> None:
        self.webhook_url = config.get("webhook_url") or __import__("os").environ.get("TEAMS_WEBHOOK_URL")
    
    def is_configured(self) -> bool:
        return bool(self.webhook_url)
    
    async def send_notification(self, event_type: str, payload: Dict[str, Any]) -> bool:
        if not self.webhook_url:
            return False
        # Implementation would send adaptive card to Teams
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Exceptions
# ═══════════════════════════════════════════════════════════════════════════════

class PluginError(Exception):
    """Base plugin error."""
    pass


class PluginDependencyError(PluginError):
    """Plugin dependency not satisfied."""
    pass


class PluginValidationError(PluginError):
    """Plugin validation failed."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Global Registry Instance
# ═══════════════════════════════════════════════════════════════════════════════

_registry: Optional[PluginRegistry] = None


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry."""
    global _registry
    if _registry is None:
        _registry = PluginRegistry()
        # Register built-in plugins
        _registry.register(PythonTypeCheckerValidator())
        _registry.register(TeamsIntegration())
    return _registry


def reset_plugin_registry() -> None:
    """Reset the global registry (for testing)."""
    global _registry
    _registry = None
