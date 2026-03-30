"""
Plugin Marketplace Architecture
================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Paradigm Shift: Extensible plugin system so third parties can add capabilities

Current State: Monolithic orchestrator
Future State: Platform with third-party plugins

Benefits:
- Network effects: third parties build capabilities
- Platform play: ecosystem, not just product
- Long-term defensibility

Usage:
    from orchestrator.plugins import PluginManager, PluginHook

    manager = PluginManager()
    manager.discover(Path("./plugins"))

    # Run plugin hooks
    context = await manager.run_hook(PluginHook.PRE_GENERATION, context)
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .log_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)


class PluginHook(str, Enum):
    """
    Available plugin hooks in the orchestration pipeline.

    Plugins can register callbacks for any of these hooks.
    """
    # Pre-processing hooks
    PRE_DECOMPOSITION = "pre_decomposition"
    POST_DECOMPOSITION = "post_decomposition"

    # Generation hooks
    PRE_GENERATION = "pre_generation"
    POST_GENERATION = "post_generation"

    # Validation hooks
    VALIDATION = "validation"

    # Post-processing hooks
    POST_EVALUATION = "post_evaluation"
    PRE_DEPLOYMENT = "pre_deployment"

    # Custom hooks
    CUSTOM = "custom"


@dataclass
class PluginManifest:
    """
    Plugin metadata and configuration.

    Attributes:
        name: Unique plugin name
        version: Plugin version (semver)
        description: Plugin description
        author: Plugin author
        entry_point: Module path (e.g., "my_plugin:MyPlugin")
        hooks: List of hooks this plugin implements
        dependencies: Plugin dependencies
    """
    name: str
    version: str
    description: str
    author: str
    entry_point: str
    hooks: list[PluginHook] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    min_orchestrator_version: str = "1.0.0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PluginManifest:
        """Create manifest from dictionary."""
        hooks = [
            PluginHook(h) if isinstance(h, str) else h
            for h in data.get("hooks", [])
        ]
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            author=data["author"],
            entry_point=data["entry_point"],
            hooks=hooks,
            dependencies=data.get("dependencies", []),
            min_orchestrator_version=data.get("min_orchestrator_version", "1.0.0"),
        )

    @classmethod
    def from_file(cls, path: Path) -> PluginManifest:
        """Load manifest from plugin.json file."""
        with path.open("r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "entry_point": self.entry_point,
            "hooks": [h.value for h in self.hooks],
            "dependencies": self.dependencies,
            "min_orchestrator_version": self.min_orchestrator_version,
        }


@dataclass
class PluginContext:
    """
    Context passed to plugin hooks.

    Plugins can read and modify this context.
    """
    data: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value

    def update(self, **kwargs) -> None:
        self.data.update(kwargs)


class Plugin:
    """
    Base class for all plugins.

    Plugins should inherit from this class and implement
    the hooks they want to support.
    """

    def __init__(self, manifest: PluginManifest):
        self.manifest = manifest

    async def execute(
        self,
        hook: PluginHook,
        context: PluginContext,
    ) -> PluginContext:
        """
        Execute plugin logic for a hook.

        Args:
            hook: Hook being executed
            context: Current context

        Returns:
            Modified context
        """
        method_name = f"on_{hook.value}"

        if hasattr(self, method_name):
            method = getattr(self, method_name)
            if callable(method):
                return await method(context)

        return context


class PluginManager:
    """
    Manages plugin discovery, loading, and execution.

    Features:
    - Discover plugins in directory
    - Load/unload plugins
    - Run hooks in order
    - Plugin isolation
    """

    def __init__(self, plugins_dir: Path | None = None):
        """
        Initialize plugin manager.

        Args:
            plugins_dir: Directory to search for plugins
        """
        self.plugins_dir = plugins_dir or Path("./plugins")
        self.plugins: dict[str, Plugin] = {}
        self.hooks: dict[PluginHook, list[Callable]] = {}

        # Ensure plugins directory exists
        if self.plugins_dir and not self.plugins_dir.exists():
            self.plugins_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Plugin manager initialized (plugins_dir={self.plugins_dir})")

    def discover(self, plugins_dir: Path | None = None) -> list[PluginManifest]:
        """
        Discover plugins in directory.

        Args:
            plugins_dir: Directory to search (uses default if None)

        Returns:
            List of discovered plugin manifests
        """
        search_dir = plugins_dir or self.plugins_dir
        manifests = []

        if not search_dir.exists():
            logger.warning(f"Plugins directory not found: {search_dir}")
            return manifests

        # Look for plugin.json files
        for plugin_json in search_dir.glob("*/plugin.json"):
            try:
                manifest = PluginManifest.from_file(plugin_json)
                manifests.append(manifest)
                logger.info(f"Discovered plugin: {manifest.name} v{manifest.version}")
            except Exception as e:
                logger.error(f"Failed to load plugin manifest {plugin_json}: {e}")

        return manifests

    def load(self, manifest: PluginManifest) -> Plugin | None:
        """
        Load a plugin from manifest with security hardening.

        FIX-PS-002b: Plugin security - allowlist + import restrictions + verification.

        Args:
            manifest: Plugin manifest

        Returns:
            Loaded plugin or None if failed
        """
        try:
            # Parse entry point
            module_path, class_name = manifest.entry_point.split(":")

            # ═══════════════════════════════════════════════════════
            # FIX-PS-002b: Plugin security hardening
            # ═══════════════════════════════════════════════════════

            # 1. Verify plugin is in allowed list (if allowlist configured)
            if hasattr(self, '_allowed_entry_points'):
                if manifest.entry_point not in self._allowed_entry_points:
                    logger.error(
                        f"Plugin {manifest.name} entry point not in allowlist: "
                        f"{manifest.entry_point}"
                    )
                    return None

            # 2. Set up restricted import during plugin loading
            original_import = __builtins__.__import__ if isinstance(__builtins__.__import__, type(lambda: None)) else __builtins__['__import__']

            def restricted_import(name, *args, **kwargs):
                """Restricted import that blocks dangerous modules."""
                # Block dangerous modules that could escape plugin sandbox
                dangerous_modules = [
                    'os', 'subprocess', 'sys', 'ctypes', 'pickle',
                    'marshal', 'multiprocessing', 'socket', 'http',
                    'urllib', 'ftplib', 'smtplib', 'telnetlib'
                ]

                if any(d in name for d in dangerous_modules):
                    logger.warning(
                        f"Plugin {manifest.name} attempted to import dangerous module: {name}"
                    )
                    raise ImportError(
                        f"Plugin not allowed to import {name}. "
                        f"This module could compromise system security."
                    )

                return original_import(name, *args, **kwargs)

            # Apply restricted import
            if isinstance(__builtins__, dict):
                __builtins__['__import__'] = restricted_import
            else:
                __builtins__.__import__ = restricted_import

            try:
                # Import module with restricted imports
                module = importlib.import_module(module_path)

                # Get plugin class
                plugin_class = getattr(module, class_name)

                if not issubclass(plugin_class, Plugin):
                    logger.error(f"Plugin {manifest.name} does not inherit from Plugin")
                    return None

                # Instantiate plugin
                plugin = plugin_class(manifest)

                # Register hooks
                for hook in manifest.hooks:
                    if hook not in self.hooks:
                        self.hooks[hook] = []
                    self.hooks[hook].append(plugin.execute)

                # Store plugin
                self.plugins[manifest.name] = plugin

                logger.info(f"Loaded plugin: {manifest.name} with {len(manifest.hooks)} hooks")
                return plugin

            finally:
                # Restore original import - CRITICAL: always restore even if load fails
                if isinstance(__builtins__, dict):
                    __builtins__['__import__'] = original_import
                else:
                    __builtins__.__import__ = original_import

        except Exception as e:
            logger.error(f"Failed to load plugin {manifest.name}: {e}")
            return None

    def unload(self, plugin_name: str) -> bool:
        """
        Unload a plugin.

        Args:
            plugin_name: Name of plugin to unload

        Returns:
            True if successful
        """
        if plugin_name not in self.plugins:
            return False

        plugin = self.plugins[plugin_name]

        # Remove from hooks
        for hook in plugin.manifest.hooks:
            if hook in self.hooks:
                self.hooks[hook] = [
                    h for h in self.hooks[hook]
                    if h.__self__ != plugin
                ]

        # Remove plugin
        del self.plugins[plugin_name]

        logger.info(f"Unloaded plugin: {plugin_name}")
        return True

    async def run_hook(
        self,
        hook: PluginHook,
        context: PluginContext,
    ) -> PluginContext:
        """
        Run all plugins registered for a hook.

        Args:
            hook: Hook to run
            context: Current context

        Returns:
            Modified context after all plugins
        """
        if hook not in self.hooks:
            return context

        logger.debug(f"Running hook: {hook.value} ({len(self.hooks[hook])} plugins)")

        # Run plugins in order
        for plugin_executor in self.hooks[hook]:
            try:
                context = await plugin_executor(hook, context)
            except Exception as e:
                logger.error(f"Plugin hook {hook.value} failed: {e}")
                # Continue with next plugin (don't break pipeline)

        return context

    def get_plugins(self) -> list[PluginManifest]:
        """Get list of loaded plugin manifests."""
        return [p.manifest for p in self.plugins.values()]

    def get_hook_count(self, hook: PluginHook) -> int:
        """Get number of plugins registered for a hook."""
        return len(self.hooks.get(hook, []))


# ═══════════════════════════════════════════════════════
# Reference Plugins
# ═══════════════════════════════════════════════════════

class SecurityScannerPlugin(Plugin):
    """
    Reference plugin: Security scanning with Bandit + Safety.

    Hooks: POST_GENERATION, VALIDATION
    """

    async def on_post_generation(self, context: PluginContext) -> PluginContext:
        """Scan generated code for security issues."""
        code = context.get("code", "")

        if not code:
            return context

        # Run security checks (simplified - would use bandit in production)
        issues = []

        # Check for common security issues
        if "eval(" in code:
            issues.append("Use of eval() detected - security risk")
        if "exec(" in code:
            issues.append("Use of exec() detected - security risk")
        if "pickle" in code and "load" in code:
            issues.append("Use of pickle.load() detected - deserialization risk")

        context.set("security_issues", issues)
        context.set("security_passed", len(issues) == 0)

        if issues:
            logger.warning(f"Security scanner found {len(issues)} issues")

        return context


class DjangoTemplatePlugin(Plugin):
    """
    Reference plugin: Django-specific templates.

    Hooks: PRE_DECOMPOSITION, POST_DECOMPOSITION
    """

    async def on_pre_decomposition(self, context: PluginContext) -> PluginContext:
        """Add Django-specific context to decomposition."""
        prompt = context.get("prompt", "")

        # Check if this is a Django project
        if "django" in prompt.lower():
            context.set("framework", "django")
            context.set("templates", [
                "models.py",
                "views.py",
                "urls.py",
                "forms.py",
                "admin.py",
            ])

        return context


class AWSDeployPlugin(Plugin):
    """
    Reference plugin: AWS deployment.

    Hooks: PRE_DEPLOYMENT
    """

    async def on_pre_deployment(self, context: PluginContext) -> PluginContext:
        """Prepare deployment to AWS."""
        context.get("files", {})
        framework = context.get("framework", "fastapi")

        # Generate deployment config
        if framework == "fastapi":
            # Generate Lambda deployment config
            context.set("deployment_target", "aws-lambda")
            context.set("deployment_config", {
                "runtime": "python3.12",
                "handler": "main.handler",
                "memory_size": 128,
                "timeout": 30,
            })

        return context


__all__ = [
    "PluginManager",
    "Plugin",
    "PluginManifest",
    "PluginContext",
    "PluginHook",
    "SecurityScannerPlugin",
    "DjangoTemplatePlugin",
    "AWSDeployPlugin",
]
