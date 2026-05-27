"""
Plugin Discovery — Scan for User + Bundled + pip Entry Points
===============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Discovers plugins from three sources in priority order:
1. Bundled: <repo_base>/orchestrator/plugins/<kind>/<name>/
2. User:    ~/.orchestrator/plugins/<kind>/<name>/
3. pip:     entry points registered as orchestrator_<kind>

User plugins of the same name override bundled ones (last-writer-wins).
This lets third parties swap out any built-in provider without a repo patch.

Currently supported kinds:
- "memory"      → MemoryProvider (see memory_provider.py)
- "context"     → ContextProvider (see context_provider.py)

Usage:
    from orchestrator.plugins.discovery import discover_plugins, load_plugin

    memory_providers = await discover_plugins("memory")
    ctx_providers = await discover_plugins("context")
"""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# Repo root for bundled plugins (resolved at module load time)
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# User plugin directory
_USER_PLUGINS_ROOT = Path.home() / ".orchestrator" / "plugins"

# Entry point group prefix
_ENTRY_POINT_PREFIX = "orchestrator_"


# ─────────────────────────────────────────────────────────────────────────────
# Core discovery
# ─────────────────────────────────────────────────────────────────────────────


def _bundled_plugin_path(kind: str) -> Path:
    """Return the path to bundled plugins of a given kind.

    Example: orchestrator/plugins/memory/ → <repo>/orchestrator/plugins/memory/
    """
    return _REPO_ROOT / "orchestrator" / "plugins" / kind


def _user_plugin_path(kind: str) -> Path:
    """Return the path to user-installed plugins of a given kind.

    Example: ~/.orchestrator/plugins/memory/
    """
    return _USER_PLUGINS_ROOT / kind


def _scan_directory(base: Path, kind: str) -> list[dict[str, Any]]:
    """Scan a plugin directory for valid plugin subdirectories.

    A valid plugin directory must contain an __init__.py file.

    Args:
        base: The base directory to scan (e.g. ~/.orchestrator/plugins/memory/).
        kind: Plugin kind for logging (e.g. "memory").

    Returns:
        List of dicts with keys: name, path.
    """
    if not base.is_dir():
        return []

    plugins: list[dict[str, Any]] = []
    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith("__") or entry.name.startswith("."):
            continue
        init_file = entry / "__init__.py"
        if not init_file.exists():
            logger.debug("Skipping %s: no __init__.py", entry.name)
            continue
        plugins.append({"name": entry.name, "path": entry})
    return plugins


def _scan_entry_points(kind: str) -> list[dict[str, Any]]:
    """Scan pip-installed entry points for plugins of the given kind.

    Entry point group: orchestrator_<kind> (e.g. orchestrator_memory)

    Each entry point should point to a module that exposes a ``register()``
    function accepting ``register(ctx)``.

    Returns:
        List of dicts with keys: name, entry_point.
    """
    group = f"{_ENTRY_POINT_PREFIX}{kind}"
    plugins: list[dict[str, Any]] = []

    try:
        eps = importlib.metadata.entry_points(group=group)
        for ep in eps:
            plugins.append({"name": ep.name, "entry_point": ep})
    except importlib.metadata.PackageNotFoundError:
        pass
    except TypeError:
        # Python < 3.12 compatibility: entry_points(group=...) may fail
        pass

    return plugins


async def discover_plugins(kind: str) -> list[dict[str, Any]]:
    """Discover all plugins of a given kind from all sources.

    Scan order (priority ascending — later overrides earlier):
    1. Bundled: <repo>/orchestrator/plugins/<kind>/
    2. User: ~/.orchestrator/plugins/<kind>/
    3. pip entry points: orchestrator_<kind>

    User plugins override bundled ones of the same name (last-writer-wins).

    Args:
        kind: Plugin kind: "memory", "context", or future kinds.

    Returns:
        List of dicts with keys: name, source (bundled|user|pip), path/entry_point.
        Ordered by priority, deduplicated by name.
    """
    bundled = _scan_directory(_bundled_plugin_path(kind), kind)
    user = _scan_directory(_user_plugin_path(kind), kind)
    pip = _scan_entry_points(kind)

    for plugin in bundled:
        plugin["source"] = "bundled"
    for plugin in user:
        plugin["source"] = "user"
    for plugin in pip:
        plugin["source"] = "pip"

    # Merge: later sources override earlier of the same name
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    # Bundled first
    for plugin in bundled + user + pip:
        if plugin["name"] not in seen:
            seen.add(plugin["name"])
            merged.append(plugin)
        else:
            # Replace existing entry with override
            for i, existing in enumerate(merged):
                if existing["name"] == plugin["name"]:
                    merged[i] = plugin
                    break

    if merged:
        logger.info(
            "Discovered %d '%s' plugins: %s",
            len(merged),
            kind,
            [p["name"] for p in merged],
        )

    return merged


async def load_plugin(
    plugin_info: dict[str, Any],
    kind: str,
) -> object | None:
    """Load and instantiate a single plugin from its discovery info.

    Args:
        plugin_info: Dict from discover_plugins() with keys: name, source, path/entry_point.
        kind: Plugin kind for locating the correct ABC.

    Returns:
        An instance of MemoryProvider/ContextProvider if loading succeeds,
        or None if the plugin fails to load.

    Raises:
        ImportError: If the plugin module cannot be imported.
        TypeError: If the module does not contain a compatible plugin class.
    """
    from .base import Plugin

    name = plugin_info["name"]
    source = plugin_info.get("source", "unknown")

    try:
        if source == "pip":
            # Entry point — load the registered object directly
            ep = plugin_info.get("entry_point")
            if ep is None:
                logger.warning("Plugin %s has no entry_point", name)
                return None
            plugin_instance = ep.load()
            if isinstance(plugin_instance, type):
                plugin_instance = plugin_instance()
            return plugin_instance

        # Directory-based plugin (bundled or user)
        plugin_path = plugin_info.get("path")
        if plugin_path is None or not plugin_path.is_dir():
            logger.warning("Plugin %s path %s is invalid", name, plugin_path)
            return None

        # Import the plugin module
        if source == "bundled":
            # orchestrator.plugins.<kind>.<name>
            module_name = f"orchestrator.plugins.{kind}.{name}"
        else:
            # User plugins: add to sys.path temporarily
            _USER_PLUGINS_ROOT.mkdir(parents=True, exist_ok=True)
            if str(_USER_PLUGINS_ROOT.parent) not in sys.path:
                sys.path.insert(0, str(_USER_PLUGINS_ROOT.parent))

            module_name = f"plugins.{kind}.{name}"

        module = importlib.import_module(module_name)

        # Find plugin class: any subclass of the appropriate ABC
        from .memory_provider import MemoryProvider
        from .context_provider import ContextProvider

        base_classes = (MemoryProvider, ContextProvider, Plugin)

        plugin_instance = None
        for _name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and issubclass(obj, base_classes) and obj not in base_classes:
                plugin_instance = obj()
                break

        if plugin_instance is None:
            logger.warning(
                "No %s subclass found in plugin %s",
                kind,
                name,
            )
            return None

        return plugin_instance

    except Exception as exc:
        logger.error("Failed to load plugin '%s' from %s: %s", name, source, exc)
        return None


async def load_all_plugins(kind: str) -> list[Any]:
    """Convenience: discover + load all plugins of a given kind.

    Args:
        kind: Plugin kind (e.g. "memory", "context").

    Returns:
        List of loaded plugin instances. Failed plugins are skipped with a warning.
    """
    discovered = await discover_plugins(kind)
    instances: list[Any] = []
    for info in discovered:
        instance = await load_plugin(info, kind)
        if instance is not None:
            instances.append(instance)
    return instances
