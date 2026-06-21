"""CLI command handler modules — extracted from orchestrator/cli.py.

Each module exposes ``register(subparsers)`` and ``execute(args)`` functions.
"""

from __future__ import annotations

# ── Registry of available subcommands ─────────────────────────────────────
# Each entry is the base module name in orchestrator/commands/ that exports
# register(subparsers) and execute(args). Add new commands here.
COMMAND_MODULES: list[str] = [
    "agent",
    "analyze",
    "build",
    "cache_stats",
    "chat",
    "codebase",
    "dashboard",
    "gateway",
    "kanban",
    "meta",
    "nash",
    "nexus",
    "nexusscope",
    "slash",
    "website",
]

# Backward-compat re-exports — pre-existing from before Phase 2 extraction
from .center import *  # noqa: F401, F403
from .integration import *  # noqa: F401, F403
from .server import *  # noqa: F401, F403
from .registry import *  # noqa: F401, F403
