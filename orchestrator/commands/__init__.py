"""CLI command handler modules — extracted from orchestrator/cli.py.

Each module exposes ``register(subparsers)`` and ``execute(args)`` functions.
New commands are auto-discovered — just add a module in this package
and it will be registered automatically.
"""

from __future__ import annotations

import pkgutil


def discover_command_modules() -> list[str]:
    """Return names of all modules in this package that expose ``register()``."""
    import os

    modules = []
    this_dir = os.path.dirname(__file__)
    for importer, modname, ispkg in pkgutil.iter_modules([this_dir]):
        if not ispkg and modname not in ("center", "integration", "server", "registry"):
            modules.append(modname)
    return sorted(modules)


# Backward-compat re-exports — pre-existing from before Phase 2 extraction
from .center import *  # noqa: F401, F403
from .integration import *  # noqa: F401, F403
from .server import *  # noqa: F401, F403
from .registry import *  # noqa: F401, F403
