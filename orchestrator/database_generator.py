"""
database_generator — Backward-compatibility shim
The canonical implementation lives in orchestrator/generators/database_generator.py.
New code should import from `orchestrator.generators.database_generator` directly.
"""

from .generators.database_generator import *  # noqa: F401, F403
