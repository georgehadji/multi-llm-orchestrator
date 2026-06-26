"""
fullstack_generator — Backward-compatibility shim
The canonical implementation lives in orchestrator/generators/fullstack_generator.py.
New code should import from `orchestrator.generators.fullstack_generator` directly.
"""

from .generators.fullstack_generator import *  # noqa: F401, F403
