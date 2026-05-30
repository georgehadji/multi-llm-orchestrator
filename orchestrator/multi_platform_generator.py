"""
multi_platform_generator — Backward-compatibility shim
The canonical implementation lives in orchestrator/generators/multi_platform_generator.py.
New code should import from `orchestrator.generators.multi_platform_generator` directly.
"""

from .generators.multi_platform_generator import *  # noqa: F401, F403
