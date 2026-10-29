"""
diff_generator — Backward-compatibility shim
The canonical implementation lives in orchestrator/generators/diff_generator.py.
New code should import from `orchestrator.generators.diff_generator` directly.
"""

from .generators.diff_generator import *  # noqa: F401, F403
