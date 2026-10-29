"""
design_to_code — Backward-compatibility shim
The canonical implementation lives in orchestrator/design/design_to_code.py.
New code should import from `orchestrator.design.design_to_code` directly.
"""

from .design.design_to_code import *  # noqa: F401, F403
