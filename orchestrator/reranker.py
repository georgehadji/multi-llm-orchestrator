"""
reranker — Backward-compatibility shim
The canonical implementation lives in orchestrator/infrastructure/reranker.py.
New code should import from `orchestrator.infrastructure.reranker` directly.
"""

from .infrastructure.reranker import *  # noqa: F401, F403
