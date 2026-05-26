"""
Dependency Resolver — Backward-compatibility shim
===================================================
The canonical DependencyResolver now lives in orchestrator/application/dependency_resolver.py.
"""

from ..application.dependency_resolver import *  # noqa: F401, F403
