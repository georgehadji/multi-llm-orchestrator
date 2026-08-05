"""Test runner adapters — framework-specific implementations of TestExecutorPort.

Framework registry acts as a factory: given a framework string, return a
configured runner instance (with a sandbox injected), or raise KeyError.
"""

from __future__ import annotations

import logging

from .base import TestRunnerBase
from .cargo_runner import CargoRunner
from .go_runner import GoRunner
from .jest_runner import JestRunner
from .pytest_runner import PytestRunner

logger = logging.getLogger(__name__)

__all__ = [
    "CargoRunner",
    "GoRunner",
    "JestRunner",
    "PytestRunner",
    "TestRunnerBase",
    "get_runner",
    "register_runner",
]

_FRAMEWORK_REGISTRY: dict[str, type[TestRunnerBase]] = {}


def register_runner(framework: str, runner_cls: type[TestRunnerBase]) -> None:
    """Register a runner class for a framework."""
    _FRAMEWORK_REGISTRY[framework] = runner_cls


def get_runner(framework: str, sandbox=None) -> TestRunnerBase:
    """Return a configured runner instance for *framework*.

    Args:
        framework: Framework identifier ('pytest', 'jest', 'go', 'cargo').
        sandbox: Optional SandboxPort; defaults to a SubprocessSandbox so
            execution is never unisolated (F-3).

    Returns:
        A runner instance implementing :class:`TestExecutorPort`.

    Raises:
        KeyError: If no runner is registered for the framework.
    """
    if not _FRAMEWORK_REGISTRY:
        _populate_registry()
    if framework not in _FRAMEWORK_REGISTRY:
        raise KeyError(f"No test runner registered for framework: {framework}")
    runner_cls = _FRAMEWORK_REGISTRY[framework]
    return runner_cls(sandbox=sandbox)


def _populate_registry() -> None:
    _FRAMEWORK_REGISTRY["pytest"] = PytestRunner
    _FRAMEWORK_REGISTRY["jest"] = JestRunner
    _FRAMEWORK_REGISTRY["go"] = GoRunner
    _FRAMEWORK_REGISTRY["cargo"] = CargoRunner
