"""Test runner adapters — framework-specific implementations of TestExecutorPort.

Framework registry acts as a factory: given a framework string, return
the matching runner or raise.
"""

from __future__ import annotations

from .base import TestRunnerBase
from .pytest_runner import PytestRunner
from .jest_runner import JestRunner
from .go_runner import GoRunner
from .cargo_runner import CargoRunner

__all__ = [
    "CargoRunner",
    "GoRunner",
    "JestRunner",
    "PytestRunner",
    "TestRunnerBase",
]


_FRAMEWORK_REGISTRY: dict[str, type[TestRunnerBase]] = {}


def register_runner(framework: str, runner_cls: type[TestRunnerBase]) -> None:
    """Register a runner for a framework."""
    _FRAMEWORK_REGISTRY[framework] = runner_cls


def get_runner(framework: str) -> type[TestRunnerBase]:
    """Get the runner class for a framework. Raises KeyError if not found."""
    if not _FRAMEWORK_REGISTRY:
        _populate_registry()
    if framework not in _FRAMEWORK_REGISTRY:
        raise KeyError(f"No test runner registered for framework: {framework}")
    return _FRAMEWORK_REGISTRY[framework]


def _populate_registry() -> None:
    _FRAMEWORK_REGISTRY["pytest"] = PytestRunner
    _FRAMEWORK_REGISTRY["jest"] = JestRunner
    _FRAMEWORK_REGISTRY["go"] = GoRunner
    _FRAMEWORK_REGISTRY["cargo"] = CargoRunner
