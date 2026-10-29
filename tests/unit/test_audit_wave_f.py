"""Wave F of the 2026-10-29 audit remediation plan: lean (dead code out)."""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit


# ── E3 follow-up: the unused retry stacks are gone ──────────────────────────


@pytest.mark.parametrize(
    "module",
    ["orchestrator.retry_utils", "orchestrator.operations.retry_utils"],
)
def test_retry_utils_is_deleted(module) -> None:
    # The SDK retries (client.with_options(max_retries=...)); no second stack.
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_run_with_resilience_is_deleted() -> None:
    import orchestrator.operations.resilience as canonical
    import orchestrator.resilience as shim

    assert not hasattr(canonical, "run_with_resilience")
    assert not hasattr(shim, "run_with_resilience")


# ── F1: dead modules are deleted, with their re-export shims ────────────────


@pytest.mark.parametrize(
    "module",
    [
        "orchestrator.engine_core.engine_deps",
        "orchestrator.engine_core.evaluation",
        "orchestrator.engine_core.escalation",
        "orchestrator.costing.tracker",
        "orchestrator.costing.analytics",
        "orchestrator.cost_tracker",
        "orchestrator.cost_analytics",
        "orchestrator.routing",
        "orchestrator.model_routing",
    ],
)
def test_dead_module_is_deleted(module) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def test_container_has_no_dep_resolver_stub() -> None:
    # ProjectPlanner already falls back when dep_resolver is None.
    import inspect

    import orchestrator.engine_core.container as container

    assert "_DepResolver" not in inspect.getsource(container)


# ── F2: one validator filter, one engine method ─────────────────────────────


def test_engine_defines_filter_validators_once() -> None:
    import ast
    from pathlib import Path

    import orchestrator.engine as engine

    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    defs = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_filter_validators_for_task"
    ]
    assert len(defs) == 1


def test_task_validator_uses_the_quality_filter() -> None:
    from unittest.mock import MagicMock

    from orchestrator.engine_core.validator import TaskValidator
    from orchestrator.models import Task, TaskType
    from orchestrator.quality.validators import filter_validators_for_task

    task = Task(
        id="t",
        type=TaskType.CODE_GEN,
        prompt="Build a React form",
        target_path="src/Form.tsx",
        hard_validators=["python_syntax", "ruff", "json_schema"],
    )
    output = "export default function Form() {}"
    validator = TaskValidator.__new__(TaskValidator)
    assert validator.filter_validators_for_task(task, output) == filter_validators_for_task(
        task, output
    )
    assert "python_syntax" not in validator.filter_validators_for_task(task, output)
