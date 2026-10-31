"""TestFirstGenerator's sandbox path runs through the runner port.

It used to call ``sandbox.execute(...)``, a DockerSandbox-only method the
injected SubprocessSandbox does not have, so every sandboxed TDD run failed.
"""

from __future__ import annotations

import pytest

from orchestrator.models import TaskType
from orchestrator.testing.first_generator import (
    TestFirstGenerator,
    TestingFramework,
    _failure_detail,
)

pytestmark = pytest.mark.unit


def test_failure_detail_keeps_assert_lines_and_drops_paths() -> None:
    message = (
        "def test_add():\n"
        ">       assert add(1, 2) == 3\n"
        "E       assert -1 == 3\n"
        "E        +  where -1 = add(1, 2)\n"
        "\n"
        "C:\\Users\\x\\AppData\\Local\\Temp\\ws_ab12\\test_main.py:3: AssertionError\n"
    )
    detail = _failure_detail(message)
    assert detail == "assert -1 == 3; +  where -1 = add(1, 2)"
    assert "Temp" not in detail


def test_failure_detail_falls_back_to_last_line() -> None:
    assert _failure_detail("Traceback...\nImportError: no add") == "ImportError: no add"
    assert _failure_detail("") == ""


@pytest.mark.asyncio
async def test_sandbox_path_never_calls_execute() -> None:
    class _ExecOnlySandbox:
        level = None

        async def execute(self, *a, **k):  # pragma: no cover - must not run
            raise AssertionError("sandbox.execute is not the SandboxPort API")

    from orchestrator.domain.testing_models import SuiteTools

    tools = SuiteTools(runner_factory=None, materializer=None)
    gen = TestFirstGenerator(client=None, sandbox=_ExecOnlySandbox(), suite_tools=tools)
    result = await gen._run_tests_and_collect_results(
        test_code="",
        implementation_code="",
        task_type=TaskType.CODE_GEN,
        framework=TestingFramework.VITEST,
    )
    assert result.passed is False
    assert result.errors == ["No sandboxed test runner for framework vitest"]


@pytest.mark.asyncio
async def test_sandbox_without_suite_tools_refuses_instead_of_running_locally() -> None:
    gen = TestFirstGenerator(client=None, sandbox=object())
    result = await gen._run_tests_and_collect_results(
        test_code="def test_x():\n    assert True\n",
        implementation_code="",
        task_type=TaskType.CODE_GEN,
    )
    assert result.passed is False
    assert result.tests_run == 0
    assert "suite_tools" in result.errors[0]
