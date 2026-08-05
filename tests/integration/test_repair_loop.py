"""
Integration tests for the E-3 repair loop (bounded, escalating).
================================================================
Uses a fake LLM client so no API keys are needed. The fake returns the
same broken implementation every time, so the loop must detect the plateau,
escalate exactly once, and stop with a diagnostic artifact.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from orchestrator.infrastructure.sandboxes import SubprocessSandbox
from orchestrator.testing.first_generator import TestFirstGenerator, TestingFramework


class _StubbornClient:
    """Always returns the same broken implementation."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def call(self, model, prompt, system, max_tokens, temperature, timeout):
        self.calls.append(getattr(model, "value", str(model)))
        return SimpleNamespace(
            text="def add(a, b):\n    return a - b  # wrong on purpose\n",
            cost_usd=0.0,
        )


@pytest.mark.integration
class TestRepairLoop:
    """The repair loop must plateau-detect, escalate once, then fail."""

    @pytest.mark.asyncio
    async def test_plateau_escalates_once_then_fails(self) -> None:
        client = _StubbornClient()
        gen = TestFirstGenerator(client=client, sandbox=SubprocessSandbox())
        gen._suite_timeout_s = 60

        tests = "from main import add\ndef test_add():\n    assert add(1, 2) == 3\n"
        broken = "def add(a, b):\n    return a - b\n"
        result, test_result, iterations = await gen._repair_to_pass_tests(
            tests=tests,
            implementation=broken,
            errors=["test_main.py::test_add: assert -1 == 3"],
            requirement="add",
            model="fake/model-a",
            framework=TestingFramework.PYTEST,
        )
        # Bounded: 2 calls at tier1 + 1 at tier2 (escalated) = 3 LLM calls max.
        assert len(client.calls) <= 3, f"unbounded repair: {client.calls}"
        assert test_result.passed is False
        # Diagnostic artifact: plateau reason recorded in errors.
        assert any(
            "plateau" in e or "escalat" in e or "collection" in e for e in test_result.errors
        )

    @pytest.mark.asyncio
    async def test_repair_response_cannot_modify_tests(self) -> None:
        """The hash lock is structural: tests passed to the runner are the
        exact bytes locked at entry (E-3)."""
        from orchestrator.application.testing.repair_policy import RepairPolicy

        policy = RepairPolicy()
        locked = "from main import add\ndef test_add():\n    assert add(1, 2) == 3\n"
        policy.lock_tests(locked)
        # Any attempt to run with modified tests must be rejected.
        with pytest.raises(Exception):
            policy.assert_tests_locked(locked + "def test_weakened():\n    assert True\n")

    @pytest.mark.asyncio
    async def test_collection_error_stops_repair(self) -> None:
        """A repair iteration that breaks collection (0 tests run) stops the
        loop — an LLM cannot fix a missing import/syntax at module level."""

        class _CollectionBreakingClient:
            """Always returns code that does NOT define the tested symbol."""

            def __init__(self) -> None:
                self.calls: list[str] = []

            async def call(self, model, prompt, system, max_tokens, temperature, timeout):
                self.calls.append(getattr(model, "value", str(model)))
                return SimpleNamespace(
                    text="def unrelated():\n    return 0\n",
                    cost_usd=0.0,
                )

        client = _CollectionBreakingClient()
        gen = TestFirstGenerator(client=client, sandbox=SubprocessSandbox())
        gen._suite_timeout_s = 60

        tests = "from main import add\ndef test_add():\n    assert add(1, 2) == 3\n"
        # The client returns code WITHOUT 'add' -> import error -> 0 tests run.
        result, test_result, iterations = await gen._repair_to_pass_tests(
            tests=tests,
            implementation="def other():\n    return 1\n",
            errors=["ImportError: cannot import name 'add'"],
            requirement="add",
            model="fake/model-a",
            framework=TestingFramework.PYTEST,
        )
        assert test_result.passed is False
        assert len(client.calls) == 1, "collection error must stop the loop after one attempt"
        assert any("collection" in e for e in test_result.errors)
