"""
Integration tests for the RED-gate execution path (E-2).
=========================================================
Runs a real suite against the NotImplementedError stub through the sandbox.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_genuine_suite_survives_red_gate() -> None:
    """A suite asserting real implementation behavior is not vacuous."""
    from orchestrator.infrastructure.sandboxes import SubprocessSandbox
    from orchestrator.testing.first_generator import TestFirstGenerator, TestingFramework

    gen = TestFirstGenerator(client=None, sandbox=SubprocessSandbox())
    gen._suite_timeout_s = 60
    gen._red_gate_mode = "enforce"

    genuine = "from main import add\ndef test_add():\n    assert add(1, 2) == 3\n"
    result = await gen._run_red_gate(genuine, TestingFramework.PYTEST)
    assert result.decision == "proceed"
    assert result.vacuous_node_ids == ()
    assert result.total == 1


@pytest.mark.asyncio
async def test_vacuous_suite_regenerates_in_enforce() -> None:
    """A suite that passes without an implementation is vacuous (E-2)."""
    from orchestrator.infrastructure.sandboxes import SubprocessSandbox
    from orchestrator.testing.first_generator import TestFirstGenerator, TestingFramework

    gen = TestFirstGenerator(client=None, sandbox=SubprocessSandbox())
    gen._suite_timeout_s = 60
    gen._red_gate_mode = "enforce"

    vacuous = "def test_independent():\n    assert True\n"
    result = await gen._run_red_gate(vacuous, TestingFramework.PYTEST)
    assert result.decision == "regenerate"
    assert result.vacuous_node_ids == ("test_main.py::test_independent",)
    assert result.vacuous_ratio == 1.0


@pytest.mark.asyncio
async def test_warn_mode_measures_but_never_blocks() -> None:
    """Default mode: vacuity is logged, never acted on."""
    from orchestrator.infrastructure.sandboxes import SubprocessSandbox
    from orchestrator.testing.first_generator import TestFirstGenerator, TestingFramework

    gen = TestFirstGenerator(client=None, sandbox=SubprocessSandbox())
    gen._suite_timeout_s = 60
    gen._red_gate_mode = "warn"

    vacuous = "def test_independent():\n    assert True\n"
    result = await gen._run_red_gate(vacuous, TestingFramework.PYTEST)
    assert result.decision == "proceed"
    assert result.discard_node_ids == ()
    assert result.diagnosis
