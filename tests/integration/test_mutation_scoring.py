"""
Tests for E-4 mutation scoring — suite quality measurement.
============================================================
A strong suite kills mutants; a vacuous suite kills none. Integration
tests run real pytest through the sandbox.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_strong_suite_scores_high() -> None:
    """A suite asserting real behavior kills the sampled mutants."""
    from orchestrator.infrastructure.mutation_sampler import MutationSampler
    from orchestrator.infrastructure.test_runners import get_runner
    from orchestrator.infrastructure.workspace_materializer import WorkspaceMaterializer

    source = (
        "def classify(n):\n"
        "    if n < 0:\n        return 'neg'\n"
        "    if n == 0:\n        return 'zero'\n"
        "    return 'pos'\n"
    )
    tests = (
        "from main import classify\n"
        "def test_neg():\n    assert classify(-1) == 'neg'\n"
        "def test_zero():\n    assert classify(0) == 'zero'\n"
        "def test_pos():\n    assert classify(5) == 'pos'\n"
    )
    sampler = MutationSampler(max_mutations=8)
    score = await sampler.score(
        source,
        tests,
        runner=get_runner("pytest"),
        materializer=WorkspaceMaterializer(),
        max_total_s=90,
    )
    assert score.total > 0, "no mutants generated for a reachable implementation"
    assert score.score >= 0.5, f"strong suite scored too low: {score.score}"


@pytest.mark.asyncio
async def test_vacuous_suite_scores_zero() -> None:
    """A suite that never touches the implementation kills nothing (E-4)."""
    from orchestrator.infrastructure.mutation_sampler import MutationSampler
    from orchestrator.infrastructure.test_runners import get_runner
    from orchestrator.infrastructure.workspace_materializer import WorkspaceMaterializer

    source = "def add(a, b):\n    return a + b\n"
    tests = "def test_independent():\n    assert True\n"
    sampler = MutationSampler(max_mutations=6)
    score = await sampler.score(
        source,
        tests,
        runner=get_runner("pytest"),
        materializer=WorkspaceMaterializer(),
        max_total_s=60,
    )
    assert score.score == 0.0, f"vacuous suite must score 0, got {score.score}"


@pytest.mark.asyncio
async def test_mutation_score_attached_when_requested() -> None:
    """score_mutation=True populates the score on TestExecutionResult."""
    import os

    from orchestrator.infrastructure.sandboxes import SubprocessSandbox
    from orchestrator.testing.first_generator import TestFirstGenerator, TestingFramework
    from orchestrator.models import TaskType

    os.environ["ORCH_MUTATION"] = "report"
    gen = TestFirstGenerator(client=None, sandbox=SubprocessSandbox())
    gen._suite_timeout_s = 60

    result = await gen._run_tests_and_collect_results(
        test_code="from main import add\ndef test_add():\n    assert add(1, 2) == 3\n",
        implementation_code="def add(a, b):\n    return a + b\n",
        task_type=TaskType.CODE_GEN,
        framework=TestingFramework.PYTEST,
        score_mutation=True,
    )
    assert result.passed is True
    assert result.mutation_score is not None
    assert 0.0 <= result.mutation_score <= 1.0


@pytest.mark.asyncio
async def test_mutation_off_skips_scoring() -> None:
    """ORCH_MUTATION=off disables scoring entirely (zero extra runs)."""
    import os

    from orchestrator.infrastructure.sandboxes import SubprocessSandbox
    from orchestrator.testing.first_generator import TestFirstGenerator, TestingFramework
    from orchestrator.models import TaskType

    os.environ["ORCH_MUTATION"] = "off"
    gen = TestFirstGenerator(client=None, sandbox=SubprocessSandbox())
    gen._suite_timeout_s = 60
    result = await gen._run_tests_and_collect_results(
        test_code="from main import add\ndef test_add():\n    assert add(1, 2) == 3\n",
        implementation_code="def add(a, b):\n    return a + b\n",
        task_type=TaskType.CODE_GEN,
        framework=TestingFramework.PYTEST,
        score_mutation=True,
    )
    assert result.passed is True
    assert result.mutation_score is None
