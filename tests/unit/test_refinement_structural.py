"""
Tests for the structural refinement tier (E-11).
==================================================
Pure AST finders, the three structural operators, and the service's
zero-model-call guarantee on a healthy workspace.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from orchestrator.application.refinement.operators._finders import (
    find_deepest_nesting_function,
    find_duplicate_pair,
    find_longest_function,
)
from orchestrator.application.refinement.operators.deduplicate import DeduplicateOperator
from orchestrator.application.refinement.operators.extract_function import (
    ExtractFunctionOperator,
)
from orchestrator.application.refinement.operators.flatten_nesting import (
    FlattenNestingOperator,
)
from orchestrator.application.refinement.operators.llm_refactor import LLMRefactorCommand
from orchestrator.application.refinement.service import RefinementService
from orchestrator.domain.refinement import MetricSnapshot, RefinementTier
from orchestrator.domain.testing_models import SuiteReport, Workspace

# ── fakes ────────────────────────────────────────────────────────────────────


@dataclass
class _FakeResponse:
    text: str
    cost_usd: float = 0.0


class _FakeClient:
    """Records calls; returns a scripted response (or raises)."""

    def __init__(self, response_text: str | None = None, raise_on_call: bool = False) -> None:
        self.response_text = response_text
        self.raise_on_call = raise_on_call
        self.calls = 0

    async def call(self, model, prompt, system=None, max_tokens=4096, temperature=0.7, **kwargs):
        self.calls += 1
        if self.raise_on_call:
            raise RuntimeError("client must not be called")
        return _FakeResponse(text=self.response_text or "")


class _RaisingClient:
    """A client that fails the test the moment it is invoked (zero-cost assertion)."""

    async def call(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("LLM client called when it should not have been")


@dataclass
class _FakeRunner:
    """TestExecutorPort fake — always green (entry gate only)."""

    async def run(self, workspace, selection=None, *, timeout_s=120.0) -> SuiteReport:
        return SuiteReport(passed=True, exit_code=0)

    def supports(self, framework: str) -> bool:
        return True


HEALTHY_SNAPSHOT = MetricSnapshot(
    cyclomatic_mean=1.0,
    cyclomatic_max=1,
    max_nesting_depth=1,
    longest_function_lines=5,
    duplicated_blocks=0,
    total_lines=10,
)


async def _healthy_collector(workspace: Workspace) -> MetricSnapshot:
    return HEALTHY_SNAPSHOT


# ── finders ──────────────────────────────────────────────────────────────────


@pytest.mark.unit
class TestFinders:
    def test_find_longest_function(self) -> None:
        src = "def a():\n    return 1\n\n\ndef b():\n" + "    x = 1\n" * 60 + "    return x\n"
        found = find_longest_function(src)
        assert found is not None
        assert found.name == "b"
        assert found.line_count > 50

    def test_find_longest_function_no_functions(self) -> None:
        assert find_longest_function("x = 1\n") is None

    def test_find_longest_function_syntax_error(self) -> None:
        assert find_longest_function("def broken(:\n") is None

    def test_find_deepest_nesting(self) -> None:
        src = (
            "def deep(x):\n"
            "    if x:\n"
            "        if x:\n"
            "            if x:\n"
            "                if x:\n"
            "                    if x:\n"
            "                        return 1\n"
            "    return 0\n"
        )
        found = find_deepest_nesting_function(src)
        assert found is not None
        assert found.name == "deep"
        assert found.metric_value == 5

    def test_find_deepest_nesting_flat_function(self) -> None:
        assert find_deepest_nesting_function("def flat():\n    return 1\n") is None

    def test_find_duplicate_pair_detected(self) -> None:
        src = (
            "def add_one(x):\n    y = x + 1\n    return y\n\n\n"
            "def add_two(z):\n    w = z + 1\n    return w\n"
        )
        pair = find_duplicate_pair(src)
        assert pair is not None
        assert {pair[0].name, pair[1].name} == {"add_one", "add_two"}

    def test_find_duplicate_pair_none_when_distinct(self) -> None:
        src = "def a():\n    return 1\n\n\ndef b():\n    return [1, 2, 3]\n"
        assert find_duplicate_pair(src) is None


# ── operators: applicable() gating ───────────────────────────────────────────


@pytest.mark.unit
class TestApplicableGating:
    def test_extract_function_gated_by_threshold(self) -> None:
        op = ExtractFunctionOperator(client=_RaisingClient(), model="m")
        assert op.applicable(HEALTHY_SNAPSHOT) is False
        over = MetricSnapshot(
            cyclomatic_mean=1.0,
            cyclomatic_max=1,
            max_nesting_depth=1,
            longest_function_lines=51,
            duplicated_blocks=0,
        )
        assert op.applicable(over) is True

    def test_flatten_nesting_gated_by_threshold(self) -> None:
        op = FlattenNestingOperator(client=_RaisingClient(), model="m")
        assert op.applicable(HEALTHY_SNAPSHOT) is False
        over = MetricSnapshot(
            cyclomatic_mean=1.0,
            cyclomatic_max=1,
            max_nesting_depth=5,
            longest_function_lines=5,
            duplicated_blocks=0,
        )
        assert op.applicable(over) is True

    def test_deduplicate_gated_by_threshold(self) -> None:
        op = DeduplicateOperator(client=_RaisingClient(), model="m")
        assert op.applicable(HEALTHY_SNAPSHOT) is False
        over = MetricSnapshot(
            cyclomatic_mean=1.0,
            cyclomatic_max=1,
            max_nesting_depth=1,
            longest_function_lines=5,
            duplicated_blocks=1,
        )
        assert op.applicable(over) is True


# ── operators: propose() ─────────────────────────────────────────────────────


LONG_FUNCTION_SOURCE = "def do_it():\n" + "    x = 1\n" * 60 + "    return x\n"
REWRITTEN_SOURCE = "def do_it():\n    return _helper()\n\n\ndef _helper():\n    return 1\n"


@pytest.mark.unit
class TestExtractFunctionPropose:
    def test_proposes_candidate_with_payload(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text(LONG_FUNCTION_SOURCE, encoding="utf-8")
        ws = Workspace(root=tmp_path, framework="pytest")
        client = _FakeClient(response_text=REWRITTEN_SOURCE)
        op = ExtractFunctionOperator(client=client, model="m")

        candidates = asyncio.run(op.propose(ws, HEALTHY_SNAPSHOT))

        assert len(candidates) == 1
        assert candidates[0].target_file == "svc.py"
        assert candidates[0].payload == REWRITTEN_SOURCE
        assert candidates[0].predicted_metric == "longest_function_lines"
        assert "do_it" in candidates[0].rationale
        assert client.calls == 1

    def test_no_functions_no_candidate_no_call(self, tmp_path: Path) -> None:
        (tmp_path / "empty.py").write_text("x = 1\n", encoding="utf-8")
        ws = Workspace(root=tmp_path, framework="pytest")
        client = _RaisingClient()
        op = ExtractFunctionOperator(client=client, model="m")

        candidates = asyncio.run(op.propose(ws, HEALTHY_SNAPSHOT))
        assert candidates == []

    def test_invalid_llm_output_rejected(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text(LONG_FUNCTION_SOURCE, encoding="utf-8")
        ws = Workspace(root=tmp_path, framework="pytest")
        client = _FakeClient(response_text="not python at all (((")
        op = ExtractFunctionOperator(client=client, model="m")

        candidates = asyncio.run(op.propose(ws, HEALTHY_SNAPSHOT))
        assert candidates == []

    def test_client_exception_yields_no_candidate(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text(LONG_FUNCTION_SOURCE, encoding="utf-8")
        ws = Workspace(root=tmp_path, framework="pytest")
        client = _FakeClient(raise_on_call=True)
        op = ExtractFunctionOperator(client=client, model="m")

        candidates = asyncio.run(op.propose(ws, HEALTHY_SNAPSHOT))
        assert candidates == []


# ── operators: command_for() ─────────────────────────────────────────────────


@pytest.mark.unit
class TestCommandFor:
    def test_command_for_builds_llm_refactor_command(self, tmp_path: Path) -> None:
        (tmp_path / "svc.py").write_text(LONG_FUNCTION_SOURCE, encoding="utf-8")
        ws = Workspace(root=tmp_path, framework="pytest")
        client = _FakeClient(response_text=REWRITTEN_SOURCE)
        op = ExtractFunctionOperator(client=client, model="m")

        candidates = asyncio.run(op.propose(ws, HEALTHY_SNAPSHOT))
        command = op.command_for(candidates[0])

        assert isinstance(command, LLMRefactorCommand)
        changed = command.apply(tmp_path)
        assert changed == ["svc.py"]
        assert (tmp_path / "svc.py").read_text(encoding="utf-8") == REWRITTEN_SOURCE

    def test_command_for_returns_none_for_foreign_candidate(self) -> None:
        from orchestrator.domain.refinement import RefinementCandidate

        op = ExtractFunctionOperator(client=_RaisingClient(), model="m")
        foreign = RefinementCandidate(
            operator="flatten_nesting",
            tier=RefinementTier.STRUCTURAL,
            target_file="x.py",
            rationale="",
            diff="",
            predicted_metric="max_nesting_depth",
            payload="def x(): pass\n",
        )
        assert op.command_for(foreign) is None

    def test_command_for_returns_none_without_payload(self) -> None:
        from orchestrator.domain.refinement import RefinementCandidate

        op = ExtractFunctionOperator(client=_RaisingClient(), model="m")
        empty_payload = RefinementCandidate(
            operator="extract_function",
            tier=RefinementTier.STRUCTURAL,
            target_file="x.py",
            rationale="",
            diff="",
            predicted_metric="longest_function_lines",
        )
        assert op.command_for(empty_payload) is None


# ── service: zero-cost on a healthy workspace ────────────────────────────────


@pytest.mark.unit
class TestServiceZeroCostOnHealthyWorkspace:
    def test_full_tier_never_calls_llm_when_snapshot_is_healthy(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ORCH_REFINE", "full")
        (tmp_path / "svc.py").write_text("def ok():\n    return 1\n", encoding="utf-8")
        ws = Workspace(root=tmp_path, framework="pytest")

        service = RefinementService(runner=_FakeRunner(), collect_snapshot=_healthy_collector)
        service.register_operator(ExtractFunctionOperator(client=_RaisingClient(), model="m"))
        service.register_operator(FlattenNestingOperator(client=_RaisingClient(), model="m"))
        service.register_operator(DeduplicateOperator(client=_RaisingClient(), model="m"))

        receipt = asyncio.run(service.refine(ws, mutation_score=1.0))

        assert receipt.started is True
        assert receipt.accepted == 0
        assert receipt.rejected == 0
