"""
Integration tests for the refinement service (E-10/E-11).
==========================================================
Real suite execution through the sandbox. Verifies the hard entry gate,
the zero-LLM mechanical tier, byte-exact revert on rejection, and the
acceptance chain behavior.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from orchestrator.application.refinement.operators.dead_code import DeadCodeOperator
from orchestrator.application.refinement.operators.formatter import FormatterOperator
from orchestrator.application.refinement.service import RefinementService
from orchestrator.domain.refinement import RefinementTier
from orchestrator.domain.testing_models import SuiteReport, Workspace
from orchestrator.infrastructure.metrics.collector import collect_snapshot
from orchestrator.infrastructure.sandboxes import SubprocessSandbox
from orchestrator.infrastructure.test_runners import get_runner
from orchestrator.infrastructure.workspace_materializer import WorkspaceMaterializer

ENV = {"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}


def _hash_tree(root: Path) -> str:
    """SHA-256 of all file contents (byte-exactness check)."""
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and "__pycache__" not in str(p) and ".ruff" not in p.name:
            h.update(p.relative_to(root).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def _make_service() -> RefinementService:
    runner = get_runner("pytest", sandbox=SubprocessSandbox())
    service = RefinementService(
        runner=runner,
        materializer=WorkspaceMaterializer(),
        collect_snapshot=collect_snapshot,
    )
    service.register_operator(DeadCodeOperator())
    service.register_operator(FormatterOperator())
    return service


@pytest.mark.integration
class TestRefinementServiceMechanical:
    """E-10: deterministic tier, zero LLM calls, byte-exact revert."""

    @pytest.mark.asyncio
    async def test_happy_path_removes_unused_import(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text(
            "import os\nimport json\n\ndef add(a, b):\n    return a + b\n",
            encoding="utf-8",
        )
        (tmp_path / "test_main.py").write_text(
            "from main import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
            encoding="utf-8",
        )
        ws = Workspace(root=tmp_path, framework="pytest", env=ENV)
        service = _make_service()
        green = SuiteReport(passed=True, exit_code=0, isolation="subprocess")

        receipt = await service.refine(ws, green, mutation_score=0.8)
        assert receipt.started is True
        assert receipt.accepted >= 1
        assert receipt.model_calls == 0  # mechanical tier is zero-LLM
        cleaned = (tmp_path / "main.py").read_text(encoding="utf-8")
        # Both `os` and `json` are unused (the test imports only `add`).
        assert "import" not in cleaned
        assert "def add" in cleaned

        # The suite still passes after the change.
        report = await get_runner("pytest", sandbox=SubprocessSandbox()).run(ws, timeout_s=60)
        assert report.passed is True

    @pytest.mark.asyncio
    async def test_entry_gate_blocks_when_suite_fails(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("import os\nx = 1\n", encoding="utf-8")
        (tmp_path / "test_main.py").write_text("def test_bad():\n    assert False\n")
        ws = Workspace(root=tmp_path, framework="pytest", env=ENV)
        service = _make_service()
        failing = SuiteReport(passed=False, exit_code=1, isolation="subprocess")
        receipt = await service.refine(ws, failing, mutation_score=0.8)
        assert receipt.started is False
        assert "not green" in receipt.reason
        # Nothing changed.
        assert "import os" in (tmp_path / "main.py").read_text(encoding="utf-8")

    @pytest.mark.asyncio
    async def test_entry_gate_blocks_below_mutation_floor(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("import os\nx = 1\n", encoding="utf-8")
        (tmp_path / "test_main.py").write_text("def test_x():\n    assert 1\n")
        ws = Workspace(root=tmp_path, framework="pytest", env=ENV)
        service = _make_service()
        green = SuiteReport(passed=True, exit_code=0, isolation="subprocess")
        receipt = await service.refine(ws, green, mutation_score=0.2)  # below 0.6
        assert receipt.started is False
        assert "mutation" in receipt.reason

    @pytest.mark.asyncio
    async def test_clean_workspace_zero_cost(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (tmp_path / "test_main.py").write_text(
            "from main import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        ws = Workspace(root=tmp_path, framework="pytest", env=ENV)
        service = _make_service()
        green = SuiteReport(passed=True, exit_code=0, isolation="subprocess")
        receipt = await service.refine(ws, green, mutation_score=0.9)
        assert receipt.accepted == 0
        assert receipt.rejected == 0
        assert (tmp_path / "main.py").read_text(encoding="utf-8") == (
            "def add(a, b):\n    return a + b\n"
        )

    @pytest.mark.asyncio
    async def test_revert_is_byte_exact_when_suite_breaks(self, tmp_path: Path) -> None:
        """A candidate that breaks the suite is reverted; the workspace hash
        matches the pre-candidate snapshot exactly."""
        (tmp_path / "main.py").write_text(
            "import os\n\ndef add(a, b):\n    return a + b\n", encoding="utf-8"
        )
        (tmp_path / "test_main.py").write_text(
            "from main import add, os\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        ws = Workspace(root=tmp_path, framework="pytest", env=ENV)
        service = _make_service()
        green = SuiteReport(passed=True, exit_code=0, isolation="subprocess")
        before_hash = _hash_tree(tmp_path)

        # The test imports `os` from main, so removing the `import os` from
        # main.py BREAKS the suite -> the candidate must be reverted.
        receipt = await service.refine(ws, green, mutation_score=0.8)
        assert receipt.rejected >= 1
        assert _hash_tree(tmp_path) == before_hash, "revert must be byte-exact"


@pytest.mark.integration
class TestRefinementServiceStructuralGate:
    """E-11: structural operators are gated behind ORCH_REFINE=structural."""

    @pytest.mark.asyncio
    async def test_mechanical_mode_skips_structural(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("ORCH_REFINE", "mechanical")
        from orchestrator.application.refinement.operators.base import RefinementOperator
        from orchestrator.domain.refinement import MetricSnapshot, RefinementCandidate

        class _FakeStructural(RefinementOperator):
            name = "structural_fake"
            tier = RefinementTier.STRUCTURAL
            called = False

            def applicable(self, snapshot: MetricSnapshot) -> bool:
                return True

            async def propose(self, workspace, snapshot):
                _FakeStructural.called = True
                return [
                    RefinementCandidate(
                        operator=self.name,
                        tier=self.tier,
                        target_file="main.py",
                        rationale="x",
                        diff="-",
                        predicted_metric="cyclomatic_max",
                    )
                ]

        (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n")
        (tmp_path / "test_main.py").write_text(
            "from main import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
        )
        ws = Workspace(root=tmp_path, framework="pytest", env=ENV)
        service = _make_service()
        service.register_operator(_FakeStructural())
        green = SuiteReport(passed=True, exit_code=0, isolation="subprocess")
        await service.refine(ws, green, mutation_score=0.8)
        assert (
            _FakeStructural.called is False
        ), "structural operator must not run in mechanical mode"
