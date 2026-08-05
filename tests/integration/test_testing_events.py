"""
Tests for E-8: testing telemetry events.
========================================
Verifies each event builder emits the expected payload with counts/hashes
only (no artifact content), and that the gate stamps test-execution
metadata on the ExecutionReceipt.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from orchestrator.application.testing import events
from orchestrator.domain.testing_models import IsolationLevel, Workspace
from orchestrator.domain.verification import CheckOutcome
from orchestrator.infrastructure.verification_checks import _make_test_execution_check
from orchestrator.application.verification_gate import VerificationGate
from orchestrator.unified_events.core import EventType


@pytest.mark.unit
class TestEventBuilders:
    """Payloads carry counts/hashes, never artifact content."""

    def test_suite_generated_payload(self) -> None:
        e = events.suite_generated("task:t1", test_count=3, vacuity_rate=0.33, mode="warn")
        assert e.event_type is EventType.TEST_SUITE_GENERATED
        assert e.aggregate_id == "task:t1"
        assert e.metadata["test_count"] == 3
        assert e.metadata["vacuity_rate"] == 0.33
        assert "test_code" not in e.metadata

    def test_suite_executed_payload(self) -> None:
        e = events.suite_executed(
            "task:t1",
            passed=True,
            executed=4,
            passed_count=4,
            failed_count=0,
            skipped_count=0,
            duration_ms=12.5,
            isolation="subprocess",
            flaky_count=1,
            mutation_score=0.8,
        )
        assert e.event_type is EventType.TEST_SUITE_EXECUTED
        assert e.metadata["isolation"] == "subprocess"
        assert e.metadata["flaky_count"] == 1
        assert e.metadata["mutation_score"] == 0.8

    def test_repair_iteration_payload_has_no_pii(self) -> None:
        e = events.repair_iteration(
            "task:t1",
            iteration=2,
            signature="AssertionError @ test_main.py::test_x",
            tier="fake/a",
            plateau=True,
        )
        assert e.event_type is EventType.TEST_REPAIR_ITERATION
        assert e.metadata["plateau"] is True
        assert len(e.metadata["signature"]) <= 200

    def test_mutation_scored_payload(self) -> None:
        e = events.mutation_scored("task:t1", total=10, killed=7, score=0.7)
        assert e.event_type is EventType.TEST_MUTATION_SCORED
        assert e.metadata["survived"] == 3

    def test_flake_quarantined_hashes_node_id(self) -> None:
        e = events.flake_quarantined(
            "task:t1", node_id="test_main.py::test_flaky", observation_count=3
        )
        assert e.event_type is EventType.TEST_FLAKE_QUARANTINED
        # Only a hash of the node id — no test content leaks.
        assert "test_main.py" not in str(e.metadata)
        assert len(e.metadata["node_id_hash"]) == 16


@pytest.mark.integration
class TestGateReceiptMetadata:
    """E-8: ExecutionReceipt carries isolation/executed/mutation/flake."""

    @pytest.mark.asyncio
    async def test_receipt_stamped_with_suite_metadata(self, tmp_path: Path) -> None:
        (tmp_path / "main.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (tmp_path / "test_main.py").write_text(
            "from main import add\ndef test_add():\n    assert add(1, 2) == 3\n",
            encoding="utf-8",
        )
        ws = Workspace(
            root=tmp_path,
            framework="pytest",
            source_files=(tmp_path / "main.py",),
            test_files=(tmp_path / "test_main.py",),
            env={"PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
        )
        gate = VerificationGate(checks=[_make_test_execution_check()])
        result = await gate.run("", workspace=ws)
        receipt = next(r for r in result.receipts if r.check_name == "test_execution")
        assert receipt.outcome is CheckOutcome.PASSED
        assert receipt.executed == 1
        assert receipt.isolation == IsolationLevel.SUBPROCESS.value
        assert receipt.flaky_count == 0
