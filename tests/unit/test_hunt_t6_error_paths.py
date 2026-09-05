"""
T6 (error-path sweep) proof-of-defect and no-regression tests.

Three VERIFIED DEFECTs from docs/hunts/t6-error-paths/inventory.md, all the
same shape: a broad `except Exception` in a money-relevant path silently
swallowed the failure with zero logging, making a real error indistinguishable
from "nothing went wrong" to anyone reading logs after the fact.

C1 — application/verbalized_sampling.py::VerbalizedSampler.sample() charged
     the caller-supplied budget for a real, already-incurred LLM cost, but a
     failure in that charge call (e.g. Budget.charge raising) was swallowed
     with a bare `except Exception: pass` — no log, so the cost silently
     never gets recorded against the budget with no trace it happened.
C2 — costing/tracker.py::CostTracker._load() silently reset the tracker's
     cumulative cost history to empty on any read/parse failure of its
     persisted JSON file, with zero logging distinguishing "no history yet"
     from "history file is corrupt, lost silently."
C3 — cost.py::BudgetHierarchy._load_from_db() silently dropped any spend
     record whose JSON value failed to parse, with zero logging identifying
     which key was skipped — a team/job's persisted spend could silently
     reset to 0 with no way to tell from the logs.
"""

from __future__ import annotations

import json
import logging
import sqlite3

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


class _FakeResponse:
    def __init__(self, text: str, cost_usd: float) -> None:
        self.text = text
        self.cost_usd = cost_usd


class _FakeClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def call(self, **kwargs):
        return self._response


class _RaisingBudget:
    async def charge(self, amount: float, label: str) -> None:
        raise RuntimeError("budget backend unavailable")


@pytest.mark.asyncio
async def test_c1_verbalized_sampler_logs_budget_charge_failure(caplog):
    from orchestrator.application.verbalized_sampling import VerbalizedSampler
    from orchestrator.models import VSConfig

    response = _FakeResponse(text='[{"text": "a", "probability": 0.5}]', cost_usd=1.23)
    sampler = VerbalizedSampler(client=_FakeClient(response), budget=_RaisingBudget())

    with caplog.at_level(logging.WARNING):
        await sampler.sample(prompt="test", model=None, cfg=VSConfig(k=1))

    assert any(
        "charge" in rec.message.lower() and "1.23" in rec.message for rec in caplog.records
    ), f"expected a warning naming the failed charge and its cost, got: {[r.message for r in caplog.records]}"


# --- C2 -----------------------------------------------------------------


def test_c2_cost_tracker_logs_load_failure(tmp_path, caplog):
    from orchestrator.costing.tracker import CostTracker

    corrupt_file = tmp_path / "cost_tracker.json"
    corrupt_file.write_text("{not valid json", encoding="utf-8")

    with caplog.at_level(logging.WARNING):
        tracker = CostTracker(storage_dir=str(tmp_path))

    assert tracker._cumulative == {}
    assert any(
        "cost_tracker.json" in rec.message or "failed to load" in rec.message.lower()
        for rec in caplog.records
    ), f"expected a warning naming the load failure, got: {[r.message for r in caplog.records]}"


# --- C3 -----------------------------------------------------------------


def test_c3_budget_hierarchy_logs_unparseable_spend_row(tmp_path, caplog):
    from orchestrator.cost import BudgetHierarchy

    db_path = tmp_path / "budget_hierarchy.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE IF NOT EXISTS budget_hierarchy (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO budget_hierarchy (key, value) VALUES (?, ?)",
        ("team:alpha", "{not valid json"),
    )
    conn.execute(
        "INSERT INTO budget_hierarchy (key, value) VALUES (?, ?)",
        ("team:beta", json.dumps(4.5)),
    )
    conn.commit()
    conn.close()

    with caplog.at_level(logging.WARNING):
        hierarchy = BudgetHierarchy(org_max_usd=100.0, db_path=str(db_path))

    assert hierarchy._team_spent.get("beta") == 4.5
    assert "alpha" not in hierarchy._team_spent
    assert any(
        "team:alpha" in rec.message for rec in caplog.records
    ), f"expected a warning naming the unparseable key, got: {[r.message for r in caplog.records]}"
