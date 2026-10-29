"""Audit remediation plan items B6 (per-call reservations) and B7 (one price table)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator.budget import Budget
from orchestrator.models import Model

pytestmark = pytest.mark.unit


def _stage(client, budget):
    from orchestrator.engine_core.stages.generate import GenerateStage

    return GenerateStage(client=client, budget=budget, selector=MagicMock())


def _ctx(task_id: str):
    from orchestrator.engine_core.pipeline import PipelineContext
    from orchestrator.models import Task, TaskType

    task = Task(id=task_id, type=TaskType.CODE_GEN, prompt="p", preferred_model=Model.GPT_4O)
    return PipelineContext(task=task, tokens_used={"input": 0, "output": 0})


# ── B6: each generation call reserves its estimate first ────────────────────


class TestPerCallReservation:
    @pytest.mark.asyncio
    async def test_parallel_tasks_cannot_overrun_the_cap(self, monkeypatch) -> None:
        from orchestrator.api_clients import APIResponse
        from orchestrator.crosscutting.config import flags

        monkeypatch.setattr(flags, "vs_generate", False, raising=False)
        probe = _stage(MagicMock(), Budget(max_usd=100.0))
        est = probe._estimate_call_cost(Model.GPT_4O, _ctx("x").task, "p", "s")
        assert est > 0

        async def call(**_):
            await asyncio.sleep(0.01)
            return APIResponse(text="ok", model=Model.GPT_4O, cost_usd=est)

        client = MagicMock(call=AsyncMock(side_effect=call))
        budget = Budget(max_usd=est * 1.5)
        stage = _stage(client, budget)
        stage._estimate_call_cost = lambda *a, **k: est

        results = await asyncio.gather(*(stage.process(_ctx(f"t{i}")) for i in range(3)))

        assert client.call.await_count == 1
        assert sorted(r.abort_reason for r in results) == ["", "budget_exhausted", "budget_exhausted"]
        assert budget.spent_usd == pytest.approx(est)
        assert budget._reserved_usd == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_reservation_released_when_the_call_fails(self, monkeypatch) -> None:
        from orchestrator.crosscutting.config import flags

        monkeypatch.setattr(flags, "vs_generate", False, raising=False)
        client = MagicMock(call=AsyncMock(side_effect=RuntimeError("provider down")))
        budget = Budget(max_usd=10.0)
        with pytest.raises(RuntimeError):
            await _stage(client, budget).process(_ctx("t"))
        assert budget._reserved_usd == pytest.approx(0.0)
        assert budget.spent_usd == 0.0

    def test_shortcut_note_is_gone(self) -> None:
        import inspect

        import orchestrator.engine_core.stages.generate as generate

        assert "shortcut:" not in inspect.getsource(generate)


# ── B7: COST_TABLE is the only price table ──────────────────────────────────


class TestOnePriceTable:
    def test_costs_json_is_gone(self) -> None:
        from pathlib import Path

        import orchestrator

        assert not (Path(orchestrator.__file__).parent / "config" / "costs.json").exists()

    def test_config_port_has_no_get_costs(self) -> None:
        from orchestrator.domain.ports import ConfigPort, NullConfig
        from orchestrator.infrastructure.adapters.config_adapter import JsonConfigAdapter

        for cls in (ConfigPort, NullConfig, JsonConfigAdapter):
            assert not hasattr(cls, "get_costs"), cls

    def test_no_lazy_cost_table_builder(self) -> None:
        import orchestrator.models as models

        assert not hasattr(models, "_build_cost_table")
        assert "COST_TABLE" not in models._LAZY_TABLES
