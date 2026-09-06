"""
T11 (infrastructure/engine_core/domain/events core) proof-of-defect and
no-regression tests.

Three VERIFIED DEFECTs from docs/hunts/t11-core-architecture/inventory.md:

C1 — config/costs.json was missing an entry for qwen/qwen3.6-flash (a real,
     active Model enum member with correct pricing already in
     models.py::COST_TABLE) — infrastructure/llm_client.py::UnifiedClient's
     real-response cost computation reads costs.json via CostService, which
     silently defaults to {"input": 0.0, "output": 0.0} for any missing
     key. Every real, non-cached call routed to this model computed and
     reported exactly $0.00 cost, silently, against a real provider key.
C2 — engine_core/project_planner.py::ProjectPlanner.get_execution_levels()
     (the live task-scheduling method, called directly by
     engine_core/pipeline_runner.py's real execution loop) had no
     completeness check: a circular dependency among tasks silently
     dropped the cyclic tasks from every returned level, with no log, no
     error, and no field naming which tasks were lost.
C3 — engine_core/pipeline_executor.py's self-consistency/ARA retry loop
     only recognized "retry_for_quality"/"ara_retry" as retry signals.
     engine_core/stages/self_consistency.py's own VS tail-escape feature
     sets ctx.abort_reason = "vs_retry_escape" to request exactly this
     kind of retry, but the loop treated that value as terminal and
     returned immediately — the retry the stage had just configured
     (a revision_context prompt for verbalized-sampling tail exploration)
     never actually ran.
"""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit


# --- C1 -----------------------------------------------------------------


def test_c1_qwen_3_6_flash_has_nonzero_cost_entry():
    from orchestrator.domain.services.config_services import CostService
    from orchestrator.infrastructure.adapters.config_adapter import JsonConfigAdapter
    from orchestrator.models import Model

    cost = CostService(JsonConfigAdapter()).get_cost(Model.QWEN_3_6_FLASH)

    assert cost["input"] > 0.0, f"qwen/qwen3.6-flash input cost silently defaulted to $0: {cost}"
    assert cost["output"] > 0.0, f"qwen/qwen3.6-flash output cost silently defaulted to $0: {cost}"


# --- C2 -----------------------------------------------------------------


def test_c2_circular_dependency_is_logged_not_silent(caplog):
    from orchestrator.engine_core.project_planner import ProjectPlanner
    from orchestrator.models import Task, TaskType

    # a -> b -> a: a genuine cycle, neither task ever reaches in_degree == 0.
    tasks = {
        "a": Task(id="a", type=TaskType.CODE_GEN, prompt="a", dependencies=["b"]),
        "b": Task(id="b", type=TaskType.CODE_GEN, prompt="b", dependencies=["a"]),
        "c": Task(id="c", type=TaskType.CODE_GEN, prompt="c", dependencies=[]),
    }

    planner = ProjectPlanner()
    with caplog.at_level(logging.ERROR):
        levels = planner.get_execution_levels(tasks)

    scheduled = {tid for level in levels for tid in level}
    assert scheduled == {"c"}, "the acyclic task must still be scheduled normally"
    assert any(
        "a" in rec.message and "b" in rec.message and "circular" in rec.message.lower()
        for rec in caplog.records
    ), f"expected an error naming the dropped cyclic tasks, got: {[r.message for r in caplog.records]}"


# --- C3 -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_c3_pipeline_executor_retries_on_vs_retry_escape():
    from orchestrator.engine_core.pipeline_executor import PipelineExecutor
    from orchestrator.models import Task, TaskStatus, TaskType

    task = Task(
        id="test-vs-escape",
        type=TaskType.CODE_GEN,
        prompt="build a widget",
        acceptance_threshold=0.7,
    )

    call_count = {"n": 0}

    async def mock_run(ctx):
        call_count["n"] += 1
        if call_count["n"] == 1:
            ctx.score = 0.3
            ctx.abort_reason = "vs_retry_escape"
            return ctx
        ctx.output = "def hello(): pass"
        ctx.score = 0.85
        ctx.tokens_used = {"input": 100, "output": 50}
        ctx.cost_usd = 0.005
        ctx.abort_reason = ""
        ctx.should_abort = True
        return ctx

    pipeline = MagicMock()
    pipeline.run = AsyncMock(side_effect=mock_run)
    selector = MagicMock()
    selector.select = MagicMock(return_value=None)

    executor = PipelineExecutor(pipeline=pipeline, selector=selector, background_tasks=set())
    result = await executor.execute(task)

    assert call_count["n"] == 2, (
        "expected the pipeline to be re-run once after a vs_retry_escape signal, "
        f"but it ran {call_count['n']} time(s)"
    )
    assert result.status == TaskStatus.COMPLETED
