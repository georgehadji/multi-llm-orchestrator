"""
Audit Wave C — durability and resume
====================================
Regression tests for docs/plans/2026-10-29-audit-remediation-plan.md, Wave C.

C3: only PARTIAL_SUCCESS runs were resumable, but runs halted by budget or time end as
    BUDGET_EXHAUSTED or TIMEOUT, so --resume re-decomposed them from scratch and
    find_resumable never offered them.
C4: resume copied the saved spend into the live Budget but returned (and saved) the old
    deserialized Budget, so the next resume saw stale spend and got fresh headroom.
C5: the engine called audit.flush() on an attribute it never set, AuditLog had no flush(),
    and flush_jsonl() never cleared, so policy-check records grew in memory and were lost.
C6: cache.db rows were never deleted and get() served responses of any age.
C8: APIServer kept every project's status dict in memory for the life of the process.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit


def _state(status, description="build a weather dashboard"):
    from orchestrator.budget import Budget
    from orchestrator.models import ProjectState

    return ProjectState(
        project_description=description,
        success_criteria="c",
        budget=Budget(),
        status=status,
    )


async def test_c3_find_resumable_offers_runs_halted_by_time_or_budget(tmp_path) -> None:
    from orchestrator.infrastructure.state import StateManager
    from orchestrator.models import ProjectStatus

    sm = StateManager(db_path=tmp_path / "state.db")
    await sm.save_project("timed-out", _state(ProjectStatus.TIMEOUT))
    await sm.save_project("broke", _state(ProjectStatus.BUDGET_EXHAUSTED))
    await sm.save_project("done", _state(ProjectStatus.SUCCESS))

    found = {row["project_id"] for row in await sm.find_resumable([])}
    await sm.close()

    assert found == {"timed-out", "broke"}


async def test_c3_resume_gate_resumes_a_timed_out_run(monkeypatch) -> None:
    from orchestrator.application.project_runner import ProjectRunner
    from orchestrator.application.project_runner_deps import ProjectRunState
    from orchestrator.budget import Budget
    from orchestrator.models import ProjectStatus

    monkeypatch.setenv("ORCH_UNATTENDED_GUARD", "false")
    halted = _state(ProjectStatus.TIMEOUT)
    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=halted)
    resumption = AsyncMock()
    resumption.resume = AsyncMock(return_value=halted)
    runner = ProjectRunner(
        callables=MagicMock(),
        run_state=ProjectRunState(entered=True),
        state_mgr=state_mgr,
        budget=Budget(),
        event_bus=None,
        resumption_svc=resumption,
        dashboard_bridge=MagicMock(),
        git_bridge=MagicMock(),
        generator=AsyncMock(),
        meta_v2=None,
        cache=AsyncMock(),
        api_health={},
    )

    await runner.run_project(project_description="d", success_criteria="c", project_id="p1")

    resumption.resume.assert_awaited_once()


async def test_c4_resumed_state_carries_the_live_budget() -> None:
    from orchestrator.application.resumption_service import ResumptionService
    from orchestrator.budget import Budget
    from orchestrator.models import Model, ProjectStatus, Task, TaskResult, TaskType

    live = Budget(max_usd=5.0)
    saved = _state(ProjectStatus.PARTIAL_SUCCESS)
    saved.budget.spent_usd = 4.0
    saved.tasks = {"t1": Task(id="t1", type=TaskType.CODE_GEN, prompt="p")}
    saved.execution_order = ["t1"]

    async def run_task(task, policy=None):
        await live.charge(1.0, "generation")
        return TaskResult(task_id=task.id, output="x", score=1.0, model_used=Model.GPT_4O_MINI)

    svc = ResumptionService(live, {}, run_task, lambda s: ProjectStatus.SUCCESS)
    resumed = await svc.resume(saved)

    assert resumed.budget is live
    assert resumed.budget.spent_usd == pytest.approx(5.0)


def _record(log, task_id="t1"):
    log.record(task_id, "m", True, True, [], "monitor", ["p"])


async def test_c5_audit_flush_writes_pending_records_once(tmp_path) -> None:
    from orchestrator.infrastructure.audit import AuditLog

    log = AuditLog()
    path = tmp_path / "audit" / "policy.jsonl"
    for i in range(3):
        _record(log, f"t{i}")

    await log.flush(path)
    await log.flush(path)

    assert len(path.read_text(encoding="utf-8").splitlines()) == 3
    assert len(log) == 0


async def test_c5_orchestrator_close_flushes_the_policy_audit_log(tmp_path, monkeypatch) -> None:
    from orchestrator.crosscutting.config import settings
    from orchestrator.engine import Orchestrator

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key-for-verification-only")
    path = tmp_path / "audit.log"
    monkeypatch.setattr(settings, "audit_log_path", str(path))
    orch = Orchestrator()
    _record(orch._policy_engine.audit_log)

    await orch.close()

    assert len(path.read_text(encoding="utf-8").splitlines()) == 1


async def _age_all_rows(cache, days):
    import time

    db = await cache._get_conn()
    await db.execute("UPDATE cache SET created_at = ?", (time.time() - days * 86400,))
    await db.commit()


async def test_c6_cache_does_not_serve_entries_older_than_30_days(tmp_path) -> None:
    from orchestrator.infrastructure.cache import DiskCache

    cache = DiskCache(tmp_path / "cache.db")
    await cache.put("m", "p", 10, "fresh answer")
    await _age_all_rows(cache, 31)

    assert await cache.get("m", "p", 10) is None
    await cache.close()


async def test_c6_cache_prunes_expired_rows_on_open(tmp_path) -> None:
    from orchestrator.infrastructure.cache import DiskCache

    path = tmp_path / "cache.db"
    cache = DiskCache(path)
    await cache.put("m", "old", 10, "old answer")
    await _age_all_rows(cache, 31)
    await cache.put("m", "new", 10, "new answer")
    await cache.close()

    reopened = DiskCache(path)
    db = await reopened._get_conn()
    async with db.execute("SELECT COUNT(*) FROM cache") as cur:
        (rows,) = await cur.fetchone()
    await reopened.close()

    assert rows == 1


async def test_c8_api_server_keeps_only_recent_finished_project_states() -> None:
    from orchestrator.api_server import APIServer

    server = APIServer(port=0, host="127.0.0.1", auth_required=False)
    server._MAX_FINISHED_STATES = 2

    async def run():
        return None

    for i in range(4):
        pid = f"p{i}"
        server._active_project_state[pid] = {"project_id": pid, "status": "accepted"}
        orch = MagicMock(close=AsyncMock())
        orch.state_mgr.close = AsyncMock()
        orch.cache.close = AsyncMock()
        await (await server._launch_background_project(pid, orch=orch, run_fn=run))

    assert list(server._active_project_state) == ["p2", "p3"]
