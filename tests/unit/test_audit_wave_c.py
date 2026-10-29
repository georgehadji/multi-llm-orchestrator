"""
Audit Wave C — durability and resume
====================================
Regression tests for docs/plans/2026-10-29-audit-remediation-plan.md, Wave C.

C1: the project lock stored time.monotonic() (resets at boot) in a persistent DB with a
    30 s expiry that was never refreshed, and was only released in __aexit__, which the
    CLI never enters.
C2: state was saved only after every task finished, so a crash at task 9 of 10 left
    nothing to resume.

C3: only PARTIAL_SUCCESS runs were resumable, but runs halted by budget or time end as
    BUDGET_EXHAUSTED or TIMEOUT, so --resume re-decomposed them from scratch and
    find_resumable never offered them.
C4: resume copied the saved spend into the live Budget but returned (and saved) the old
    deserialized Budget, so the next resume saw stale spend and got fresh headroom.
C5: the engine called audit.flush() on an attribute it never set, AuditLog had no flush(),
    and flush_jsonl() never cleared, so policy-check records grew in memory and were lost.
C6: cache.db rows were never deleted and get() served responses of any age.
C7: IDE sessions were saved to disk but never loaded back after a restart, saves truncated
    the file before writing, and update_file reported success for a missing path.
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


async def test_c1_lease_outlives_the_lock_ttl_while_held(tmp_path) -> None:
    import asyncio

    from orchestrator.infrastructure.state import StateManager

    holder, rival = StateManager(db_path=tmp_path / "s.db"), StateManager(db_path=tmp_path / "s.db")
    holder._LOCK_TTL = rival._LOCK_TTL = 0.3

    async with holder.project_lease("p") as acquired:
        assert acquired
        await asyncio.sleep(0.8)  # well past the TTL; the heartbeat must keep it
        assert await rival.acquire_project_lock("p") is False

    assert await rival.acquire_project_lock("p") is True  # released on exit
    await holder.close()
    await rival.close()


async def test_c1_lock_left_by_a_previous_boot_is_stale(tmp_path) -> None:
    import os
    import time

    from orchestrator.infrastructure.state import StateManager

    sm = StateManager(db_path=tmp_path / "s.db")
    db = await sm._get_conn()
    # A crashed run before a reboot: its monotonic stamp is ahead of today's clock.
    await db.execute(
        "INSERT INTO project_locks (project_id, locked_at, locked_by) VALUES (?, ?, ?)",
        ("p", time.monotonic() + 3600, os.getpid()),
    )
    await db.commit()

    assert await sm.acquire_project_lock("p") is True
    await sm.close()


async def test_c1_run_project_releases_the_lock_without_a_context_manager(monkeypatch) -> None:
    from unittest.mock import AsyncMock

    from orchestrator.engine import Orchestrator

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test-key-for-verification-only")
    orch = Orchestrator()
    orch._project_runner.run_project = AsyncMock(return_value=None)

    await orch.run_project("build x", "it works", project_id="cli-run")

    assert await orch.state_mgr.acquire_project_lock("cli-run") is True
    await orch.state_mgr.release_project_lock("cli-run")
    await orch.close()


async def test_c2_pipeline_runner_persists_after_each_task() -> None:
    from orchestrator.engine_core.pipeline_runner import PipelineRunner
    from orchestrator.engine_core.project_planner import ProjectPlanner
    from orchestrator.models import Model, Task, TaskResult, TaskType

    saved: list[set[str]] = []

    async def persist(results):
        saved.append(set(results))

    async def run_task(task):
        if task.id == "t2":
            raise RuntimeError("process killed here")
        return TaskResult(task_id=task.id, output="x", score=1.0, model_used=Model.GPT_4O_MINI)

    tasks = {
        "t1": Task(id="t1", type=TaskType.CODE_GEN, prompt="p"),
        "t2": Task(id="t2", type=TaskType.CODE_GEN, prompt="p", dependencies=["t1"]),
    }
    runner = PipelineRunner(pipeline=None, planner=ProjectPlanner(), max_parallel_tasks=1)
    await runner.execute_all(
        tasks, ["t1", "t2"], "d", "c", execute_task_fn=run_task, persist_fn=persist
    )

    assert saved[0] == {"t1"}  # t1 was durable before t2 ever ran
    assert saved[-1] == {"t1", "t2"}


async def test_c2_project_runner_wires_a_partial_state_save(monkeypatch) -> None:
    from orchestrator.application.project_runner import ProjectRunner
    from orchestrator.application.project_runner_deps import ProjectRunState
    from orchestrator.budget import Budget
    from orchestrator.models import ProjectStatus

    monkeypatch.setenv("ORCH_UNATTENDED_GUARD", "false")
    captured = {}

    async def execute_all(*args, **kwargs):
        captured["persist_fn"] = kwargs.get("persist_fn")
        await kwargs["persist_fn"]({})
        return _state(ProjectStatus.SUCCESS)

    callables = MagicMock()
    callables.execute_all = execute_all
    callables.generate_architecture_rules = AsyncMock(return_value="")
    callables.make_state = MagicMock(return_value=_state(ProjectStatus.PARTIAL_SUCCESS))
    callables.determine_final_status = MagicMock(return_value=ProjectStatus.SUCCESS)
    state_mgr = AsyncMock()
    state_mgr.load_project = AsyncMock(return_value=None)
    generator = AsyncMock()
    generator.decompose = AsyncMock(
        return_value=MagicMock(succeeded=True, tasks={"t1": MagicMock()}, error="")
    )
    runner = ProjectRunner(
        callables=callables,
        run_state=ProjectRunState(entered=True),
        state_mgr=state_mgr,
        budget=Budget(),
        event_bus=None,
        resumption_svc=AsyncMock(),
        dashboard_bridge=MagicMock(),
        git_bridge=MagicMock(),
        generator=generator,
        meta_v2=None,
        cache=AsyncMock(),
        api_health={},
    )

    await runner.run_project(project_description="d", success_criteria="c", project_id="p1")

    assert captured["persist_fn"] is not None
    first_save = state_mgr.save_project.await_args_list[0]
    assert first_save.args[0] == "p1"


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


async def test_c7_ide_session_survives_a_restart(tmp_path) -> None:
    from orchestrator.ide_backend.session_manager import SessionManager

    created = await SessionManager(tmp_path).create_session(project_name="demo", owner_id="u1")

    restarted = SessionManager(tmp_path)
    session = await restarted.get_session(created.id)

    assert session is not None and session.project_name == "demo"
    # Once loaded it is cached, so later mutations find it too.
    assert await restarted.update_session(created.id, status="active") is not None


async def test_c7_ide_session_save_never_leaves_a_truncated_file(tmp_path, monkeypatch) -> None:
    import json

    from orchestrator.ide_backend import session_manager as sm_mod

    manager = sm_mod.SessionManager(tmp_path)
    session = await manager.create_session(project_name="before", owner_id="u1")
    path = tmp_path / f"{session.id}.json"

    def crash(*args, **kwargs):
        raise OSError("disk full mid-write")

    monkeypatch.setattr(sm_mod.json, "dump", crash)
    session.project_name = "after"
    await manager._save_session(session)

    assert json.loads(path.read_text(encoding="utf-8"))["project"]["name"] == "before"


async def test_c7_update_file_reports_a_missing_path(tmp_path) -> None:
    from orchestrator.ide_backend.session_manager import SessionManager

    manager = SessionManager(tmp_path)
    session = await manager.create_session(owner_id="u1")

    assert await manager.update_file(session.id, "does/not/exist.py", "x") is False


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
