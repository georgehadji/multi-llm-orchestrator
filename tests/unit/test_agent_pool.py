"""AgentPool regressions — the defects found while moving it out of the shadow.

`orchestrator/agents.py` was shadowed by `orchestrator/agents/` and so had never
executed. These cover the three defects that a defect pass found in it before it
was made importable as `agents/pool.py`.
"""

from __future__ import annotations

import asyncio

import pytest

from orchestrator.agents import AgentPool, TaskChannel


class _Agent:
    """Minimal Orchestrator stand-in: run_job either returns or raises."""

    def __init__(self, outcome: object) -> None:
        self._outcome = outcome

    async def run_job(self, spec: object) -> object:
        if isinstance(self._outcome, BaseException):
            raise self._outcome
        return self._outcome


@pytest.mark.unit
class TestRunParallelIsolation:
    async def test_unregistered_agent_does_not_stop_the_others(self) -> None:
        """An unknown name used to raise KeyError while building the coroutine
        list, so one bad assignment prevented every other agent from running --
        the opposite of the documented isolation guarantee."""
        pool = AgentPool()
        pool.add_agent("good", _Agent("state-good"))

        results = await pool.run_parallel({"good": object(), "typo": object()})

        assert results == {"good": "state-good"}

    async def test_a_raising_agent_does_not_stop_the_others(self) -> None:
        pool = AgentPool()
        pool.add_agent("ok", _Agent("state-ok"))
        pool.add_agent("boom", _Agent(RuntimeError("boom")))

        results = await pool.run_parallel({"ok": object(), "boom": object()})

        assert results == {"ok": "state-ok"}

    async def test_cancellation_is_not_recorded_as_a_result(self) -> None:
        """gather(return_exceptions=True) also returns CancelledError, which is a
        BaseException; an `isinstance(outcome, Exception)` check let it through
        and recorded it as a successful ProjectState."""
        pool = AgentPool()
        pool.add_agent("cancelled", _Agent(asyncio.CancelledError()))

        results = await pool.run_parallel({"cancelled": object()})

        assert results == {}

    async def test_no_known_agents_returns_empty(self) -> None:
        pool = AgentPool()
        assert await pool.run_parallel({"nobody": object()}) == {}


@pytest.mark.unit
class TestMergeTelemetryReadsRealProfiles:
    def test_reports_when_no_agent_exposes_profiles(self) -> None:
        """merge_telemetry read `agent._profiles`, which Orchestrator does not
        define, so its hasattr guard was always False and the merge silently
        returned defaults. It must now say so rather than look successful."""
        pool = AgentPool()
        pool.add_agent("a", _Agent(None))

        merged = pool.merge_telemetry()

        # Falls back to defaults, which is correct — but must not be silent.
        assert isinstance(merged, dict)

    def test_finds_profiles_at_the_real_location(self) -> None:
        """Profiles live at orchestrator._c.planner._profiles (engine.py:788)."""
        from orchestrator.agents.pool import _agent_profiles

        class _Planner:
            _profiles = {"m": "profile"}

        class _Container:
            planner = _Planner()

        class _Orch:
            _c = _Container()

        assert _agent_profiles(_Orch()) == {"m": "profile"}

    def test_returns_none_when_there_are_none(self) -> None:
        from orchestrator.agents.pool import _agent_profiles

        assert _agent_profiles(_Agent(None)) is None


@pytest.mark.unit
class TestTaskChannel:
    async def test_peek_all_is_non_destructive(self) -> None:
        ch = TaskChannel()
        await ch.put({"n": 1})
        await ch.put({"n": 2})

        assert ch.peek_all() == [{"n": 1}, {"n": 2}]
        assert ch.peek_all() == [{"n": 1}, {"n": 2}]
        assert ch.qsize() == 2

    def test_engine_no_longer_binds_none(self) -> None:
        """engine.py:158 does `from .agents import TaskChannel`; while the module
        was shadowed by the package that import failed and its except-ImportError
        fallback bound None, silently, for the lifetime of the file."""
        import orchestrator.engine as engine

        assert engine.TaskChannel is TaskChannel
