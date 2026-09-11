"""
T4 (Concurrency & resource lifecycle) — regression guards.

Each test fires ONE audited defect from the V3 tier-4 audit
(docs/audits/v3/T4/). Every test was verified RED against the pre-fix tree
and traced to its own defect's mechanism before the matching fix was written.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── T4B4-01 — _rule_based_selection builds an enum from a list index ─────────
@pytest.mark.unit
def test_t4b4_01_rule_based_fallback_does_not_raise_on_unmapped_key() -> None:
    """ComplexityLevel/RiskLevel are str-enums; constructing them from an
    integer list index raises ValueError instead of stepping down a level.

    Violated property: _rule_based_selection's own contract — it returns a
    list[ReasoningMethod] for every (task_type, complexity, risk) triple.
    """
    from orchestrator.engine_core.method_selector import (
        METHOD_SELECTION_RULES,
        ComplexityLevel,
        MethodSelector,
        RiskLevel,
    )
    from orchestrator.models import TaskType

    # Find a triple genuinely absent from the rules table so the step-down
    # fallback is the path under test.
    missing = [
        (tt, c, r)
        for tt in TaskType
        for c in ComplexityLevel
        for r in RiskLevel
        if (tt, c, r) not in METHOD_SELECTION_RULES and c is not ComplexityLevel.LOW
    ]
    assert missing, "no unmapped key exists; this test no longer exercises the fallback"
    task_type, complexity, risk = missing[0]

    selector = MethodSelector()
    methods = selector._rule_based_selection(task_type, complexity, risk)
    assert isinstance(methods, list) and methods, (
        f"expected a non-empty list of ReasoningMethod for "
        f"({task_type}, {complexity}, {risk}), got {methods!r}"
    )


# ── T4B4-02 — a gate-aborted pipeline is reported COMPLETED ──────────────────
@pytest.mark.unit
async def test_t4b4_02_aborted_pipeline_is_not_reported_completed() -> None:
    """ConstitutionGate sets ctx.should_abort=True, but PipelineExecutor only
    maps abort_reason values starting with 'stage_error' to FAILED — so a
    deliberately blocked task is delivered as COMPLETED.

    Violated property: a task whose pipeline was aborted by a delivery gate
    must not be reported as successfully completed.
    """
    from orchestrator.engine_core.pipeline_executor import PipelineExecutor
    from orchestrator.models import Model, Task, TaskStatus, TaskType

    task = Task(
        id="t1",
        type=TaskType.CODE_GEN,
        prompt="write something the constitution forbids",
        preferred_model=Model.GPT_4O_MINI,
    )

    class AbortingPipeline:
        async def run(self, ctx):
            ctx.output = "forbidden output"
            ctx.score = 0.99  # high score: only the abort should matter
            ctx.should_abort = True
            ctx.abort_reason = "constitution: no network calls"
            return ctx

    executor = PipelineExecutor(pipeline=AbortingPipeline(), selector=None)
    result = await executor.execute(task)

    assert result.status is not TaskStatus.COMPLETED, (
        "gate-aborted task was reported COMPLETED "
        f"(status={result.status}, output={result.output!r})"
    )


# ── T4B4-03 — the app's own modules are written into requirements.txt ────────
@pytest.mark.unit
def test_t4b4_03_local_modules_are_not_written_to_requirements(tmp_path: Path) -> None:
    """_is_third_party only excludes stdlib and a 7-name blocklist, so a
    generated app's own sibling modules are emitted as PyPI requirements.

    Violated property: a name written into the delivered requirements.txt must
    be an installable third-party distribution, not a local module.
    """
    from orchestrator.engine_core.dep_resolver import DependencyResolver

    (tmp_path / "routers").mkdir()
    (tmp_path / "routers" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "schemas.py").write_text("X = 1\n", encoding="utf-8")
    (tmp_path / "main.py").write_text(
        "import requests\nfrom routers import api\nimport schemas\n",
        encoding="utf-8",
    )

    DependencyResolver().resolve(tmp_path)
    written = (tmp_path / "requirements.txt").read_text(encoding="utf-8").split()

    assert any(w.startswith("requests") for w in written), f"real dependency dropped: {written}"
    assert "routers" not in written, f"local package emitted as a requirement: {written}"
    assert "schemas" not in written, f"local module emitted as a requirement: {written}"


# ── T4B7-01 — a successful probe locks the breaker HALF_OPEN forever ─────────
@pytest.mark.unit
async def test_t4b7_01_breaker_recovers_after_successful_probes() -> None:
    """record_success() in HALF_OPEN keeps probe_in_flight=True until
    success_threshold is met, but check() rejects every caller while that flag
    is set — so with the default success_threshold=2 no second probe can ever
    run and the breaker never closes.

    Violated property: module docstring — 'HALF_OPEN -> one probe call allowed
    to test recovery', and a recovered dependency must re-close the breaker.
    """
    from orchestrator.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(name="t4b7-01", failure_threshold=1, reset_timeout=0.0, success_threshold=2)
    await cb.record_failure(RuntimeError("boom"))  # -> OPEN

    # First probe is admitted and succeeds.
    await cb.check()
    await cb.record_success()

    # The dependency has recovered; a second probe must be admissible so the
    # breaker can reach success_threshold and close.
    await cb.check()
    await cb.record_success()

    assert cb.is_closed, f"breaker never recovered; state={cb.state}"


# ── T4B8-01 — events are written but can never be read back ──────────────────
@pytest.mark.unit
async def test_t4b8_01_async_event_store_can_replay_what_it_persisted(
    tmp_path: Path,
) -> None:
    """events/async_event_store.py imports '.unified_events.core' — one dot
    short of orchestrator.unified_events — and swallows the ImportError, so
    HAS_UNIFIED_EVENTS is pinned False and every read path returns [].

    Violated property: a value this store persists must be readable back by
    this store.
    """
    from orchestrator.events import async_event_store as aes
    from orchestrator.unified_events.core import DomainEvent, EventType

    store = aes.AsyncEventStore(db_path=tmp_path / "events.db")
    event = DomainEvent(
        event_type=EventType.TASK_COMPLETED,
        aggregate_id="proj-1",
        metadata={"task_id": "t1"},
    )
    await store.append(event)
    replayed = await store.get_aggregate("proj-1")
    await store.close()

    assert len(replayed) == 1, (
        f"store persisted the event but replayed {len(replayed)} "
        f"(HAS_UNIFIED_EVENTS={aes.HAS_UNIFIED_EVENTS})"
    )


# ── T4B9-01 — Experiment.to_dict() silently drops every recorded outcome ─────
@pytest.mark.unit
def test_t4b9_01_experiment_roundtrip_preserves_outcomes() -> None:
    """to_dict() persists control_count/treatment_count but never emits
    'outcomes', while from_dict() reads data['outcomes'] — so every recorded
    A/B outcome is destroyed at process exit and the reloaded experiment
    reports counts it has no data for.

    Violated property: a value this module writes must be readable back by it.
    """
    from orchestrator.events.ab_testing import Experiment, ExperimentOutcome, Variant

    proposal = MagicMock()
    proposal.to_dict.return_value = {}

    exp = Experiment(
        experiment_id="e1",
        proposal=proposal,
        traffic_split=0.5,
        min_samples=10,
        significance_level=0.05,
        start_time=0.0,
    )
    exp.outcomes.append(
        ExperimentOutcome(
            outcome_id="o1",
            experiment_id="e1",
            variant=Variant.TREATMENT,
            project_id="p1",
            success=True,
            score=0.9,
            cost_usd=0.01,
            latency_ms=120.0,
        )
    )
    exp.treatment_count = 1

    payload = exp.to_dict()
    assert "outcomes" in payload, f"to_dict() dropped outcomes entirely: {sorted(payload)}"
    assert len(payload["outcomes"]) == 1, f"outcomes not persisted: {payload['outcomes']!r}"


# ── T4B1-01 — concurrent run_project calls erase each other's results ────────
@pytest.mark.unit
async def test_t4b1_01_concurrent_runs_do_not_erase_each_others_results() -> None:
    """run_project()/run_project_with_tasks() call _run_ctx.reset(), which does
    results.clear(), outside any lock (only run_job takes _job_lock). api_server
    shares one long-running Orchestrator across background per-request runs with
    distinct project_ids, so the advisory per-project lock does not serialise
    them and the second run wipes the first run's results mid-flight.

    Violated property: a run's accumulated results belong to that run.
    """
    from orchestrator.engine import Orchestrator
    from orchestrator.models import Budget, Model, TaskResult, TaskStatus, TaskType

    orch = Orchestrator(budget=Budget(max_usd=1.0, max_time_seconds=60))
    orch.state_mgr = None  # no advisory project lock; isolates the results race

    observed: dict[str, int] = {}

    class RecordingRunner:
        async def run_project(self, *, project_id: str, **kwargs):
            orch.results[f"{project_id}-task"] = TaskResult(
                task_id=f"{project_id}-task",
                output="x",
                score=1.0,
                model_used=Model.GPT_4O_MINI,
                status=TaskStatus.COMPLETED,
                task_type=TaskType.CODE_GEN.value,
            )
            await asyncio.sleep(0.05)  # let the sibling run start and reset()
            # Count only THIS run's own result: a bare len() cannot tell
            # "my entry survived" from "my entry was wiped and replaced".
            observed[project_id] = int(f"{project_id}-task" in orch.results)
            return MagicMock()

    orch._project_runner = RecordingRunner()

    await asyncio.gather(
        orch.run_project("build a", "done", project_id="alpha"),
        orch.run_project("build b", "done", project_id="beta"),
    )

    assert observed == {"alpha": 1, "beta": 1}, (
        "a concurrent run erased results belonging to another run "
        f"(1 = own result survived, 0 = erased): {observed}"
    )


# ── T4B7-02 — the container never gets a real event bus ─────────────────────
@pytest.mark.unit
def test_t4b7_02_container_wires_the_real_unified_event_bus() -> None:
    """container.py imported '.unified_events.core' from inside engine_core —
    one dot short of orchestrator.unified_events — and the ImportError was
    swallowed, so every container fell back to NullEventBus/NullHookRegistry
    and the whole unified event system was dead.

    Violated property: container.py's own comment — 'event_bus -> async face
    (publish DomainEvents)'; a null adapter is the documented *fallback*, not
    the normal path.
    """
    from unittest.mock import AsyncMock

    from orchestrator.engine_core.container import ServiceContainer
    from orchestrator.models import Budget
    from orchestrator.unified_events.core import UnifiedEventBus

    container = ServiceContainer.build(
        budget=Budget(max_usd=1.0, max_time_seconds=60),
        cache=AsyncMock(),
        state_manager=AsyncMock(),
    )
    assert isinstance(
        container.event_bus, UnifiedEventBus
    ), f"container fell back to {type(container.event_bus).__name__}"


# ── T4B2-03 — shutdown() probes for close(), the bus only has stop() ────────
@pytest.mark.unit
async def test_t4b2_03_shutdown_stops_the_event_bus_loop() -> None:
    """ServiceContainer.shutdown() guards on hasattr(event_bus, 'close'), but
    UnifiedEventBus exposes stop() — so the guard is always False, shutdown
    silently skips the bus, and its _process_loop task outlives the container.

    Violated property: shutdown()'s own docstring — 'Release all
    container-managed resources.'
    """
    from unittest.mock import AsyncMock

    from orchestrator.engine_core.container import ServiceContainer
    from orchestrator.models import Budget
    from orchestrator.unified_events.core import UnifiedEventBus

    container = ServiceContainer.build(
        budget=Budget(max_usd=1.0, max_time_seconds=60),
        cache=AsyncMock(),
        state_manager=AsyncMock(),
    )
    bus = container.event_bus
    assert isinstance(bus, UnifiedEventBus), "precondition: real bus required"

    await bus.start()
    assert bus._process_task is not None and not bus._process_task.done()

    await container.shutdown()
    await asyncio.sleep(0)

    assert bus._process_task.done(), (
        "shutdown() left the event bus processing task running — it leaks for "
        "the lifetime of the process"
    )
