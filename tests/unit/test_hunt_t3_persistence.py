"""
V3 precision-defect-audit campaign — Tier T3 (persistence & resume).
docs/audits/v3/T3/batch1-core_state_checkpoint_resume.md (state/checkpoint/resume primitives)
docs/audits/v3/T3/batch2-workspace_watcher_telemetry.md (workspace/session_watcher/telemetry)

Sole home for T3's regression tests, organized by finding ID.

Four tests below are rewritten from the audit's own proposed regression test
because the original did not actually discriminate pre-fix from post-fix
behavior (caught during RED verification, same discipline as T2's rounds):
  - SESSION-1: the original monkeypatched builtins.open to raise BEFORE any
    real truncation occurred, so the original bug (truncate-then-crash) was
    never actually reproduced. Rewritten to let the real open() truncate the
    file, then fail the first .write() call after that.
  - T3B1-09: the original never injected a failure at all -- two ordinary
    sequential capture() calls followed by a reload pass identically whether
    or not the fix is applied. Rewritten to truncate the on-disk file mid
    "write" the way write_text() itself would if killed partway through.
  - T3B1-05: the original used a Barrier + real threads with no forced delay
    inside the actual check-then-act window (there is nothing to inject a
    sleep into between "if self._initialized" and "self._initialized = True"),
    so it does not reliably force the race. Rewritten to hold the shared lock
    externally and prove __init__ now blocks on it (it does not, pre-fix).
  - WORKSPACE-1: the original monkeypatched the `datetime` module's own
    `datetime` attribute, but workspace.py does `from datetime import
    datetime`, which already bound its own module-level name at import time
    -- patching the source module doesn't touch that binding. Rewritten to
    patch orchestrator.state_mgmt.workspace's own `datetime` name directly.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


# ════════════════════════════════════════════════════════════════════════
# Batch 1 — core state / checkpoint / resume primitives
# ════════════════════════════════════════════════════════════════════════


# ── T3B1-02: _recency_factor missing upper clamp ──────────────────────────


def test_t3b1_02_recency_factor_clamped_for_future_timestamp() -> None:
    """Fires T3B1-02 without the fix; passes with it. Violated property:
    _recency_factor's own documented contract, 'Returns: Recency score
    between 0.0 and 1.0' -- a project timestamped later than `now` (clock
    skew / NTP step) must not push the score above 1.0."""
    from orchestrator.state_mgmt.resume_detector import _recency_factor

    now = 1_000_000.0
    future_created_at = now + 10 * 86400  # 10 days "in the future" of `now`
    score = _recency_factor(future_created_at, reference_time=now)
    assert 0.0 <= score <= 1.0, f"recency_score {score} escaped documented [0.0, 1.0] range"


# ── T3B1-03: find_resumable_project lacks error handling ─────────────────


class _BadRowStateManager:
    async def find_resumable(self, keywords):
        return [
            {
                "project_id": "p1",
                "description": "a project",
                "keywords": ["a", "project"],
                "updated_at": "not-a-number",
            }
        ]


@pytest.mark.asyncio
async def test_t3b1_03_find_resumable_project_survives_malformed_updated_at() -> None:
    """Fires T3B1-03 without the fix; passes with it. Violated property:
    find_resumable_project()'s own docstring contract ('None otherwise')
    must hold even when a row's updated_at is not a parseable number,
    mirroring cli_dispatch.py::_check_resume()'s proven-live guard."""
    from orchestrator.state_mgmt.resume_detector import ResumeDetector

    detector = ResumeDetector(state_manager=_BadRowStateManager())
    result = await detector.find_resumable_project("a project", "some criteria")
    assert result is None or isinstance(result, dict)


# ── T3B1-04: CapabilityEvent writes a timestamp get_stats() can't parse ──


def test_t3b1_04_capability_event_timestamp_round_trips_through_get_stats_parsing() -> None:
    """Fires T3B1-04 without the fix; passes with it. Violated property:
    a value CapabilityEvent.create() writes must be parseable by this same
    module's own get_stats() reader (the exact expression it uses)."""
    from datetime import datetime

    from orchestrator.state_mgmt.capability_logger import CapabilityEvent, CapabilityType

    event = CapabilityEvent.create(capability=CapabilityType.TASK_STARTED)
    line = json.dumps(event.to_dict(), default=str)
    event_data = json.loads(line)

    try:
        datetime.fromisoformat(event_data["timestamp"].replace("Z", "+00:00")).timestamp()
    except ValueError:
        pytest.fail(
            "get_stats()'s own parse expression could not round-trip a timestamp "
            "produced by CapabilityEvent.create() in the same module"
        )


# ── T3B1-05: CapabilityLogger's __init__ re-entry guard is unsynchronized ─


def test_t3b1_05_init_reentry_is_lock_guarded(tmp_path: Path) -> None:
    """Fires T3B1-05 without the fix; passes with it. Violated property:
    CapabilityLogger's own docstring claim, 'Thread-safe logger' -- the
    __init__ re-entry guard/mutation must be serialized on the same lock
    __new__ already trusts, not run unguarded. Verified deterministically by
    holding the shared lock externally and confirming a forced re-entry into
    __init__ blocks on it rather than proceeding."""
    from orchestrator.state_mgmt import capability_logger as cl_module

    cl_module.CapabilityLogger._instance = None
    inst = cl_module.CapabilityLogger(log_dir=str(tmp_path))
    inst._initialized = False  # force a re-entry through __init__'s guarded body

    finished = threading.Event()

    with cl_module.CapabilityLogger._lock:

        def construct() -> None:
            cl_module.CapabilityLogger(log_dir=str(tmp_path))
            finished.set()

        t = threading.Thread(target=construct)
        t.start()
        finished_while_locked = finished.wait(timeout=0.3)

    t.join(timeout=2)
    assert not finished_while_locked, (
        "defect still present: __init__ mutated shared state while "
        "type(self)._lock was held externally -- it is not lock-guarded"
    )
    assert finished.wait(timeout=2), "construction never completed after the lock was released"


# ── T3B1-06: CheckpointManager filename collision + no corrupt-fallback ──


@pytest.mark.asyncio
async def test_t3b1_06_load_checkpoint_falls_back_past_a_corrupt_latest_file(
    tmp_path: Path,
) -> None:
    """Fires T3B1-06(b) without the fix; passes with it. Violated property:
    a corrupt newest checkpoint file must not hide older, valid history for
    the same task_id that is still on disk."""
    from orchestrator.checkpoints import CheckpointManager

    mgr = CheckpointManager(checkpoint_dir=str(tmp_path))
    await mgr.save_checkpoint({"step": 1}, "task-x")

    bad_file = tmp_path / "checkpoint_task-x_99999999_235959.json"
    bad_file.write_text('{"task_id": "task-x", "data": {"step": 2}, "timesta', encoding="utf-8")

    result = await mgr.load_checkpoint("task-x")
    assert result is not None, "valid older checkpoint was hidden by a corrupt newer one"
    assert result.data == {"step": 1}


# ── T3B1-07: silent partial-failure signaling in Named/ContentCheckpointManager ──


class _FailingSnapshotStore:
    async def create(self, label, source_dir, metadata):
        return "snap-1"

    async def restore(self, snapshot_id, output_dir):
        return False  # simulates a failed content restore


@pytest.mark.asyncio
async def test_t3b1_07_rollback_signals_failed_content_restore(tmp_path: Path) -> None:
    """Fires T3B1-07(b) without the fix; passes with it. Violated property:
    rollback()'s own docstring promise that it 'actually restores' file
    contents -- a caller must be able to tell a failed restore from a
    successful one, not receive the same truthy result for both."""
    from orchestrator.checkpoints import ContentCheckpointManager

    mgr = ContentCheckpointManager(
        checkpoint_dir=str(tmp_path), snapshot_store=_FailingSnapshotStore()
    )
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.txt").write_text("hello", encoding="utf-8")
    await mgr.create_snapshot(name="snap1", output_dir=str(out))

    with pytest.raises(RuntimeError):
        await mgr.rollback("snap1", str(out))


# ── T3B1-08: RestorePointManager point_id collision ───────────────────────


@pytest.mark.asyncio
async def test_t3b1_08_rapid_captures_get_distinct_point_ids(tmp_path: Path) -> None:
    """Fires T3B1-08 without the fix; passes with it. Violated property:
    point_id must uniquely identify the restore point the caller created,
    even across two capture() calls issued within the same wall-clock
    second (the original int(time.time()) truncation collides here)."""
    from orchestrator.state_mgmt.restore_points import RestorePointManager

    mgr = RestorePointManager(storage_dir=str(tmp_path))
    p1 = await mgr.capture(label="first")
    p2 = await mgr.capture(label="second")  # same wall-clock second as p1

    assert p1.point_id != p2.point_id, "two captures in the same second collided"

    discarded = await mgr.discard_from(p2.point_id)
    assert discarded == 1, "discard_from acted on the wrong (older) colliding point"
    assert mgr.point_count == 1


# ── T3B1-09: RestorePointManager non-atomic write ─────────────────────────


@pytest.mark.asyncio
async def test_t3b1_09_corrupt_write_does_not_erase_prior_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fires T3B1-09 without the fix; passes with it. Violated property: a
    crash mid-write must not destroy already-persisted restore-point
    history. Simulates the crash by truncating the on-disk file to half its
    new content then raising -- exactly what write_text() itself leaves
    behind if the process dies partway through -- and confirms the prior
    point survives a fresh reload (as a process restart would perform)."""
    from orchestrator.state_mgmt.restore_points import RestorePointManager

    mgr = RestorePointManager(storage_dir=str(tmp_path))
    await mgr.capture(label="before-refactor")

    real_write_text = Path.write_text

    def flaky_write_text(self, data, *a, **kw):
        if not str(self).endswith(".tmp"):
            # Simulate a crash mid-write on the direct target path: only
            # half the new content actually lands on disk before death.
            real_write_text(self, data[: len(data) // 2], *a, **kw)
            raise OSError("simulated crash mid-write")
        return real_write_text(self, data, *a, **kw)

    monkeypatch.setattr(Path, "write_text", flaky_write_text)
    try:
        await mgr.capture(label="after-refactor")
    except OSError:
        pass

    fp = tmp_path / "restore_points.json"
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        pytest.fail(
            "defect still present: restore_points.json left truncated/corrupt after a crash mid-write"
        )
    assert len(data) >= 1, "the first capture's history was destroyed by the second save's crash"


# ── T3B1-01: state_mgmt/session_lifecycle.py is an orphaned full duplicate ──


def test_t3b1_01_state_mgmt_session_lifecycle_is_the_root_class() -> None:
    """Fires T3B1-01 without the fix; passes with it. Violated property:
    single source of truth -- orchestrator.state_mgmt.session_lifecycle's
    SessionLifecycleManager must be the identical object the live engine.py
    path uses (orchestrator.session_lifecycle), not an independent copy
    that can silently diverge from it."""
    from orchestrator.session_lifecycle import SessionLifecycleManager as Root
    from orchestrator.state_mgmt.session_lifecycle import SessionLifecycleManager as Shimmed

    assert (
        Shimmed is Root
    ), "state_mgmt copy is an independent class, not a re-export of the live one"


# ════════════════════════════════════════════════════════════════════════
# Batch 2 — workspace / session_watcher / telemetry
# ════════════════════════════════════════════════════════════════════════


# ── TELEM-1: record_snapshots_batch's only production caller passes the wrong shape ──


@pytest.mark.regression
@pytest.mark.asyncio
async def test_telem1_record_snapshots_batch_reproducer(tmp_path: Path) -> None:
    """Fires TELEM-1 without the fix; passes with it. Violated property:
    record_snapshots_batch's list[tuple[Model, ModelProfile]] contract must
    match what its only production caller (engine.py's get_active_profiles_fn,
    reproduced inline here) actually supplies."""
    from orchestrator.models import Model
    from orchestrator.policy import ModelProfile
    from orchestrator.state_mgmt.telemetry_store import TelemetryStore

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    profile = ModelProfile(
        model=list(Model)[0],
        provider="anthropic",
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
        call_count=5,
    )

    broken_shape = [profile]
    result = await store.record_snapshots_batch("proj-1", broken_shape)
    assert result["success"] == 0, "unpacking a bare ModelProfile should not silently succeed"

    fixed_shape = [(profile.model, profile)]
    result2 = await store.record_snapshots_batch("proj-1", fixed_shape)
    assert result2["success"] == 1
    assert result2["failed"] == 0


def test_telem1_engine_active_profiles_lambda_yields_tuples() -> None:
    """Fires TELEM-1 without the fix; passes with it. Directly checks the
    actual production lambda in engine.py's _get_snapshotter, not just a
    reproduction of its shape, so a future edit to the lambda that
    reintroduces the bug is caught here too."""
    from orchestrator.models import Model
    from orchestrator.policy import ModelProfile

    class _FakeProfiles(dict):
        def values(self):  # noqa: D102
            return super().values()

    class _FakePlanner:
        def __init__(self, profiles):
            self._profiles = profiles

    class _FakeContainer:
        def __init__(self, planner):
            self.planner = planner
            self.telemetry_store = None

    profile = ModelProfile(
        model=list(Model)[0],
        provider="anthropic",
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
        call_count=1,
    )
    profiles = _FakeProfiles({profile.model: profile})

    import orchestrator.engine as engine_module

    orch = engine_module.Orchestrator.__new__(engine_module.Orchestrator)
    orch._c = _FakeContainer(_FakePlanner(profiles))
    orch._background_tasks = set()
    orch._snapshotter = None

    snapshotter = orch._get_snapshotter()
    active = snapshotter._get_active_profiles()
    assert active == [(profile.model, profile)], (
        f"defect still present: get_active_profiles_fn yielded {active!r}, "
        "expected a list of (Model, ModelProfile) tuples"
    )


# ── SESSION-1: non-atomic _save_session destroys prior history on a crash ──


def test_session1_save_session_is_atomic_reproducer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fires SESSION-1 without the fix; passes with it. Violated property:
    _save_session must never leave file_path observably truncated/corrupted
    -- it must be either the fully-intact old content or the fully-intact
    new content, never a mix, even if the write is interrupted partway.
    Lets the real open() truncate the file (as mode "w" always does), then
    fails the very next .write() call on that handle -- reproducing a crash
    that happens AFTER truncation, which is the actual bug mechanism."""
    from orchestrator.state_mgmt.session_watcher import SessionWatcher

    watcher = SessionWatcher(storage_path=tmp_path)
    session_id = watcher.start_session("proj-1")
    watcher.record_interaction(session_id, "input-1", "output-1", "code_generation")

    file_path = watcher._session_file_path(session_id)
    original_content = file_path.read_text(encoding="utf-8")
    assert original_content.count("\n") == 2  # header + 1 interaction

    real_open = open

    def flaky_open(path, mode="r", *a, **kw):
        f = real_open(path, mode, *a, **kw)
        is_target_write = "w" in mode and str(path) == str(file_path)
        if is_target_write:

            def dying_write(_s):
                f.close()
                raise OSError("simulated crash mid-write, after truncation")

            f.write = dying_write
        return f

    monkeypatch.setattr("builtins.open", flaky_open)
    try:
        watcher.record_interaction(session_id, "input-2", "output-2", "code_generation")
    except OSError:
        pass

    content = file_path.read_text(encoding="utf-8")
    lines = [line for line in content.splitlines() if line.strip()]
    for line in lines:
        json.loads(line)  # every persisted line must be complete, valid JSON
    assert len(lines) in (2, 3), f"file was left in a partial/truncated state: {lines!r}"


def test_session_watcher_root_reexports_the_state_mgmt_class() -> None:
    """The live orchestrator.session_watcher (root) must be a re-export shim
    of orchestrator.state_mgmt.session_watcher (the fixed, timezone-aware
    copy) rather than an independently-forked duplicate -- otherwise
    SESSION-1/2/3's fixes only apply to the copy nothing calls."""
    from orchestrator.session_watcher import SessionWatcher as RootWatcher
    from orchestrator.state_mgmt.session_watcher import SessionWatcher as CanonicalWatcher

    assert (
        RootWatcher is CanonicalWatcher
    ), "root session_watcher.py is an independent copy, not a shim"


# ── TELEM-3: flush() can silently drop a concurrently-buffered write ─────


@pytest.mark.regression
@pytest.mark.asyncio
async def test_telem3_flush_does_not_drop_concurrent_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fires TELEM-3 without the fix; passes with it. Violated property: an
    item appended to the buffer during flush()'s own I/O must survive to be
    picked up by a subsequent flush(), never be silently dropped."""
    from orchestrator.state_mgmt.telemetry_store import TelemetryStore

    store = TelemetryStore(
        db_path=tmp_path / "telemetry.db", batch_size=999, flush_interval_seconds=999
    )
    store._snapshot_buffer.append(
        {
            "project_id": "p1",
            "model": "m1",
            "task_type": "code_generation",
            "quality_score": 0.9,
            "trust_factor": 1.0,
            "avg_latency_ms": 100.0,
            "latency_p95_ms": 200.0,
            "success_rate": 1.0,
            "avg_cost_usd": 0.01,
            "call_count": 1,
            "failure_count": 0,
            "validator_fail_count": 0,
            "recorded_at": 0.0,
        }
    )

    import aiosqlite

    real_commit = aiosqlite.Connection.commit
    late_item = {
        "project_id": "p2",
        "model": "m2",
        "task_type": "code_generation",
        "quality_score": 0.5,
        "trust_factor": 1.0,
        "avg_latency_ms": 50.0,
        "latency_p95_ms": 90.0,
        "success_rate": 1.0,
        "avg_cost_usd": 0.02,
        "call_count": 1,
        "failure_count": 0,
        "validator_fail_count": 0,
        "recorded_at": 0.0,
    }

    async def patched_commit(self):
        store._snapshot_buffer.append(late_item)  # the "concurrent" append
        return await real_commit(self)

    monkeypatch.setattr(aiosqlite.Connection, "commit", patched_commit)

    await store.flush()

    assert (
        late_item in store._snapshot_buffer
    ), "concurrently-appended item was lost by flush()'s clear()"


# ── TELEM-4: flush_snapshots bypasses the crash-safe WAL path ─────────────


@pytest.mark.regression
@pytest.mark.asyncio
async def test_telem4_flush_snapshots_is_durable_before_background_task_runs(
    tmp_path: Path,
) -> None:
    """Fires TELEM-4 without the fix; passes with it. Violated property: the
    write-intent for a snapshot must be durable (present in pending_writes)
    the instant flush_snapshots() returns, independent of whether its
    background task has run yet -- so a crash right after this call still
    lets drain_queue() recover the data on next startup."""
    import aiosqlite

    from orchestrator.infrastructure.telemetry_snapshotter import TelemetrySnapshotter
    from orchestrator.models import Model
    from orchestrator.policy import ModelProfile
    from orchestrator.state_mgmt.telemetry_store import TelemetryStore

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    model = list(Model)[0]
    profile = ModelProfile(
        model=model,
        provider="anthropic",
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
        call_count=3,
    )
    snapshotter = TelemetrySnapshotter(
        telemetry_store=store,
        get_active_profiles_fn=lambda: [(model, profile)],
    )

    await snapshotter.flush_snapshots("proj-1")  # returns before its background task runs

    async with (
        aiosqlite.connect(store._db_path) as db,
        db.execute("SELECT COUNT(*) FROM pending_writes") as cur,
    ):
        (count,) = await cur.fetchone()
    assert count == 1, "enqueue_snapshot must be durable before flush_snapshots() returns"


# ── TELEM-2: task_type is hardcoded to CODE_GEN ───────────────────────────


@pytest.mark.regression
@pytest.mark.asyncio
async def test_telem2_task_type_is_not_hardcoded_reproducer(tmp_path: Path) -> None:
    """Fires TELEM-2 without the fix; passes with it. Violated property: a
    snapshot's stored task_type must reflect what the caller says the model
    was used for, not an unconditional CODE_GEN literal."""
    import aiosqlite

    from orchestrator.models import Model, TaskType
    from orchestrator.policy import ModelProfile
    from orchestrator.state_mgmt.telemetry_store import TelemetryStore

    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    model = list(Model)[0]
    profile = ModelProfile(
        model=model,
        provider="anthropic",
        cost_per_1m_input=3.0,
        cost_per_1m_output=15.0,
        call_count=2,
    )

    await store.record_snapshots_batch("proj-1", [(model, profile)], task_type=TaskType.EVALUATE)

    async with (
        aiosqlite.connect(store._db_path) as db,
        db.execute(
            "SELECT task_type FROM model_snapshots WHERE project_id = ?", ("proj-1",)
        ) as cur,
    ):
        (stored_task_type,) = await cur.fetchone()
    assert stored_task_type == TaskType.EVALUATE.value, (
        f"expected {TaskType.EVALUATE.value!r}, got {stored_task_type!r} "
        "(task_type was silently hardcoded to CODE_GEN)"
    )


# ── SESSION-2: one corrupt line discards the whole session, forever ───────


def test_session2_one_corrupt_line_does_not_discard_whole_session(tmp_path: Path) -> None:
    """Fires SESSION-2 without the fix; passes with it. Violated property:
    a load failure for one interaction record must not discard other,
    already-successfully-parsed records from the same session file."""
    from orchestrator.state_mgmt.session_watcher import SessionWatcher

    session_id = "sess-corrupt-1"
    file_path = tmp_path / f"{session_id}.jsonl"
    header = {
        "id": session_id,
        "project_id": "proj-1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "last_activity": "2026-01-01T00:00:00+00:00",
        "status": "active",
        "summary": None,
        "metadata": {},
    }
    good_interaction = {
        "id": "i1",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "task_input": "hello",
        "task_output": "world",
        "task_type": "code_generation",
        "model": None,
        "tokens_used": None,
        "duration_ms": None,
        "metadata": {},
    }
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(header) + "\n")
        f.write(json.dumps(good_interaction) + "\n")
        f.write(
            '{"id": "i2", "timestamp": "trunc'
        )  # simulated crash-truncated line, no trailing \n

    watcher = SessionWatcher(storage_path=tmp_path)

    assert session_id in watcher._sessions, "the whole session must not be discarded"
    assert (
        len(watcher._sessions[session_id].interactions) == 1
    ), "the one valid interaction must survive even though a later line is corrupt"


# ── WORKSPACE-1: create_workspace has no ID-collision guard ──────────────


def test_workspace1_collision_raises_instead_of_overwriting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fires WORKSPACE-1 without the fix; passes with it. Violated property:
    create_workspace must never silently reuse an existing workspace's ID
    and directory tree. Patches the `datetime` name actually bound inside
    orchestrator.state_mgmt.workspace (via `from datetime import datetime`)
    directly, since patching the datetime module itself does not affect a
    name already imported by value into another module's namespace."""
    import datetime as dt_module
    import uuid as uuid_module

    import orchestrator.state_mgmt.workspace as workspace_module
    from orchestrator.state_mgmt.workspace import WorkspaceManager

    manager = WorkspaceManager(base_dir=str(tmp_path))

    fixed_now = dt_module.datetime(2026, 1, 1, 0, 0, 0)
    fixed_uuid = uuid_module.UUID("12345678-1234-5678-1234-567812345678")

    class _FixedDateTime(dt_module.datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_now

    monkeypatch.setattr(workspace_module, "datetime", _FixedDateTime)
    monkeypatch.setattr(uuid_module, "uuid4", lambda: fixed_uuid)

    ws1 = manager.create_workspace("proj", "alice")
    assert ws1 is not None

    with pytest.raises(RuntimeError, match="collision"):
        manager.create_workspace("proj", "alice")  # identical inputs -> identical id


# ── SESSION-3: get_context(limit=0) returns everything instead of nothing ──


@pytest.mark.regression
@pytest.mark.asyncio
async def test_session3_get_context_limit_zero_returns_nothing(tmp_path: Path) -> None:
    """Fires SESSION-3 without the fix; passes with it. Violated property:
    limit=0 must return zero interactions, not every interaction."""
    from orchestrator.state_mgmt.session_watcher import SessionWatcher

    watcher = SessionWatcher(storage_path=tmp_path)
    session_id = watcher.start_session("proj-1")
    watcher.record_interaction(session_id, "in-1", "out-1", "code_generation")
    watcher.record_interaction(session_id, "in-2", "out-2", "code_generation")

    result = await watcher.get_context(session_id, limit=0)

    assert result == [], f"limit=0 must return no interactions, got {len(result)}"


# ── WORKSPACE-2: delete_workspace returns True even when deletion fails ──


def test_workspace2_delete_returns_false_on_rmtree_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fires WORKSPACE-2 without the fix; passes with it. Violated property:
    delete_workspace's return value must accurately report whether file
    deletion succeeded, per its own documented contract."""
    import shutil

    from orchestrator.state_mgmt.workspace import WorkspaceManager

    manager = WorkspaceManager(base_dir=str(tmp_path))
    ws = manager.create_workspace("proj", "alice")

    def failing_rmtree(path, *a, **kw):
        raise OSError("simulated: file locked by another process")

    monkeypatch.setattr(shutil, "rmtree", failing_rmtree)

    result = manager.delete_workspace(ws.id, delete_files=True)

    assert result is False, "delete_workspace must report False when rmtree fails"
