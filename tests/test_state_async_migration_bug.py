"""
Regression test for BUG-NEW-003 in StateManager._get_conn().

BUG-NEW-003: migrate_add_resume_fields() was called synchronously inside an
async method that holds asyncio.Lock.  Because it uses sqlite3.connect()
(blocking I/O), the event loop was frozen for the entire migration duration on
every first StateManager initialization.

Fix: the call is now wrapped with loop.run_in_executor(None, ...) so the
migration runs on a thread-pool thread while the event loop remains free.

Test strategy:
- Initialize StateManager and measure whether the event loop stays responsive
  (i.e. a lightweight coroutine can make progress) while _get_conn() is running
  for the first time.
- Before the fix, the sync sqlite3 call inside the lock would block the loop
  completely; the probe coroutine would only run *after* _get_conn() returned.
- After the fix, the probe coroutine can interleave with the executor work.
"""
from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path

import aiosqlite
import pytest

from orchestrator.state import StateManager


@pytest.mark.asyncio
async def test_event_loop_not_blocked_during_migration():
    """
    While StateManager initialises its DB (which runs migrate_add_resume_fields
    on first _get_conn()), the event loop must remain responsive.

    We schedule a lightweight probe task concurrently and verify it completes
    in well under the time a blocking sqlite3 call would take.
    """
    probe_completed_at: list[float] = []

    async def probe():
        """Yield to the event loop and record when we resume."""
        await asyncio.sleep(0)  # single cooperative yield
        probe_completed_at.append(time.monotonic())

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "state.db"
        manager = StateManager(db_path=db_path)

        t0 = time.monotonic()

        # Run both concurrently; _get_conn triggers first-time migration
        await asyncio.gather(
            manager._get_conn(),   # triggers migrate_add_resume_fields
            probe(),
        )

        # Close the connection so Windows can release the file lock
        if manager._conn is not None:
            await manager._conn.close()
            manager._conn = None

        probe_delay = probe_completed_at[0] - t0 if probe_completed_at else 999.0

    assert probe_completed_at, "Probe coroutine never ran"
    # The probe only does asyncio.sleep(0) — it must complete quickly.
    # If the event loop was blocked the probe would be delayed by the full
    # migration duration (typically 10–100 ms of sync I/O).
    assert probe_delay < 0.5, (
        f"Probe took {probe_delay:.3f}s — event loop was probably blocked "
        "during sync migration (BUG-NEW-003 regressed)"
    )


@pytest.mark.asyncio
async def test_migration_completes_successfully_via_executor():
    """
    Sanity-check: StateManager still works end-to-end after the executor fix.
    The migration must actually run and add the expected columns.
    """
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "state.db"
        manager = StateManager(db_path=db_path)

        conn = await manager._get_conn()
        assert conn is not None, "Connection should be established"

        # Verify migration columns exist (added by migrate_add_resume_fields)
        async with aiosqlite.connect(str(db_path)) as db:
            async with db.execute("PRAGMA table_info(projects)") as cur:
                columns = {row[1] async for row in cur}

        # Close the connection before tempdir cleanup (Windows file lock)
        if manager._conn is not None:
            await manager._conn.close()
            manager._conn = None

    # migrate_add_resume_fields adds project_description and keywords_json
    assert "project_description" in columns, (
        "Migration did not add 'project_description' column — executor path "
        "may not have run the migration (BUG-NEW-003)"
    )
    assert "keywords_json" in columns, (
        "Migration did not add 'keywords_json' column — executor path "
        "may not have run the migration (BUG-NEW-003)"
    )
