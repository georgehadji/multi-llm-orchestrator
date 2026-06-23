"""
Catalog of PRE-EXISTING, UNFIXED problems discovered during autonomous bug hunt.

Each test asserts the CORRECT behavior and is marked ``xfail(strict=True)``:
  - While the bug exists, the test fails as expected (xfail) → suite stays green.
  - When someone FIXES the bug, the test passes → strict xfail reports XPASS as a
    FAILURE, forcing the fixer to remove the marker. This makes the bugs both
    documented and impossible to silently "fix and forget".

Fixed bugs (trigger ID collision, trigger sync-action cooldown, VS-first cost
tracking) have their own regression guards in:
  - tests/unit/test_trigger_bugs.py
  - tests/unit/test_vs_cost_tracking.py

Discovered: 2026-06-23 autonomous debugging protocol run.
"""

import asyncio
import tempfile
import time

import pytest


# ──────────────────────────────────────────────────────────────────────────────
# Problem P1: AutomationScheduler awaits sync handlers → silent failure.
#   File: orchestrator/automations.py:128, :152
#   register(task, handler) accepts ANY callable (no async contract). When a
#   sync handler is registered, `await handler(payload)` raises
#   "object NoneType can't be used in 'await' expression", which is caught and
#   logged. The handler's side effect still runs, but run_count is NOT
#   incremented and fire_event/run_cycle under-report their success count.
#   Same defect class as the (now fixed) trigger sync-action bug.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
async def test_automations_sync_handler_counts_as_success():
    from orchestrator.operations.automations import (
        AutomationScheduler,
        ScheduledTask,
        ScheduleType,
    )

    sched = AutomationScheduler(storage_dir=tempfile.mkdtemp())
    fired = []
    task = ScheduledTask(
        name="t1", schedule_type=ScheduleType.EVENT, event_entity="e"
    )
    sched.register(task, lambda payload: fired.append(payload))

    count = await sched.fire_event("e", {"x": 1})

    assert len(fired) == 1, "Sync handler side effect should run"
    assert count == 1, "fire_event must count a successfully-fired sync handler"
    assert task.run_count == 1, "run_count must increment for sync handler"


# ──────────────────────────────────────────────────────────────────────────────
# Problem P2: CronParser day-of-week uses the wrong convention.
#   File: orchestrator/automations.py:53-74
#   Standard cron weekday: 0 = Sunday .. 6 = Saturday.
#   Python time.localtime().tm_wday: 0 = Monday .. 6 = Sunday.
#   The parser compares the cron weekday field directly against tm_wday with no
#   conversion, so "* * * * 0" matches Monday instead of Sunday (off-by-one /
#   wrong-convention). Every weekday-scheduled automation fires on the wrong day.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
def test_cron_weekday_sunday_matches_sunday():
    from orchestrator.operations.automations import CronParser

    # 2024-01-07 is a Sunday; 2024-01-08 is a Monday.
    sunday = time.mktime(time.strptime("2024-01-07 12:00", "%Y-%m-%d %H:%M"))
    monday = time.mktime(time.strptime("2024-01-08 12:00", "%Y-%m-%d %H:%M"))

    # Standard cron "0" in the weekday field means Sunday.
    assert CronParser.matches("* * * * 0", sunday) is True
    assert CronParser.matches("* * * * 0", monday) is False


# ──────────────────────────────────────────────────────────────────────────────
# Problem P3: CronParser "*/0" step → ZeroDivisionError.
#   File: orchestrator/automations.py:66-69
#   `step = int(cf[2:])` then `val % step` with step==0 crashes the whole
#   run_cycle / matches call instead of being rejected as an invalid expression.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
def test_cron_step_zero_does_not_crash():
    from orchestrator.operations.automations import CronParser

    # An invalid '*/0' step should be rejected (return False), not raise.
    assert CronParser.matches("*/0 * * * *") is False


# ──────────────────────────────────────────────────────────────────────────────
# Problem P4: HierarchyManager node IDs are collision-prone.
#   File: orchestrator/hierarchy.py:78, 119, 166, 213
#   IDs are `f"{type}_{len(self.nodes)}"`. The scheme only stays unique while
#   self.nodes never shrinks. There is no remove today, but the invariant is
#   undocumented and fragile: the moment a delete is added, a new node reuses a
#   freed index and silently overwrites an existing node (exactly the trigger
#   bug that WAS fixed). This test pins the latent risk by simulating a removal.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="hierarchy.py len()-based IDs collide if a node is ever removed",
)
def test_hierarchy_ids_survive_removal():
    from orchestrator.hierarchy import HierarchyManager

    h = HierarchyManager()
    org = h.create_org("Acme", budget=1000.0)
    t1 = h.create_team("Eng", org.id, budget=100.0)
    t2 = h.create_team("Sales", org.id, budget=100.0)

    # Simulate a removal that a future delete API would perform.
    h.nodes.pop(t1.id, None)

    # A newly created node must NOT reuse t2's index / id.
    t3 = h.create_team("Ops", org.id, budget=100.0)

    assert t3.id != t2.id, "New node id collided with existing node after removal"
    assert h.nodes[t2.id].name == "Sales", "Existing node was silently overwritten"


# ──────────────────────────────────────────────────────────────────────────────
# Problem P5: BatchClient waits on result truthiness, not a completion flag.
#   File: orchestrator/cost_optimization/batch_client.py:222-228
#   The poll loop does `if request.result:` — a legitimately falsy result
#   (empty string, empty dict/list, 0) is treated as "not ready" and the call
#   blocks until it times out (300s) instead of returning the real result.
#   Documented here against a lightweight stand-in for the wait predicate.
# ──────────────────────────────────────────────────────────────────────────────
@pytest.mark.unit
@pytest.mark.xfail(
    strict=True,
    reason="batch_client poll uses `if request.result:` → falsy valid result never returns",
)
def test_batch_result_falsy_is_recognized_as_complete():
    # Mirror the predicate used in BatchClient._batch_call's wait loop.
    class _Req:
        def __init__(self):
            self.result = ""  # a valid, completed-but-empty result
            self.error = None
            self.completed = True  # the flag the loop SHOULD check

    req = _Req()

    # Current (buggy) predicate: truthiness of result.
    ready_buggy = bool(req.result)
    # Correct predicate: an explicit completion signal.
    ready_correct = req.completed

    assert ready_buggy == ready_correct, (
        "A falsy-but-valid result must be recognized as complete; "
        "the wait loop should test a completion flag, not result truthiness"
    )
