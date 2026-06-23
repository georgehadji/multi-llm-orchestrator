"""
Regression tests for trigger bugs:
  - Trigger ID collision after delete+create
  - Sync callable in custom action (cooldown bypass)
"""

import asyncio

import pytest

from orchestrator.events.triggers import TriggerManager


@pytest.mark.unit
def test_trigger_id_unique_after_delete():
    """
    Reproducer: create 3 triggers, delete middle one, create 4th.
    Before fix: 4th trigger reuses deleted ID → overwrites trigger 3.
    After fix:  4th trigger gets a unique ID, trigger 3 survives.
    """
    tm = TriggerManager()
    t1 = tm.create_trigger("A", "x > 0", "log", {})
    t2 = tm.create_trigger("B", "x > 1", "log", {})
    t3 = tm.create_trigger("C", "x > 2", "log", {})

    tm.remove_trigger(t2.id)
    t4 = tm.create_trigger("D", "x > 3", "log", {})

    # All three remaining triggers must be distinct
    assert t4.id != t3.id, "New trigger overwrote existing trigger C"
    assert t1.id in tm.triggers, "Trigger A was unexpectedly removed"
    assert t3.id in tm.triggers, "Trigger C was silently overwritten"
    assert t4.id in tm.triggers, "Trigger D was not registered"
    assert tm.triggers[t3.id].name == "C", "Trigger C name was overwritten"


@pytest.mark.unit
def test_trigger_id_no_collision_many_creates():
    """All IDs must be unique across many creates."""
    tm = TriggerManager()
    ids = [tm.create_trigger(f"T{i}", "x > 0", "log", {}).id for i in range(50)]
    assert len(set(ids)) == 50, "Duplicate trigger IDs detected"


@pytest.mark.unit
async def test_sync_action_returns_true_and_applies_cooldown():
    """
    Reproducer: register sync callable as custom action; fire the trigger.
    Before fix: await sync_fn() → TypeError caught → returns False, no cooldown.
    After fix:  action executes, trigger returns True, last_triggered set.
    """
    tm = TriggerManager()
    fired = []
    tm.register_custom_action("capture", lambda ctx: fired.append(ctx))

    t = tm.create_trigger("Fire", "x > 5", "capture", {"extra": 1})

    result = await tm.evaluate_trigger(t.id, {"x": 10})

    assert result is True, "Trigger should return True when action succeeds"
    assert len(fired) == 1, "Action should have been called exactly once"
    assert tm.triggers[t.id].last_triggered is not None, "Cooldown timestamp not set"


@pytest.mark.unit
async def test_sync_action_cooldown_prevents_refiring():
    """After a sync action fires, cooldown prevents immediate re-evaluation."""
    tm = TriggerManager()
    fired = []
    tm.register_custom_action("capture", lambda ctx: fired.append(1))

    t = tm.create_trigger("Fire", "x > 5", "capture", {}, cooldown_period=9999.0)

    await tm.evaluate_trigger(t.id, {"x": 10})
    await tm.evaluate_trigger(t.id, {"x": 10})

    assert len(fired) == 1, "Action should fire only once during cooldown period"


@pytest.mark.unit
async def test_async_action_still_works():
    """Async custom actions still execute correctly after the fix."""
    tm = TriggerManager()
    fired = []

    async def async_action(ctx):
        fired.append(ctx)

    tm.register_custom_action("async_capture", async_action)

    t = tm.create_trigger("Fire", "x > 5", "async_capture", {})
    result = await tm.evaluate_trigger(t.id, {"x": 10})

    assert result is True
    assert len(fired) == 1
