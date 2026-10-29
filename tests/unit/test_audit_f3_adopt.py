"""F3 batch 4: root twins that diverged from their subpackage copies.

The owner chose to keep the subpackage behaviour. These tests import through
the root paths, which are the ones whose behaviour changes when the root
module becomes a re-export shim.
"""

from __future__ import annotations

import warnings

import pytest

pytestmark = pytest.mark.unit


# ── triggers: root orchestrator.triggers gains the events/ hardening ────────


def test_root_trigger_ids_stay_unique_after_delete() -> None:
    from orchestrator.triggers import TriggerManager

    tm = TriggerManager()
    first = tm.create_trigger(name="a", condition="x > 1", action="log", context={})
    tm.create_trigger(name="b", condition="x > 1", action="log", context={})
    tm.remove_trigger(first.id)
    tm.create_trigger(name="c", condition="x > 1", action="log", context={})
    assert len(tm.triggers) == 2, "a new trigger must not overwrite an existing one"


@pytest.mark.asyncio
async def test_root_trigger_supports_not() -> None:
    from orchestrator.triggers import TriggerManager

    tm = TriggerManager()
    assert await tm._evaluate_condition("not paused", {"paused": False}) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("condition", ["x.__class__ == 1", "x[0] > 1", "len(x) > 1"])
async def test_root_trigger_rejects_non_allowlisted_nodes(condition) -> None:
    from orchestrator.triggers import TriggerManager

    fired: list[dict] = []
    tm = TriggerManager()
    tm.register_custom_action("record", fired.append)
    trigger = tm.create_trigger(name="t", condition=condition, action="record", context={})
    assert await tm.evaluate_trigger(trigger.id, {"x": [5]}) is False
    assert fired == []


@pytest.mark.asyncio
async def test_root_trigger_runs_sync_custom_action() -> None:
    from orchestrator.triggers import TriggerManager

    fired: list[dict] = []
    tm = TriggerManager()
    tm.register_custom_action("record", fired.append)
    trigger = tm.create_trigger(name="t", condition="x > 1", action="record", context={})
    assert await tm.evaluate_trigger(trigger.id, {"x": 2}) is True
    assert len(fired) == 1


# ── capability_logger: root gains the state_mgmt/ fixes ─────────────────────


def test_root_capability_event_timestamp_is_tz_aware_without_utcnow() -> None:
    from datetime import datetime

    from orchestrator.capability_logger import CapabilityEvent, CapabilityType

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        event = CapabilityEvent.create(capability=CapabilityType.TASK_STARTED)
    assert datetime.fromisoformat(event.timestamp).utcoffset() is not None


def test_root_capability_logger_accepts_log_dir(tmp_path, monkeypatch) -> None:
    from orchestrator import capability_logger

    monkeypatch.setattr(capability_logger.CapabilityLogger, "_instance", None)
    log = capability_logger.CapabilityLogger(log_dir=str(tmp_path))
    assert log.log_dir == tmp_path


# ── design_to_code: keep the root's current model name ──────────────────────


def test_canonical_vision_models_name_a_declared_claude_model() -> None:
    from orchestrator.design.design_to_code import DesignToCodePipeline
    from orchestrator.models import Model

    declared = {m.value.split("/", 1)[-1] for m in Model}
    claude = [k for k in DesignToCodePipeline.VISION_MODELS if k.startswith("claude-")]
    assert claude and all(k in declared for k in claude), claude
