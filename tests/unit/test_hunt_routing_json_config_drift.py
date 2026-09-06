"""
Out-of-band fix — orchestrator/config/routing.json config drift.

Flagged but not investigated during hunt T11 ("a large routing.json
config-drift condition... large enough to warrant its own dedicated
investigation"). Resolved here using a live OpenRouter catalogue the user
supplied directly (this sandbox has no outbound network access to
openrouter.ai), which let tests/unit/test_openrouter_model_audit.py's
snapshot-based check run for real instead of skipping, and separately
motivated re-running .claude/skills/orchestrator-diagnostics-and-tooling/
scripts/check_config_drift.py.

routing.json had 21 dead top-level keys shaped like ``{model_id:
[task_type, ...]}`` — the inverse of the file's real ``{task_type:
[model_id, ...]}`` shape. orchestrator/models.py::_build_routing_table's
``if k in TaskType._value2member_map_`` guard (the exact mechanism
check_config_drift.py simulates) silently drops any key that isn't a real
TaskType value, so all 21 were always inert. Of the 50 (model, task_type)
pairs those dead keys carried, 44 duplicated a pairing already present in
the correct list; 6 did not and were the actual defect — genuinely live
models silently absent from the task types they were meant to be routed
for. Fixed by adding the 6 missing pairs to their correct TaskType-keyed
lists and deleting all 21 dead keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchestrator.models import Model, ROUTING_TABLE, TaskType

pytestmark = pytest.mark.unit

ROUTING_JSON = Path(__file__).resolve().parents[2] / "orchestrator" / "config" / "routing.json"

# (model id, TaskType enum member) — the 6 pairs that were only ever
# recorded in a dead, silently-dropped routing.json key.
_PREVIOUSLY_MISSING_PAIRS = [
    ("meituan/longcat-2.0", TaskType.CODE_REVIEW),
    ("meituan/longcat-2.0", TaskType.WRITING),
    ("google/gemini-3.5-flash-lite", TaskType.CODE_REVIEW),
    ("google/gemini-3.5-flash-lite", TaskType.WRITING),
    ("anthropic/claude-opus-5-fast", TaskType.CODE_REVIEW),
    ("z-ai/glm-5.1", TaskType.CODE_GEN),
]


@pytest.mark.parametrize("model_id,task_type", _PREVIOUSLY_MISSING_PAIRS)
def test_previously_dropped_pair_now_routes(model_id: str, task_type: TaskType) -> None:
    assert Model(model_id) in ROUTING_TABLE[task_type]


def test_routing_json_has_no_dead_model_id_keys() -> None:
    """Every top-level routing.json key must be a real TaskType value —
    anything else is silently dropped by _build_routing_table's own guard
    and is dead weight at best, a routing gap at worst."""
    raw = json.loads(ROUTING_JSON.read_text(encoding="utf-8"))
    valid_task_types = {t.value for t in TaskType}
    assert set(raw.keys()) <= valid_task_types


def test_no_regression_existing_routing_entry_still_present() -> None:
    """A pre-existing, correctly-keyed routing entry is unaffected by the cleanup."""
    assert Model("anthropic/claude-sonnet-5") in ROUTING_TABLE[TaskType.CODE_GEN]
