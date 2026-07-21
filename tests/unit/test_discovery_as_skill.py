"""
Tests for ENH-5: Discovery-as-skill automations.

Loop Engineering §VI Move 1 (Discovery): "Every task the agent performs should
be a reusable skill, not an inline prompt." This applies to scheduled tasks too —
a task should reference a named skill rather than embedding a prompt string.

Adds `skill_name` field to ScheduledTask:
- When set, the skill resolver provides the prompt (indirection layer)
- Inline prompts continue to work for backward compatibility
- to_dict() / round-trip through JSON preserves skill_name
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from orchestrator.operations.automations import ScheduledTask, ScheduleType

# ── ScheduledTask.skill_name field ───────────────────────────────────────────


class TestScheduledTaskSkillName:
    def test_skill_name_defaults_to_none(self):
        task = ScheduledTask(
            name="t1",
            schedule_type=ScheduleType.CRON,
        )
        assert task.skill_name is None

    def test_skill_name_can_be_set(self):
        task = ScheduledTask(
            name="t2",
            schedule_type=ScheduleType.CRON,
            skill_name="discover_new_apis",
        )
        assert task.skill_name == "discover_new_apis"

    def test_skill_name_in_to_dict(self):
        task = ScheduledTask(
            name="t3",
            schedule_type=ScheduleType.ONCE,
            skill_name="audit_dependencies",
        )
        d = task.to_dict()
        assert "skill_name" in d
        assert d["skill_name"] == "audit_dependencies"

    def test_to_dict_skill_name_none_when_not_set(self):
        task = ScheduledTask(name="t4", schedule_type=ScheduleType.INTERVAL)
        d = task.to_dict()
        assert d.get("skill_name") is None

    def test_uses_skill_when_skill_name_set(self):
        """has_skill_name property / bool check for dispatch logic."""
        task_with_skill = ScheduledTask(
            name="s1",
            schedule_type=ScheduleType.CRON,
            skill_name="my_skill",
        )
        task_inline = ScheduledTask(name="s2", schedule_type=ScheduleType.CRON)
        assert task_with_skill.skill_name is not None
        assert task_inline.skill_name is None


# ── Round-trip persistence ────────────────────────────────────────────────────


class TestScheduledTaskRoundTrip:
    def test_roundtrip_with_skill_name(self):
        """to_dict() must include skill_name so _load() can restore it."""
        original = ScheduledTask(
            name="weekly_scan",
            schedule_type=ScheduleType.CRON,
            cron_expr="0 9 * * 1",
            skill_name="security_scan",
        )
        d = original.to_dict()
        # Simulate _load() round-trip
        restored = ScheduledTask(
            name=d["name"],
            schedule_type=ScheduleType(d["type"]),
            cron_expr=d.get("cron", ""),
            interval_seconds=d.get("interval", 0),
            event_entity=d.get("entity", ""),
            webhook_path=d.get("webhook", ""),
            last_run=d.get("last_run", 0.0),
            run_count=d.get("run_count", 0),
            enabled=d.get("enabled", True),
            skill_name=d.get("skill_name"),
        )
        assert restored.skill_name == "security_scan"

    def test_roundtrip_without_skill_name(self):
        """skill_name=None preserved across round-trip (backward compat)."""
        original = ScheduledTask(
            name="ping", schedule_type=ScheduleType.INTERVAL, interval_seconds=60
        )
        d = original.to_dict()
        restored = ScheduledTask(
            name=d["name"],
            schedule_type=ScheduleType(d["type"]),
            interval_seconds=d.get("interval", 0),
            skill_name=d.get("skill_name"),
        )
        assert restored.skill_name is None
