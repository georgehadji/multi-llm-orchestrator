"""
Tests for governance improvements:
  - EnforcementMode (MONITOR / SOFT / HARD)
  - PolicyHierarchy (4-level merge)
  - RateLimit (data structure)
  - AuditLog + AuditRecord
  - AuditLog integration with PolicyEngine

All tests are synchronous (no pytest-asyncio required).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from orchestrator.audit import AuditLog, AuditRecord
from orchestrator.models import Model, TaskType
from orchestrator.policy import (
    EnforcementMode,
    ModelProfile,
    Policy,
    PolicyHierarchy,
    PolicySet,
    RateLimit,
)
from orchestrator.policy_engine import PolicyEngine, PolicyViolationError


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_profile(
    model: Model = Model.GPT_4O,
    provider: str = "openai",
    region: str = "global",
    tags: list[str] | None = None,
    latency: float = 2000.0,
) -> ModelProfile:
    return ModelProfile(
        model=model,
        provider=provider,
        cost_per_1m_input=1.0,
        cost_per_1m_output=5.0,
        capable_task_types={TaskType.CODE_GEN: 0},
        region=region,
        compliance_tags=tags or [],
        avg_latency_ms=latency,
    )


# ── EnforcementMode ───────────────────────────────────────────────────────────

class TestEnforcementMode:

    def test_enum_values(self):
        assert EnforcementMode.MONITOR.value == "monitor"
        assert EnforcementMode.SOFT.value    == "soft"
        assert EnforcementMode.HARD.value    == "hard"

    def test_monitor_mode_allows_violating_model(self):
        """MONITOR: even a blocked provider should pass (with violations logged)."""
        engine = PolicyEngine()
        profile = _make_profile(provider="openai")
        policy = Policy(
            name="no_openai",
            blocked_providers=["openai"],
            enforcement_mode=EnforcementMode.MONITOR,
        )
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is True          # allowed by MONITOR
        assert len(result.violations) == 1    # violation still recorded

    def test_hard_mode_blocks_violating_model(self):
        """HARD: blocked provider is denied."""
        engine = PolicyEngine()
        profile = _make_profile(provider="openai")
        policy = Policy(
            name="no_openai",
            blocked_providers=["openai"],
            enforcement_mode=EnforcementMode.HARD,
        )
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is False
        assert len(result.violations) == 1

    def test_none_mode_defaults_to_hard(self):
        """Policy with enforcement_mode=None behaves as HARD."""
        engine = PolicyEngine()
        profile = _make_profile(provider="openai")
        policy = Policy(name="no_openai", blocked_providers=["openai"])
        # enforcement_mode is None by default → HARD
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is False

    def test_soft_mode_blocks_provider_violation(self):
        """SOFT: provider block is a hard violation → still denied."""
        engine = PolicyEngine()
        profile = _make_profile(provider="openai")
        policy = Policy(
            name="no_openai",
            blocked_providers=["openai"],
            enforcement_mode=EnforcementMode.SOFT,
        )
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is False

    def test_soft_mode_allows_latency_violation(self):
        """SOFT: latency violation is a soft violation → allowed through."""
        engine = PolicyEngine()
        profile = _make_profile(latency=9000.0)  # very slow
        policy = Policy(
            name="fast_sla",
            max_latency_ms=1000.0,
            enforcement_mode=EnforcementMode.SOFT,
        )
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is True          # soft violation allowed
        assert len(result.violations) == 1    # still recorded

    def test_monitor_mode_overrides_multiple_violations(self):
        """MONITOR with multiple violations: all logged but all allowed."""
        engine = PolicyEngine()
        profile = _make_profile(provider="openai", region="us")
        policy = Policy(
            name="strict",
            blocked_providers=["openai"],
            allowed_regions=["eu"],
            enforcement_mode=EnforcementMode.MONITOR,
        )
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is True
        assert len(result.violations) == 2

    def test_most_restrictive_mode_wins_across_policies(self):
        """When multiple policies have different modes, most restrictive wins (HARD beats MONITOR)."""
        engine = PolicyEngine()
        profile = _make_profile(provider="openai")
        hard_policy    = Policy(name="hard", blocked_providers=["openai"],
                                enforcement_mode=EnforcementMode.HARD)
        monitor_policy = Policy(name="monitor", allowed_regions=["eu"],
                                enforcement_mode=EnforcementMode.MONITOR)
        # HARD is more restrictive than MONITOR → effective mode = HARD
        result = engine.check(Model.GPT_4O, profile, [hard_policy, monitor_policy])
        assert result.passed is False  # HARD overrides MONITOR (most restrictive wins)


# ── PolicyHierarchy ───────────────────────────────────────────────────────────

class TestPolicyHierarchy:

    def test_org_only_returns_org_policies(self):
        gdpr = Policy(name="gdpr", allow_training_on_output=False)
        hier = PolicyHierarchy(org=[gdpr])
        policies = hier.policies_for()
        assert len(policies) == 1
        assert policies[0].name == "gdpr"

    def test_org_plus_team_merged(self):
        gdpr    = Policy(name="gdpr")
        eu_only = Policy(name="eu_only", allowed_regions=["eu"])
        hier = PolicyHierarchy(
            org=[gdpr],
            team={"eng": [eu_only]},
        )
        policies = hier.policies_for(team="eng")
        assert len(policies) == 2
        names = [p.name for p in policies]
        assert "gdpr" in names
        assert "eu_only" in names

    def test_all_four_levels_merged(self):
        p_org  = Policy(name="p_org")
        p_team = Policy(name="p_team")
        p_job  = Policy(name="p_job")
        p_node = Policy(name="p_node")
        hier = PolicyHierarchy(
            org=[p_org],
            team={"t1": [p_team]},
            job={"j1": [p_job]},
            node={"n1": [p_node]},
        )
        policies = hier.policies_for(team="t1", job_id="j1", task_id="n1")
        assert len(policies) == 4
        names = [p.name for p in policies]
        assert names == ["p_org", "p_team", "p_job", "p_node"]

    def test_missing_keys_return_empty(self):
        """Non-existent team/job/task keys contribute zero policies."""
        hier = PolicyHierarchy(org=[Policy(name="gdpr")])
        policies = hier.policies_for(team="nonexistent", job_id="also_none", task_id="xyz")
        assert len(policies) == 1  # only org policy

    def test_empty_hierarchy_returns_empty(self):
        hier = PolicyHierarchy()
        assert hier.policies_for() == []

    def test_as_policy_set_returns_correct_type(self):
        hier = PolicyHierarchy(org=[Policy(name="gdpr")])
        ps = hier.as_policy_set()
        assert isinstance(ps, PolicySet)
        assert len(ps.global_policies) == 1

    def test_as_policy_set_with_all_levels(self):
        hier = PolicyHierarchy(
            org=[Policy(name="org")],
            team={"x": [Policy(name="team")]},
        )
        ps = hier.as_policy_set(team="x")
        assert len(ps.global_policies) == 2

    def test_org_policies_applied_before_team(self):
        """Org policies come first in the merged list."""
        hier = PolicyHierarchy(
            org=[Policy(name="org_first")],
            team={"t": [Policy(name="team_second")]},
        )
        policies = hier.policies_for(team="t")
        assert policies[0].name == "org_first"
        assert policies[1].name == "team_second"


# ── RateLimit ─────────────────────────────────────────────────────────────────

class TestRateLimit:

    def test_all_fields_none_by_default(self):
        rl = RateLimit()
        assert rl.calls_per_minute is None
        assert rl.cost_usd_per_hour is None
        assert rl.tokens_per_day is None

    def test_fields_set_correctly(self):
        rl = RateLimit(calls_per_minute=60, cost_usd_per_hour=1.0, tokens_per_day=100_000)
        assert rl.calls_per_minute == 60
        assert rl.cost_usd_per_hour == 1.0
        assert rl.tokens_per_day == 100_000

    def test_rate_limit_attached_to_policy(self):
        rl = RateLimit(calls_per_minute=10)
        policy = Policy(name="limited", rate_limit=rl)
        assert policy.rate_limit is not None
        assert policy.rate_limit.calls_per_minute == 10

    def test_policy_without_rate_limit_is_none(self):
        policy = Policy(name="no_limit")
        assert policy.rate_limit is None


# ── AuditLog ──────────────────────────────────────────────────────────────────

class TestAuditLog:

    def test_initially_empty(self):
        log = AuditLog()
        assert len(log) == 0

    def test_record_appends(self):
        log = AuditLog()
        log.record(
            task_id="t1",
            model="gpt-4o",
            passed=True,
            raw_passed=True,
            violations=[],
            enforcement_mode="hard",
            policies_applied=["gdpr"],
        )
        assert len(log) == 1

    def test_record_content_correct(self):
        log = AuditLog()
        log.record(
            task_id="t1", model="gpt-4o", passed=False, raw_passed=False,
            violations=["[p] provider 'openai' blocked"],
            enforcement_mode="hard", policies_applied=["p"],
        )
        rec = log.records()[0]
        assert isinstance(rec, AuditRecord)
        assert rec.task_id == "t1"
        assert rec.model == "gpt-4o"
        assert rec.passed is False
        assert len(rec.violations) == 1
        assert rec.enforcement_mode == "hard"
        assert rec.policies_applied == ["p"]
        assert rec.timestamp > 0

    def test_records_returns_copy(self):
        log = AuditLog()
        log.record("t1", "m", True, True, [], "hard", [])
        snap = log.records()
        snap.clear()
        assert len(log) == 1  # original not affected

    def test_to_list_returns_dicts(self):
        log = AuditLog()
        log.record("t1", "m", True, True, [], "hard", ["p"])
        items = log.to_list()
        assert isinstance(items, list)
        assert isinstance(items[0], dict)
        assert items[0]["task_id"] == "t1"

    def test_flush_jsonl_writes_valid_jsonl(self):
        log = AuditLog()
        log.record("t1", "m1", True, True, [], "hard", ["p1"])
        log.record("t2", "m2", False, False, ["v1"], "monitor", ["p2"])
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            path = f.name
        log.flush_jsonl(path)
        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) == 2
        parsed = [json.loads(line) for line in lines]
        assert parsed[0]["task_id"] == "t1"
        assert parsed[1]["passed"] is False
        assert parsed[1]["violations"] == ["v1"]

    def test_flush_jsonl_appends_on_repeated_calls(self):
        log = AuditLog()
        log.record("t1", "m", True, True, [], "hard", [])
        with tempfile.NamedTemporaryFile(mode="r", suffix=".jsonl", delete=False) as f:
            path = f.name
        log.flush_jsonl(path)
        log.flush_jsonl(path)  # second flush appends
        lines = Path(path).read_text().strip().splitlines()
        assert len(lines) == 2  # two writes of 1 record each

    def test_clear_empties_log(self):
        log = AuditLog()
        log.record("t1", "m", True, True, [], "hard", [])
        log.clear()
        assert len(log) == 0


# ── AuditLog integration with PolicyEngine ────────────────────────────────────

class TestAuditLogInPolicyEngine:

    def test_check_emits_one_record(self):
        log = AuditLog()
        engine = PolicyEngine(audit_log=log)
        profile = _make_profile()
        policy = Policy(name="allow_all")
        engine.check(Model.GPT_4O, profile, [policy])
        assert len(log) == 1

    def test_record_reflects_passed_true(self):
        log = AuditLog()
        engine = PolicyEngine(audit_log=log)
        profile = _make_profile(provider="openai")
        policy = Policy(name="allow_all")
        engine.check(Model.GPT_4O, profile, [policy])
        rec = log.records()[0]
        assert rec.passed is True
        assert rec.raw_passed is True
        assert rec.violations == []

    def test_record_reflects_violation(self):
        log = AuditLog()
        engine = PolicyEngine(audit_log=log)
        profile = _make_profile(provider="openai")
        policy = Policy(name="no_openai", blocked_providers=["openai"])
        engine.check(Model.GPT_4O, profile, [policy])
        rec = log.records()[0]
        assert rec.passed is False
        assert rec.raw_passed is False
        assert len(rec.violations) == 1

    def test_monitor_mode_raw_vs_effective_differ(self):
        """MONITOR: raw_passed=False but passed=True (mode override)."""
        log = AuditLog()
        engine = PolicyEngine(audit_log=log)
        profile = _make_profile(provider="openai")
        policy = Policy(name="no_openai", blocked_providers=["openai"],
                        enforcement_mode=EnforcementMode.MONITOR)
        engine.check(Model.GPT_4O, profile, [policy])
        rec = log.records()[0]
        assert rec.raw_passed is False   # genuine violation
        assert rec.passed is True        # overridden by MONITOR
        assert rec.enforcement_mode == "monitor"

    def test_task_id_propagated_to_record(self):
        log = AuditLog()
        engine = PolicyEngine(audit_log=log)
        profile = _make_profile()
        engine.check(Model.GPT_4O, profile, [], task_id="my_task_123")
        rec = log.records()[0]
        assert rec.task_id == "my_task_123"

    def test_policies_applied_list_correct(self):
        log = AuditLog()
        engine = PolicyEngine(audit_log=log)
        profile = _make_profile()
        policies = [Policy(name="p1"), Policy(name="p2")]
        engine.check(Model.GPT_4O, profile, policies)
        rec = log.records()[0]
        assert rec.policies_applied == ["p1", "p2"]

    def test_no_audit_log_still_works(self):
        """PolicyEngine without audit_log behaves as before (no errors)."""
        engine = PolicyEngine()
        profile = _make_profile()
        result = engine.check(Model.GPT_4O, profile, [])
        assert result.passed is True

    def test_multiple_checks_accumulate_records(self):
        log = AuditLog()
        engine = PolicyEngine(audit_log=log)
        profile = _make_profile()
        engine.check(Model.GPT_4O, profile, [])
        engine.check(Model.GPT_4O, profile, [])
        engine.check(Model.GPT_4O, profile, [])
        assert len(log) == 3

    def test_enforce_still_raises_on_violation(self):
        """enforce() still raises PolicyViolationError when mode is HARD."""
        engine = PolicyEngine()
        profile = _make_profile(provider="openai")
        policy = Policy(name="no_openai", blocked_providers=["openai"])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.enforce(Model.GPT_4O, profile, [policy])
        assert "no_openai" in str(exc_info.value)
