"""
Tests for the policy-driven constraint solver stack.
=======================================================
Covers: ConstraintPlanner, PolicyEngine, TelemetryCollector,
        PolicyViolationError, and build_default_profiles().

Run with:
    cd "E:\\Documents\\Vibe-Coding\\Ai Orchestrator"
    python -m pytest tests/test_constraint_planner.py -v
"""
from __future__ import annotations

import pytest

from orchestrator.models import (
    Model, TaskType, COST_TABLE, ROUTING_TABLE, build_default_profiles,
)
from orchestrator.policy import ModelProfile, Policy, PolicySet, JobSpec, Budget
from orchestrator.policy_engine import PolicyEngine, PolicyViolationError
from orchestrator.planner import ConstraintPlanner
from orchestrator.telemetry import TelemetryCollector


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_profile(
    model: Model,
    provider: str,
    cost_in: float = 1.0,
    cost_out: float = 5.0,
    task_types: dict | None = None,
    region: str = "global",
    tags: list | None = None,
    quality: float = 0.8,
    trust: float = 1.0,
    latency: float = 2000.0,
) -> ModelProfile:
    """Create a minimal ModelProfile for testing."""
    return ModelProfile(
        model=model,
        provider=provider,
        cost_per_1m_input=cost_in,
        cost_per_1m_output=cost_out,
        capable_task_types=task_types or {TaskType.CODE_GEN: 0},
        region=region,
        compliance_tags=tags or [],
        quality_score=quality,
        trust_factor=trust,
        avg_latency_ms=latency,
    )


def _all_healthy(profiles: dict) -> dict[Model, bool]:
    """Return an api_health dict with every model marked healthy."""
    return {m: True for m in profiles}


def _make_planner(profiles, all_healthy=True) -> ConstraintPlanner:
    engine = PolicyEngine()
    health = {m: all_healthy for m in profiles}
    return ConstraintPlanner(profiles, engine, health)


# ─────────────────────────────────────────────────────────────────────────────
# TC-1  EU-only policy
# ─────────────────────────────────────────────────────────────────────────────

class TestEUOnlyPolicy:
    """TC-1a/b: EU-only region constraint."""

    def _setup(self):
        profiles = {
            Model.GPT_4O:       _make_profile(Model.GPT_4O,       "openai",    region="us"),
            Model.CLAUDE_SONNET:_make_profile(Model.CLAUDE_SONNET, "anthropic", region="eu"),
            Model.GEMINI_PRO:   _make_profile(Model.GEMINI_PRO,   "google",    region="us"),
        }
        policy = Policy(name="eu_only", allowed_regions=["eu"])
        planner = _make_planner(profiles)
        return planner, [policy]

    def test_tc1a_eu_blocks_global_models(self):
        """TC-1a: EU-only policy blocks US-region models → only EU model can be selected."""
        planner, policies = self._setup()
        # GPT-4o and Gemini are in "us" region; only Claude Sonnet is in "eu"
        result = planner.select_model(TaskType.CODE_GEN, policies, budget_remaining=100.0)
        assert result == Model.CLAUDE_SONNET

    def test_tc1b_eu_returns_none_when_no_eu_model(self):
        """TC-1b: When no model has allowed region, select_model returns None."""
        profiles = {
            Model.GPT_4O:     _make_profile(Model.GPT_4O,     "openai", region="us"),
            Model.GEMINI_PRO: _make_profile(Model.GEMINI_PRO, "google", region="us"),
        }
        policy = Policy(name="eu_only", allowed_regions=["eu"])
        planner = _make_planner(profiles)
        result = planner.select_model(TaskType.CODE_GEN, [policy], budget_remaining=100.0)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TC-2  Budget constraints
# ─────────────────────────────────────────────────────────────────────────────

class TestBudgetConstraints:
    """TC-2a/b: Budget filtering."""

    def test_tc2a_tight_budget_forces_cheapest(self):
        """TC-2a: Very tight budget ($0.001) forces selection of cheapest model."""
        # CODE_GEN typical: 800 input + 1200 output tokens
        # cheap model: 0.10/1M in + 0.40/1M out  → cost ≈ 0.000560
        # expensive model: 10.0/1M in + 50.0/1M out → cost ≈ 0.0680 (over budget)
        profiles = {
            Model.GPT_4O_MINI: _make_profile(
                Model.GPT_4O_MINI, "openai",
                cost_in=0.10, cost_out=0.40,
                task_types={TaskType.CODE_GEN: 1},
            ),
            Model.CLAUDE_OPUS: _make_profile(
                Model.CLAUDE_OPUS, "anthropic",
                cost_in=10.0, cost_out=50.0,
                task_types={TaskType.CODE_GEN: 0},
            ),
        }
        planner = _make_planner(profiles)
        # Budget of $0.001 — only GPT_4O_MINI's est ~$0.00056 fits
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=0.001)
        assert result == Model.GPT_4O_MINI

    def test_tc2b_zero_budget_returns_none(self):
        """TC-2b: Zero budget → no model can afford the task → returns None."""
        profiles = {
            Model.GPT_4O_MINI: _make_profile(
                Model.GPT_4O_MINI, "openai",
                cost_in=0.10, cost_out=0.40,
            ),
        }
        planner = _make_planner(profiles)
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=0.0)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TC-3  Scoring: quality vs trust
# ─────────────────────────────────────────────────────────────────────────────

class TestScoring:
    """TC-3a/b: Multi-objective scoring."""

    def test_tc3a_high_quality_wins(self):
        """TC-3a: High quality_score model wins over equal-cost lower-quality model."""
        profiles = {
            Model.GPT_4O: _make_profile(
                Model.GPT_4O, "openai",
                cost_in=1.0, cost_out=5.0,
                quality=0.95, trust=1.0,
                task_types={TaskType.CODE_GEN: 0},
            ),
            Model.GEMINI_PRO: _make_profile(
                Model.GEMINI_PRO, "google",
                cost_in=1.0, cost_out=5.0,
                quality=0.70, trust=1.0,
                task_types={TaskType.CODE_GEN: 1},
            ),
        }
        planner = _make_planner(profiles)
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert result == Model.GPT_4O

    def test_tc3b_degraded_trust_avoids_model(self):
        """TC-3b: Very low trust_factor (0.01) avoids model even with high quality."""
        profiles = {
            Model.CLAUDE_SONNET: _make_profile(
                Model.CLAUDE_SONNET, "anthropic",
                cost_in=1.0, cost_out=5.0,
                quality=0.99, trust=0.01,  # super degraded
                task_types={TaskType.CODE_GEN: 0},
            ),
            Model.GPT_4O: _make_profile(
                Model.GPT_4O, "openai",
                cost_in=1.0, cost_out=5.0,
                quality=0.80, trust=1.0,
                task_types={TaskType.CODE_GEN: 1},
            ),
        }
        planner = _make_planner(profiles)
        # Claude Sonnet: 0.99 × 0.01 / cost ≈ 0.0099/cost
        # GPT-4o:        0.80 × 1.00 / cost ≈ 0.80/cost  → GPT-4o wins
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert result == Model.GPT_4O


# ─────────────────────────────────────────────────────────────────────────────
# TC-4  replan()
# ─────────────────────────────────────────────────────────────────────────────

class TestReplan:
    """TC-4a/b: Fallback after failure."""

    def test_tc4a_replan_returns_different_model(self):
        """TC-4a: replan() returns a model different from the failed one."""
        profiles = {
            Model.GPT_4O:       _make_profile(Model.GPT_4O,       "openai",    task_types={TaskType.CODE_GEN: 0}),
            Model.CLAUDE_SONNET:_make_profile(Model.CLAUDE_SONNET, "anthropic", task_types={TaskType.CODE_GEN: 1}),
        }
        planner = _make_planner(profiles)
        result = planner.replan(
            failed_model=Model.GPT_4O,
            task_type=TaskType.CODE_GEN,
            policies=[],
            budget_remaining=100.0,
        )
        assert result is not None
        assert result != Model.GPT_4O

    def test_tc4b_replan_returns_none_when_no_alternative(self):
        """TC-4b: replan() returns None when only the failed model was available."""
        profiles = {
            Model.GPT_4O: _make_profile(Model.GPT_4O, "openai", task_types={TaskType.CODE_GEN: 0}),
        }
        planner = _make_planner(profiles)
        result = planner.replan(
            failed_model=Model.GPT_4O,
            task_type=TaskType.CODE_GEN,
            policies=[],
            budget_remaining=100.0,
        )
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TC-5  PolicyEngine
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyEngine:
    """TC-5a/b/c: PolicyViolationError, violations list, impossible policy."""

    def test_tc5a_enforce_raises_on_blocked_provider(self):
        """TC-5a: PolicyEngine.enforce() raises PolicyViolationError on blocked provider."""
        engine = PolicyEngine()
        profile = _make_profile(Model.GPT_4O, "openai")
        policy = Policy(name="no_openai", blocked_providers=["openai"])
        with pytest.raises(PolicyViolationError) as exc_info:
            engine.enforce(Model.GPT_4O, profile, [policy])
        assert "no_openai" in str(exc_info.value)
        assert "openai" in str(exc_info.value)

    def test_tc5b_check_returns_violations_list(self):
        """TC-5b: PolicyEngine.check() returns PolicyCheckResult with violations list."""
        engine = PolicyEngine()
        profile = _make_profile(
            Model.GPT_4O, "openai",
            region="us",
            tags=[],  # no compliance tags
        )
        policies = [
            Policy(name="p1", blocked_providers=["openai"]),
            Policy(name="p2", allowed_regions=["eu"]),
            Policy(name="p3", allow_training_on_output=False),
        ]
        result = engine.check(Model.GPT_4O, profile, policies)
        assert result.passed is False
        assert len(result.violations) == 3  # all three policies violated
        violation_text = " ".join(result.violations)
        assert "p1" in violation_text
        assert "p2" in violation_text
        assert "p3" in violation_text

    def test_tc5c_impossible_policy_returns_none(self):
        """TC-5c: When all providers are blocked, select_model() returns None."""
        profiles = {
            Model.GPT_4O:       _make_profile(Model.GPT_4O,       "openai"),
            Model.CLAUDE_SONNET:_make_profile(Model.CLAUDE_SONNET, "anthropic"),
            Model.GEMINI_PRO:   _make_profile(Model.GEMINI_PRO,   "google"),
        }
        policy = Policy(
            name="block_all",
            blocked_providers=["openai", "anthropic", "google", "kimi"],
        )
        planner = _make_planner(profiles)
        result = planner.select_model(TaskType.CODE_GEN, [policy], budget_remaining=100.0)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TC-6  Trust factor dynamics
# ─────────────────────────────────────────────────────────────────────────────

class TestTrustFactor:
    """TC-6a/b: Degradation and recovery."""

    def test_tc6a_trust_degrades_after_failures(self):
        """TC-6a: trust_factor = 1.0 × 0.95^10 ≈ 0.5987 after 10 failures."""
        profiles = {Model.GPT_4O: _make_profile(Model.GPT_4O, "openai")}
        telemetry = TelemetryCollector(profiles)
        for _ in range(10):
            telemetry.record_call(Model.GPT_4O, 1000.0, 0.01, success=False)
        trust = profiles[Model.GPT_4O].trust_factor
        expected = 1.0 * (0.95 ** 10)
        assert abs(trust - expected) < 1e-6, f"Expected {expected:.6f}, got {trust:.6f}"

    def test_tc6b_trust_recovers_and_never_exceeds_cap(self):
        """TC-6b: trust_factor recovers after successes and never exceeds 1.0."""
        profiles = {Model.GPT_4O: _make_profile(Model.GPT_4O, "openai")}
        telemetry = TelemetryCollector(profiles)
        # First degrade it
        for _ in range(5):
            telemetry.record_call(Model.GPT_4O, 1000.0, 0.01, success=False)
        trust_after_failures = profiles[Model.GPT_4O].trust_factor
        assert trust_after_failures < 1.0

        # Then recover with many successes
        for _ in range(1000):
            telemetry.record_call(Model.GPT_4O, 1000.0, 0.01, success=True)
        trust_after_recovery = profiles[Model.GPT_4O].trust_factor
        # Should recover close to 1.0 but never exceed it
        assert trust_after_recovery <= 1.0
        assert trust_after_recovery > trust_after_failures


# ─────────────────────────────────────────────────────────────────────────────
# TC-7  select_reviewer()
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectReviewer:
    """TC-7a/b: Cross-provider preference, same-provider fallback."""

    def test_tc7a_prefers_cross_provider_reviewer(self):
        """TC-7a: select_reviewer() returns a model from a different provider."""
        profiles = {
            Model.GPT_4O:       _make_profile(Model.GPT_4O,       "openai",    task_types={TaskType.CODE_REVIEW: 0}),
            Model.CLAUDE_SONNET:_make_profile(Model.CLAUDE_SONNET, "anthropic", task_types={TaskType.CODE_REVIEW: 1}),
            Model.GEMINI_PRO:   _make_profile(Model.GEMINI_PRO,   "google",    task_types={TaskType.CODE_REVIEW: 2}),
        }
        planner = _make_planner(profiles)
        # Generator is GPT-4o (openai); reviewer should NOT be openai
        reviewer = planner.select_reviewer(
            generator=Model.GPT_4O,
            task_type=TaskType.CODE_REVIEW,
            policies=[],
            budget_remaining=100.0,
        )
        assert reviewer is not None
        assert reviewer != Model.GPT_4O
        from orchestrator.models import get_provider
        assert get_provider(reviewer) != "openai"

    def test_tc7b_falls_back_to_same_provider(self):
        """TC-7b: select_reviewer() falls back to same-provider when cross-provider unavailable."""
        # Only two openai models available for CODE_REVIEW
        profiles = {
            Model.GPT_4O:      _make_profile(Model.GPT_4O,      "openai", task_types={TaskType.CODE_REVIEW: 0}),
            Model.GPT_4O_MINI: _make_profile(Model.GPT_4O_MINI, "openai", task_types={TaskType.CODE_REVIEW: 1}),
        }
        planner = _make_planner(profiles)
        reviewer = planner.select_reviewer(
            generator=Model.GPT_4O,
            task_type=TaskType.CODE_REVIEW,
            policies=[],
            budget_remaining=100.0,
        )
        # Should fall back to GPT_4O_MINI (same provider, but different model)
        assert reviewer == Model.GPT_4O_MINI


# ─────────────────────────────────────────────────────────────────────────────
# TC-8  Quality EMA
# ─────────────────────────────────────────────────────────────────────────────

class TestQualityEMA:
    """TC-8a/b: Exponential Moving Average for quality_score."""

    def test_tc8a_quality_ema_one_sample(self):
        """TC-8a: quality_score EMA: 0.1×1.0 + 0.9×0.8 = 0.82 after one perfect score."""
        profiles = {Model.GPT_4O: _make_profile(Model.GPT_4O, "openai", quality=0.8)}
        telemetry = TelemetryCollector(profiles)
        telemetry.record_call(Model.GPT_4O, 1000.0, 0.01, success=True, quality_score=1.0)
        expected = 0.1 * 1.0 + 0.9 * 0.8
        actual = profiles[Model.GPT_4O].quality_score
        assert abs(actual - expected) < 1e-9, f"Expected {expected}, got {actual}"

    def test_tc8b_quality_unchanged_when_none_passed(self):
        """TC-8b: quality_score unchanged when quality_score=None is passed."""
        profiles = {Model.GPT_4O: _make_profile(Model.GPT_4O, "openai", quality=0.8)}
        telemetry = TelemetryCollector(profiles)
        telemetry.record_call(Model.GPT_4O, 1000.0, 0.01, success=True, quality_score=None)
        assert profiles[Model.GPT_4O].quality_score == 0.8


# ─────────────────────────────────────────────────────────────────────────────
# TC-9  build_default_profiles()
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildDefaultProfiles:
    """TC-9a/b/c: Profile factory correctness."""

    def test_tc9a_covers_all_models(self):
        """TC-9a: build_default_profiles() produces a profile for every Model enum value."""
        profiles = build_default_profiles()
        for model in Model:
            assert model in profiles, f"Missing profile for {model}"

    def test_tc9b_costs_match_cost_table(self):
        """TC-9b: Profile costs match COST_TABLE exactly."""
        profiles = build_default_profiles()
        for model, costs in COST_TABLE.items():
            profile = profiles[model]
            assert profile.cost_per_1m_input == costs["input"], (
                f"{model}: expected input cost {costs['input']}, "
                f"got {profile.cost_per_1m_input}"
            )
            assert profile.cost_per_1m_output == costs["output"], (
                f"{model}: expected output cost {costs['output']}, "
                f"got {profile.cost_per_1m_output}"
            )

    def test_tc9c_rank0_model_has_correct_priority(self):
        """TC-9c: Priority rank 0 model in ROUTING_TABLE has capable_task_types[type]=0."""
        profiles = build_default_profiles()
        for task_type, model_list in ROUTING_TABLE.items():
            if model_list:
                first_model = model_list[0]
                profile = profiles[first_model]
                assert task_type in profile.capable_task_types, (
                    f"{first_model} should be capable of {task_type}"
                )
                assert profile.capable_task_types[task_type] == 0, (
                    f"{first_model} should have rank 0 for {task_type}, "
                    f"got {profile.capable_task_types[task_type]}"
                )


# ─────────────────────────────────────────────────────────────────────────────
# TC-10  PolicySet.policies_for() merging
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicySetMerging:
    """Verifies that global + node-level policies are correctly merged."""

    def test_global_and_node_policies_merged(self):
        """policies_for() returns global policies + node-specific policies combined."""
        global_policy = Policy(name="global_gdpr", allow_training_on_output=False)
        node_policy = Policy(name="task_eu", allowed_regions=["eu"])
        policy_set = PolicySet(
            global_policies=[global_policy],
            node_policies={"task-42": [node_policy]},
        )
        merged = policy_set.policies_for("task-42")
        assert global_policy in merged
        assert node_policy in merged
        assert len(merged) == 2

    def test_policies_for_unknown_task_returns_global_only(self):
        """policies_for() for an unknown task_id returns only global policies."""
        global_policy = Policy(name="global", allow_training_on_output=False)
        policy_set = PolicySet(global_policies=[global_policy])
        result = policy_set.policies_for("nonexistent-task")
        assert result == [global_policy]


# ─────────────────────────────────────────────────────────────────────────────
# TC-11  ModelProfile.estimate_cost()
# ─────────────────────────────────────────────────────────────────────────────

class TestModelProfileEstimateCost:
    """Verifies cost estimation arithmetic."""

    def test_cost_estimation_arithmetic(self):
        """estimate_cost(1000, 1000) with 1.0/1M in + 5.0/1M out = 0.000001 + 0.000005 = 0.000006."""
        profile = _make_profile(
            Model.GPT_4O, "openai",
            cost_in=1.0, cost_out=5.0,
        )
        cost = profile.estimate_cost(1_000_000, 1_000_000)
        # 1.0 + 5.0 = 6.0 USD for 1M/1M tokens
        assert abs(cost - 6.0) < 1e-9

    def test_cost_estimation_fractional(self):
        """estimate_cost(800, 1200) as used for CODE_GEN typical tokens."""
        profile = _make_profile(
            Model.CLAUDE_HAIKU, "anthropic",
            cost_in=0.80, cost_out=4.0,
        )
        cost = profile.estimate_cost(800, 1200)
        expected = (800 * 0.80 + 1200 * 4.0) / 1_000_000
        assert abs(cost - expected) < 1e-12


# ─────────────────────────────────────────────────────────────────────────────
# TC-12  Unhealthy models excluded
# ─────────────────────────────────────────────────────────────────────────────

class TestApiHealthFiltering:
    """Verifies that unhealthy models are excluded from selection."""

    def test_unhealthy_model_excluded(self):
        """An unhealthy model should never be selected."""
        profiles = {
            Model.GPT_4O:       _make_profile(Model.GPT_4O,       "openai",    task_types={TaskType.CODE_GEN: 0}, quality=0.99),
            Model.CLAUDE_SONNET:_make_profile(Model.CLAUDE_SONNET, "anthropic", task_types={TaskType.CODE_GEN: 1}, quality=0.80),
        }
        engine = PolicyEngine()
        # GPT-4o is unhealthy
        health = {Model.GPT_4O: False, Model.CLAUDE_SONNET: True}
        planner = ConstraintPlanner(profiles, engine, health)
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert result == Model.CLAUDE_SONNET

    def test_all_unhealthy_returns_none(self):
        """When all models are unhealthy, select_model returns None."""
        profiles = {
            Model.GPT_4O: _make_profile(Model.GPT_4O, "openai"),
        }
        engine = PolicyEngine()
        health = {Model.GPT_4O: False}
        planner = ConstraintPlanner(profiles, engine, health)
        result = planner.select_model(TaskType.CODE_GEN, [], budget_remaining=100.0)
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# TC-13  record_policy_violation()
# ─────────────────────────────────────────────────────────────────────────────

class TestRecordPolicyViolation:
    """TC-13: trust_factor degrades on policy violation."""

    def test_policy_violation_degrades_trust(self):
        """record_policy_violation() degrades trust_factor by ×0.95."""
        profiles = {Model.GPT_4O: _make_profile(Model.GPT_4O, "openai", trust=1.0)}
        telemetry = TelemetryCollector(profiles)
        telemetry.record_policy_violation(Model.GPT_4O)
        expected = 1.0 * 0.95
        assert abs(profiles[Model.GPT_4O].trust_factor - expected) < 1e-9

    def test_policy_violation_unknown_model_no_error(self):
        """record_policy_violation() on unknown model should not raise."""
        profiles = {Model.GPT_4O: _make_profile(Model.GPT_4O, "openai")}
        telemetry = TelemetryCollector(profiles)
        # Should not raise
        telemetry.record_policy_violation(Model.KIMI_K2_5)


# ─────────────────────────────────────────────────────────────────────────────
# TC-14  pii_allowed and no_train compliance tags
# ─────────────────────────────────────────────────────────────────────────────

class TestComplianceTags:
    """Verifies PII and training consent tag checks."""

    def test_no_train_tag_required_when_training_disallowed(self):
        """Policy allow_training_on_output=False requires 'no_train' tag."""
        engine = PolicyEngine()
        profile = _make_profile(Model.GPT_4O, "openai", tags=[])  # no no_train tag
        policy = Policy(name="gdpr", allow_training_on_output=False)
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is False
        assert any("no_train" in v for v in result.violations)

    def test_model_with_no_train_tag_passes_training_policy(self):
        """A model with 'no_train' tag passes allow_training_on_output=False policy."""
        engine = PolicyEngine()
        profile = _make_profile(Model.GPT_4O, "openai", tags=["no_train"])
        policy = Policy(name="gdpr", allow_training_on_output=False)
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is True

    def test_pii_tag_required_when_pii_disallowed(self):
        """Policy pii_allowed=False requires 'pii_allowed' tag on the model."""
        engine = PolicyEngine()
        profile = _make_profile(Model.GPT_4O, "openai", tags=[])
        policy = Policy(name="no_pii", pii_allowed=False)
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is False
        assert any("pii_allowed" in v for v in result.violations)

    def test_model_with_pii_tag_passes_pii_policy(self):
        """A model with 'pii_allowed' tag passes pii_allowed=False policy."""
        engine = PolicyEngine()
        profile = _make_profile(Model.GPT_4O, "openai", tags=["pii_allowed"])
        policy = Policy(name="no_pii", pii_allowed=False)
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is True


# ─────────────────────────────────────────────────────────────────────────────
# TC-15  Latency SLA
# ─────────────────────────────────────────────────────────────────────────────

class TestLatencySLA:
    """Verifies latency constraint enforcement."""

    def test_high_latency_model_excluded_by_policy(self):
        """Model with avg_latency_ms above max_latency_ms fails the policy check."""
        engine = PolicyEngine()
        profile = _make_profile(Model.GPT_4O, "openai", latency=5000.0)
        policy = Policy(name="fast_sla", max_latency_ms=1000.0)
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is False
        assert any("latency" in v.lower() for v in result.violations)

    def test_low_latency_model_passes_sla(self):
        """Model within latency SLA passes the check."""
        engine = PolicyEngine()
        profile = _make_profile(Model.GPT_4O, "openai", latency=500.0)
        policy = Policy(name="fast_sla", max_latency_ms=1000.0)
        result = engine.check(Model.GPT_4O, profile, [policy])
        assert result.passed is True

    def test_latency_sla_affects_model_selection(self):
        """In ConstraintPlanner, high-latency models are excluded when SLA policy active."""
        profiles = {
            Model.GPT_4O:       _make_profile(Model.GPT_4O,       "openai",    latency=5000.0, task_types={TaskType.CODE_GEN: 0}),
            Model.CLAUDE_SONNET:_make_profile(Model.CLAUDE_SONNET, "anthropic", latency=500.0,  task_types={TaskType.CODE_GEN: 1}),
        }
        policy = Policy(name="fast_sla", max_latency_ms=1000.0)
        planner = _make_planner(profiles)
        result = planner.select_model(TaskType.CODE_GEN, [policy], budget_remaining=100.0)
        assert result == Model.CLAUDE_SONNET


# ─────────────────────────────────────────────────────────────────────────────
# TC-16  JobSpec
# ─────────────────────────────────────────────────────────────────────────────

class TestJobSpec:
    """Verifies JobSpec dataclass construction."""

    def test_jobspec_construction(self):
        """JobSpec can be constructed with all required fields."""
        budget = Budget(max_usd=5.0)
        policy_set = PolicySet(
            global_policies=[Policy(name="gdpr", allow_training_on_output=False)]
        )
        spec = JobSpec(
            project_description="Build a REST API",
            success_criteria="All endpoints return 200",
            budget=budget,
            policy_set=policy_set,
        )
        assert spec.project_description == "Build a REST API"
        assert spec.budget.max_usd == 5.0
        assert len(spec.policy_set.global_policies) == 1

    def test_jobspec_default_policy_set(self):
        """JobSpec uses an empty PolicySet by default."""
        spec = JobSpec(
            project_description="Test project",
            success_criteria="Works",
            budget=Budget(),
        )
        assert isinstance(spec.policy_set, PolicySet)
        assert spec.policy_set.global_policies == []
