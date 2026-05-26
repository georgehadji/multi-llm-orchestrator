"""
Regression tests for three confirmed bugs.

BUG-001: AgentCache.save() did not persist timestamp → disk-loaded entries got
         a fresh TTL, bypassing expiry for up to 1 hour after restart.

BUG-002: KnowledgeGraph.find_similar_projects() checked edge.source == nid
         but all edges have task_type as TARGET, so the function always returned [].
         Additionally, it used the "used_with" edge weight (always 1.0) instead of
         the average "produced_score" weight.

BUG-003: BudgetEnforcer._check_budget_thresholds() divided by budget.max_usd
         without guarding against zero, raising ZeroDivisionError.
         get_budget_status() had the same flaw.
"""

import os
import tempfile
import time

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# BUG-001: AgentCache timestamp persistence
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentCacheTimestampPersistence:
    """BUG-001 — timestamp must survive a save/load round-trip."""

    def test_expired_entry_stays_expired_after_disk_roundtrip(self):
        """Entry that was past TTL before save must still be expired after load."""
        from orchestrator.learning.agent_cache import AgentCache, CACHE_TTL_SECONDS

        cache = AgentCache()
        key = cache.make_key("stale goal", "ctx", "model-x")
        cache.put(key, "cached output", 0.9)

        # Force the entry to look ancient (2× TTL ago)
        entry = cache._cache[key]
        entry.timestamp = time.time() - CACHE_TTL_SECONDS * 2

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            cache.save(path)
            loaded = AgentCache.load(path)
            # Without the fix, load() resets timestamp to now → hit instead of miss
            assert loaded.get(key) is None, (
                "Disk-loaded entry that was past TTL should remain expired"
            )
        finally:
            os.unlink(path)

    def test_fresh_entry_survives_disk_roundtrip(self):
        """Entry that is within TTL must still be valid after save/load."""
        from orchestrator.learning.agent_cache import AgentCache

        cache = AgentCache()
        key = cache.make_key("fresh goal", "ctx", "model-y")
        cache.put(key, "fresh output", 0.8)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
            path = fh.name
        try:
            cache.save(path)
            loaded = AgentCache.load(path)
            result = loaded.get(key)
            assert result is not None
            assert result.output == "fresh output"
        finally:
            os.unlink(path)

    def test_timestamp_is_written_to_json(self):
        """The saved JSON file must contain a 'timestamp' field."""
        import json
        from orchestrator.learning.agent_cache import AgentCache

        cache = AgentCache()
        key = cache.make_key("timestamped", "", "")
        cache.put(key, "some text", 0.5)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as fh:
            path = fh.name
        try:
            cache.save(path)
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            assert key in data
            assert "timestamp" in data[key], "timestamp field must be present in serialized JSON"
            assert isinstance(data[key]["timestamp"], float)
        finally:
            os.unlink(path)


# ─────────────────────────────────────────────────────────────────────────────
# BUG-002: KnowledgeGraph.find_similar_projects edge direction + score
# ─────────────────────────────────────────────────────────────────────────────


class TestKnowledgeGraphFindSimilarProjects:
    """BUG-002 — find_similar_projects must return results and correct scores."""

    def test_returns_nonempty_after_recording_success(self):
        """Must not always return [] — pre-fix it always did due to reversed edge check."""
        from orchestrator.learning.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.record_success("code_gen", "model-a", "cove", 0.9)

        # "write code" overlaps with "code" from "code_gen"
        results = kg.find_similar_projects("write code for a todo app")
        assert len(results) > 0, (
            "find_similar_projects returned [] even though code_gen was recorded"
        )

    def test_score_reflects_produced_score_not_used_with_weight(self):
        """Score must be the average produced_score, not the used_with edge weight (1.0)."""
        from orchestrator.learning.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.record_success("code_gen", "model-a", "cove", 0.8)
        kg.record_success("code_gen", "model-b", "basic", 0.6)

        results = kg.find_similar_projects("write code")
        assert len(results) > 0
        score = results[0]["score"]
        # Average of 0.8 and 0.6 per model edge + 0.8 and 0.6 per method edge = 0.7
        # (4 produced_score edges, all with weights 0.8 or 0.6)
        assert score != 1.0, "Score must not be the used_with edge weight (1.0)"
        assert 0.0 < score <= 1.0, f"Score must be in (0, 1], got {score}"

    def test_returns_empty_for_unrelated_goal(self):
        """Goals with no keyword overlap with 'code_gen' or 'reasoning' return []."""
        from orchestrator.learning.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.record_success("code_gen", "model-a", "cove", 0.9)

        results = kg.find_similar_projects("translate french to english")
        assert results == [], "Unrelated goal should return no results"

    def test_limit_is_respected(self):
        """Result list must not exceed the specified limit."""
        from orchestrator.learning.knowledge_graph import KnowledgeGraph

        kg = KnowledgeGraph()
        kg.record_success("code_gen", "model-a", "cove", 0.9)
        kg.record_success("reasoning", "model-b", "debate", 0.85)

        results = kg.find_similar_projects("code reasoning", limit=1)
        assert len(results) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# BUG-003: BudgetEnforcer division by zero
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetEnforcerZeroDivision:
    """BUG-003 — _check_budget_thresholds and get_budget_status must not crash on max_usd=0."""

    def _make_enforcer(self, max_usd: float):
        from orchestrator.application.budget_enforcer import BudgetEnforcer
        from orchestrator.models import Budget

        budget = Budget(max_usd=max_usd, max_time_seconds=3600.0)
        return BudgetEnforcer(budget=budget)

    def test_check_budget_zero_max_does_not_raise(self):
        """check_budget() must not raise ZeroDivisionError when max_usd=0."""
        enforcer = self._make_enforcer(max_usd=0.0)
        try:
            result = enforcer.check_budget()
            assert isinstance(result, tuple)
        except ZeroDivisionError:
            pytest.fail("check_budget() raised ZeroDivisionError with max_usd=0")

    def test_get_budget_status_zero_max_does_not_raise(self):
        """get_budget_status() must not raise ZeroDivisionError when max_usd=0."""
        enforcer = self._make_enforcer(max_usd=0.0)
        try:
            status = enforcer.get_budget_status()
            assert status["spent_percentage"] == 0.0
        except ZeroDivisionError:
            pytest.fail("get_budget_status() raised ZeroDivisionError with max_usd=0")

    def test_threshold_warnings_still_fire_for_normal_budget(self):
        """Threshold logic must still work correctly when max_usd > 0."""
        enforcer = self._make_enforcer(max_usd=10.0)
        enforcer.budget.spent_usd = 8.0  # 80% → above WARNING_THRESHOLD (75%)
        # Must not raise
        enforcer._check_budget_thresholds()
        assert "budget_75" in enforcer._warned_thresholds

    def test_should_halt_zero_budget_does_not_raise(self):
        """should_halt() must not crash when max_usd=0."""
        enforcer = self._make_enforcer(max_usd=0.0)
        try:
            result = enforcer.should_halt()
            assert isinstance(result, bool)
        except ZeroDivisionError:
            pytest.fail("should_halt() raised ZeroDivisionError with max_usd=0")


# ─────────────────────────────────────────────────────────────────────────────
# BUG-004: BudgetEnforcer.record_time() referenced nonexistent Budget.elapsed_time
# ─────────────────────────────────────────────────────────────────────────────


class TestBudgetEnforcerRecordTimeRemoved:
    """BUG-004 — record_time() referenced Budget.elapsed_time which does not exist.

    The method was unreachable (zero callers) so it was deleted.  This test
    ensures it is never re-introduced accidentally.
    """

    def test_record_time_does_not_exist(self):
        """BudgetEnforcer must not expose record_time() — it would raise AttributeError."""
        from orchestrator.application.budget_enforcer import BudgetEnforcer

        assert not hasattr(BudgetEnforcer, "record_time"), (
            "record_time() was re-added to BudgetEnforcer; "
            "it references Budget.elapsed_time which does not exist and would "
            "raise AttributeError at runtime."
        )


# ─────────────────────────────────────────────────────────────────────────────
# BUG-005: assumption_gate.surface_assumptions() called client.get_cheapest_model()
#          which does not exist on UnifiedClient → AttributeError every run,
#          caught silently, making assumption checking permanently broken.
# ─────────────────────────────────────────────────────────────────────────────


class TestAssumptionGateModelReference:
    """BUG-005 — _ASSUMPTION_MODEL must be a valid Model enum value, not a
    nonexistent client method call."""

    def test_assumption_model_is_valid_model_enum(self):
        """_ASSUMPTION_MODEL must be a Model enum member."""
        from orchestrator.assumption_gate import _ASSUMPTION_MODEL
        from orchestrator.models import Model

        assert isinstance(_ASSUMPTION_MODEL, Model), (
            "_ASSUMPTION_MODEL must be a Model enum; got "
            f"{type(_ASSUMPTION_MODEL).__name__}"
        )

    def test_assumption_gate_imports_without_error(self):
        """assumption_gate module must import cleanly (no AttributeError at import time)."""
        import importlib
        try:
            importlib.import_module("orchestrator.assumption_gate")
        except AttributeError as exc:
            pytest.fail(f"assumption_gate raised AttributeError on import: {exc}")

    def test_surface_assumptions_uses_model_not_client_method(self):
        """surface_assumptions must not call get_cheapest_model() on the client."""
        import inspect
        from orchestrator import assumption_gate

        src = inspect.getsource(assumption_gate.surface_assumptions)
        assert "get_cheapest_model" not in src, (
            "surface_assumptions() still calls client.get_cheapest_model() "
            "which does not exist on UnifiedClient"
        )
