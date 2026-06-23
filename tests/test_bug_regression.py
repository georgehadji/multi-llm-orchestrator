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
            assert (
                loaded.get(key) is None
            ), "Disk-loaded entry that was past TTL should remain expired"
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
        assert (
            len(results) > 0
        ), "find_similar_projects returned [] even though code_gen was recorded"

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
            "_ASSUMPTION_MODEL must be a Model enum; got " f"{type(_ASSUMPTION_MODEL).__name__}"
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


# ─────────────────────────────────────────────────────────────────────────────
# BUG-006: self_consistency.py used FALLBACK_CHAIN.get(model, model) — the
#          default is the model itself, so for all v3.0 primary routing models
#          not yet in FALLBACK_CHAIN the retry reused the same model (useless).
# ─────────────────────────────────────────────────────────────────────────────


class TestFallbackChainV3Models:
    """BUG-006 — every ROUTING_TABLE primary model must have a distinct
    FALLBACK_CHAIN entry so self-consistency retries actually switch models."""

    def test_all_routing_table_primaries_have_fallback(self):
        """Primary model of each task type must appear in FALLBACK_CHAIN."""
        from orchestrator.models import FALLBACK_CHAIN, ROUTING_TABLE

        missing = []
        for task_type, models in ROUTING_TABLE.items():
            primary = models[0]
            if primary not in FALLBACK_CHAIN:
                missing.append(f"{task_type.value}: {primary.value}")

        assert not missing, (
            "ROUTING_TABLE primary models with no FALLBACK_CHAIN entry "
            "(self-consistency retry would reuse same model):\n  " + "\n  ".join(missing)
        )

    def test_fallback_is_different_from_primary(self):
        """Each ROUTING_TABLE primary's fallback must differ from itself."""
        from orchestrator.models import FALLBACK_CHAIN, ROUTING_TABLE

        same_model_fallbacks = []
        for task_type, models in ROUTING_TABLE.items():
            primary = models[0]
            fb = FALLBACK_CHAIN.get(primary)
            if fb is not None and fb == primary:
                same_model_fallbacks.append(f"{task_type.value}: {primary.value}")

        assert not same_model_fallbacks, (
            "These primary models fall back to themselves "
            "(self-consistency retry is a no-op):\n  " + "\n  ".join(same_model_fallbacks)
        )

    def test_new_v3_models_specifically_have_fallback(self):
        """The four v3.0 models that were specifically missing must be present."""
        from orchestrator.models import FALLBACK_CHAIN, Model

        v3_primaries = [
            Model.XIAOMI_MIMO_V2_FLASH,
            Model.XAI_GROK_4_20,
            Model.STEPFUN_STEP_3_5_FLASH,
            Model.ZHIPU_GLM_5_2,
        ]
        for model in v3_primaries:
            assert model in FALLBACK_CHAIN, (
                f"{model.value} missing from FALLBACK_CHAIN; "
                "self_consistency retry would reuse same model"
            )
            assert (
                FALLBACK_CHAIN[model] != model
            ), f"{model.value} falls back to itself in FALLBACK_CHAIN"


# ─────────────────────────────────────────────────────────────────────────────
# BUG-007: persistent_workspace.py called asyncio.ensure_future() from sync
#          write_file() / record_decision() — without a running event loop
#          this emits DeprecationWarning (Python 3.10+) and the coroutine is
#          never awaited, silently losing the persistence call.
# ─────────────────────────────────────────────────────────────────────────────


class TestPersistentWorkspaceSyncSafe:
    """BUG-007 — write_file() and record_decision() must not raise or emit
    DeprecationWarning when called from a synchronous (non-async) context."""

    def _make_workspace(self):
        import tempfile
        from pathlib import Path
        from orchestrator.workspace.persistent_workspace import PersistentWorkspace

        tmp = tempfile.mkdtemp()
        return PersistentWorkspace(root=Path(tmp))

    def test_write_file_no_deprecation_warning_in_sync_context(self):
        """write_file() must not emit DeprecationWarning about missing event loop."""
        import warnings

        ws = self._make_workspace()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            # Must not raise DeprecationWarning: "There is no current event loop"
            try:
                ws.write_file("a.py", "x=1", author="test")
            except DeprecationWarning as exc:
                pytest.fail(f"write_file() emitted DeprecationWarning: {exc}")

    def test_record_decision_no_deprecation_warning_in_sync_context(self):
        """record_decision() must not emit DeprecationWarning about missing event loop."""
        import warnings

        ws = self._make_workspace()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                ws.record_decision("title", "decision", "rationale")
            except DeprecationWarning as exc:
                pytest.fail(f"record_decision() emitted DeprecationWarning: {exc}")

    def test_write_file_returns_file_version_in_sync_context(self):
        """write_file() must succeed (return FileVersion) even without event loop."""
        from orchestrator.workspace.workspace import FileVersion

        ws = self._make_workspace()
        result = ws.write_file("b.py", "y=2", author="bot")
        assert isinstance(result, FileVersion)
        assert result.content == "y=2"


# ─────────────────────────────────────────────────────────────────────────────
# BUG-008: ExperienceBuffer.successes / .failures grew without bound —
#          no cap applied, unlike AgentMemory which caps at 50.
#          Architecture spec documents 200 entries as the target limit.
# ─────────────────────────────────────────────────────────────────────────────


class TestExperienceBufferBoundedLists:
    """BUG-008 — successes and failures lists must not grow without bound."""

    def test_successes_capped_at_max_audit_size(self):
        """After _MAX_AUDIT_SIZE+N records, len(successes) == _MAX_AUDIT_SIZE."""
        from orchestrator.learning.experience_buffer import ExperienceBuffer

        buf = ExperienceBuffer()
        limit = buf._MAX_AUDIT_SIZE
        for i in range(limit + 50):
            buf.record_success("code_gen", "cove", f"model-{i}", 0.9)

        assert (
            len(buf.successes) == limit
        ), f"successes list not capped: len={len(buf.successes)}, expected {limit}"

    def test_failures_capped_at_max_audit_size(self):
        """After _MAX_AUDIT_SIZE+N records, len(failures) == _MAX_AUDIT_SIZE."""
        from orchestrator.learning.experience_buffer import ExperienceBuffer

        buf = ExperienceBuffer()
        limit = buf._MAX_AUDIT_SIZE
        for i in range(limit + 50):
            buf.record_failure("reasoning", "debate", f"model-{i}", 0.1)

        assert (
            len(buf.failures) == limit
        ), f"failures list not capped: len={len(buf.failures)}, expected {limit}"

    def test_best_model_unaffected_by_cap(self):
        """Capping the audit list must not affect best_model_for() accuracy."""
        from orchestrator.learning.experience_buffer import ExperienceBuffer

        buf = ExperienceBuffer()
        # Record 250 entries for model-a at score 0.5
        for i in range(250):
            buf.record_success("code_gen", "cove", "model-a", 0.5)
        # Record 50 entries for model-b at score 0.9 (recent wins)
        for i in range(50):
            buf.record_success("code_gen", "cove", "model-b", 0.9)

        # model_scores dict is NOT capped, so it still has all data
        best = buf.best_model_for("code_gen")
        assert (
            best == "model-b"
        ), f"best_model_for() should return model-b (score 0.9), got {best!r}"

    def test_max_audit_size_is_at_least_50(self):
        """_MAX_AUDIT_SIZE must be >= 50 (documented minimum in architecture)."""
        from orchestrator.learning.experience_buffer import ExperienceBuffer

        assert ExperienceBuffer._MAX_AUDIT_SIZE >= 50
