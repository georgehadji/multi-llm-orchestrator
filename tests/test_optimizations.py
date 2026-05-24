"""
Tests for optimization modules A-2, B-5, C-8, D-10.
"""

import pytest


class TestAgentCache:
    """D-10: Agent call caching."""

    def test_cache_miss(self):
        from orchestrator.learning.agent_cache import AgentCache
        cache = AgentCache()
        key = cache.make_key("write hello world")
        result = cache.get(key)
        assert result is None

    def test_cache_hit(self):
        from orchestrator.learning.agent_cache import AgentCache
        cache = AgentCache()
        key = cache.make_key("write hello")
        cache.put(key, "print('hello')", 0.85)
        result = cache.get(key)
        assert result is not None
        assert result.output == "print('hello')"

    def test_cache_key_uniqueness(self):
        from orchestrator.learning.agent_cache import AgentCache
        cache = AgentCache()
        k1 = cache.make_key("task1", "ctx1", "model-a")
        k2 = cache.make_key("task2", "ctx2", "model-b")
        assert k1 != k2

    def test_cache_clear(self):
        from orchestrator.learning.agent_cache import AgentCache
        cache = AgentCache()
        cache.put("k", "v", 1.0)
        cache.clear()
        assert cache.size == 0

    def test_expired_entry(self):
        from orchestrator.learning.agent_cache import AgentCache, CachedResponse, CACHE_TTL_SECONDS
        import time
        cache = AgentCache()
        key = "test_expired"
        cache.put(key, "test", 0.5)
        # Force expiry by moving timestamp back
        entry = cache._cache.get(key)
        if entry:
            entry.timestamp = time.time() - CACHE_TTL_SECONDS - 1
        result = cache.get(key)
        assert result is None


class TestAgentRateLimiter:
    """C-8: Rate limiting."""

    def test_allows_first_call(self):
        from orchestrator.agents.rate_limiter import AgentRateLimiter, RateLimit
        limiter = AgentRateLimiter()
        limiter.set_limit("dev", RateLimit(max_calls=3))
        assert limiter.check("dev") is True

    def test_blocks_excess_calls(self):
        from orchestrator.agents.rate_limiter import AgentRateLimiter, RateLimit
        limiter = AgentRateLimiter()
        limiter.set_limit("dev", RateLimit(max_calls=2))
        assert limiter.check("dev") is True
        assert limiter.check("dev") is True
        assert limiter.check("dev") is False

    def test_allows_unlimited_when_no_limit(self):
        from orchestrator.agents.rate_limiter import AgentRateLimiter
        limiter = AgentRateLimiter()
        for _ in range(100):
            assert limiter.check("unknown") is True

    def test_blocks_on_cost(self):
        from orchestrator.agents.rate_limiter import AgentRateLimiter, RateLimit
        limiter = AgentRateLimiter()
        limiter.set_limit("expensive", RateLimit(max_cost_usd=5.0))
        assert limiter.check("expensive", cost=4.0) is True
        assert limiter.check("expensive", cost=2.0) is False


class TestKnowledgeGraph:
    """A-3: Knowledge graph integration."""

    def test_record_success(self):
        from orchestrator.learning.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.record_success("code_gen", "gpt-4o", "cove", 0.85)
        assert len(kg.nodes) == 3
        assert len(kg.edges) >= 4

    def test_best_method(self):
        from orchestrator.learning.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.record_success("code_gen", "model_a", "basic", 0.6)
        kg.record_success("code_gen", "model_b", "cove", 0.95)
        best = kg.best_method_for("code_gen")
        assert best == "cove"

    def test_best_method_none(self):
        from orchestrator.learning.knowledge_graph import KnowledgeGraph
        assert KnowledgeGraph().best_method_for("unknown") is None

    def test_failures_for_model(self):
        from orchestrator.learning.knowledge_graph import KnowledgeGraph
        kg = KnowledgeGraph()
        kg.add_edge("model:gpt-4o", "tt:code_review", "failed_on")
        kg.add_edge("model:gpt-4o", "tt:code_gen", "failed_on")
        fails = kg.failures_for_model("gpt-4o")
        assert len(fails) == 2


class TestAuditTrail:
    """C-9: Security audit trail."""

    def test_record_entry(self):
        from orchestrator.workspace.audit import AuditTrail
        audit = AuditTrail()
        entry = audit.record("dev", "FILE_CREATED", "main.py")
        assert entry.agent == "dev"
        assert entry.action == "FILE_CREATED"

    def test_get_recent(self):
        from orchestrator.workspace.audit import AuditTrail
        audit = AuditTrail()
        audit.record("a", "TOOL_EXECUTED", "shell cmd")
        assert len(audit.get_recent(limit=10)) == 1

    def test_export_json(self):
        from orchestrator.workspace.audit import AuditTrail
        audit = AuditTrail()
        audit.record("dev", "MODEL_CALL", "gpt-4o", duration_ms=1500)
        exported = audit.export_json()
        assert "gpt-4o" in exported
        assert "dev" in exported

    def test_clear(self):
        from orchestrator.workspace.audit import AuditTrail
        audit = AuditTrail()
        audit.record("dev", "TOOL_EXECUTED", "ls")
        audit.clear()
        assert len(audit.get_recent()) == 0
