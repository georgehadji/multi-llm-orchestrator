"""Tests for nadirclaw.cache — prompt caching for chat completions."""

import time

from nadirclaw.cache import PromptCache, _make_cache_key


class TestMakeCacheKey:
    def test_same_messages_same_key(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _make_cache_key("gpt-4", msgs)
        k2 = _make_cache_key("gpt-4", msgs)
        assert k1 == k2

    def test_different_model_different_key(self):
        msgs = [{"role": "user", "content": "hello"}]
        k1 = _make_cache_key("gpt-4", msgs)
        k2 = _make_cache_key("gpt-3.5", msgs)
        assert k1 != k2

    def test_different_messages_different_key(self):
        k1 = _make_cache_key("gpt-4", [{"role": "user", "content": "hello"}])
        k2 = _make_cache_key("gpt-4", [{"role": "user", "content": "world"}])
        assert k1 != k2

    def test_key_is_hex_string(self):
        key = _make_cache_key("model", [{"role": "user", "content": "test"}])
        assert isinstance(key, str)
        assert len(key) == 64  # sha256 hex


class TestPromptCache:
    def test_put_and_get(self):
        cache = PromptCache(max_size=10, ttl=60)
        msgs = [{"role": "user", "content": "hello"}]
        response = {"content": "hi", "finish_reason": "stop", "prompt_tokens": 5, "completion_tokens": 2}

        cache.put("gpt-4", msgs, response)
        result = cache.get("gpt-4", msgs)
        assert result == response

    def test_miss_returns_none(self):
        cache = PromptCache(max_size=10, ttl=60)
        result = cache.get("gpt-4", [{"role": "user", "content": "hello"}])
        assert result is None

    def test_ttl_expiry(self):
        cache = PromptCache(max_size=10, ttl=1)
        msgs = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4", msgs, {"content": "hi", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})

        # Should hit
        assert cache.get("gpt-4", msgs) is not None

        # Wait for expiry
        time.sleep(1.1)
        assert cache.get("gpt-4", msgs) is None

    def test_lru_eviction(self):
        cache = PromptCache(max_size=2, ttl=60)

        cache.put("m", [{"role": "user", "content": "a"}], {"content": "1", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})
        cache.put("m", [{"role": "user", "content": "b"}], {"content": "2", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})
        cache.put("m", [{"role": "user", "content": "c"}], {"content": "3", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})

        # "a" should be evicted
        assert cache.get("m", [{"role": "user", "content": "a"}]) is None
        assert cache.get("m", [{"role": "user", "content": "b"}]) is not None
        assert cache.get("m", [{"role": "user", "content": "c"}]) is not None

    def test_stats(self):
        cache = PromptCache(max_size=10, ttl=60)
        msgs = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4", msgs, {"content": "hi", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})

        cache.get("gpt-4", msgs)  # hit
        cache.get("gpt-4", [{"role": "user", "content": "miss"}])  # miss

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5
        assert stats["entries"] == 1

    def test_clear(self):
        cache = PromptCache(max_size=10, ttl=60)
        cache.put("m", [{"role": "user", "content": "a"}], {"content": "x", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})
        cache.clear()

        stats = cache.get_stats()
        assert stats["entries"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_different_model_no_hit(self):
        cache = PromptCache(max_size=10, ttl=60)
        msgs = [{"role": "user", "content": "hello"}]
        cache.put("gpt-4", msgs, {"content": "hi", "finish_reason": "stop", "prompt_tokens": 0, "completion_tokens": 0})

        assert cache.get("gpt-3.5", msgs) is None
