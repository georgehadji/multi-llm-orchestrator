# tests/test_semantic_cache.py
import asyncio
from orchestrator.semantic_cache import SemanticCache, DuplicationDetector


def test_exact_prompt_returns_cached():
    cache = SemanticCache()
    asyncio.run(cache.put("t1", "code_gen", "def hello(): return 42",
                           prompt="Write hello world function"))
    result = asyncio.run(cache.get("t2", "code_gen",
                                   "Write hello world function"))
    assert result == "def hello(): return 42"


def test_different_task_type_no_hit():
    cache = SemanticCache()
    asyncio.run(cache.put("t1", "code_gen", "some code",
                           prompt="Write hello world"))
    result = asyncio.run(cache.get("t2", "code_review",
                                   "Write hello world"))
    assert result is None   # different task type — no cross-type reuse


def test_no_hit_on_unrelated_prompt():
    cache = SemanticCache()
    asyncio.run(cache.put("t1", "code_gen", "def hello(): return 42",
                           prompt="Write hello world function"))
    result = asyncio.run(cache.get("t2", "code_gen",
                                   "Analyze the database schema for performance"))
    assert result is None


def test_cache_get_returns_none_when_empty():
    cache = SemanticCache()
    result = asyncio.run(cache.get("t1", "code_gen", "anything"))
    assert result is None


def test_cache_size_increments():
    cache = SemanticCache()
    assert cache.size() == 0
    asyncio.run(cache.put("t1", "code_gen", "output", prompt="prompt"))
    assert cache.size() == 1
    asyncio.run(cache.put("t2", "code_gen", "output2", prompt="prompt2"))
    assert cache.size() == 2


def test_duplication_detector_finds_similar():
    det = DuplicationDetector(similarity_threshold=0.85)
    tasks = {
        "t1": {"prompt": "Write a Dockerfile for a Python FastAPI app"},
        "t2": {"prompt": "Write a Dockerfile for a FastAPI Python application"},
        "t3": {"prompt": "Write unit tests for the auth service"},
    }
    groups = det.find_duplicate_groups(tasks)
    # t1 and t2 should be grouped together
    assert any("t1" in g and "t2" in g for g in groups), f"groups={groups}"
    # t3 should NOT be grouped with t1
    assert not any("t1" in g and "t3" in g for g in groups)


def test_duplication_detector_no_duplicates():
    det = DuplicationDetector()
    tasks = {
        "t1": {"prompt": "Write a Dockerfile"},
        "t2": {"prompt": "Analyze the database schema"},
        "t3": {"prompt": "Create unit tests for authentication"},
    }
    groups = det.find_duplicate_groups(tasks)
    assert len(groups) == 0 or all(len(g) == 1 for g in groups)


def test_duplication_detector_empty():
    det = DuplicationDetector()
    groups = det.find_duplicate_groups({})
    assert groups == []
