"""
Hunt T20 — wiring gaps: registered but never read.

`knowledge_rerank_enabled` was declared in `crosscutting/config.py` and read
by nothing in the repository. The feature behind it was fully built and
tested: `KnowledgeBase.find_similar(rerank=..., fetch_k=...)` implements the
two-stage cosine → LLM rerank, `infrastructure/reranker.py` provides the
reranker, and `tests/unit/test_knowledge_rerank.py` covers five cases.

`implementation_plan_reranking.md` spells out the final step that was never
taken:

    Caller (`find_similar` consumers, e.g. `find_similar(top_k=3)`) passes
    `rerank=flags.knowledge_rerank_enabled`.

That consumer is `KnowledgeBase.get_recommendations`, which `CAPABILITIES.md`
and `USAGE_GUIDE.md` both document as the public way to query the knowledge
base. So a user who set `ORCH_KNOWLEDGE_RERANK_ENABLED=true` and followed the
guide got no reranking and no warning.
"""

from __future__ import annotations

import pytest

from orchestrator import knowledge_base as kb_mod
from orchestrator.knowledge_base import KnowledgeBase

pytestmark = pytest.mark.unit


@pytest.fixture
def kb(tmp_path, monkeypatch):
    monkeypatch.setattr(KnowledgeBase, "_load_index", lambda self: None)
    return KnowledgeBase(storage_path=tmp_path)


async def _captured_rerank(kb, monkeypatch, enabled: bool):
    """Run get_recommendations and report the `rerank` it passed down."""
    seen: dict[str, object] = {}

    async def fake_find_similar(query, *a, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(kb, "find_similar", fake_find_similar)
    monkeypatch.setattr(kb_mod.flags, "knowledge_rerank_enabled", enabled)
    await kb.get_recommendations("build a payment service")
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [True, False])
async def test_flag_reaches_find_similar(kb, monkeypatch, enabled):
    """The flag must actually gate stage-2 rerank, not sit unread."""
    seen = await _captured_rerank(kb, monkeypatch, enabled)

    assert "rerank" in seen, (
        "get_recommendations must forward the rerank flag; without it "
        "knowledge_rerank_enabled has no effect anywhere in the codebase"
    )
    assert seen["rerank"] is enabled


def test_flag_is_read_somewhere_in_the_package():
    """Guards the general shape: a declared flag nobody consults."""
    import pathlib

    hits = [
        p
        for p in pathlib.Path("orchestrator").rglob("*.py")
        if "knowledge_rerank_enabled" in p.read_text(encoding="utf-8", errors="ignore")
        and p.name != "config.py"
    ]
    assert hits, "knowledge_rerank_enabled is declared but read by no module"
