"""Unified memory query for agent prompt enrichment."""

from __future__ import annotations
from typing import Any


class AgentPromptEnricher:
    def __init__(self, buffer=None, graph=None, memories=None, cache=None):
        self._buffer = buffer
        self._graph = graph
        self._memories = memories or {}
        self._cache = cache

    async def enrich(self, task: Any, agent_id: str = "") -> str:
        context = getattr(task, "context", "")
        parts = []
        target = getattr(task, "goal", "")[:50]
        if self._graph and target:
            try:
                best = self._graph.best_method_for(target)
                if best:
                    parts.append(f"[Best method: {best}]")
            except Exception:
                pass
        if agent_id and agent_id in self._memories:
            lesson = self._memories[agent_id].lesson()
            if lesson:
                parts.append(lesson)
        if self._buffer and target:
            try:
                m = self._buffer.best_method_for(target)
                if m:
                    parts.append(f"[Experience: {m} works best]")
            except Exception:
                pass
        enriched = (context + " " + " ".join(parts)).strip()
        return enriched if parts else context
