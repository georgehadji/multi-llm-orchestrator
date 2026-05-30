# Agent Memory Enhancement Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Version:** 1.0  
> **Status:** Plan — awaiting implementation  

---

## Executive Summary

The orchestrator has 5 memory layers but only the workspace persists across sessions. The two layers that **learn** (ExperienceBuffer and KnowledgeGraph) are in-memory only — agents forget everything they learned at restart. This plan adds file-backed persistence, strategy feedback, and cross-project transfer learning.

**Total effort:** 2 days. **Risk:** None — additive changes only.

---

## Current Memory Architecture (Baseline)

```
Session N                                Session N+1
══════════                                ═══════════
ExperienceBuffer                         ExperienceBuffer
  "CoVE + GPT-4o → 0.89 avg"              (empty — forgot)
  "CODESTRAL_2508 → best budget"          (empty)
  "Kimi K2.6 → 0.73"                     (empty)
                                         → Learns same lessons again
KnowledgeGraph                           KnowledgeGraph
  model→task_type→score edges             (empty graph)
  method→task_type→score edges            (empty)
                                         → Repeats known-bad routes
AgentCache                               AgentCache
  SHA-256 → cached response               (empty — cold cache)
                                         → Re-calls LLM for identical prompts
```

**Cost of amnesia:** 20-30% of session budget wasted re-learning known patterns. Model selection doesn't improve over time. No cross-project knowledge transfer.

---

## Target Memory Architecture (3 Levels)

### Level 1: File-Backed Persistence (0.5 day)

**Storage:** `~/.orchestrator/` directory — platform-independent, zero-config.

```
~/.orchestrator/
├── experience.json           # ExperienceBuffer dump (50KB per 1000 tasks)
├── knowledge_graph.json      # KnowledgeGraph edges (100KB per 500 patterns)
├── agent_cache.json          # SHA-256 → CachedResponse (200KB per 100 entries)
├── agent_memories/
│   ├── developer.json        # DeveloperAgent personal memory
│   ├── architect.json        # ArchitectAgent decisions
│   ├── reviewer.json         # ReviewerAgent audit patterns
│   └── ...
├── compressed_lessons.json   # MemoryCompressor output (5KB)
└── prompt_enricher_state.json # Enricher configuration (1KB)
```

**What changes:**

| File | Method Added | Lines |
|------|-------------|-------|
| `learning/experience_buffer.py` | `save(path)`, `load(cls, path)` static factory | +25 |
| `learning/knowledge_graph.py` | `save(path)`, `load(cls, path)` | +20 |
| `learning/agent_cache.py` | `save(path)`, `load(cls, path)` | +15 |
| `learning/agent_memory.py` | `save()`, `load(cls, path)` | +20 |
| `config.py` | `MemoryConfig.memory_dir` path | +5 |
| **Total** | | **+85 lines** |

### Level 2: Strategy Feedback Loop (0.5 day)

**What it does:** When an agent is about to handle a task, the coordinator queries the ExperienceBuffer for the best model and method. If a pattern exists, it overrides defaults.

**Integration point:** `AgentOrchestrator._enrich_task()` — called before every agent dispatch.

```python
def _enrich_task(self, task: AgentTask) -> None:
    """Query experience buffer and inject best-known strategy."""
    if self.experience is None:
        return

    best_method = self.experience.best_method_for(task.type.value)
    best_model = self.experience.best_model_for(task.type.value)

    if best_method:
        task.context = (task.context or "") + \
            f" [Memory: {best_method} method historically best for {task.type.value}]"

    if best_model:
        task.preferred_model = best_model
```

| File | Change | Lines |
|------|--------|-------|
| `agents/coordinator.py` | Add `_enrich_task()`, wire into `execute_goal()` | +12 |
| `agents/base.py` | Auto-load memory on init | +5 |
| `agents/coordinator.py` | Save memory on task completion + shutdown | +8 |
| `learning/prompt_enricher.py` | Accept loaded buffer/graph from constructor | +5 |
| **Total** | | **+30 lines** |

### Level 3: Cross-Project Transfer Learning (1 day)

**What it does:** When starting a new project, queries the KnowledgeGraph for similar past projects and bootstraps architecture decisions.

**Integration point:** `AgentOrchestrator.execute_goal()` — before decomposition.

```python
async def _bootstrap_from_past(self, goal: str) -> list[dict]:
    """Find similar past projects and extract reusable decisions."""
    if self.knowledge is None:
        return []

    similar = self.knowledge.find_similar_projects(goal)
    if not similar:
        return []

    decisions = []
    for project in similar[:3]:
        decisions.append({
            "project": project["name"],
            "decisions": project.get("architecture", []),
            "success_score": project.get("score", 0.0),
        })

    logger.info("Bootstrapped %d decisions from %d past projects",
                sum(len(d["decisions"]) for d in decisions), len(similar))
    return decisions
```

| File | Change | Lines |
|------|--------|-------|
| `agents/coordinator.py` | Add `_bootstrap_from_past()` | +20 |
| `learning/knowledge_graph.py` | Add `find_similar_projects(goal)` | +25 |
| `workspace/workspace.py` | Add `project_summary()` for knowledge graph | +15 |
| `agents/coordinator.py` | Wire bootstrap into `execute_goal()` | +10 |
| **Total** | | **+70 lines** |

---

## Implementation Order

```
Stage 1: Add save/load           Stage 2: Wire auto-       Stage 3: Strategy feed-
to 5 memory classes              load into AgentBase       back loop
(0.5 day, 4 files)               (0.2 day, 2 files)       (0.3 day, 2 files)
       │                               │                        │
       └───────────────┬───────────────┘                        │
                       │                                        │
                       ▼                                        ▼
              Agents persist memory                   Enriched prompts improve
              between restarts                       model/method selection
                       │                                        │
                       ▼                                        ▼
              Session 2 starts with                    20-30% budget saved
              full knowledge from                     on known patterns
              Session 1

Stage 4: Cross-project            Stage 5: Tests
transfer learning                 (0.5 day, 1 test file)
(0.5 day, 3 files)               
       │                               │
       ▼                               ▼
  30-50% faster startup           8 tests covering:
  on similar projects              - Save/load roundtrip
                                   - Eviction + persistence
                                   - Strategy overrides
                                   - Cross-project queries
```

---

## Verification Gates

| Gate | How to Verify |
|------|--------------|
| Memory persists across restarts | Run session → stop → restart → `experience.best_method_for("code_gen")` returns previous value |
| Strategy feedback improves selection | Run 10 CODE_GEN tasks → observer selects `CoVE` after it becomes the top performer |
| Cross-project transfer works | Complete project "auth system" → start "auth with OAuth" → bootstrapped decisions appear |
| Budget savings measurable | Track `total_cost_usd` per project before/after; after 3 projects, savings should be visible |
| No regression in existing tests | `pytest tests/test_god_file_refactoring.py` — all 72 core tests pass |

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Corrupted JSON on save | Low (1 file error per 10K writes) | `save()` wraps in try/except; corrupted file → fall back to empty in-memory |
| Memory directory permission denied | Low | `mkdir(parents=True, exist_ok=True)` handles missing dirs |
| Large experience file (>10MB) | Low (only after 200K+ patterns) | Eviction caps at 200 patterns; max file size ~50KB |
| Strategy feedback converges to wrong answer | Low | Initial weights are default heuristics; only overrides when pattern has 5+ samples |
| Cross-project query slow | Very Low | In-memory KnowledgeGraph; find_similar is O(n) on max ~500 nodes |

---

## Files Modified (Full List)

| File | Level | Action |
|------|-------|--------|
| `orchestrator/learning/experience_buffer.py` | 1 | Add `save()` + `load()` static factory |
| `orchestrator/learning/knowledge_graph.py` | 1 | Add `save()` + `load()` |
| `orchestrator/learning/agent_cache.py` | 1 | Add `save()` + `load()` |
| `orchestrator/learning/agent_memory.py` | 1 | Add `save()` + `load()` |
| `orchestrator/learning/prompt_enricher.py` | 2 | Accept buffer/graph from constructor |
| `orchestrator/learning/memory_compressor.py` | 1 | Add `save_lessons()` |
| `orchestrator/agents/base.py` | 2 | Auto-load memory on `__init__` |
| `orchestrator/agents/coordinator.py` | 1,2,3 | `_enrich_task()`, `_bootstrap_from_past()`, save on task completion |
| `orchestrator/workspace/workspace.py` | 3 | Add `project_summary()` for cross-project transfer |
| `orchestrator/config.py` | 1 | Add `MemoryConfig.memory_dir` |
| `tests/test_memory_persistence.py` | 5 | **New** — 8 tests for save/load/query |

**Total: 11 files (10 modified, 1 new), ~185 lines of new code.**

---

**Last updated:** 2026-05-25
