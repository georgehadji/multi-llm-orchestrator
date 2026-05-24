# Agent Memory Optimization Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-24  
> **Version:** 1.0  
> **Status:** Plan → Implementation Ready  

---

## Executive Summary

The AI Orchestrator has a functional but fragmented memory architecture. Agents collect data silently but never learn from it. This plan delivers **6 optimizations across 6 days** that transform the system from data-collecting to genuinely memory-aware.

---

## Current State (5 Memory Layers)

```
Layer 1: ProjectWorkspace     → In-memory, per-session file versions + decisions
Layer 2: PersistentWorkspace  → SQLite-backed for crash recovery
Layer 3: ExperienceBuffer     → Per-task-type success/failure patterns
Layer 4: KnowledgeGraph       → Relational (TaskType, Model, Method, Score) edges
Layer 5: AgentCache           → SHA-256 hash → LLM response, TTL-based expiry
```

**The gap:** Layers 3-5 collect data but **never feed back** into agent prompts or decisions. The `AgentOrchestrator` dispatches tasks with the same blank context every time. The system records "CoVE + GPT_5 works best for CODE_GEN" but never uses that knowledge.

---

## Optimizations

### Opt 1: Memory Injection into Agent Prompts (HIGHEST PRIORITY)

**Current:** `ExperienceBuffer.record_success()` stores data, but `AgentOrchestrator` never reads it.

**Target:** Before dispatching a task, query the `ExperienceBuffer` and inject relevant memories into the agent's context.

**Implementation:** ~20 lines in `AgentOrchestrator._enrich_prompt()`

```python
async def _enrich_prompt(self, task: AgentTask, buffer: ExperienceBuffer) -> str:
    """Inject relevant past experiences into task context."""
    context = task.context
    best_method = buffer.best_method_for(task.goal[:50])
    best_model = buffer.best_model_for(task.goal[:50])
    if best_method:
        context += f"\n[Past experience: {best_method} method worked best for similar tasks]"
    if best_model:
        context += f"\n[Best model for this: {best_model}]"
    return context
```

**Effort:** 0.5 day | **Files:** `agents/coordinator.py`

### Opt 2: Agent-Specific Memory (HIGH PRIORITY)

**Current:** All agents share a single `ProjectWorkspace`. No per-agent "private" memory.

**Target:** Each agent gets an `AgentMemory` that tracks:
- Personal successes/failures
- Preferred approaches learned over time
- Tool call patterns that worked best

**Implementation:**

```python
@dataclass
class AgentMemory:
    agent_id: str
    successes: list[SuccessPattern]  # Agent-specific success patterns
    failures: list[FailurePattern]   # What this agent tends to get wrong
    tool_effectiveness: dict[str, float]  # Tool → success rate
    last_n_tasks: list[AgentTaskResult]  # Sliding window of recent results

    def lesson_learned(self, task_type: str) -> str | None:
        """Get a lesson this agent has learned for this task type."""
        # Return the most recent lesson matching this task type
```

**Integration:** `AgentBase.__init__` creates an `AgentMemory`. `DeveloperAgent.handle_task()` calls `self.memory.record_success()` on completion.

**Effort:** 1 day | **Files:** `agents/base.py`, `learning/agent_memory.py`

### Opt 3: Priority-Based Memory Eviction (MEDIUM PRIORITY)

**Current:** All memories are equal. No eviction strategy.

**Target:** LRU + importance-weighted eviction for long-running sessions. Lower-priority patterns (low score impact) are evicted before high-priority ones.

**Implementation:**

```python
class EvictionPolicy:
    """Decides which memories to forget when capacity is reached."""
    
    def evict(self, memories: list, max_size: int) -> list:
        # 1. Sort by priority (critical > high > medium > low)
        # 2. Within same priority, evict oldest (LRU)
        # 3. Preserve memories with highest score impact
```

**Effort:** 0.5 day | **Files:** `learning/experience_buffer.py`

### Opt 4: Cross-Session Knowledge Persistence (MEDIUM PRIORITY)

**Current:** `KnowledgeGraph` is in-memory, lost on restart.

**Target:** `KnowledgeGraph` persists to a `.orchestrator/knowledge.db` SQLite file. On construction, loads existing state. On `record_success()`, auto-saves.

**Implementation:** Same pattern as `PersistentWorkspace` — SQLite table with nodes + edges, async save after each mutation.

**Effort:** 0.5 day | **Files:** `learning/knowledge_graph.py`

### Opt 5: Memory Compression via Summarization (LOW PRIORITY)

**Current:** Raw experiences stored forever. After 100+ runs, the buffer is diluted with noise.

**Target:** When a task type has >10 similar success patterns, compress them into a single "lesson learned" entry: "For CODE_GEN tasks, CoVE + GPT_5 has a 0.92 avg success rate over 12 runs."

**Implementation:**

```python
class MemoryCompressor:
    def compress(self, buffer: ExperienceBuffer, task_type: str) -> str:
        """Summarize similar patterns into a lesson."""
        patterns = [p for p in buffer.successes if p.task_type == task_type]
        if len(patterns) > 10:
            avg_score = sum(p.score for p in patterns) / len(patterns)
            best_method = max(set(p.method for p in patterns), key=...)
            return f"[Lesson] {task_type}: {best_method} achieves {avg_score:.2f} avg score over {len(patterns)} runs"
```

**Effort:** 1 day | **Files:** `learning/memory_compressor.py`

### Opt 6: AgentPromptEnricher — Unified Memory Query (HIGH PRIORITY)

**Current:** No single interface to query all memory layers before agent dispatch.

**Target:** A `AgentPromptEnricher` that queries all memory layers (ExperienceBuffer + KnowledgeGraph + AgentMemory + AgentCache) and returns an enriched context string.

**Implementation:**

```python
class AgentPromptEnricher:
    def __init__(self, buffer, graph, agent_memories, cache):
        ...
    
    async def enrich(self, task: AgentTask, agent_id: str) -> str:
        context = task.context
        # 1. Query experience buffer for best method/model
        # 2. Query knowledge graph for similar past patterns
        # 3. Query agent memory for personal lessons
        # 4. Check cache for identical task (instant return)
        # 5. Combine into enriched context string
        return context
```

**Effort:** 0.5 day | **Files:** `learning/prompt_enricher.py`

---

## Implementation Order

| # | Optimization | Days | Depends On | Priority |
|---|-------------|------|------------|----------|
| Opt 1 | Memory injection into prompts | 0.5 | — | **HIGHEST** |
| Opt 6 | AgentPromptEnricher (unified query) | 0.5 | Opt 1 | **HIGH** |
| Opt 2 | Agent-specific memory | 1 | — | **HIGH** |
| Opt 3 | Priority-based eviction | 0.5 | Opt 2 | MEDIUM |
| Opt 4 | Cross-session persistence | 0.5 | — | MEDIUM |
| Opt 5 | Memory compression | 1 | Opt 3 | LOW |
| **Total** | | **4 days** | | |

**First sprint (2 days):** Opt 1 + Opt 6 + Opt 2 — immediate impact: agents actually learn.

**Second sprint (2 days):** Opt 3 + Opt 4 + Opt 5 — scale and persistence.

---

## Files to Create / Modify

| File | Action | Optimization |
|------|--------|-------------|
| `orchestrator/agents/coordinator.py` | Modify (add _enrich_prompt) | Opt 1 |
| `orchestrator/learning/prompt_enricher.py` | **Create** | Opt 6 |
| `orchestrator/learning/agent_memory.py` | **Create** | Opt 2 |
| `orchestrator/agents/base.py` | Modify (AgentMemory field) | Opt 2 |
| `orchestrator/learning/experience_buffer.py` | Modify (eviction) | Opt 3 |
| `orchestrator/learning/knowledge_graph.py` | Modify (SQLite persistence) | Opt 4 |
| `orchestrator/learning/memory_compressor.py` | **Create** | Opt 5 |

**Total: 7 files (3 new, 4 modified)**

---

## Verification Gates

- [ ] Agent receives enriched prompt with past experience injected
- [ ] `AgentPromptEnricher.enrich()` returns context with past success patterns
- [ ] DeveloperAgent has `self.memory` tracking personal successes
- [ ] After 15 similar tasks, memory compressor summarizes into a lesson
- [ ] Knowledge graph persists across sessions (restart → graph intact)
- [ ] Eviction removes lowest-priority memories first when buffer is full
- [ ] All existing 176+ tests pass

---

**Last updated:** 2026-05-24
