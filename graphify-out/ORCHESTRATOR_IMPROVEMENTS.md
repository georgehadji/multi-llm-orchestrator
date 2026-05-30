# AI Orchestrator — Improvement Vectors from Hermes Agent

**Date:** 2026-05-25
**Analysis:** Cross-project comparison between Multi-LLM Orchestrator (v6.0.0) and Hermes Agent (v0.14.0)

---

## Overview

The AI Orchestrator and Hermes Agent solve different problems but share the same substrate: Python-based LLM orchestration with tool calling, provider routing, and budget management. The Orchestrator excels at **deterministic task decomposition → generate → critique → revise → evaluate** cycles. Hermes excels at **general-purpose agent scaffolding** — learning loops, multi-platform delivery, plugin depth, and subagent delegation.

Below are 10 concrete improvement vectors, ranked by impact-to-effort ratio.

---

## Priority 1 — High Impact, Moderate Effort

### 1. Closed Learning Loop — Pattern Extraction from Successful Runs

**What Hermes does:**
- After complex tasks, agents autonomously call `skill_manage(action="create")` to capture reusable patterns as SKILL.md files
- Skills self-improve during use (tracked per-skill `use_count`, `view_count`, `patch_count` in `.usage.json`)
- Curator background process reviews, archives stale skills (never deletes)
- `created_by: "agent"` provenance prevents curator from touching bundled skills

**What the Orchestrator has:**
- `telemetry_store.py` — persistence for cross-run learning
- `evaluator.py` — 2-pass self-consistency evaluation with quality scoring
- `bm25_search.py` — SQLite FTS5 search infrastructure
- Task execution produces `TaskResult` with scores, attempt history, critique

**What it doesn't have:**
- No mechanism to capture *reusable patterns* from successful task executions across runs
- No skill/materialization of "what worked well last time"
- No feedback loop where generated code patterns inform future generations

**Implementation outline:**

```
orchestrator/
  └── pattern_learner/           # NEW subpackage
      ├── __init__.py
      ├── extractor.py           # PatternExtractor — analyzes TaskResult history,
      │                            extracts successful prompt→code→validation patterns
      ├── pattern_store.py       # SQLite-backed pattern library (pattern_id,
      │                            task_type, prompt_fingerprint, generated_code_hash,
      │                            quality_score, reuse_count, last_used_at)
      ├── curator.py             # PatternCurator — periodic review: archive stale,
      │                            promote high-scoring patterns, merge duplicates
      └── injector.py            # PatternInjector — at task generation time,
      │                            injects relevant past patterns into the prompt
      │                            context (behind an opt-in flag)
```

**Key design decisions to borrow from Hermes:**
- Patterns get `provenance: "agent"` vs `provenance: "bundled"` — curator only touches agent-created
- Never delete — archive to `.archive/` with restore capability
- Usage tracking: `reuse_count`, `last_reused_at`, `avg_score_on_reuse`
- Pattern injection is ephemeral (not persisted to project state) — preserves prompt caching

**Effort estimate:** ~8-12 new files, ~600-900 lines of new code  
**Impact:** Transforms the Orchestrator from a "stateless generator" to a system that improves with use

---

### 2. Subagent Delegation — Batch Parallelism for Task Groups

**What Hermes does:**
- `delegate_task` spawns isolated subagents with role gating
- Batch mode: `tasks: [...]` spawns concurrent subagents, parent waits for all
- Roles: `leaf` (focused worker, cannot delegate/clarify/memory) vs `orchestrator` (can spawn own workers)
- Config: `max_concurrent_children` (default 3), `max_spawn_depth` (default 2)
- Isolated context + terminal session per subagent

**What the Orchestrator has:**
- `engine_core/task_executor.py` — executes one task at a time
- `concurrency_controller.py` — `TaskConcurrencyGuard` limits concurrent tasks
- `_max_parallel_tasks` — serial execution enforced (`max_parallel_tasks=1`)
- Dependency resolver (`engine_core/dependency_resolver.py`) — topological sort

**What it doesn't have:**
- No concept of subagents — all tasks run in the same engine
- No role gating — every task has full tool access
- No spawn depth limit — could theoretically chain infinitely
- No batch parallelism for independent task groups

**Implementation outline:**

```
orchestrator/
  └── delegation/               # NEW subpackage
      ├── __init__.py
      ├── subagent.py           # SubAgent — lightweight AIAgent wrapper
      │                          with isolated context, toolsets, and
      │                          iteration budget
      ├── batch_runner.py       # BatchRunner — spawns N subagents for
      │                          independent leaf tasks, gathers results
      └── depth_tracker.py      # SpawnDepthTracker — prevents infinite
      │                          delegation chains
```

**Key design decisions to borrow:**
- Default batch parallelism: 3 concurrent subagents (matches Hermes)
- Leaf tasks: can use `CODE_GEN`, `CODE_REVIEW` tools but cannot themselves `delegate_task`
- Orchestrator mode: the main Orchestrator instance manages subagent spawns
- Each subagent gets its own budget slice from the parent budget
- Synchronicity: parent waits for batch completion (Hermes is sync too)

**Effort estimate:** ~5-7 new files, ~400-600 lines  
**Impact:** Parallelizes independent tasks within dependency levels; reduces wall-clock time for projects with many independent tasks

---

## Priority 2 — High Impact, Higher Effort

### 3. Context Compression — LLM-Powered Summarization vs Fixed Truncation

**What Hermes does:**
- `agent/context_compressor.py` — `ContextCompressor` class
- Preflight compression: before the main loop, checks if loaded history exceeds model context threshold
- Mid-loop compression: when nearing limits, summarises middle N turns via LLM
- Configurable: `protect_first_n`, `protect_last_n`, `threshold_tokens`
- Compression creates a new session — SQLite records the transition

**What the Orchestrator has:**
- `context_truncation_limit` — fixed at 40,000 characters, hard truncation
- Dependency resolver passes truncated context to child tasks
- Comment in code: "raised from 20000: code_generation outputs routinely reach 25000+ chars and truncation causes code_review tasks to miss the tail of the source"

**What it doesn't have:**
- No LLM-powered summarization of truncated content
- No adaptive threshold based on model context window
- Fixed character limit regardless of model being used

**Implementation outline:**

```
orchestrator/
  └── context_compressor.py     # NEW file
      ContextCompressor:
        - compress(text: str, max_chars: int) -> str
        - Uses a cheap model (GPT-4o-mini / Phi-4) to summarise
          truncated content while preserving key symbols/functions
        - Caches summaries by content hash
        - Fallback: hard truncation if summarization fails
        - Configurable via ORCH_CONTEXT_COMPRESSION=true/false
```

**Key design decisions:**
- Only compress dependency context passed to code_review/evaluation tasks — never compress prompts
- Use cheapest available model (Phi-4 at $0.07/1M input) for summarization
- Cache summarizations at L2 (48h) to avoid re-summarizing the same code across review iterations
- Compression is opt-in: `ORCH_CONTEXT_COMPRESSION=true` (default off for cost predictability)

**Effort estimate:** ~3-4 new files, ~300-400 lines  
**Impact:** Eliminates the biggest source of code_review failures ("source code was not provided" due to truncation)

---

### 4. Cross-Session Memory — Pluggable Memory Providers

**What Hermes does:**
- `agent/memory_manager.py` — orchestrates multiple `MemoryProvider` implementations
- Providers: Honcho (dialectic user modeling), Mem0, Supermemory, ByteRover, Hindsight, Holographic, OpenViking, RetainDB
- FTS5 session search via `hermes_state.py` / `SessionDB`
- Periodic memory nudges: `memory.nudge_interval` prompts agent to review and consolidate
- Prefetch: external providers prefetch relevant context at conversation start

**What the Orchestrator has:**
- `state.py` — project state persistence (SQLite)
- `telemetry_store.py` — cross-run learning data
- `bm25_search.py` — FTS5 full-text search infrastructure
- `hybrid_search_pipeline.py` — BM25 + vector RRF fusion
- `memory_tier.py` — `MemoryTierManager` with HOT/WARM/COLD tiers

**What it doesn't have:**
- No concept of "session memory" that persists across *projects*
- No consolidation of lessons learned across runs
- No memory provider abstraction (all memory is in-tree)

**This is lower priority than pattern learning** because `telemetry_store` + `bm25_search` already provide the infrastructure. The gap is the *abstraction* (provider interface) and the *consolidation loop* (periodic review).

**Implementation outline:**

```
orchestrator/
  └── memory/                   # NEW subpackage
      ├── __init__.py
      ├── provider.py           # MemoryProvider ABC with sync_turn(),
      │                          prefetch(), shutdown()
      ├── builtin_provider.py   # BuiltinMemoryProvider — uses existing
      │                          telemetry_store + bm25_search
      ├── memory_manager.py     # MemoryManager — orchestrates providers,
      │                          periodic nudge logic, prefetch_all()
      └── consolidation.py      # ConsolidationLoop — after N projects,
      │                          triggers LLM review of telemetry_store
      │                          to extract cross-project insights
```

**Effort estimate:** ~5-6 new files, ~400-600 lines  
**Impact:** Turns per-project learning into cross-project memory; enables insights like "project X failed because of pattern Y seen in 3 prior projects"

---

## Priority 3 — Moderate Impact, Lower Effort

### 5. Tool Guardrails — Runtime Safety Checks

**What Hermes does:**
- `agent/tool_guardrails.py` — `ToolCallGuardrailController`
  - Per-turn counter reset: `reset_for_turn()`
  - Repeat detection: same tool + same args → guardrail
  - Destructive-pattern detection: `rm -rf`, `DEL /F`, format commands
  - Configurable halt threshold, synthetic result injection

**What the Orchestrator has:**
- `validators.py` — deterministic validation (python_syntax, pytest, ruff)
- Circuit breaker (3-consecutive-failure threshold for models)
- No runtime tool-call guards

**What it doesn't have:**
- No detection of repeated identical tool calls within a task iteration
- No destructive-pattern detection for generated code
- No per-turn guardrail state reset

**Implementation outline:**

```
orchestrator/
  └── tool_guardrails.py        # NEW file
      ToolGuardrailController:
        - reset_for_turn() - called at each iteration start
        - check(code: str) -> ToolGuardrailDecision
        - Decisions: ALLOW, WARN, BLOCK
        - Detects:
            - Repeated identical generations (agent stuck in loop)
            - Destructive patterns in generated code (rm -rf, os.system("format"))
            - Output size regression (new output < 10% of previous output size)
        - Integrates at _execute_task() level — checked before validators
```

**Effort estimate:** ~1-2 new files, ~150-250 lines  
**Impact:** Prevents runaway generation loops; catches accidental destructive code before it reaches validation

---

### 6. Plugin System Depth — Memory-Provider and Context-Engine Surfaces

**What Hermes does:**
- Separate plugin surfaces with their own ABCs: `MemoryProvider`, `ContextEngine`, `ModelProvider`
- Each surface has its own discovery mechanism (not all routed through one PluginManager)
- Model providers: last-writer-wins registration, user plugins override bundled
- Context engines: plug into `agent/context_engine.py`
- Policy: plugins MUST NOT modify core files; expand the plugin surface instead

**What the Orchestrator has:**
- `plugins/base.py` — `Plugin` ABC with lifecycle hooks (INIT, PRE/POST task, PRE/POST project)
- `PluginRegistry` — manages registration, dependency ordering, hook execution
- Globally consistent: all plugins use the same `PluginContext`

**What it doesn't have:**
- Domain-specific plugin surfaces (memory, context, model)
- No discovery of plugins from user directories (`~/.orchestrator/plugins/`)
- Model-provider plugins are hardcoded in `models.py`

**Implementation outline:**

```
orchestrator/
  └── plugins/
      ├── base.py               # EXISTING — general plugin hooks
      ├── memory_provider.py    # NEW — MemoryProvider ABC
      │   - Abstract: sync_turn(result), prefetch(query), shutdown()
      │   - Discovery: ~/.orchestrator/plugins/memory/*/
      ├── context_provider.py   # NEW — ContextProvider ABC
      │   - Abstract: enrich(prompt, task_type) -> str
      │   - Discovery: ~/.orchestrator/plugins/context/*/
      └── cost_optimization.py  # EXISTING
```

**Effort estimate:** ~3-4 new files, ~200-350 lines (ABCs + discovery)  
**Impact:** Enables community plugins without touching core; follows Hermes's "expand the surface, don't hardcode plugins" rule

---

## Priority 4 — Nice to Have

### 7. Dependency Pinning Policy Formalization

**Hermes context:**
- Exact pins (`==X.Y.Z`) for all direct deps in `pyproject.toml`
- Policy rationale documented: supply-chain attacks (litellm compromise, Mini Shai-Hulud worm on mistralai)
- `uv.lock` regenerated on every dep change
- Lazy deps (`tools/lazy_deps.py`) for optional backends — reduce blast radius

**Orchestrator gap:**
- `requirements.txt` and `requirements-dev.txt` are pinned but the rationale isn't documented
- No `uv.lock` equivalent (pip-based)
- No lazy-dep mechanism for optional providers

**Action:** Add a `DEPENDENCY_POLICY.md` to `/docs/` documenting the rationale, pinning strategy, and supply-chain hardening posture. Low effort (<1 file, ~50 lines).

---

### 8. Multi-Platform Gateway

**Hermes has:** 18+ platform adapters (Telegram, Discord, Slack, WhatsApp, Signal, etc.) via `gateway/run.py` + `gateway/platforms/`

**Orchestrator could use:** A gateway that:
- Accepts project specs via Telegram/Discord bot
- Reports task progress and results to messaging platforms
- Allows `/approve`/`/deny` for expensive model choices

**Effort:** High (~800-1200 lines + platform deps). Only worth it if the Orchestrator needs continuous operation patterns.

---

### 9. Slash Command Registry Pattern

**Hermes pattern:** Single `COMMAND_REGISTRY` list of `CommandDef` objects drives CLI, gateway, Telegram menus, Slack, autocomplete — all from one source of truth.

**Orchestrator could adopt:** A unified command registry for CLI commands (`orchestrator/cli.py`) + future gateway commands. Less critical since the Orchestrator is currently CLI-only.

---

### 10. Kanban Work Queue

**Hermes has:** `plugins/kanban/` — SQLite-backed multi-agent work queue with dispatcher running inside the gateway.

**Orchestrator could use:** A persistent work queue for long-running orchestration projects — submit project specs, workers pick them up, coordinator tracks progress. The Orchestrator already has `state.py` (SQLite) and `concurrency_controller.py` — the queue management layer is the missing piece.

---

## Summary: Prioritized Implementation Roadmap

| Pri | Vector | Files | Lines | Impact |
|-----|--------|-------|-------|--------|
| **P1** | Pattern Learner (closed learning loop) | 8-12 | 600-900 | Transformational — makes the Orchestrator improve with use |
| **P1** | Subagent Delegation (batch parallelism) | 5-7 | 400-600 | Parallelizes independent tasks; reduces wall-clock time |
| **P2** | Context Compression (LLM summarization) | 3-4 | 300-400 | Fixes the #1 cause of code_review failures |
| **P2** | Memory Provider Abstraction | 5-6 | 400-600 | Cross-project memory consolidation |
| **P3** | Tool Guardrails | 1-2 | 150-250 | Prevents runaway loops and destructive code |
| **P3** | Plugin Surface Expansion | 3-4 | 200-350 | Enables community plugins |
| **P4** | Dependency Policy Docs | 1 | ~50 | Documents existing posture |
| **P4** | Gateway | 8-12 | 800-1200 | Multi-platform access (high effort, lower urgency) |
| **P4** | Slash Command Registry | 2-3 | 150-250 | Unified command source of truth |
| **P4** | Kanban Work Queue | 5-7 | 400-600 | Persistent multi-project work queue |

**Total estimated addition:** ~40-60 files, ~3,500-5,000 lines of new code across all priorities.

---

## Comparison: Architecture Philosophy

| Dimension | AI Orchestrator | Hermes Agent | Insight |
|-----------|----------------|--------------|---------|
| **Loop style** | Async (asyncio) | Sync (threads) | Orchestrator's async is correct for its use case; Hermes's sync is simpler but less I/O efficient |
| **State management** | SQLite via StateManager | SQLite via SessionDB (FTS5) | Both use SQLite; Hermes adds FTS5 for search — Orchestrator already has FTS5 via BM25 |
| **Testing** | pytest-asyncio, 12% coverage | pytest with subprocess isolation, ~17K tests | Hermes's test isolation plugin and "no change-detector tests" rule are directly adoptable |
| **Config** | `.env` for everything | `config.yaml` + `.env` (secrets only) | Hermes's separation of config from secrets is cleaner |
| **Plugin model** | Single `Plugin` ABC + lifecycle | Multiple domain-specific ABCs | Hermes's approach is more extensible; Orchestrator should add domain-specific surfaces |
| **Prompt caching** | `prompt_cache.py`, `semantic_cache.py` | System prompt persisted in SQLite | Both have caching; Hermes's session-level persistence is worth adopting |
| **Error handling** | `exceptions.py` hierarchy + circuit breaker | `classify_api_error()` + failover routing | Both have solid approaches; complementary |

---

## Files Examined

### AI Orchestrator:
- `AGENTS.md` (full)
- `orchestrator/engine.py` (~5,100 lines)
- `orchestrator/models.py` (full — ~1,500 lines)
- `orchestrator/budget.py` (full)
- `orchestrator/plugins/base.py` (full)
- `orchestrator/services/executor.py` (full)

### Hermes Agent:
- `README.md` (full)
- `AGENTS.md` — development guide (full, ~1,000 lines)
- `pyproject.toml` — dependencies and build config (full)
- `run_agent.py` — `AIAgent` class + conversation loop (start, ~12K LOC)
- `agent/conversation_loop.py` — `run_conversation()` extraction (start, ~3,900 LOC)
- `agent/__init__.py` (full)
