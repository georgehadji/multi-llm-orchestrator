# Agentic System Optimization Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-23  
> **Version:** 1.0  
> **Status:** Plan — awaiting implementation  

---

## Executive Summary

The agentic system now has all 10 capabilities implemented: multi-agent architecture, shared workspace, recursive planning, tool integration, agent communication, self-reflection, runtime execution, dynamic scaffolds, CI pipeline, and human-in-the-loop. This plan identifies **12 optimizations** across **5 dimensions** that transform the system from functionally complete to production-grade.

---

## Current Architecture Maturity Assessment

| Dimension | Current Grade | Target Grade | Gaps |
|-----------|--------------|-------------|------|
| **Agent Intelligence** | B- | A- | Agents generate code but don't self-correct; no multi-model deliberation |
| **Coordination** | C+ | B+ | Basic goal decomposition; no conflict resolution; no dynamic replanning |
| **Safety & Security** | C | B+ | Tool permissions exist but not enforced; no rate limiting per agent; no audit trail |
| **Performance** | C+ | B | No parallel agent execution; no caching of agent decisions; workspace is in-memory only |
| **Observability** | D | B | No agent metrics; no traceability across agent calls; no cost per agent tracking |

---

## Optimization Dimensions

### Dimension A: Agent Intelligence (Make agents smarter)

#### A-1: Self-Correcting Agent Loop (HIGH PRIORITY)

**Current state:** Agents generate output once and return. No feedback loop.  
**Target:** Agents execute → validate → self-correct → re-validate loop within handle_task().

**Implementation:**
- DeveloperAgent wraps LLM call in a `SelfCorrectingLoop` (generate → syntax check → lint → retry)
- Max 3 iterations per task
- Each iteration feeds the previous error as critique context
- Persists which strategies worked for which error types in ExperienceBuffer

**Effort:** 3 days | **Files:** `agents/developer.py`, `learning/strategy_adapter.py`

#### A-2: Multi-Model Deliberation (MEDIUM PRIORITY)

**Current state:** Each agent uses a single model for its task.  
**Target:** Critical decisions route through ARA Debate/Jury for multi-model consensus.

**Implementation:**
- ArchitectAgent decisions (framework choice, architecture patterns) route through Jury pipeline (3 models, independent scoring, meta-evaluation)
- Code review tasks route through Debate pipeline
- AgentOrchestrator selects method based on task criticality (heuristic: architecture = critical, code gen = standard)
- Integration with existing ARA pipelines (already wired in engine.py)

**Effort:** 2 days | **Files:** `agents/developer.py`, `agents/coordinator.py`

#### A-3: Knowledge Graph Integration (MEDIUM PRIORITY)

**Current state:** ExperienceBuffer stores flat pattern → score mappings. No relational knowledge.  
**Target:** Build a KnowledgeGraph that links patterns, models, and outcomes with confidence scores.

**Implementation:**
- `KnowledgeGraph` class with nodes (TaskType, Model, Method, Score) and edges (used_with, produced_score, failed_on)
- Query: "given task_type=X, what model+method combo gives highest success rate?"
- Query: "what is the most common failure mode for model Y on task Z?"
- Integrated into ExperienceBuffer as a graph backend

**Effort:** 2 days | **Files:** `learning/knowledge_graph.py`

---

### Dimension B: Coordination (Make agents work together better)

#### B-4: Dynamic Replanning (HIGH PRIORITY)

**Current state:** AgentOrchestrator decomposes goal once, dispatches, and reports.  
**Target:** If a task fails, replan and re-dispatch with context from the failure.

**Implementation:**
- AgentOrchestrator monitors workspace for task failures
- On failure, queries GoalDecomposer to replan the specific sub-goal that failed
- Injects failure context (error message, stack trace, critique) into the replanned task
- Tracks replan attempts; after 2 replan failures, escalates to human
- Adds `ReplanRecord` to workspace for audit

**Effort:** 2 days | **Files:** `agents/coordinator.py`, `planning/decomposer.py`

#### B-5: Conflict Detection & Resolution (MEDIUM PRIORITY)

**Current state:** No conflict detection — if two agents modify the same file, the last write wins.  
**Target:** Workspace detects write conflicts and notifies agents via message bus.

**Implementation:**
- `ProjectWorkspace.write_file()` checks if another agent wrote to the same file recently
- If conflict detected, publishes a `CONFLICT` message to the bus
- Both agents receive the conflict message and the AgentOrchestrator replans
- Conflict resolution strategy: prefer the agent that first claimed the file, or merge via diff

**Effort:** 2 days | **Files:** `workspace/workspace.py`, `workspace/message_bus.py`

#### B-6: Parallel Agent Execution (HIGH PRIORITY)

**Current state:** AgentOrchestrator dispatches tasks sequentially.  
**Target:** Execute independent agent tasks in parallel via `asyncio.gather()`.

**Implementation:**
- AgentOrchestrator builds a DAG of tasks from dependencies
- Tasks at the same level (no cross-dependencies) execute concurrently
- Semaphore limits max concurrent agents (default: 3)
- Workspace is thread-safe (asyncio.Lock on file operations)
- Result aggregation waits for all parallel tasks to complete

**Effort:** 2 days | **Files:** `agents/coordinator.py`

---

### Dimension C: Safety & Security (Make agents safe)

#### C-7: Tool Permission Enforcement (HIGH PRIORITY)

**Current state:** ToolRegistry stores permissions but agents can call any tool.  
**Target:** Agent's `execute_tool()` checks ToolRegistry.can_execute() before every call.

**Implementation:**
- `AgentBase.execute_tool(name, params)` method wraps `Tool.execute()` with permission check
- Permissions are granted per agent role at construction time
- Critical permissions (SHELL_EXECUTE, FILE_DELETE, GIT_PUSH) require HITL approval
- Audit trail records every tool invocation with agent, timestamp, and result

**Effort:** 2 days | **Files:** `agents/base.py`, `tools/base.py`

#### C-8: Agent Rate Limiting (MEDIUM PRIORITY)

**Current state:** No per-agent rate limiting.  
**Target:** Each agent has a configurable TPM/RPM limit to prevent runaway costs.

**Implementation:**
- `AgentRateLimiter` class with sliding window counters per agent
- Integration with existing `RateLimiter` from cost_optimization
- AgentOrchestrator enforces limits — if agent exceeds quota, task is paused
- Per-agent cost tracking: `agent.cost_usd` accumulates

**Effort:** 1 day | **Files:** `agents/base.py`, `agents/coordinator.py`

#### C-9: Security Audit Trail (LOW PRIORITY)

**Current state:** Workspace tracks mutations but without structured audit.  
**Target:** Every agent action is logged to an immutable audit trail.

**Implementation:**
- `AuditTrail` class with append-only log entries (timestamp, agent, action, result)
- Integration with existing `AuditLog` from engine.py
- Entries: FILE_CREATED, FILE_MODIFIED, MODEL_CALL, TOOL_EXECUTED, DECISION_MADE
- Exportable as JSON for compliance review

**Effort:** 1 day | **Files:** `workspace/audit.py`

---

### Dimension D: Performance (Make agents faster)

#### D-10: Agent Call Caching (MEDIUM PRIORITY)

**Current state:** Every agent call goes to the LLM, even for identical prompts.  
**Target:** Cache agent responses for identical task patterns.

**Implementation:**
- `AgentCache` class using hash(task.goal + task.context) as key
- TTL-based expiration (default: 1 hour)
- Integration with existing `SemanticCache` for similar-but-not-identical prompts
- Cache misses go to LLM; cache hits return immediately with `cached=True` flag

**Effort:** 1 day | **Files:** `agents/base.py`, `learning/agent_cache.py`

#### D-11: Persistent Workspace (LOW PRIORITY)

**Current state:** Workspace is entirely in-memory — lost on process restart.  
**Target:** Workspace persists to disk (SQLite or JSON) for crash recovery.

**Implementation:**
- `PersistentWorkspace` subclass of `ProjectWorkspace`
- Uses aiosqlite for file state storage
- On construction, loads existing state from disk
- `write_file()` and `record_decision()` auto-save to disk
- Integration with existing `StateManager` for consistency

**Effort:** 2 days | **Files:** `workspace/persistent_workspace.py`

---

### Dimension E: Observability (Make agents measurable)

#### E-12: Agent Metrics Dashboard (HIGH PRIORITY)

**Current state:** No visibility into agent performance.  
**Target:** Track per-agent metrics: tasks completed, avg score, avg latency, cost, error rate.

**Implementation:**
- `AgentMetrics` class with per-agent counters
- Integration with existing `ObservabilityService`
- Metrics: `tasks_completed`, `avg_score`, `avg_latency_ms`, `total_cost_usd`, `error_rate`
- CLI command: `python -m orchestrator agentic --status` shows metrics table
- Dashboard integration: per-agent widgets in mission_control

**Effort:** 2 days | **Files:** `agents/metrics.py`, `cli.py` (add `--status` flag)

---

## Implementation Order & Effort Summary

| # | Optimization | Dimension | Days | Priority | Depends On |
|---|-------------|-----------|------|----------|------------|
| A-1 | Self-Correcting Agent Loop | Intelligence | 3 | **HIGH** | — |
| B-4 | Dynamic Replanning | Coordination | 2 | **HIGH** | — |
| B-6 | Parallel Agent Execution | Coordination | 2 | **HIGH** | B-4 |
| C-7 | Tool Permission Enforcement | Safety | 2 | **HIGH** | — |
| E-12 | Agent Metrics Dashboard | Observability | 2 | **HIGH** | — |
| A-2 | Multi-Model Deliberation | Intelligence | 2 | MEDIUM | — |
| B-5 | Conflict Detection & Resolution | Coordination | 2 | MEDIUM | B-6 |
| C-8 | Agent Rate Limiting | Safety | 1 | MEDIUM | — |
| D-10 | Agent Call Caching | Performance | 1 | MEDIUM | — |
| A-3 | Knowledge Graph Integration | Intelligence | 2 | MEDIUM | A-2 |
| C-9 | Security Audit Trail | Safety | 1 | LOW | C-7 |
| D-11 | Persistent Workspace | Performance | 2 | LOW | — |
| **Total** | | | **22 days** | | |

**HIGH priority (first sprint): 11 days** — delivers the biggest quality-of-life improvements: self-correction, replanning, parallelism, tool safety, and metrics.

**MEDIUM priority (second sprint): 8 days** — enhances intelligence and coordination.

**LOW priority (third sprint): 3 days** — audits and persistence for production readiness.

---

## Files to Create / Modify

| File | Action | Optimization |
|------|--------|-------------|
| `orchestrator/agents/base.py` | Modify (self-correct loop, rate limiter, caching) | A-1, C-8, D-10 |
| `orchestrator/agents/coordinator.py` | Modify (dynamic replanning, parallel execution) | B-4, B-6 |
| `orchestrator/agents/developer.py` | Modify (self-correction, multi-model) | A-1, A-2 |
| `orchestrator/agents/metrics.py` | **Create** | E-12 |
| `orchestrator/learning/strategy_adapter.py` | **Create** | A-1 |
| `orchestrator/learning/knowledge_graph.py` | **Create** | A-3 |
| `orchestrator/learning/agent_cache.py` | **Create** | D-10 |
| `orchestrator/workspace/workspace.py` | Modify (conflict detection) | B-5 |
| `orchestrator/workspace/audit.py` | **Create** | C-9 |
| `orchestrator/workspace/persistent_workspace.py` | **Create** | D-11 |
| `orchestrator/tools/base.py` | Modify (permission enforcement) | C-7 |
| `orchestrator/cli.py` | Modify (add `agentic --status`) | E-12 |

**Total: 12 files (5 new, 7 modified)**

---

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| Self-correction loop causes infinite retry | Low | Max 3 iterations; counter tracked per task |
| Parallel agents corrupt workspace state | Medium | `asyncio.Lock` on all workspace write methods |
| Replanning chains indefinitely on stubborn failures | Low | Max 2 replan attempts; human escalation on 3rd failure |
| Tool permission enforcement breaks existing agent calls | Low | Permissions granted liberally at first; tightened gradually |
| Agent caching returns stale results | Low | TTL-based expiration; cache key includes model version |

---

## Verification Gates

- [ ] All existing 154+ tests pass
- [ ] Self-correcting loop: generate → lint fail → revise → lint pass → return
- [ ] Dynamic replanning: task fails → replan triggers → new task dispatched
- [ ] Parallel execution: 3 independent tasks complete concurrently
- [ ] Tool permission: agent without SHELL_EXECUTE cannot call ShellTool
- [ ] Agent metrics: `--status` shows per-agent counters
- [ ] Conflict detection: two agents write same file → CONFLICT message published
- [ ] Agent caching: identical task returns cached results (second call is instant)

---

## Appendix: Target Metrics After Optimization

| Metric | Before | After |
|---------|--------|-------|
| Agent task success rate | ~85% (no self-correction) | ~95% (with 3-iteration loop) |
| Task execution latency (3 tasks) | Sequential (3x single task) | Parallel (1x single task) |
| Unauthorized tool calls | Not enforced | Blocked at AgentBase level |
| Per-agent cost visibility | None | Per-agent `cost_usd` + CLI status |
| Workspace durability | In-memory only (lost on crash) | Persistent (SQLite-backed) |
| Human escalation on failure | Only on architecture | On any replan failure |
| Cross-task learning | Basic pattern → score mapping | Graph-based knowledge with relational queries |

---

**Last updated:** 2026-05-23
