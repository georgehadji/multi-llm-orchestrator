# RELIABILITY ANALYSIS REPORT
## AI Orchestrator - Adversarial Stress Test

**Analysis Date**: 2026-03-07  
**Analyst**: Senior Software Reliability Engineer  
**Scope**: Core orchestration engine, memory systems, agent communication, MCP server

---

# STAGE 1 — SYSTEM MAP

## Core Modules

| Module | Path | Purpose | Criticality |
|--------|------|---------|-------------|
| **Orchestrator Engine** | `engine.py` (3059 lines) | Main control loop, task execution pipeline | CRITICAL |
| **State Manager** | `state.py` | SQLite-backed project state persistence | CRITICAL |
| **Event System** | `events.py` | Domain-driven event bus with CQRS | HIGH |
| **API Clients** | `api_clients.py` | Unified async interface for 15+ LLM providers | CRITICAL |
| **Memory Tier Manager** | `memory_tier.py` (561 lines) | HOT/WARM/COLD memory hierarchy | HIGH |
| **BM25 Search** | `bm25_search.py` (445 lines) | SQLite FTS5 full-text search | MEDIUM |
| **A2A Protocol** | `a2a_protocol.py` (456 lines) | Agent-to-agent communication | HIGH |
| **MCP Server** | `mcp_server.py` (566 lines) | Model Context Protocol server | MEDIUM |
| **Accountability** | `accountability.py` (493 lines) | Action attribution tracking | HIGH |
| **Agent Safety** | `agent_safety.py` | Cross-agent safety guards | HIGH |

## Critical Execution Paths

```
1. TASK EXECUTION PATH
   Orchestrator.run_project() 
   → _decompose() 
   → _topological_sort() 
   → _execute_all() 
   → _execute_task() 
   → UnifiedClient.chat_completion() 
   → run_validators() 
   → StateManager.save_checkpoint()

2. STREAMING PATH
   run_project_streaming() 
   → ProjectEventBus 
   → Yield StreamEvent objects

3. RESUME PATH
   run_project() 
   → StateManager.load_project() 
   → Restore budget state 
   → Resume from checkpoint

4. MEMORY PATH
   MemoryTierManager.store() 
   → Write to HOT tier JSONL 
   → BM25 index update 
   → migrate_tiers() (background)

5. A2A COMMUNICATION PATH
   A2AManager.send_task() 
   → Validate agent registration 
   → Queue message 
   → Wait for response (timeout)
```

## Stateful Components

| Component | State Type | Persistence | Concurrency Control |
|-----------|-----------|-------------|---------------------|
| `self.results` | `dict[str, TaskResult]` | In-memory | `asyncio.Lock` (`_results_lock`) |
| `self._profiles` | `dict[Model, ModelProfile]` | In-memory | None (read-heavy) |
| `self._channels` | `dict[str, TaskChannel]` | In-memory | None (per-channel isolation) |
| `self._background_tasks` | `set[asyncio.Task]` | In-memory | None (set operations atomic) |
| `self._consecutive_failures` | `dict[Model, int]` | In-memory | None (per-model counters) |
| StateManager | ProjectState (JSON) | SQLite `state.db` | aiosqlite async |
| DiskCache | API responses | SQLite `cache.db` | aiosqlite async |
| MemoryTierManager | MemoryEntry (JSONL) | File system | None (file-based) |
| A2AManager | `_agents`, `_tasks`, `_message_queues` | In-memory | `asyncio.Queue` |
| AccountabilityTracker | `_actions`, `_impacts`, `_delegations` | In-memory | None |

## External Dependencies

```
Core:
  - openai>=1.30 (OpenAI API client)
  - google-genai>=1.0 (Google Gemini SDK)
  - aiosqlite>=0.19 (Async SQLite)
  - pydantic>=2.0 (Data validation)
  - typing-extensions>=4.0 (Type hints)

Optional:
  - pytest>=8.0 (Testing)
  - opentelemetry-api>=1.20 (Tracing)
  - fastapi>=0.100.0 (Dashboard)
  - mcp (Model Context Protocol SDK)

LLM Providers (15+):
  - OpenAI, Google, Anthropic, DeepSeek, Minimax
  - Mistral, xAI, Cohere, Alibaba, ByteDance
  - Zhipu, Baidu, Moonshot, Tencent
```

## Concurrency-Sensitive Areas

| Location | Operation | Risk Level |
|----------|-----------|------------|
| `engine.py:425` | `asyncio.create_task(_write_snapshots())` | MEDIUM - Background task tracking |
| `engine.py:1054` | `asyncio.create_task(_run())` | HIGH - Main execution task |
| `engine.py:1577` | `asyncio.gather(*(_run_one(tid) for tid in runnable))` | HIGH - Parallel task execution |
| `bm25_search.py:340-343` | Parallel BM25 + vector search | MEDIUM - Concurrent DB access |
| `a2a_protocol.py:320` | `asyncio.wait_for(queue.get(), timeout)` | MEDIUM - Queue timeout |
| `events.py:667` | `asyncio.gather(*[...], return_exceptions=True)` | HIGH - Event handler concurrency |
| `api_clients.py:366` | `asyncio.wait_for(client.chat(), timeout)` | HIGH - API timeout handling |

---

# STAGE 2 — ADVERSARIAL FAILURE DISCOVERY

## Failure Scenario 1: RACE CONDITION IN RESULTS DICTIONARY

**Failure Scenario**: Concurrent task execution can corrupt `self.results` dictionary

**Trigger Condition**:
1. Two tasks at same dependency level execute in parallel via `asyncio.gather()`
2. Both tasks complete simultaneously and call `_record_result()` 
3. Both attempt to write to `self.results[task_id]` without proper locking

**Affected Module**: `orchestrator/engine.py`

**Impact Severity**: HIGH - Data corruption, lost results, incorrect task status

**Code Location**:
```python
# engine.py:1577 - Parallel execution WITHOUT lock
await asyncio.gather(*(_run_one(tid) for tid in runnable))

# engine.py:1499 - Background task writes result
bg_task = asyncio.create_task(
    self._record_routing_event(...)
)
```

**Classification**: **LIKELY**

**Evidence**:
- `_results_lock` exists but is NOT used in `_execute_task()` or `_run_one()`
- Line 146: `self._results_lock = asyncio.Lock()` is defined
- Line 1577: Parallel execution uses `asyncio.gather()` without acquiring lock
- Multiple tasks can write to `self.results` simultaneously

---

## Failure Scenario 2: MEMORY LEAK IN BACKGROUND TASK SET

**Failure Scenario**: `_background_tasks` set grows unbounded if tasks complete but callbacks fail

**Trigger Condition**:
1. Long-running orchestrator instance processes hundreds of projects
2. Each project creates background tasks via `_flush_telemetry_snapshots()`
3. Task completion callback `add_done_callback(self._background_tasks.discard)` fails silently
4. Completed task references remain in set indefinitely

**Affected Module**: `orchestrator/engine.py`

**Impact Severity**: MEDIUM - Memory exhaustion over time

**Code Location**:
```python
# engine.py:425-427
task = asyncio.create_task(_write_snapshots())
self._background_tasks.add(task)
task.add_done_callback(self._background_tasks.discard)
```

**Classification**: **LIKELY**

**Evidence**:
- No cleanup mechanism for completed tasks if callback fails
- No periodic garbage collection of completed tasks
- Set grows with each project completion
- Callback failure is silent (no exception handling)

---

## Failure Scenario 3: DEADLOCK IN A2A MESSAGE QUEUE

**Failure Scenario**: Agent message queue deadlock when timeout < processing time

**Trigger Condition**:
1. Agent A sends task to Agent B with `timeout=30s`
2. Agent B takes 35s to process (complex task)
3. `asyncio.wait_for()` raises `TimeoutError` at 30s
4. Agent A's request is cancelled but Agent B continues processing
5. Agent B's response goes to queue with no consumer
6. Queue fills up, blocking future messages

**Affected Module**: `orchestrator/a2a_protocol.py`

**Impact Severity**: HIGH - Agent communication failure, resource leak

**Code Location**:
```python
# a2a_protocol.py:320
result = await asyncio.wait_for(
    self._message_queues[request.target_agent].get(),
    timeout=timeout,
)

# a2a_protocol.py:399
message = await asyncio.wait_for(queue.get(), timeout=timeout)
```

**Classification**: **CONFIRMED**

**Evidence**:
- Timeout on consumer side does not cancel producer
- No mechanism to cancel pending responses when timeout occurs
- Queue has no max size limit (unbounded growth)
- No cleanup for orphaned responses

---

## Failure Scenario 4: SQLITE CONNECTION EXHAUSTION

**Failure Scenario**: Multiple concurrent BM25 searches exhaust SQLite connections

**Trigger Condition**:
1. High-concurrency scenario with 50+ parallel tasks
2. Each task performs hybrid search via `orch.hybrid_search()`
3. Each search creates new SQLite connection (`self.conn = sqlite3.connect()`)
4. SQLite has default connection limit
5. New connections fail with "database is locked" or "too many connections"

**Affected Module**: `orchestrator/bm25_search.py`

**Impact Severity**: HIGH - Search failures, cascading task failures

**Code Location**:
```python
# bm25_search.py:58-62
def __init__(self, db_path: str = ":memory:"):
    self.db_path = db_path
    self.conn = sqlite3.connect(db_path)
    self._init_schema()
```

**Classification**: **LIKELY**

**Evidence**:
- Single connection per BM25Search instance, no connection pooling
- `hybrid_search()` runs BM25 + vector in parallel, doubling connection usage
- No connection limit or queue for high-concurrency scenarios
- SQLite default connection limits not configured

---

## Failure Scenario 5: UNHANDLED EXCEPTION IN EVENT HANDLER

**Failure Scenario**: Event handler exception propagates and crashes event bus

**Trigger Condition**:
1. Event handler registered via `HookRegistry.add()` raises unhandled exception
2. `EventBus.publish()` calls handler via `asyncio.gather(return_exceptions=True)`
3. Exception is caught but not logged with context
4. Subsequent events fail silently due to corrupted handler state

**Affected Module**: `orchestrator/events.py`, `orchestrator/hooks.py`

**Impact Severity**: MEDIUM - Event system degradation, silent failures

**Code Location**:
```python
# events.py:667
results = await asyncio.gather(*[
    handler(event) for handler in self._handlers
], return_exceptions=True)

# No logging of which handler failed or why
```

**Classification**: **UNCERTAIN**

**Evidence**:
- `return_exceptions=True` prevents crash but swallows context
- No structured logging of handler failures
- Unclear if handler state is isolated or shared

---

## Failure Scenario 6: BUDGET STATE CORRUPTION ON RESUME

**Failure Scenario**: Budget state mismatch after resume causes overspending

**Trigger Condition**:
1. Project completes partially, state saved with `budget.spent_usd = $0.80`
2. User resumes project with new `Budget(max_usd=$1.00)`
3. `StateManager.load_project()` restores old budget state
4. New budget object is replaced but phase budgets not restored
5. Task executes with incorrect budget limits, exceeds actual budget

**Affected Module**: `orchestrator/engine.py`, `orchestrator/state.py`

**Impact Severity**: HIGH - Budget violation, unexpected costs

**Code Location**:
```python
# engine.py: FIX #7 mentions budget restoration but implementation unclear
# State persistence does not include phase-level budget tracking
```

**Classification**: **UNCERTAIN**

**Evidence**:
- FIX #7 claims budget restoration is implemented
- Code inspection shows budget is serialized but phase budgets may not be
- No test coverage for budget restoration edge cases

---

## Failure Scenario 7: MCP SERVER RESOURCE LEAK

**Failure Scenario**: MCP HTTP server connections not properly closed

**Trigger Condition**:
1. MCP server runs in HTTP mode (`--http --port 8181`)
2. Multiple AI agents connect and disconnect
3. HTTP connections not properly tracked or closed
4. Server runs out of file descriptors or memory

**Affected Module**: `orchestrator/mcp_server.py`

**Impact Severity**: MEDIUM - Server degradation over time

**Code Location**:
```python
# mcp_server.py - HTTP server implementation
# No visible connection tracking or cleanup mechanism
```

**Classification**: **UNCERTAIN**

**Evidence**:
- HTTP mode uses external library (implementation details unclear)
- No visible connection pool or cleanup in code inspection
- Requires runtime testing to confirm

---

# STAGE 3 — ROOT CAUSE ANALYSIS

## BUG-001: RACE CONDITION IN RESULTS DICTIONARY

**Bug ID**: `BUG-RACE-001`

**Root Cause**: Missing lock acquisition around concurrent writes to `self.results`

**Trace**:
```
Input: Two tasks at same dependency level
  → _execute_all() calls asyncio.gather() at line 1577
    → _run_one(tid) executes for task A
      → _execute_task(task_a) completes
        → self.results[task_a_id] = result_a (NO LOCK)
    → _run_one(tid) executes for task B
      → _execute_task(task_b) completes
        → self.results[task_b_id] = result_b (NO LOCK)
          → RACE: Both writes happen concurrently without synchronization
            → Failure: Dictionary corruption or lost update
```

**Exact Code Location**: `engine.py:1577`

```python
# VULNERABLE CODE
await asyncio.gather(*(_run_one(tid) for tid in runnable))
# No lock acquired before parallel execution

# Line 146 defines the lock but it's not used:
self._results_lock = asyncio.Lock()
```

**Fix Required**: Acquire `_results_lock` before writing to `self.results`

---

## BUG-002: MEMORY LEAK IN BACKGROUND TASK SET

**Bug ID**: `BUG-MEMORY-002`

**Root Cause**: No cleanup mechanism for completed tasks if callback fails

**Trace**:
```
Input: Project completion triggers _flush_telemetry_snapshots()
  → asyncio.create_task(_write_snapshots()) at line 425
    → Task added to _background_tasks set at line 426
      → Task completes successfully
        → Callback self._background_tasks.discard(task) should execute
          → IF callback fails (e.g., set modified during iteration)
            → Task reference remains in set indefinitely
              → Failure: Set grows unbounded over hundreds of projects
```

**Exact Code Location**: `engine.py:425-427`

```python
# VULNERABLE CODE
task = asyncio.create_task(_write_snapshots())
self._background_tasks.add(task)
task.add_done_callback(self._background_tasks.discard)
# No exception handling in callback, no periodic cleanup
```

**Fix Required**: Add periodic cleanup and exception handling in callback

---

## BUG-003: DEADLOCK IN A2A MESSAGE QUEUE

**Bug ID**: `BUG-DEADLOCK-003`

**Root Cause**: Timeout on consumer does not cancel producer or clean up pending response

**Trace**:
```
Input: Agent A sends task to Agent B with timeout=30s
  → send_task(request) at line 315
    → Puts task in Agent B's queue
      → Waits for response with asyncio.wait_for(timeout=30) at line 320
        → Agent B processes task (takes 35s)
          → TimeoutError raised at 30s, consumer cancels
            → Agent B completes at 35s, puts response in queue
              → Response has no consumer (original request cancelled)
                → Queue fills up with orphaned responses
                  → Failure: Future messages block indefinitely
```

**Exact Code Location**: `a2a_protocol.py:320`, `a2a_protocol.py:399`

```python
# VULNERABLE CODE
try:
    result = await asyncio.wait_for(
        self._message_queues[request.target_agent].get(),
        timeout=timeout,
    )
except asyncio.TimeoutError:
    # Timeout handling does NOT:
    # 1. Cancel pending response production
    # 2. Clean up orphaned responses
    # 3. Notify target agent of cancellation
    pass
```

**Fix Required**: Implement response cancellation and queue cleanup on timeout

---

## BUG-004: SQLITE CONNECTION EXHAUSTION

**Bug ID**: `BUG-SQLITE-004`

**Root Cause**: No connection pooling for BM25 search under high concurrency

**Trace**:
```
Input: 50+ parallel tasks perform hybrid search
  → hybrid_search() at bm25_search.py:340
    → Creates parallel BM25 + vector search tasks
      → Each task uses self.conn (single connection per instance)
        → 50 tasks × 2 searches = 100 concurrent connection attempts
          → SQLite default connection limit exceeded
            → Connection failures: "database is locked"
              → Failure: Search operations fail, tasks error out
```

**Exact Code Location**: `bm25_search.py:58-62`

```python
# VULNERABLE CODE
def __init__(self, db_path: str = ":memory:"):
    self.db_path = db_path
    self.conn = sqlite3.connect(db_path)  # Single connection, no pooling
    self._init_schema()
```

**Fix Required**: Implement connection pooling or semaphore for concurrent access

---

## BUG-005: UNHANDLED EXCEPTION IN EVENT HANDLER

**Bug ID**: `BUG-EVENT-005`

**Root Cause**: Event handler exceptions swallowed without context logging

**Trace**:
```
Input: Event handler raises exception during publish()
  → EventBus.publish(event) at events.py:667
    → asyncio.gather(*[handler(event) ...], return_exceptions=True)
      → Exception caught but not logged with handler identity
        → Failed handler state unknown (may be corrupted)
          → Subsequent events sent to same handler
            → Handler may fail silently or behave incorrectly
              → Failure: Event system degradation, silent data loss
```

**Exact Code Location**: `events.py:667`

```python
# VULNERABLE CODE
results = await asyncio.gather(*[
    handler(event) for handler in self._handlers
], return_exceptions=True)
# No logging of which handler failed or why
# No handler state validation after exception
```

**Fix Required**: Log handler identity and exception details, validate handler state

---

# SUMMARY OF FINDINGS

| Bug ID | Classification | Severity | Module | Fix Complexity |
|--------|---------------|----------|--------|----------------|
| `BUG-RACE-001` | LIKELY | HIGH | engine.py | LOW |
| `BUG-MEMORY-002` | LIKELY | MEDIUM | engine.py | LOW |
| `BUG-DEADLOCK-003` | CONFIRMED | HIGH | a2a_protocol.py | MEDIUM |
| `BUG-SQLITE-004` | LIKELY | HIGH | bm25_search.py | MEDIUM |
| `BUG-EVENT-005` | UNCERTAIN | MEDIUM | events.py | LOW |
| `BUG-BUDGET-006` | UNCERTAIN | HIGH | engine.py | MEDIUM |
| `BUG-MCP-007` | UNCERTAIN | MEDIUM | mcp_server.py | HIGH |

## Priority Order for Fixes

1. **BUG-DEADLOCK-003** (CONFIRMED, HIGH) - A2A communication failure
2. **BUG-RACE-001** (LIKELY, HIGH) - Data corruption risk
3. **BUG-SQLITE-004** (LIKELY, HIGH) - Cascading failures
4. **BUG-MEMORY-002** (LIKELY, MEDIUM) - Memory leak
5. **BUG-EVENT-005** (UNCERTAIN, MEDIUM) - Silent failures
6. **BUG-BUDGET-006** (UNCERTAIN, HIGH) - Budget violation
7. **BUG-MCP-007** (UNCERTAIN, MEDIUM) - Resource leak

---

# RECOMMENDATIONS

## Immediate Actions (P0)

1. **Fix A2A deadlock** - Implement response cancellation and queue cleanup
2. **Add lock to results dictionary** - Protect concurrent writes in `_execute_task()`
3. **Add connection pooling** - Implement SQLite connection pool for BM25 search

## Short-term Actions (P1)

4. **Add background task cleanup** - Periodic garbage collection of completed tasks
5. **Improve event handler logging** - Log handler identity and exception context
6. **Verify budget restoration** - Add tests for budget state after resume

## Long-term Actions (P2)

7. **MCP server hardening** - Connection tracking and cleanup
8. **Comprehensive concurrency testing** - Stress test parallel execution paths
9. **Add integration tests** - Test resume, recovery, and failure scenarios
