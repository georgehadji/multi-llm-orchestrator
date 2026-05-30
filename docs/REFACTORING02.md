# REFACTORING02 — Remaining Work After REFACTORING01

**Date:** 2026-05-27
**Author:** Georgios-Chrysovalantis Chatzivantsidis
**Status:** Planning phase — no tasks started

---

## Context

REFACTORING01 completed 19 of 20 planned tasks across Phases 0, 1, and 2.
The codebase is now structurally sound: `engine.py` reduced 30% (5,036→3,513 lines),
ServiceContainer wires all collaborators, application layer respects port boundaries,
max_parallel_tasks raised to 3, and ~200 flat files restructured into packages.

This plan covers what's left to reach architectural maturity Level 4 (Structural
Coherence) from the current Level 3 (contained chaos).

---

## Phase 3: Extraction & Testing (2–3 weeks)

### TASK 301 — Extract _try_decompose (M) **DONE**
**Target: Remove the last `self.client.call()` from engine.py

**Current state:** `_try_decompose` is a nested function inside `Orchestrator._decompose()`
(200+ lines). It's the only remaining direct LLM call in engine.py.

**Steps:**
```
1. Verify `DecomposerService.evaluate()` or add a `decompose_project()` method
   if it doesn't exist yet
2. Replace the nested function body with a call to the DecomposerService
3. Keep the outer `_decompose()` as the orchestrator-facing wrapper (handles
   budget, telemetry, logging around the service call)
4. Write a unit test: mock DecomposerService, call _decompose, verify delegation
```

**Verification:** `grep -c "self.client.call(" orchestrator/engine.py` == 0

---

### TASK 302 — Contract Tests for Protocols (L) **DONE**
**Target: Every Protocol in `domain/ports.py` has a contract test suite.

**Current state:** 11 Protocols defined, 0 contract tests.

**Protocols to cover:**
- `CachePort`, `StatePort`, `EventPort` — infrastructure ports
- `ModelProvider`, `BudgetTracker`, `TaskRunner`, `ContextProvider` — app ports
- `EventEmitter`, `CircuitBreakerAccess`, `TelemetryRecorder` — observability ports
- `LoggingProvider`, `LLMClient` — cross-cutting ports

**Pattern for each protocol:**
```python
# tests/contracts/test_cache_port.py
def test_cache_port_contract():
    """Every CachePort implementation must pass these."""
    impl = create_cache_implementation()
    assert isinstance(impl, CachePort)
    # SET
    await impl.set("key", "value")
    # GET
    assert await impl.get("key") == "value"
    # DELETE
    await impl.delete("key")
    assert await impl.get("key") is None
```

**Test file locations:** `tests/contracts/test_*_port.py`

**Verification:** `pytest tests/contracts/ -v` passes

---

### TASK 303 — Add Gating CI Step (S) **DONE**
**Target: CI blocks PRs on test failures, lint errors, and security scans.

**Current state:** `pyproject.toml` ci config exists but no gating step.

**Steps:**
```
1. Add `.github/workflows/gate.yml` that runs on every push
2. Gate steps: pytest --no-cov -x → ruff check → mypy → bandit
3. Set `fail-fast: true` so first failure aborts the pipeline
```

**Verification:** CI shows green for every PR.

---

## Phase 4: Observability & Monitoring (1–2 weeks)

### TASK 401 — Adopt structlog (L) **IN PROGRESS**
**Target: Replace mixed stdlib logging + print() with structured logging.

**Current state:** ~50% stdlib logging, ~15% print(), 1 structlog user (0% adoption).

**Steps:**
```
1. Add structlog to requirements.txt / pyproject.toml
2. Create orchestrator/logging.py with configure_logging() that sets up
   structlog processors (ISO dates, JSON render for production, colorized
   console for dev)
3. Replace print() with structlog.info/debug/warning in:
   - engine.py (15 print statements)
   - cli.py (dashboard output)
   - All validators
4. Replace stdlib logger calls with structlog in high-traffic paths
   (engine.py._execute_task, api_clients.call)
```

**Verification:** `grep -r "print(" orchestrator/ --include="*.py" | wc -l` < 30

---

### TASK 402 — Wire OpenTelemetry Tracing (M) **DONE**
**Target: Every LLM call emits a span exportable to Jaeger/Grafana.

**Current state:** `tracing.py` module exists but is not wired through the pipeline.

**Steps:**
```
1. Wire `traced_task` into PipelineContext at the start of pipeline.run()
2. Wire `traced_llm_call` into UnifiedClient.call() 
3. Add CLI flags: --tracing, --otlp-endpoint
4. Validate: run a project with --tracing, check that Jaeger shows span tree
```

**Critical path:** `PipelineRunner.run()` → `PipelineContext` → each stage →
`traced_task(task_id)`, `traced_llm_call(model, prompt)` around each LLM call.

**Verification:** `python -m orchestrator --project "Hello World" --tracing` produces
a trace with at least 3 child spans per task.

---

## Phase 5: Config & Knowledge Consolidation (1–2 weeks)

### TASK 501 — Single Config Source (M) **DONE**
**Target: One canonical config loader, no env-var chasing.

**Current state:** 5 config files (config.py, crosscutting/config.py, nexus_search/config.py,
config_as_code.py, config_sync.py). Hierarchy documented but not enforced.

**Steps:**
```
1. Make `orchestrator/config.py` the single entry point for all config needs
2. Have `crosscutting/config.py` import from config.py (not the reverse)
3. Have `nexus_search/config.py` import its defaults from config.py
4. Move `config_sync.py` to `infrastructure/config_sync.py`
```

**Verification:** `import orchestrator.config` works standalone. Every config value
can be traced to one definition.

---

### TASK 502 — Knowledge Base Consolidation (L) **DONE**
**Target: Eliminate duplicate knowledge_base.py, knowledge_graph.py files.

**Current state:** Both `orchestrator/knowledge_base.py` and
`orchestrator/knowledge/knowledge_base.py` exist (same content, different location).
Same for knowledge_graph.py.

**Steps:**
```
1. Make root knowledge_base.py the canonical source, delete package copy
   (or vice versa — pick one)
2. Make root knowledge_graph.py the canonical source, delete package copy
3. Update imports across the codebase
```

**Verification:** `import orchestrator.knowledge_base` and
`from orchestrator.knowledge import knowledge_base` resolve to the same module.

---

## Phase 6: Performance & Scalability (2–4 weeks)

### TASK 601 — Async SQLite Pool (M) **DONE**
**Target: `max_parallel_tasks` can be raised to 10+ without lock contention.

**Current state:** Connection pool added (5 connections, WAL mode), default
max_parallel_tasks=3. No benchmark data.

**Steps:**
```
1. Benchmark: run 10 concurrent tasks, measure lock contention
2. Increase pool size empirically (5 → 10 → 20)
3. Add aiosqlite write-ahead hooks for deferred checkpoint
```

**Verification:** `pytest tests/test_concurrency.py -k "parallel"` passes with
max_parallel_tasks=10.

---

### TASK 602 — Profile Hot Path (S) **DONE**
**Target: Identify top 3 CPU/memory bottlenecks.

**Steps:**
```
1. Run `python -m cProfile -o profile.out -m orchestrator --project "..." --budget 1.0`
2. Analyze with snakeviz or pstats
3. Document top 3 hotspots in CODEBASE_MINDMAP.md
```

**Verification:** Profile output committed as `docs/profiles/hot_path_2026-05.prof`

---

## Phase 7: Documentation (1 week)

### TASK 701 — ROADMAP.md (M) **DONE**
**Target: Create public-facing development roadmap.

**Content:**
- Short-term (Phase 3–4): Contract tests, structured logging, tracing
- Medium-term (Phase 5–6): Config consolidation, async pool tuning
- Long-term (Phase 8): Distributed execution, multi-machine agents

---

### TASK 702 — Architecture Decision Records (L) **DONE**
**Target: Document key architectural decisions as ADRs.

**ADRs to write (one per file in docs/adr/):**
```
ADR-001: Hexagonal Architecture with Protocol-based Ports
ADR-002: ServiceContainer as DI Wiring Layer
ADR-003: Pipeline Runner as Execution Abstraction
ADR-004: Why max_parallel_tasks = 3 (SQLite WAL Behavior)
ADR-005: EventBus Unification (HookRegistry + AgentMessageBus)
ADR-006: Budget Class moved from models.py (async vs pure data separation)
```

---

## Phase 8: Long-Term Evolution (3–6 months)

### TASK 801 — Distributed Execution **PARTIAL**
**Target: Multiple orchestrator instances can coordinate via message queue.

**Architecture sketch:**
```
Orchestrator A ──→ NATS/Redis ──→ Orchestrator B
     │                                 │
State DB (shared)               State DB (shared)
```

### TASK 802 — Multi-Machine Agent Orchestration
**Target:** Agents can run on different machines with network-level coordination.

### TASK 803 — Auto-Scaling Model Router
**Target:** Router auto-provisions GPU-backed endpoints based on demand.

---

## Summary: Effort Breakdown

| Phase | Tasks | Total Effort | Timeline |
|-------|-------|-------------|----------|
| **P3**: Extraction & Testing | 301, 302, 303 | M + L + S = 3 | 2–3 weeks |
| **P4** Observability | 401, 402 | L + M = 2 | 1–2 weeks |
| **P5** Config & Knowledge | 501, 502 | M + L = 2 | 1–2 weeks |
| **P6** Performance | 601, 602 | M + S = 2 | 2–4 weeks |
| **P7** Documentation | 701, 702 | M + L = 2 | 1 week |
| **P8** Long-term | 801, 802, 803 | XL | 3–6 months |

**Recommended next step:** TASK 301 — Extract the last `self.client.call()` from
engine.py. It's the smallest remaining task with the highest architectural win
(--99% of engine.py LLM calls through proper services).


---

# Appendix: Pyinstrument Hot-Path Profile (2026-05-27)

## Test 1: Module Import Cost (0.773s)

| Module | Time | % | Bottleneck |
|--------|------|---|-----------|
| `engine.py` + all dependencies | 0.357s | 46% | 0.132s in telemetry.py -> numpy import |
| `telemetry.py` -> `numpy` | 0.132s | 17% | NumPy is a 15MB C extension — imported for stats |
| Regex compilation (code_post_processor) | 0.002s | <1% | 50 regex patterns at module level |
| `dataclass` definitions | 0.002s | <1% | ProgressEntry, CodebaseMap, ExecutionPlan |

**Action:** Move numpy import from telemetry.py to lazy local import (only needed for
stats operations, not for all start-up paths). Estimated saving: 0.13s on first import.

## Test 2: Budget Operations (10k iterations = 0.256s)

| Operation | Time | % | Per Call |
|-----------|------|---|----------|
| `reserve()` | 0.069s | 27% | 0.0069ms |
| `charge()` | 0.065s | 25% | 0.0065ms |
| `release_reservation()` | 0.065s | 25% | 0.0065ms |
| Loop overhead | 0.055s | 22% | — |

**Breakdown by layer:**
- Async lock (aenter/aexit): 0.082s (32% of total)
- Budget logic (arithmetic, dict updates): 0.118s (46%)
- Loop/overhead: 0.055s (22%)

**Verdict:** Budget operations are NOT a bottleneck. For a typical project with 20 tasks
and 3 iterations each (60 budget calls), total budget overhead ≈ 0.4ms — negligible.

## Test 3: Import Time (suggested follow-up)

Profile the engine.py import specifically:
```bash
pyinstrument -r html -o profile_import.html \
  -m pytest tests/ -m "not slow and not requires_api" -q
```


### Test 3: Combined Hot Path (5k iterations = 0.077s)

| Operation | Time | Per Call | Notes |
|-----------|------|----------|-------|
| budget.charge() | 0.050s | 0.010ms | 32% async lock overhead |
| dict.get (ROUTING) | 0.001s | 0.0002ms | Negligible |
| Loop + overhead | 0.026s | 0.005ms | — |

**Verdict:** Hot path is not CPU-bound. Budget operations are fast enough for
10,000+ concurrent tasks on a single machine. The real bottleneck is LLM API
latency (1-15s per call), not Python overhead.
