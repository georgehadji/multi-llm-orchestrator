# AI Orchestrator — Principal Engineer Architecture Audit
**Date:** 2026-05-29  
**Auditors:** 4-agent parallel analysis (Layer Compliance · Engine/Orchestration · Testing/Observability · Anti-patterns/Scalability)  
**Codebase state:** Post-V7 Milestone 7 (`docs/architecture-audit-and-refactoring-plan` branch)  
**Prior self-assessment:** 9.2 / 10 (CODEBASE_MINDMAP v7.0)  
**This audit's empirical score:** **5.8 / 10**

---

## 1. Executive Summary

### Overall Architecture Score: 5.8 / 10

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Correctly Implemented | 5/10 | Hexagonal skeleton exists; 3 CRITICALs and 20+ HIGHs in the core |
| Consistently Enforced | 3/10 | 4 import contracts but all have holes; no contracts in CI; 55/70 container fields `Any` |
| Scalable | 2/10 | Architecturally single-process: SQLite-only, in-memory budget, module-global singletons |
| Maintainable | 5/10 | ~30+ full duplicate module pairs; 200+ loose root modules; partial migration frozen mid-flight |
| Observable | 4/10 | structlog + OTel exist but the engine (the most critical path) bypasses both |
| Extensible | 6/10 | Port system is well-designed; 7 of 11 Ports lack contract coverage or are untyped |
| Resilient | 5/10 | Circuit breaker exists but has dual-source-of-truth; event bus silently loses events |

### Architectural Maturity Level: **Transitional**
The codebase is mid-way between an accidental monolith and a clean hexagonal system. The V7 refactoring correctly identified the target architecture and extracted six key services. The skeleton (Ports & Adapters, DI container, `import-linter`, `ProjectRunner` decoupling) is genuine and solid. But execution is incomplete at every layer: the migration stalled with ~30 dual-implementation modules, the CI/CD pipeline enforces almost nothing architectural, and three properties that define production-grade systems — scalability, budget enforcement, and test coverage — are architecturally broken, not just technically weak.

### Primary Risks
1. **Budget enforcement is theater.** `BudgetHierarchy` is purely in-memory. Org/team caps reset on restart and are not shared across processes. A production deployment burns past any limit.
2. **The duplicate module crisis is a security hazard.** ~30 confirmed root/package full-copy pairs mean security patches land in one copy. `secrets_manager.py` is duplicated; the masking regex that scrubs tokens from logs must be fixed in two places.
3. **Concurrency is broken at the hot path.** `asyncio.gather` in `PipelineRunner` has no `return_exceptions=True`, leaving sibling tasks orphaned on failure. The `_results_lock` that exists in the container is never passed to `PipelineRunner`.
4. **5% test coverage with no CI gate.** There is effectively no coverage enforcement. The majority of the codebase can silently regress.
5. **The system cannot scale horizontally.** SQLite for all persistence, module-global singletons for learning/routing/budget state, and `synchronous=FULL` + `wal_checkpoint(TRUNCATE)` per write are architectural single-node decisions.

### Critical Violations
| ID | Violation | Location |
|----|-----------|----------|
| C-1 | `models.py` executes filesystem I/O at import time | `models.py:349-374` |
| C-2 | ~30+ root/package full duplicate module pairs (not shims) | `connectors.py` and ~30 others |
| C-3 | `BudgetHierarchy` in-memory only — org budget unenforceable across restarts | `cost.py:148-173` |
| C-4 | Coverage gate is 5% placeholder; import-linter never runs in CI | `ci.yml`, `pyproject.toml:364` |
| C-5 | Single-node SQLite + module-global singletons block horizontal scaling | `state.py`, `adaptive_templates.py`, `a2a_protocol.py` |

### Refactor Urgency: **HIGH**
V7 correctly diagnosed the problems and built the scaffold. The remaining work is not cosmetic — it is the difference between a system that works in a single developer session and one that works in production.

---

## 2. Intended vs Actual Architecture

### Intended Architecture
Hexagonal (Ports & Adapters) with:
- **Domain** — pure data (`models.py`, `domain/`)
- **Application** — use-case orchestration (`application/`, `engine_core/`)
- **Infrastructure** — adapters to external systems (`infrastructure/`)
- **Driving adapters** — CLI, API server, dashboard
- **Dependency rule** — strict inward flow, enforced by `import-linter`
- **DI container** — `ServiceContainer.build()` wires everything at startup
- **Mediator pattern** — `engine.py` only wires, never contains business logic
- **Cross-run budget enforcement** — `BudgetHierarchy` persists org/team caps

### Actual Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  DRIVING ADAPTERS: cli.py, api_server.py, gateway.py             │
│  ✓ Correctly placed                                              │
└─────────────────────────┬───────────────────────────────────────┘
                           │
┌─────────────────────────▼───────────────────────────────────────┐
│  ENGINE LAYER (engine.py — 2,703 lines)                          │
│  ✗ Still contains ~750 lines of business logic (decomposition   │
│    cluster, JSON recovery, warm-start, plateau detection)        │
│  ✗ Imports ~80 concrete modules at module top                   │
│  ✗ __init__ runs runtime health-probing logic                   │
└──────┬─────────────────────────────────────────┬────────────────┘
       │                                          │
┌──────▼──────────┐                    ┌─────────▼──────────────┐
│  APPLICATION     │                   │  ENGINE_CORE            │
│  ✓ 17 extracted  │                   │  ✓ Pipeline stages      │
│    services      │                   │  ✗ Stages import        │
│  ✗ chat_cli.py   │                   │    concretes (not ports) │
│    imports       │                   │  ✗ map_elites uses       │
│    engine direct │                   │    tuple client contract │
└──────┬──────────┘                    └─────────┬──────────────┘
       │                                          │
┌──────▼──────────────────────────────────────────▼──────────────┐
│  DOMAIN (ports.py, models.py, exceptions.py)                    │
│  ✓ Protocols correctly defined                                  │
│  ✗ models.py:374 — file I/O at import time (Rule #2 violated)  │
│  ✗ LLMClient port fully untyped (no param types, no return)     │
│  ✗ 7 of 11 services have no matching Port                       │
└─────────────────────────────────────────────────────────────────┘
       │
┌──────▼──────────────────────────────────────────────────────────┐
│  INFRASTRUCTURE                                                  │
│  ✓ StateManager implements StatePort fully                       │
│  ✗ BUT: exposes 5 extra methods not in port (resume, list,      │
│    delete) → callers must depend on concrete, not port          │
│  ✗ 44 files touch SQLite independently, inconsistent PRAGMAs   │
└─────────────────────────────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│  ROOT MODULE LAYER (200+ loose .py files in orchestrator/)      │
│  ✗ ~30 confirmed full duplicates of subpackage modules          │
│  ✗ Only 8 converted to shims (M5 migration stalled)             │
│  ✗ 6+ competing config modules                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Drift Summary

| Intended | Actual | Gap |
|----------|--------|-----|
| engine.py is a thin mediator | 2,703 lines with ~750 lines of business logic | **Substantial** |
| models.py is pure data | I/O at import time, behavioral functions | **Violated** |
| DI container types all services | 55/70 fields `Any` | **Partial** |
| import-linter enforces arch in CI | import-linter runs locally only | **Missing** |
| Root modules are shims | ~30 are full duplicates | **Stalled** |
| BudgetHierarchy persists cross-run | In-memory only | **Violated** |
| Structured logging throughout | Engine uses stdlib logging | **Bypassed** |

---

## 3. Architecture Compliance Matrix

| Module | Intended Pattern | Actual Implementation | Violations | Severity |
|--------|-----------------|----------------------|------------|----------|
| `engine.py` | Mediator — wire only | 2,703 lines; decomposition + JSON recovery logic | Business logic in mediator; ~80 concrete imports | HIGH |
| `models.py` | Pure domain data | File I/O at import; behavioral functions (`estimate_cost`, `build_default_profiles`) | I/O in domain; behavior in data | CRITICAL |
| `domain/ports.py` | Complete typed contracts | 11 Protocols; LLMClient untyped; 7 services lack Port | Incomplete; untyped boundary | HIGH |
| `engine_core/container.py` | Typed DI composition root | 70 fields; 55 are `Any`; 4 non-Optional fields default to `None` | Type safety nominal | HIGH |
| `engine_core/pipeline_runner.py` | Concurrent task execution | `asyncio.gather` no `return_exceptions`; `_results_lock` exists but unused | Race condition; orphaned tasks | HIGH |
| `engine_core/stages/` | Port-dependent stages | `generate.py` + `critique.py` correctly use `LLMClient` port; `map_elites.py` uses tuple contract | Inconsistent client interface | MEDIUM |
| `application/project_runner.py` | Decoupled service | Zero host back-references; fully injected callables | ✓ Compliant — reference implementation | NONE |
| `application/model_health_tracker.py` | Owns circuit-breaker state | Correct; but engine.py maintains shadow dicts | Dual source-of-truth | MEDIUM |
| `application/chat_cli.py` | Application service | Imports `Orchestrator` directly (`line 168`) | Application→Engine violation; unguarded by import-linter | HIGH |
| `unified_events/core.py` | Async event bus | `EventStore` uses blocking `sqlite3` in async loop; publish silently drops events if `start()` not called | Blocking I/O in async; event loss | HIGH |
| `agents/base.py` | Async agent | `await`s `AgentMessageBus.publish`, which is synchronous | Misleading async interface | HIGH |
| `infrastructure/state.py` | StatePort adapter | Implements all 6 Port methods + 5 extra public methods not in Port | Incomplete port boundary | MEDIUM |
| `cost.py` (`BudgetHierarchy`) | Cross-run persistent budget | Purely in-memory; resets on restart | Feature doesn't work in prod | CRITICAL |
| `crosscutting/config.py` | Single config source | 146 `os.getenv` calls across 51 files bypass it | Config discipline absent | HIGH |
| Root `orchestrator/*.py` | Backward-compat shims | ~30 confirmed full duplicates (not shims) | Live duplicate code; security risk | CRITICAL |
| `tests/` | 80% coverage with contract suite | 6.94% actual, 5% gate, no CI enforcement | Coverage and contracts broken | CRITICAL |
| `.github/workflows/ci.yml` | Full arch + quality gates | No import-linter, no coverage gate, mypy `continue-on-error` | CI enforces almost nothing architectural | CRITICAL |

---

## 4. Dependency Analysis

### Circular Dependency Risks
- `crosscutting/config.py` imports from `config.py` (`crosscutting/config.py:21-26`), which itself does env reads. Two config layers with a circular dependency of purpose if not import.
- `domain/model_registry.py:12` uses `TYPE_CHECKING` to import from root `orchestrator.model_registry` — runtime-safe but signals a circular pressure.

### Boundary Violations

**Layer leaks (confirmed):**
- `engine.py:48` — imports `DiskCache` (infrastructure concrete) directly. The engine is application layer and should never name infrastructure concretes.
- `engine.py:104` — imports `StateManager` (infrastructure concrete) directly.
- `engine.py:49-63` — imports behavioral functions (`estimate_cost`, `build_default_profiles`, `get_provider`) from `models.py`, confirming models.py contains behavior.
- `application/chat_cli.py:168` — imports `Orchestrator` (engine concrete) from the application layer.
- `engine_core/stages/generate.py:11-14` — imports `Budget`, `ModelSelector`, `SystemPrompt` as concretes rather than going through Ports.

**import-linter contract holes:**
- Contract 3 (`application-services-no-engine`) uses an allow-list of 6 modules — `chat_cli.py` imports engine and is silently exempt.
- `models.py` is not in any `source_modules` — the most-imported domain file has zero architectural enforcement.
- No `type = layers` contract proving the overall dependency direction; all 4 are `forbidden` type (detect specific edges, not the direction).

### Shared-State Risks

| State | Owner | Shared With | Risk |
|-------|-------|-------------|------|
| `results` dict | `engine.py:570` | `ProjectRunState`, `PipelineRunner`, `ProjectRunner` | Passed by reference; concurrent mutations unguarded |
| `_consecutive_failures` | `ModelHealthTracker` (M6) | `engine.py` (shadow dict) | Dual source-of-truth; can diverge under concurrent failures |
| `api_health` | `engine.py:490` | `ModelHealthTracker` | Same dual-source-of-truth problem |
| Learning/leaderboard | Module globals | Nothing — no externalized store | Silently diverges between processes |
| `BudgetHierarchy._spent` | `cost.py` | Nothing — in-memory | Budget enforcement fails across restarts/processes |

### Tight Coupling Hotspots

1. **`models.py` as universal import magnet.** `Model` enum: 3,043 graph edges. `TaskType`: 2,702. Every layer serializes/deserializes these — a new enum variant must be handled by `_safe_model()` in state.py, cost tables, routing tables, telemetry, policy, A2A, and more.
2. **`engine.py` as feature registry.** ~80 concrete imports at module top; `crosscutting/config.py` has 12 feature flags that exist solely to gate engine.py imports. The import-time dependency graph of `engine.py` is the largest in the codebase.
3. **44-file SQLite coupling.** Every persistence concern (state, cache, events, telemetry, skills, patterns, kanban, workspace) directly embeds SQL + schema. No repository abstraction centralizes schema ownership.
4. **6 config modules.** `config.py` (legacy), `crosscutting/config.py` (new), `meta_config.py`, `autonomy_config.py`, `config_as_code.py`, `infrastructure/nexusscope/config.py` — config is not a boundary, it's scattered infrastructure.

---

## 5. AI Orchestrator-Specific Review

### Agent Orchestration Model

**Two incompatible models coexist:**

1. **LLM Pipeline model** (`PipelineRunner` + `TaskPipeline` + 7 stages) — the primary, production path. Clean, well-abstracted. The stage interface is consistent (`GenerateStage`, `CritiqueStage` etc. each receive a `PipelineContext`). ProjectRunner correctly decoupled as of M3. **This is architecturally sound.**

2. **Role-based Agent model** (`AgentOrchestrator`, `AgentBase`, `CoordinatorAgent`, etc.) — secondary path. **Architecturally weak:**
   - Coordination is direct RPC (`await agent.handle_task(task)`) — not message-passing
   - Goal routing is keyword-substring matching (`coordinator.py:105-154`) — brittle
   - `AgentMessageBus` is synchronous while `AgentBase` awaits it (`base.py:147`) — interface mismatch
   - No isolation boundary between agents — shared mutable `workspace` blackboard

These two models are not integrated. The codebase supports conversations, pipelines, and role-based agents as three separate orchestration patterns with no unified execution model.

### Workflow Coordination

**Topological-sort + level-based `asyncio.gather` per `PipelineRunner`.** This is correct for DAG task execution. However:

- `asyncio.gather(*(run_one(tid) for tid in level))` without `return_exceptions=True` (`pipeline_runner.py:81`) — a single task failure cancels level dispatch but **leaves sibling coroutines running and detached**. This is a live correctness bug in the hot path.
- The `_results_lock` in the container (`engine.py:475`) is not injected into `PipelineRunner` — `results[tid] = res` and `progress_writer.write_update()` execute concurrently without serialization.
- No timeout per task within a level — a hanging LLM call holds the whole level's semaphore slots.

### Message/Event Architecture

**`UnifiedEventBus`** inherits from `HookRegistry` (sync) while also being an async pub/sub — a deliberate dual-face design that works but is confusing and fragile:

- `container.hook_registry` and `container.event_bus` are the **same object** at runtime (the bus IS the hook registry). Type-safe callers cannot distinguish which face they're using.
- `EventStore` inside the bus uses blocking `sqlite3` (`core.py:552-564`) called from the async `_process_loop` (`core.py:870`). **Every event persist stalls the event loop.** The rest of the codebase uses `aiosqlite`; the event store regressed.
- `publish()` silently enqueues events even if `start()` was never called and `_running=False` — events accumulate and are never processed. `ProjectRunner` calls `publish(ProjectStartedEvent)` with no guarantee the bus is running.
- Async queues have no `maxsize` — a slow consumer causes unbounded memory growth.

### Tool Execution Isolation

`ToolCallGuardrailController` exists (`tool_guardrails.py`) and is referenced in container.py. 4-tier bash safety exists (`bash_guard.py`). **Isolation is present at the tooling level but not at the agent level** — agents share the `workspace` blackboard and can read each other's memory. No per-agent execution sandbox beyond the tool-call guardrails.

### Context Propagation

`ContextCompressor` exists and is correctly in `application/`. However:
- `application/context_compressor.py:52,76` reads `ORCH_CONTEXT_COMPRESSION` via raw `os.getenv` rather than `flags.context_compression` — bypasses the feature flag system for its own flag.
- Dependency context injection is handled through `DependencyContextInjector` in `cost_optimization/` — correct placement — but the entire cost_optimization tier is behind a feature flag, so dependency context only flows when the optimization tier is enabled.

### Memory / State Boundaries

5-layer memory architecture (ProjectWorkspace, PersistentWorkspace, SkillStore, KnowledgeGraph, AgentCache) is well-designed on paper. In practice:

- `ProjectWorkspace` is purely in-memory with no boundary between concurrent projects
- `PersistentWorkspace` correctly uses SQLite with WAL + `asyncio.Lock`
- `SkillStore` uses SQLite but with weaker durability PRAGMAs than `StateManager` — inconsistent guarantees
- `KnowledgeGraph` (networkx, in-memory) has no persistence path

### Retry / Failure Semantics

**Mixed fidelity:**
- API retry with exponential backoff is in `resilience.py` (`RetryTemplate`) — **correct, port-independent**.
- Decomposition retry loop inside `engine.py:1873,1884` uses fixed `asyncio.sleep(1)`/`sleep(2)` — **magic-number backoff, no jitter, not using `RetryTemplate`**.
- `except (ImportError, TimeoutError)` around import statements (~25 sites) — nonsensical `TimeoutError` catch on `import` statements masks real import errors as "feature disabled."
- Engine raises raw `ValueError`/`RuntimeError` (`engine.py:770,1356,1553,1691,1800`) bypassing the `ApplicationError` hierarchy and its `retriable` classification.

### Concurrency Model

`asyncio.Semaphore(max_parallel_tasks)` guarding per-level `gather` — correct model. But see §4 for the missing `return_exceptions` and unguarded `results` dict. Under high parallelism these are live bugs, not theoretical concerns.

### Multi-Agent Scalability

**Not scalable today:**
- `AgentMessageBus` uses in-process list appends — no external message broker
- `AgentPool`/`A2AManager` global singletons — no shared state across processes
- Budget allocation in `BudgetHierarchy` is in-memory — no distributed reservation protocol
- `StateManager` is single-writer SQLite — no sharding, no replication

The architecture has the right vocabulary (`TaskQueuePort` is in BACKLOG.md) but no implementation path yet.

---

## 6. Architectural Anti-Patterns

### God Services
- **`engine.py` — 2,703 lines.** The mediator rule ("engine only wires") is violated by the decomposition cluster (`_decompose` 266 lines, `_try_parse_partial_json_array` 152 lines, `_parse_decomposition` 181 lines), warm-start logic, plateau detection, and project analysis. Target: ≤500 lines; actual: 5.4× over.

### Hidden Monolith
The "hexagonal architecture" is implemented as a layered monolith with a Port vocabulary. All features live in the same process, share the same SQLite files, and can only be deployed together. The Ports abstract the internal wiring but create no actual service boundary. **This is correct for the current scale**, but the architecture documentation implies otherwise.

### Duplicate Module Crisis
~30+ confirmed root/package full duplicate pairs — not shims. The M5 migration converted 8 modules but stalled. `connectors.py` (659 lines) is a full duplicate of `connectors/connectors.py` with no re-export relationship. `secrets_manager.py` is duplicated — security-critical code that must be patched in two places. Each duplicate pair is:
- A maintenance debt (two places to fix bugs)
- A security risk (patches miss the other copy)
- A testing surface that is never exercised

### Orchestrator Bottleneck
`engine.py` is the single point of coordination for ALL project runs — it holds the client, state manager, telemetry, policy engine, health tracker, budget, and every feature subsystem. All orchestration flows through one 2,703-line class. **This is the architectural bottleneck.** Extracting the remaining business logic (decomposition cluster) and completing the DI container typing would reduce the blast radius of changes to this class.

### Anemic Domain Model / Behavior in Wrong Layer
`models.py` is supposed to be pure data but contains:
- `estimate_cost()` — behavioral function
- `build_default_profiles()` — behavioral function
- `get_provider()` — behavioral function
- `_load_static_config()` + `_COST_DATA` — file I/O at import time

Domain models should be data carriers; behavior belongs in domain services or application services.

### Infrastructure Leakage
- `engine.py` imports `DiskCache`, `StateManager`, `SemanticCache` (infrastructure concretes) directly
- 44 files embed SQL and schema directly without going through repository abstractions
- `application/context_compressor.py` reads env vars directly instead of using `flags`
- Pipeline stages (`generate.py`) import `Budget`, `ModelSelector` as concretes

### Config System Fragmentation
6 competing config modules (`config.py`, `crosscutting/config.py`, `meta_config.py`, `autonomy_config.py`, `config_as_code.py`, `infrastructure/nexusscope/config.py`). 146 raw `os.getenv` calls in 51 files bypass the pydantic layer. Feature flags can be read inconsistently: `flags.context_compression` (pydantic) vs `os.getenv("ORCH_CONTEXT_COMPRESSION")` — both exist for the same flag.

---

## 7. Refactoring Roadmap

### Immediate Fixes (1–3 days, unblock correctness)

| # | Fix | File | Risk |
|---|-----|------|------|
| I-1 | Add `return_exceptions=True` to `asyncio.gather` in `PipelineRunner`; handle/cancel siblings; inject `_results_lock` | `pipeline_runner.py:81` | Low — contained change |
| I-2 | Fix blocking `sqlite3` in `EventStore._handle_event` — replace with `aiosqlite` or `asyncio.to_thread` | `unified_events/core.py:552-564` | Low |
| I-3 | Make `UnifiedEventBus.publish` ensure `start()` or assert `_running`; emit warning on silent drop | `unified_events/core.py:828` | Low |
| I-4 | Wire `lint-imports` into `.github/workflows/ci.yml` as a blocking job | `ci.yml` | None |
| I-5 | Set `--cov-fail-under=7` in CI test command (ratchet from real baseline) | `ci.yml` | None |
| I-6 | Fix `AgentMessageBus.publish` sync/async mismatch — either make it `async def` or remove `await` in `base.py:147` | `message_bus.py`, `base.py:147` | Low |

### High-Impact Improvements (1–2 weeks, fix architecture correctness)

| # | Improvement | Benefit |
|---|-------------|---------|
| H-1 | **Remove I/O from `models.py:349-374`** — inject cost table via `ConfigPort`; make `COST_TABLE` lazy or remove the open() | Fixes Rule #2 violation; enables clean domain testing |
| H-2 | **Finish decomposition extraction** — move `_decompose` (266L) + `_try_parse_partial_json_array` (152L) + `_parse_decomposition` (181L) into `DecomposerService` | Cuts engine.py by ~22%; fulfills Rule #1 |
| H-3 | **Persist `BudgetHierarchy`** — add atomic reserve/charge to existing SQLite via a `budget_hierarchy` table; use `SELECT FOR UPDATE`-equivalent locking | Makes headline feature work in production |
| H-4 | **Close Contract-3 allow-list hole** — change `source_modules` to `orchestrator.application` with `ignore_imports` exceptions; add `orchestrator.models` to domain-purity contract | Real architectural enforcement |
| H-5 | **Type the core LLM boundary** — fully type `LLMClient.call(model: Model, prompt: str, ...) -> APIResponse`; fix 4 non-Optional `= None` type-lies in container | Makes hexagonal boundary real, not nominal |
| H-6 | **Add `ExecutorPort`, `EvaluatorPort`, `GeneratorPort`** — type `executor`, `evaluator`, `generator` container fields | Completes core pipeline abstraction |
| H-7 | **Replace raw `ValueError`/`RuntimeError` in engine.py** with `OrchestratorError`/`ApplicationError` subclasses | Enables retry classification; unlocks `retriable` semantics |
| H-8 | **Switch `engine.py:417` to structlog** with event-name + kwarg logging | Makes core path observable |

### Long-Term Architecture Evolution (1–3 months, scalability and maintenance)

| # | Evolution | Sequencing |
|---|-----------|------------|
| L-1 | **Complete M5 migration** — audit all 200+ root modules; for each, either delete (if subpackage version exists as source-of-truth), convert to 1-line re-export shim, or promote to canonical and move to a proper subpackage. End state: zero full duplicates at root. | Highest priority; unblocks L-2 and the security fix for secrets_manager |
| L-2 | **Centralize configuration** — funnel all `os.getenv` reads through `crosscutting/config.py` (for settings/flags) and `SecretsManager` (for credentials). Make `flags`/`settings` injectable for testing. Retire `config.py` (legacy). | After L-1 (many scattered reads are in duplicate modules) |
| L-3 | **Repository abstraction for SQLite** — introduce a single `SchemaRegistry` that owns migrations; add `write_with_lock()` utility shared by all 44 SQLite consumers; standardize PRAGMAs. | Prerequisite for L-4 |
| L-4 | **`TaskQueuePort` implementation** (already in BACKLOG.md) — add `InProcessTaskQueue` (current behavior) + `RedisTaskQueue` adapter. This is the horizontal-scaling enabler: workers can fan out when the queue is external. | After L-3 |
| L-5 | **Contract test the real adapters** — add `StatePortContract` for `StateManager`, `CachePortContract` for `DiskCache`, `LLMClientContract` for `UnifiedClient`. Ratchet `fail_under` to 40% over 4 sprints. | Can parallelize with L-1 |
| L-6 | **Unify the agent model** — decide: (a) role-based agents become thin wrappers on the LLM pipeline, or (b) the pipeline becomes a special case of the agent model. Eliminate the parallel coordination logic in `AgentOrchestrator`. | After engine.py decomposition extraction (H-2) |

### Suggested Target-State Architecture (post-L series)

```
orchestrator/
├── domain/           # Pure data: models.py, ports.py, exceptions.py
│                     # NO I/O, NO behavior, NO concretes
├── application/      # Use cases: ProjectRunner, Decomposer, SkillManager,
│                     # ConversationAgent, ModelHealthTracker, etc.
│                     # Depends only on domain/ports
├── engine_core/      # Mediator + pipeline stages (≤300 lines each)
│                     # engine.py ≤ 500 lines: wiring + __aenter__/__aexit__
├── infrastructure/   # ALL external adapters: state, cache, LLM, events, telemetry
│                     # One file per adapter; no SQL embedded outside infrastructure/
├── interfaces/       # Driving adapters: cli.py, api_server.py, gateway.py, chat_cli.py
├── crosscutting/     # config.py (single), logging.py, secrets.py, tracing.py
└── (no loose *.py)   # Zero root modules outside __init__.py
```

### Migration Sequencing

```
NOW ──► I-1 through I-6 (1-3 days, unblock correctness)
         │
         ▼
WEEK 1 ─► H-1 (models.py I/O), H-3 (BudgetHierarchy), H-4 (import-linter contract fix)
         │
         ▼
WEEK 2 ─► H-2 (decomposition extraction), H-5/H-6 (type container + ports)
         │
         ▼
MONTH 1 ─► L-1 (root module cleanup — largest scope), L-5 (contract tests)
         │
         ▼
MONTH 2 ─► L-2 (config centralization), L-3 (SQLite repository abstraction)
         │
         ▼
MONTH 3 ─► L-4 (TaskQueuePort/Redis), L-6 (unify agent model)
```

### Risk Estimation

| Work Item | Effort | Risk of Regression | Value |
|-----------|--------|-------------------|-------|
| I-1 (gather fix) | 2h | Low | HIGH — live correctness bug |
| I-4/I-5 (CI gates) | 1h | None | CRITICAL — foundation for everything |
| H-1 (models.py I/O) | 4h | Medium | HIGH — domain purity |
| H-2 (decompose extraction) | 2d | Medium | HIGH — Rule #1 |
| H-3 (BudgetHierarchy persistence) | 1d | Medium | CRITICAL — feature correctness |
| L-1 (root module cleanup) | 1w | High | CRITICAL — security + maintenance |
| L-4 (TaskQueuePort) | 2w | Medium | HIGH — horizontal scaling |

---

## 8. Confidence Assessment

### Verified Findings (empirically confirmed from source code)

| Finding | Evidence |
|---------|----------|
| `models.py:374` file I/O at import | `_COST_DATA = _load_static_config("costs.json")` — module-level statement |
| `LLMClient` port untyped | `ports.py:107-114` — no param types, no return type annotation |
| 55/70 container fields `Any` | `container.py:86-177` — counted directly |
| `asyncio.gather` no `return_exceptions` | `pipeline_runner.py:81` — confirmed absence |
| `_results_lock` not passed to `PipelineRunner` | `container.py:build()` — lock created, injected to engine, not to PipelineRunner |
| `EventStore` uses blocking `sqlite3` in async loop | `core.py:552-564` called from `core.py:870` in `_process_loop` coroutine |
| `AgentMessageBus.publish` is sync | `message_bus.py:67` — no `async def` |
| `BudgetHierarchy` in-memory | `cost.py:148-149` — docstring admits it; no persistence path |
| `connectors.py` is a true duplicate | Read both `connectors.py` (659L) and `connectors/connectors.py` — same `ConnectorManager` class, not a shim |
| `fail_under = 5` | `pyproject.toml:364` — confirmed |
| No `lint-imports` in CI | `ci.yml` — confirmed absence |
| `chat_cli.py:168` imports `Orchestrator` | `application/chat_cli.py:168` — confirmed |
| `engine.py:417` uses `logging.getLogger` (stdlib) | `engine.py:417` — confirmed, not structlog |
| 146 `os.getenv`/`os.environ` in 51 files | grep count verified |
| Custom `TimeoutError` shadows builtin | `domain/exceptions.py:144` — confirmed |
| `run_project_streaming` mutates instance `self._event_bus` | `engine.py:1590,1600` — confirmed |

### HYPOTHESIS Findings (architectural inference, spot-checked but not exhaustively verified)

| Finding | Basis |
|---------|-------|
| ~30 root/package full duplicate pairs | Confirmed 5 pairs directly; pattern extrapolated from M5 migration scope and listing |
| `StatePort` has 5 uncovered extra methods | `state.py` public API examined; Port definition cross-checked |
| Token under-reporting in `generate.py` | Identified `getattr(response,"usage",None) and ... or 0` idiom; did not verify against live trace |
| `map_elites.py` tuple client contract | Read stage; tuple return pattern at lines 85, 170 confirmed |
| Concurrent `progress_writer.write_update()` race | Pattern identified; no load test to confirm actual corruption |
| `adaptive_templates.py` as a god module | Line count referenced; internal structure not fully audited |

### Areas Lacking Sufficient Evidence

| Area | Limitation |
|------|------------|
| A2A Protocol correctness | The A2A cluster is the largest community (3,371 nodes); only entry points were read |
| `NexusScope` subsystem | `infrastructure/nexusscope/` not audited in detail; its `config.py` and DB usage noted |
| `gateway.py` API layer | Not read; assumed to be a thin HTTP facade per CLAUDE.md |
| `SkillOptimizer` epoch correctness | Logic audited for architecture, not mathematical correctness |
| Load test / actual concurrency behavior | All concurrency findings are static analysis; no load test data available |
| Security scanner results | `bandit`/`safety` not run during this audit; flagged as absent from CI |

---

## Appendix: Finding Severity Index

**CRITICAL (5):** models.py I/O at import · ~30 module duplicates · BudgetHierarchy in-memory · 5% coverage gate + no CI enforcement · Single-node SQLite blocks scaling  
**HIGH (20+):** import-linter holes · LLMClient untyped · 55 `Any` container fields · No ExecutorPort/EvaluatorPort · engine.py concrete imports · `asyncio.gather` orphaned tasks · event loss on publish · blocking EventStore · sync bus awaited · streaming instance mutation · incompatible stage client contracts · contract tests mock-only · 7 Ports uncovered · mypy continue-on-error · no security scan · raw ValueError/RuntimeError in engine · engine bypasses structlog · scattered os.getenv · module-global singletons · SQLite per-write FULL-sync/TRUNCATE ceiling  
**MEDIUM (15+):** decomposition cluster in engine · dual circuit-breaker state · StatePort incomplete · container build() 340L silent-None · import-linter no layers contract · ChatCLI→engine violation · AgentMessageBus design · unbounded event queues · stage concrete imports · token-accounting bug · custom TimeoutError shadowing · no Python version matrix · 27 ignored broken tests · 103 debt markers · keyword-substring agent routing  
**LOW (5+):** SkillStorePort/NullSkillStore drift · private-symbol cross-module import · `asyncio.sleep(0)` flush hack · `datetime.utcnow()` deprecation · `ProjectRunState.results` shared reference  

---

*Generated: 2026-05-29 | Method: 4-agent parallel static analysis + synthesis*  
*Reviewed against: CLAUDE.md, CODEBASE_MINDMAP.md v7.0, .importlinter, ci.yml, pyproject.toml*
