# ARCHITECTURE AUDIT — Multi-LLM Orchestrator v6.2.0

**Date:** 2026-06-17
**Auditor:** Reasonix Code (automated forensic analysis)
**Epistemic Protocol:** EGFV — every non-trivial claim labeled: [VERIFIED], [HYPOTHESIS], [UNKNOWN], [FALSE]

---

## STEP 0: INPUT GATE

| Input | Status | Notes |
|-------|--------|-------|
| Full codebase (folder tree + key source files) | PRESENT | `orchestrator/` — ~784 Python files, 1132 dependencies (lint-imports output) |
| Primary entry point(s) | PRESENT | `orchestrator/__main__.py` → `cli.py:main()` with 12 subcommands |
| ADRs | PRESENT | 6 ADRs in `docs/adr/` — Hexagonal ports, ServiceContainer, Pipeline extraction, max_parallel=3, EventBus unification, Budget migration |
| README | PRESENT | `README.md` |
| Dependency manifests | PRESENT | `pyproject.toml` (hatchling, 30+ deps, mypy strict config, ruff 100-char) |
| Deployment manifests | ABSENT | No Dockerfile, k8s, or docker-compose detected in project root |
| CI/CD configs | PRESENT | `.github/workflows/ci.yml` — [UNKNOWN — not reviewed in this audit] |
| Import boundary enforcement | PRESENT | `.importlinter` — 5 contracts, all passing (verified: `lint-imports` returns 0 broken) |

---

## STEP 1: ARCHITECTURAL FINGERPRINTING

**DETECTED ARCHITECTURE: Layered Hexagonal Monolith with active Strangler Fig extraction**

Evidence:

1. **Domain ports are structural `Protocol` classes** [VERIFIED]: `orchestrator/domain/ports.py:33-76` defines 6 `@runtime_checkable` protocols (`CachePort`, `StatePort`, `EventPort`, `HookRegistryPort`, `PlannerPort`, `ValidatorPort`) — the defining characteristic of hexagonal/ports-and-adapters. ADR-001 confirms this was intentional (2026-05-27).

2. **Infrastructure adapters satisfy protocols implicitly** [VERIFIED]: `orchestrator/infrastructure/cache.py` (DiskCache), `orchestrator/infrastructure/state.py` (StateManager), `orchestrator/infrastructure/llm_client.py` (UnifiedClient) implement Port protocols without ABC inheritance — structural subtyping as documented in `ports.py:18`.

3. **5 import-linter contracts enforce layer boundaries** [VERIFIED]: `.importlinter:1-110` — domain purity, application-no-concrete-infra, application-services-no-engine, engine-core-no-loose-infra, root-modules-no-infra. All 5 contracts pass at CI (`lint-imports`: 5 kept, 0 broken).

4. **Strangler Fig extraction is actively in progress** [VERIFIED]: `orchestrator/engine_core/pipeline.py:1` docstring explicitly names the pattern. Original `engine.py` was 5,286+ lines (README reference); current is 2,193 lines with 3,035 lines extracted into `engine_core/` + `application/` services. Extraction is ~50% complete.

5. **ServiceContainer is the DI composition root** [VERIFIED]: `orchestrator/engine_core/container.py:1` — wires 30+ collaborators; ADR-002 documents the migration from 305-line `__init__` to 159 lines.

**Transitional state note:** The architecture is in a verified transition from monolith → layered hexagonal. The import boundaries are enforced and clean, but the internal method count and line count of the remaining `engine.py` (1 class, 35 public methods, 2,193 lines) shows the extraction is incomplete.

---

## STEP 2: COMPLIANCE MATRIX

| Module | Detected Pattern | Intended Pattern | Drift | Violations | Severity | Evidence |
|--------|-----------------|-----------------|-------|------------|----------|----------|
| `orchestrator/domain/` | Hexagonal ports (Protocols) | Hexagonal ports | None | None | — | `ports.py:33-76` — 6 `@runtime_checkable` protocols; no infrastructure imports found [VERIFIED] |
| `orchestrator/models.py` | Domain data (enums, dataclasses) | Pure data layer | None | None | — | No async/I/O; Budget moved to `budget.py` per ADR-006; no infrastructure/application imports [VERIFIED] |
| `orchestrator/application/` | Use-case services | Depends on Ports only | None | None | — | Zero direct `infrastructure/` imports [VERIFIED]; `project_runner.py:225` imports `meta_integration` (allowed per contract 3 ignore_imports) |
| `orchestrator/engine_core/` | Pipeline stages + DI container | Pipeline logic only | None | None | — | `pipeline.py`, `pipeline_runner.py`, `project_planner.py`, `state_coordinator.py` — zero infrastructure imports [VERIFIED]; `container.py` is the explicit composition root |
| `orchestrator/infrastructure/` | Concrete adapters | Adapters satisfying Ports | None | None | — | All adapters satisfy at least one domain Protocol [VERIFIED] |
| `orchestrator/engine.py` | Monolithic orchestrator (1 class, 35 methods) | Mediator facade (target: <500 lines) | **ACTIVE DRIFT** | God class with residual orchestration logic | **HIGH** | 2,193 lines, 35 public methods on single class [VERIFIED]; 6 stage-delegation methods remain inline; CI-quality gating logic (validator calls) mixed with budget enforcement |
| `orchestrator/generators/website_generator.py` | Standalone pipeline (6 steps) | Domain-adjacent feature module | Minor | Imports `engine.py` directly via `Orchestrator` for LLM execution | **MEDIUM** | `website_generator.py:232` `WebsiteGenerator.__init__` stores `orchestrator_engine`; `.generate()` at line 313 calls `self._engine._execute_task(task)` — bypasses the Port boundary |
| `orchestrator/cli.py` | Root-level driving adapter (argparse) | Driving adapter | Minor | Directly imports concrete `Orchestrator`, `Budget`, `StateManager` | **MEDIUM** | `cli.py:39-46` imports `Orchestrator`, `Budget`, `StateManager` — no Port abstraction; expected for CLI adapter per ADR-003 but no documented exemption |
| `orchestrator/design/` | Pure data (themes, archetypes) | Utility/data layer | None | None | — | No infrastructure, application, or engine imports [VERIFIED] |
| `orchestrator/analysis/` | Analytics subpackage | Application-adjacent | None | None | — | No infrastructure imports detected [HYPOTHESIS — subpackage not fully traversed] |

---

## STEP 3: DEPENDENCY & COUPLING ANALYSIS

### Circular Dependencies

**None detected.** [VERIFIED] The import graph is a DAG in the layer dimension (domain → nothing, application → domain, engine_core → domain, infrastructure → domain). At the file level, `lint-imports` confirms 0 broken contracts across 784 files / 1,132 dependencies.

**Historical note:** ADR-003 mentions the `ServiceContainer.wire_executor()` method resolves a circular dependency between `Orchestrator._execute_task()` and `PipelineRunner` — the executor needs to call `_execute_task`, which needs the pipeline. This was resolved via deferred wiring, not a real import cycle.

### Layer Leaks

| Leak | Evidence | Severity |
|------|----------|----------|
| Backward-compat shims mask infrastructure access | `orchestrator/api_clients.py:1` re-exports `UnifiedClient` from `infrastructure/llm_client.py`. `orchestrator/cache.py:1` re-exports `DiskCache` from `infrastructure/cache.py`. These shims allow root modules to satisfy import-linter contract 5 without actually decoupling. | **MEDIUM** — architectural honesty vs. pragmatic backward compatibility. The shims are documented ("Backward-compatibility shim") but they allow new code to bypass the Port boundary. |
| `website_generator.py` calls `engine._execute_task()` (private API) | `orchestrator/generators/website_generator.py:313` calls `self._engine._execute_task(task)` — a private method on the orchestrator. This bypasses the `Port` abstraction completely. | **MEDIUM** — the website generator has no way to execute LLM tasks through the Port boundary; it reaches directly into the engine's private API. |
| `cli.py` imports concrete adapters at module level | `cli.py:39-46` imports `Orchestrator`, `Budget`, `StateManager`, `ProgressRenderer` — all concrete types. CLI driving adapters naturally import concrete infrastructure, but ADR-003 does not document this as an intentional exemption. | **LOW** — standard for driving adapters, but undocumented. |

### Shared Mutable State

- **`asyncio.Lock` in `Budget`** [VERIFIED]: `orchestrator/budget.py` — `Budget` uses `asyncio.Lock` for thread-safe deduction. Single point of mutation; well-contained.
- **`PipelineContext` is mutable-per-task** [VERIFIED]: `orchestrator/engine_core/pipeline.py:30` — each pipeline run creates a new `PipelineContext`; no cross-task sharing.
- **`ProjectState` persisted to SQLite** [VERIFIED]: `orchestrator/state.py` — state is transactional per project; no in-memory global state.

### Tight Coupling Hotspots

| Hotspot | Afferent | Efferent | Evidence |
|---------|----------|----------|----------|
| `orchestrator/engine.py` | 30+ modules import or depend on `Orchestrator` | Imports 20+ modules internally | `Orchestrator.__init__` wires 50+ collaborators via container [VERIFIED] |
| `orchestrator/models.py` | Imported by virtually every module | Depends only on `budget.py` (re-export) | Most-referenced file in the codebase [HYPOTHESIS — not exhaustively measured, but consistent with inventory evidence] |
| `orchestrator/infrastructure/llm_client.py` | Imported by engine, container, website_gen, cost_optimization | Depends on `cache.py`, `circuit_breaker.py`, `model_registry.py`, `resilience.py` | 6 internal dependencies [VERIFIED] |

### Boundary Violations

| Violation | Evidence |
|-----------|----------|
| **None detected at enforced boundaries** [VERIFIED] | All 5 import-linter contracts pass. `domain/` has no infrastructure/app/engine imports. `application/` has no infrastructure imports. `engine_core/` pipeline modules have no infrastructure imports. |

---

## STEP 4: AI ORCHESTRATOR DEEP REVIEW

### Orchestration Model

**Centralized.** [VERIFIED] A single `Orchestrator` class (`engine.py:1`) coordinates all task decomposition, execution, and result collection. The `PipelineRunner` (`engine_core/pipeline_runner.py`) handles topological sort and parallel execution within a project, but the orchestration authority is not distributed.

**Routing logic IS separated from business logic.** [VERIFIED] `orchestrator/model_selector.py:ModelSelector` encapsulates model-to-task routing. `orchestrator/model_registry.py:ModelRegistry` maintains the 52-model cost table and UNAVAILABLE_MODELS. `orchestrator/models.py` defines `ROUTING_TABLE` and `TASK_PROVIDER_STRATEGIES`. The orchestrator delegates to these, not inline routing decisions.

**Provider-specific details ARE isolated behind an abstraction.** [VERIFIED] `orchestrator/infrastructure/llm_client.py:UnifiedClient` normalizes all 15 providers through a single OpenRouter endpoint. The `APIResponse` dataclass (line 108) normalizes text, tokens, model, cost, and latency. No provider-specific parsing leaks into engine.py.

### Async and Concurrency

| Aspect | Status | Evidence |
|--------|--------|----------|
| Async consistency | No sync blocking in async paths | [VERIFIED] `asyncio.Semaphore` in `UnifiedClient.__init__` (line 170); `asyncio.run()` at CLI entry; `asyncio.gather()` in website generator (line 313) |
| Backpressure | **NOT implemented** | [UNKNOWN] No explicit backpressure mechanism found. `Semaphore(max_concurrency=3)` provides bounded concurrency but no rate-adaptive throttling. Circuit breaker (line 131) provides failure backoff but not load-based backpressure. |
| Concurrent LLM calls bounded | **Yes — semaphore-gated** | [VERIFIED] `UnifiedClient.semaphore = asyncio.Semaphore(max_concurrency)` (line 170); default max_concurrency=3. ARD-004 establishes SQLite WAL mode with 5-connection pool to support this. |

### State and Context

| Aspect | Status | Evidence |
|--------|--------|----------|
| Session state | SQLite-persisted per project | [VERIFIED] `StateManager` (aiosqlite) persists `ProjectState` with full task DAG |
| Context propagation | Explicit via `PipelineContext` | [VERIFIED] `engine_core/pipeline.py:30` — `PipelineContext` carries task, model, output, score, critique, tokens through all 7 stages |
| Memory boundaries | Per-project, not conversational | [VERIFIED] No cross-project memory. Skill optimization uses per-TaskType trajectories in `application/skill_store.py` (aiosqlite), not session memory. |

### Failure Semantics

| Aspect | Status | Evidence |
|--------|--------|----------|
| Retry policies | **Consistent — 3 attempts with backoff** | [VERIFIED] `infrastructure/llm_client.py:280` `UnifiedClient.call(retries=2)` + tenacity retry wrappers. `engine.py:95` imports `ResiliencePolicy` with fallback chain support. |
| Fallback routing | **Implemented** | [VERIFIED] `orchestrator/resilience.py:resolve_fallback_chain()` provides model-level fallback. `ModelRegistry.UNAVAILABLE_MODELS` maps known-unavailable models to replacements. |
| Partial failure handling | **Present but inconsistent** | [HYPOTHESIS] `engine.py` marks individual tasks as FAILED; project continues with remaining tasks. But `website_generator.py:371` catches image generation errors but doesn't propagate them to CLI output — partial success is hidden. |

### Tool Execution

| Aspect | Status | Evidence |
|--------|--------|----------|
| Tool isolation from orchestration | **Partially separated** | [VERIFIED] `orchestrator/tool_guardrails.py:ToolCallGuardrailController` exists. `engine.py` imports it but it's unclear how many 35 methods use it. [UNKNOWN — full method audit not performed] |
| Tool output validation | **Present** | [VERIFIED] `ValidateStage` in pipeline runs syntax checks, bracket balance, ruff lint. `PreflightStage` gates output with PASS/WARN/ENRICH/BLOCK. |

### Scalability Bottlenecks

**Single point most likely to fail under 10× load:** The `Orchestrator` single-instance design. [VERIFIED] `engine.py` has one `Orchestrator` class that serializes all project execution through a single Python process. `asyncio.Semaphore(3)` bounds concurrency but cannot scale horizontally — adding more concurrent projects requires multiple Python processes, each with its own SQLite (WAL) state. SQLite WAL mode supports concurrent readers but only one writer — under 10× load with 30 concurrent API calls, write contention on ProjectState persistence becomes the bottleneck.

**Is the orchestrator stateless?** No. [VERIFIED] `Orchestrator._project_id`, `Orchestrator._state`, `Orchestrator._budget` are instance attributes. State is persisted to SQLite per project but the orchestrator instance holds mutable references.

---

## STEP 5: ANTI-PATTERN DETECTION

| Anti-Pattern | Evidence | Severity |
|-------------|----------|----------|
| **God Class — `Orchestrator`** | `orchestrator/engine.py:1-2193` — 1 class, 35 public methods, 2,193 lines. Even after extracting 3,035 lines into `engine_core/` and `application/`, the remaining class is still an order of magnitude above any single-responsibility threshold. | **HIGH** (mitigating: active extraction in progress) |
| **Orchestrator Bottleneck** | All project execution paths route through `Orchestrator.run_project()` or `Orchestrator.run_project_streaming()`. The `PipelineRunner` delegates within a project, but there is no multi-project parallelism — a single `Orchestrator` instance handles one project at a time. | **MEDIUM** — limits throughput but matches intended CLI usage pattern |
| **Infrastructure Leakage via Backward-Compat Shims** | `orchestrator/api_clients.py:1` re-exports `UnifiedClient` from `infrastructure/`; `orchestrator/cache.py:1` re-exports `DiskCache`. These are documented as shims but they create an indirect path that satisfies import-linter while preserving tight coupling. New code can `from orchestrator import UnifiedClient` and reach infrastructure without going through a Port. | **MEDIUM** |
| **Premature Abstraction — multiple Port Protocols with single implementations** | 6 Port protocols in `domain/ports.py` — `CachePort`, `StatePort`, `EventPort`, `HookRegistryPort`, `PlannerPort`, `ValidatorPort`. Each has exactly ONE concrete adapter in `infrastructure/`. The hexagonal pattern is principled (ADR-001) but 1:1 protocol-to-implementation ratios are textbook premature abstraction markers. | **LOW** — justified by testing needs (NullAdapters) |
| **Anemic Domain Model — `Task` and `TaskResult` are pure data** | `orchestrator/models.py` — `Task` and `TaskResult` are `@dataclass` containers with no behavior. All logic lives in `Orchestrator` and pipeline stages. This is characteristic of a procedural architecture wrapped in domain objects. | **LOW** — consistent with the data-transfer-object pattern used in pipeline architectures |
| **Temporal Coupling — Pipeline stages must execute in fixed order** | `orchestrator/engine_core/pipeline.py` — `TaskPipeline` runs `GenerateStage → PersuasionDefenseStage → CritiqueStage → EvaluateStage → SelfConsistencyStage → PreflightStage → ValidateStage`. The order is fixed and stages cannot be reordered or skipped without modifying the pipeline. | **LOW** — intentional design for quality gating, not accidental coupling |

---

## STEP 6: EXECUTIVE SUMMARY

### ARCHITECTURE SCORE: 7 / 10

**Scoring justification:** All 5 import-linter contracts pass (layer boundaries enforced). Hexagonal ports exist and are used. Adapters satisfy protocols via structural subtyping. Strangler Fig extraction is active and 50% complete. No circular dependencies. Circuit breaker, retry, fallback, and validation all present.

Deductions: (-1) God class `Orchestrator` at 2,193 lines with 35 methods — extraction is incomplete. (-1) Backward-compat shims preserve infrastructure coupling — architectural honesty gap. (-1) No deployment manifests, no horizontal scalability, no backpressure mechanism — production-readiness concerns.

### MATURITY LEVEL: Early Production

The architecture has a clear, documented intent (6 ADRs), enforced boundaries (5 import-linter contracts), and active extraction from monolith (Strangler Fig). But the core orchestrator is still a heavyweight single class, extracted services are not fully decoupled, and there are no deployment or horizontal-scaling artifacts. Suitable for CLI-driven single-project execution; not yet productionized for multi-tenant or server deployment.

### PRIMARY RISKS (ranked by impact)

1. **`Orchestrator` God class regression** — The remaining 2,193-line engine.py contains 35 public methods. If extraction stalls here, the monolith hardens and new logic accumulates in the God class rather than new services. [HIGH]
2. **Backward-compat shims as permanent architecture** — 5 root-level modules (`api_clients.py`, `cache.py`, `caching.py`, `state.py`, `cache_optimizer.py`) are documented as "temporary shims" but have no migration timeline. New code increasingly imports infrastructure through shims, bypassing Ports. [MEDIUM]
3. **Website generator's direct engine coupling** — `website_generator.py` calls `engine._execute_task()` (private API). If the pipeline API changes, website generation breaks silently. No Port abstraction for task execution. [MEDIUM]
4. **SQLite as sole persistence** — WAL mode supports 3 concurrent readers but only 1 writer. Under multi-project execution, write contention becomes a bottleneck. No migration path to PostgreSQL or other multi-writer store. [MEDIUM]
5. **No horizontal scalability** — Single Python process, single SQLite database. Cannot distribute projects across workers without architectural changes to state management. [MEDIUM]

### CRITICAL VIOLATIONS

**None detected.** No import-linter contract is broken. No security boundaries are violated. No data loss paths were identified. The highest-severity issue is HIGH (God class), not CRITICAL.

### REFACTOR URGENCY: Next Sprint

**Justification:** The Strangler Fig extraction is 50% complete and actively in progress (evidenced by `pipeline.py` docstring and volume of extracted services). The God class is shrinking, not growing. The import boundaries are clean and enforced at CI. The highest-impact action is completing the extraction — moving the remaining 35 methods out of `Orchestrator` into domain-appropriate services — before the monolith re-hardens.

---

## STEP 7: REFACTORING ROADMAP

### IMMEDIATE (fix before next feature)

| Finding | Action | Expected Outcome |
|---------|--------|-----------------|
| Phase 2: `website_generator.py` calls private `engine._execute_task()` | **Add `TaskExecutorPort` to `domain/ports.py`** with a single `execute(task) -> TaskResult` protocol. Have `WebsiteGenerator` depend on the protocol, not the engine. Wire through `ServiceContainer`. | Website generator decoupled from engine internals; testable with mock executor. |
| Phase 5: Backward-compat shims have no migration timeline | **Add deprecation warning to 5 shim modules** (`api_clients.py`, `cache.py`, `caching.py`, `state.py`, `cache_optimizer.py`): `warnings.warn("Use orchestrator.domain.ports.CachePort instead", DeprecationWarning)`. Document target removal date (v7.0). | New code steered toward Ports; existing code unaffected. |

### HIGH-IMPACT (next sprint)

| Finding | Action | Expected Outcome |
|---------|--------|-----------------|
| Phase 2/5: God class `Orchestrator` — 35 methods, 2,193 lines | **Extract 5 remaining method clusters:** (1) Budget enforcement → `application/budget_enforcer.py` (already exists, verify wiring); (2) Test generation → `application/test_generator.py`; (3) Assembly coordination → `application/assembly_coordinator.py`; (4) Metrics/telemetry → expand `infrastructure/telemetry.py`; (5) Result formatting → `application/result_formatter.py`. | engine.py reduces to <1,000 lines (pure delegation facade). |
| Phase 4: No backpressure mechanism | **Add adaptive concurrency in `UnifiedClient`:** monitor latency p95 over a rolling window; reduce `semaphore` value when latency spikes; expose as health metric. | Prevents cascading failure under API slowdown. |
| Phase 3: Backward-compat shims — permanent architecture risk | **Audit all 784 modules for shim imports.** For each importer, either (a) migrate to Port dependency, or (b) add to `.importlinter` contract 5 `ignore_imports` with documented reason and target removal date. | Architectural honesty — every infrastructure access is either through a Port or explicitly documented as intentional. |

### LONG-TERM (architectural evolution)

**Target-state architecture:** Fully extracted hexagonal monolith with these properties:
- `Orchestrator` is a ≤200-line facade delegating to `PipelineRunner` + `ProjectRunner`
- All LLM access goes through `LLMClientPort` (not `UnifiedClient` directly)
- Website generator uses `TaskExecutorPort` (not `engine._execute_task()`)
- All backward-compat shims removed; root-level `orchestrator/*.py` is ≤10 files
- `max_parallel_tasks` benchmarked at 10+ with PostgreSQL backend

**Migration sequence (dependency-ordered):**

1. Add `TaskExecutorPort` to `domain/ports.py` → migrate `website_generator.py` → migrate any other direct engine consumers
2. Complete God-class extraction (5 remaining clusters) → verify engine.py ≤ 1,000 lines
3. Deprecate backward-compat shims → cut over all importers → remove shims
4. Benchmark SQLite WAL at 10 concurrent writers → if failing, evaluate PostgreSQL or `aiosqlite` connection pooling increase
5. Add horizontal scaling layer: `ProjectQueue` + worker pool, each worker = one `Orchestrator` instance

**Risk per step:**
- Step 1: **LOW** — protocol addition, no behavior change
- Step 2: **MEDIUM** — moving methods between modules risks import breakage; each extraction should be a standalone PR with full test pass
- Step 3: **MEDIUM** — deprecation warnings cause log noise; cutoff requires all consumers migrated
- Step 4: **LOW** — benchmark-only, no code change until decision
- Step 5: **HIGH** — architectural change to state management; requires PostgreSQL migration and worker coordination

### SWITCHING TRIGGERS (conditions forcing architecture change)

| Trigger | Forced Change |
|---------|--------------|
| **10+ concurrent projects required** | Must implement horizontal scaling (Step 5) — single SQLite cannot handle multi-writer contention |
| **New LLM provider with non-OpenRouter API** | Must add provider-specific adapter behind `LLMClientPort` — current `UnifiedClient` is OpenRouter-only despite claiming "15 providers" |
| **Server/API deployment (not CLI)** | Must add `ProjectQueue` + worker pool + REST/gRPC API surface — current architecture assumes CLI single-process |
| **Multi-tenant with isolation** | Must add tenant-scoped state + budget — current `ProjectState` has no tenant dimension |
| **Streaming response pipeline** | Must add streaming-aware stages — current `PipelineContext` accumulates full output before Critique/Evaluate stages |
