# Architecture Audit — ARCH-AUDIT-V2

**Project:** Multi-LLM Orchestrator
**Date:** 2026-06-25
**Branch:** `feat/response-healing`
**Method:** ARCH-AUDIT-V2 with EGFV epistemic protocol

---

## Phase 1: Architectural Fingerprinting

### DETECTED ARCHITECTURE: Layered Hexagonal with Legacy Monolith Core

**Evidence:**
1. **Port/Adapter boundary**: `domain/ports.py` defines 15+ runtime-checkable protocols ([VERIFIED](orchestrator/domain/ports.py:1)). `infrastructure/cache.py` implements `CachePort`, `infrastructure/state.py` implements `StatePort` — standard hexagonal adapter pattern.
2. **DI Container**: `engine_core/container.py` is a 708-line `ServiceContainer` dataclass wiring 45+ collaborators ([VERIFIED](orchestrator/engine_core/container.py:75)). Single composition root.
3. **Layer contracts**: `.importlinter` enforces 5 contracts: domain purity, application-no-infra, application-no-engine, engine-core-no-infra, root-no-infra ([VERIFIED](.importlinter:1)).
4. **Legacy monolith**: 257 root `orchestrator/*.py` files (75% of LOC) are the legacy dump being strangulated into subpackages ([VERIFIED](.)). `engine.py` at 1,311 LOC is the primary orchestrator monolith.
5. **Entry points**: `__main__.py` → `cli.py` → `entrypoints/cli_dispatch.run()` → `Orchestrator` (engine). Also `api_server.py` for HTTP mode. ([VERIFIED](orchestrator/__main__.py:1))

**Summary:** Intended architecture is Clean Architecture (ports/adapters with DI). Detected architecture is Clean Architecture at the edges (domain, infrastructure, application) with a **heavy legacy monolith at the root** being actively refactored.

---

## Phase 2: Compliance Matrix

| Module | Detected Pattern | Intended Pattern | Drift | Violations | Severity | Evidence |
|---|---|---|---|---|---|---|
| `domain/` | Pure protocols | Pure protocols | None | — | — | [VERIFIED](orchestrator/domain/ports.py:1) — stdlib-only imports |
| `application/` | Use cases + services | Use cases + services | 2 engine imports via transitive meta_integration | Contract 3 exemptions | LOW | [VERIFIED](.importlinter:62) — `cli_helpers`, `project_runner` exempted |
| `engine_core/stages/` | Pipeline stages | Stages via ports only | None after C1 refactor | — | — | [VERIFIED] Zero `orchestrator.application` imports in stages |
| `engine_core/container.py` | DI Container | DI Container | 45+ fields, all typed `Any` | No compile-time type safety | MEDIUM | [VERIFIED](orchestrator/engine_core/container.py:75) |
| `engine_core/decomposer.py` | Engine core module | Engine core only | Imports from `application.decomposer` | Contract 4 (indirect) | HIGH | [VERIFIED](orchestrator/engine_core/decomposer.py:10) |
| `engine.py` | Mediator monolith | < 400 LOC mediator | 1,311 LOC; 40+ instance attrs | CRITICAL (size) + HIGH (stateful) | CRITICAL | [VERIFIED](orchestrator/engine.py) |
| `infrastructure/` | Adapters | Pure adapters | 11,909 LOC — heaviest layer | Overgrown | MEDIUM | [VERIFIED] |
| `entrypoints/` | Driving adapters | Driving adapters | Clean after C3 | None | — | [VERIFIED](orchestrator/entrypoints/) |
| Root `*.py` dump | Legacy dump | ≤ 15 kernel files | **257 files** (75% of code) | **CRITICAL** | CRITICAL | [VERIFIED] 298 → 258 after A3/A4/A5 |
| `generators/website_generator.py` | Service | Service | 3,639 LOC | **God file** | HIGH | [VERIFIED](orchestrator/generators/website_generator.py) |
| `reasoning/ara_pipelines.py` | Service | Service | 4,288 LOC | **God file** | HIGH | [VERIFIED](orchestrator/reasoning/ara_pipelines.py) |

---

## Phase 3: Dependency & Coupling Analysis

### Circular Dependencies

| Cycle | Modules | Evidence | Severity |
|---|---|---|---|
| engine_core ↔ application (residual) | `engine_core/decomposer.py` → `application.decomposer` | [VERIFIED](orchestrator/engine_core/decomposer.py:10) | HIGH |
| Root → Sub (legacy only, being resolved) | ~37 root identicals deleted, 20 converted to shims | [VERIFIED] A3/A4/A5 results | MEDIUM (decreasing) |

**No circular dependency found in stages.** C1 (VSSamplerPort) successfully severed the `engine_core/stages` → `application` cycle. [VERIFIED]

### Layer Leaks

| Source | Target | Evidence | Severity |
|---|---|---|---|
| `application/skill_store.py` (legacy) | `aiosqlite` (infrastructure) | [VERIFIED] C2 refactored this — now `infrastructure/skill_store_adapter.py` | ~~HIGH~~ FIXED |
| `engine_core/stages/*.py` (legacy) | `application.verbalized_sampling` | [VERIFIED] C1 refactored — now `VSSamplerPort` injection | ~~HIGH~~ FIXED |

### Shared Mutable State

| State | Location | Risk | Evidence |
|---|---|---|---|
| **Orchestrator instance state** | `engine.py` — 40+ instance attrs (budget, telemetry, circuit-breaker) | **HIGH** — concurrent runs share mutable state | [VERIFIED](orchestrator/engine.py) |
| Budget tracking | `engine.py` mutates `self.budget` per run | MEDIUM — budget leaks across runs | [VERIFIED](orchestrator/engine.py) |

### High-Coupling Hotspots

| Module | Afferent (imported by) | Efferent (imports) | Hotspot score |
|---|---|---|---|
| `engine.py` | ~25 importers | ~15 modules | **HIGH** — main orchestrator bottleneck |
| `models.py` | ~180 importers | ~0 (stdlib) | **HIGH** — natural domain model hub |
| `container.py` | 1 (engine.py) | ~30 | MEDIUM — composition root |
| `domain/ports.py` | ~40 importers | ~0 (stdlib) | MEDIUM — natural port hub |

---

## Phase 4: AI Orchestrator Deep Review

### Orchestration Model

| Aspect | Assessment |
|---|---|
| Centralized vs Distributed | **Centralized** — `Orchestrator` (engine.py) is the single coordinator for all task execution |
| Routing vs Business Logic | **Separated** — `ModelSelector`, `TieredModelRouter`, `ConstraintPlanner` handle routing independently [VERIFIED](orchestrator/model_selector.py:1) |
| Provider abstraction | **Isolated** — `UnifiedClient` (api_clients.py) wraps all LLM providers behind a single async interface [VERIFIED](orchestrator/api_clients.py:1) |

### Async & Concurrency

| Aspect | Assessment |
|---|---|
| Async consistency | ✅ Consistent `async/await` throughout. No sync-over-async detected in hot paths. `__main__.py` uses `asyncio.run()`. |
| Backpressure | ⚠️ **HYPOTHESIS** — `TaskGuard` (concurrency controller) bounds parallel LLM calls. Semaphore in `PipelineRunner` bounds parallel tasks. No explicit backpressure mechanism for overloaded API. |
| Concurrent LLM bounds | ✅ `max_concurrency` and `max_parallel_tasks` wired through `ServiceContainer` → `PipelineRunner.execute_all()` [VERIFIED](orchestrator/engine_core/pipeline_runner.py:1) |

### State & Context

| Aspect | Assessment |
|---|---|
| Session state | `Orchestrator.__init__` assigns 40+ mutable instance attrs — **unsafe under concurrency** [VERIFIED](orchestrator/engine.py) |
| Context propagation | `PipelineContext` dataclass carries mutable state through stages — explicit, well-defined [VERIFIED](orchestrator/engine_core/pipeline.py:35) |
| Memory boundaries | `MemoryTierManager` with HOT/WARM/COLD tiers — documented in codebase |

### Failure Semantics

| Aspect | Assessment |
|---|---|
| Retry policies | ✅ Unified constants after D3: `STAGE_RETRY_GENERATE=2`, `STAGE_RETRY_CRITIQUE=1`, etc. [VERIFIED](orchestrator/operations/resilience.py:155) |
| Fallback routing | ✅ `ModelSelector.fallback()` with tier escalation. `FallbackHandler` in application/ [VERIFIED](orchestrator/application/fallback_handler.py:1) |
| Partial failure handling | ⚠️ `TaskStatus` enum has COMPLETED, DEGRADED, FAILED — partial success is tracked. **UNKNOWN** how well partial project failure is surfaced to users. |

### Scalability Bottleneck

| Bottleneck | Location | Risk |
|---|---|---|
| **Orchestrator statefulness** | `engine.py` — mutable per-run state on shared instance | **PRIMARY 10x FAILURE POINT** — B1 RunContext refactor deferred |
| Root dump | 257 files at root level | Architectural debt but not a runtime bottleneck |
| Generator god files | `website_generator.py` (3,639 LOC), `ara_pipelines.py` (4,288 LOC) | Maintenance bottleneck, not runtime |

---

## Phase 5: Anti-Pattern Detection

| Anti-pattern | Evidence | Severity |
|---|---|---|
| **God Service** — `engine.py` | 1,311 LOC mediator with 40+ instance attrs, all paths route through it | [VERIFIED] CRITICAL |
| **God File** — `website_generator.py` | 3,639 LOC single file | [VERIFIED](orchestrator/generators/website_generator.py) HIGH |
| **God File** — `ara_pipelines.py` | 4,288 LOC single file | [VERIFIED](orchestrator/reasoning/ara_pipelines.py) HIGH |
| **Root-level dump** | 257 files at `orchestrator/` root (75% of code) | [VERIFIED] CRITICAL |
| **Orchestrator Bottleneck** | All execution paths route through `engine.py` | [VERIFIED] HIGH |
| **Temporal coupling** | `Orchestrator.__init__` / `__aenter__` / `run_project` sequence has implicit ordering | [HYPOTHESIS] MEDIUM |
| **Infrastructure leakage** | C1+C2 fixed the major leaks. Residual: `infrastructure/` at 11,909 LOC — largest layer | [VERIFIED] MEDIUM |
| **Shared mutable state** | `engine.py` instance attrs | [VERIFIED] HIGH |
| **Premature abstraction** | `container.py` 45+ fields all typed `Any` — no compile-time type safety | [VERIFIED] MEDIUM |
| **God Container** | `ServiceContainer` (708 lines) wires 40+ deps, has shutdown/get_or_create/wire_* methods | [VERIFIED] MEDIUM |

**Not detected:** Hidden monolith, shared database coupling, anemic domain model, overengineering.

---

## Phase 6: Executive Summary

### ARCHITECTURE SCORE: 5 / 10

**Scoring rationale:**
- ✅ Clean ports/adapters boundary (domain/, infrastructure/)
- ✅ Import-linter contracts enforced in CI (5/5 passing)
- ✅ Async/await consistently used, concurrency bounded
- ✅ Provider abstraction (UnifiedClient) — clean
- ✅ C1+C2+C3 remediations applied (cycle broken, ports cleaned)
- ❌ Root-level dump (257 files, CRITICAL) caps score at 7
- ❌ `engine.py` god service (CRITICAL) caps score at 6
- ❌ Two god files > 3,500 LOC (website_generator, ara_pipelines — HIGH)
- ❌ Orchestrator stateful under concurrency (HIGH — RunContext deferred)
- ❌ Infrastructure layer overgrown (11,909 LOC)

### MATURITY LEVEL: Early Production

### PRIMARY RISKS (ranked)

| # | Risk | Impact | Workstream |
|---|---|---|---|
| 1 | `engine.py` god service — all paths route through 1,311 LOC stateful mediator | CRITICAL — 10x load failure point | B1 RunContext deferred |
| 2 | 257 root files (75% of code) outside any subpackage | CRITICAL — architecture boundary undefined | A5 ongoing |
| 3 | Two 3,500+ LOC god files (website_generator, ara_pipelines) | HIGH — maintenance bottleneck | Not yet planned |
| 4 | Orchestrator instance state mutable per run | HIGH — concurrency unsafe | B1 RunContext deferred |
| 5 | `engine_core/decomposer.py` → `application.decomposer` residual cycle | HIGH — single import but core path | A4 divergent reconciliation |

### CRITICAL VIOLATIONS

1. **Root-level module dump** — 257 files at `orchestrator/*.py` (75% of code) — [VERIFIED] — A3/A4/A5 in progress
2. **`engine.py` god service** — 1,311 LOC, 40+ mutable instance attrs, single failure point — [VERIFIED] — B1 RunContext deferred

### REFACTOR URGENCY: Immediate

The root dump + god service are CRITICAL violations that block any score above 7. A3/A4/A5 are actively reducing the root dump (298→258). B1 RunContext is the single highest-impact remaining item but is deferred. Without RunContext, the architecture cannot reach 7+ regardless of other improvements.

---

## Phase 7: Refactoring Roadmap

### IMMEDIATE (fix before next feature)

| Ref | Finding | Action | Outcome |
|---|---|---|---|
| Phase 2: CRITICAL-1 | Root dump (257 files) | Continue A5: move root files to subpackages. 103 remaining. | Root files < 15 (kernel) |
| Phase 3: HIGH | engine_core→application cycle (decomposer.py) | Reconcil A4: make root a re-export shim | Cycle eliminated |
| Phase 5: HIGH | Orchestrator statefulness | **B1 RunContext design + implementation** | Stateless Orchestrator |

### HIGH-IMPACT (next sprint)

| Ref | Finding | Action | Outcome |
|---|---|---|---|
| Phase 5: HIGH | God files (website_generator 3,639 LOC) | Extract generator steps into subpackage | WebsiteGenerator < 1,000 LOC |
| Phase 2: MEDIUM | Container 45+ `Any` fields | Type annotate all container fields | Compile-time safety |
| Phase 5: HIGH | Orchestrator bottleneck | B2 entrypoint pooling: singleton container per request | O(1) per-request cost |

### LONG-TERM

| Step | Migration |
|---|---|
| **Target architecture** | Fully layered hexagonal — root kernel (<15 files), all logic in subpackages, ports/adapters everywhere, orchestrator stateless |
| **Sequence** | A5 (root moves) → B1 (RunContext) → B2 (pooling) → B3 (load test) → C4 (residual cycles) → F1 (contract lockdown) |
| **Risk** | RunContext touches 100+ call sites. Integration test required before/after. Safe revert via monolith facade. |

### SWITCHING TRIGGERS

| Condition | Action |
|---|---|
| Root files exceed 260 again | Revert CI freeze, block commit |
| New feature requires modifying engine.py | Extract to new service first (mediator pattern) |
| Concurrent run test fails with budget bleed | Block until B1 RunContext implemented |
