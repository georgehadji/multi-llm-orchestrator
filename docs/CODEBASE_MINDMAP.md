# Multi-LLM Orchestrator — Architecture Mindmap

> **Version:** v7.0.0 (2026-05-29)  
> **engine.py:** 2,672 lines (−49% from v5.0)  
> **Application services:** 16 extracted classes  
> **Import contracts:** 4 (domain purity, app-no-infra, app-services-no-engine, engine-core-no-loose-infra)  
> **Architecture compliance score:** 9.2 / 10 (up from 5.5 after 7-milestone refactor)

---

## Hexagonal Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Multi-LLM Orchestrator v7.0               │
│       Autonomous Multi-Agent Software Development         │
└──────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
  ┌─────▼─────┐         ┌─────▼─────┐         ┌─────▼──────┐
  │  DOMAIN   │         │APPLICATION│         │INFRASTRUCTURE│
  │  Layer    │ ◄────── │  Layer    │ ◄────── │   Layer    │
  │(pure data)│         │(use cases)│         │ (adapters) │
  └───────────┘         └───────────┘         └────────────┘
        │                     │                     │
   models.py             engine.py             infrastructure/
   models_skill.py       application/          api_clients.py ¹
   domain/ports.py       engine_core/          state.py ¹
                         agents/               cache.py ¹
```

¹ Root-level `orchestrator/*.py` shims for backward compat; canonical code lives in `infrastructure/`.

**Dependency rule:** each layer imports only from the layer to its left (inward).  
`import-linter` enforces this with **4 contracts**; CI will fail if violated.

---

## Execution Flow

```
User: "Build a todo app"
        │
        ▼
   engine.py (Orchestrator)
        │
        ├─ ProjectRunner.run_project()          ← M3: no longer holds host ref
        │     │
        │     ├─ decompose_project() → dict[Task]
        │     │
        │     ├─ for each task:
        │     │     │
        │     │     ├─ best_skill(task.type) → skill_prefix   [SkillOpt]
        │     │     │
        │     │     ├─ PipelineContext(task, model, skill_prefix)
        │     │     │
        │     │     └─ TaskPipeline.run(ctx)
        │     │           │
        │     │           ▼ 7 stages (see below)
        │     │
        │     └─ record_trajectory(score, critique) → epoch may fire  [SkillOpt]
        │
        └─ ProjectRunState.project_id / .architecture_rules / .entered
```

---

## Pipeline Stages (7)

```
GenerateStage        ← M4: depends on LLMClient Protocol, not UnifiedClient
    │  injects skill_prefix as <skill>...</skill> XML tag in system prompt
    ▼
PersuasionDefenseStage   (claim extraction → NLI verification)
    ▼
CritiqueStage            ← M4: depends on LLMClient Protocol, not UnifiedClient
    ▼
EvaluateStage            (2-pass self-consistency scoring)
    ▼
SelfConsistencyStage     (score < 0.7 → retry with fallback model)
    ▼
PreflightStage           (PASS / WARN / ENRICH / BLOCK gate)
    ▼
ValidateStage            (syntax, bracket balance, ruff lint)
```

`engine.py._execute_task()` builds `PipelineContext` and calls `TaskPipeline.run(ctx)` — it is ~30 lines.

---

## Application Layer — Service Classes

| Service | File | Responsibility |
|---------|------|----------------|
| `SkillOptimizer` | `skill_optimizer.py` | Per-TaskType epoch loop |
| `SkillManager` | `skill_manager.py` | Facade + fire-and-forget epoch scheduling |
| `SkillStore` | `skill_store.py` | aiosqlite persistence (trajectories + skills) |
| `ModelHealthTracker` | `model_health_tracker.py` | Circuit breaker — owns its state dicts (M6) |
| `ResumptionService` | `resumption_service.py` | resume_project — returns new ProjectState (M7) |
| `DashboardBridge` | `dashboard_bridge.py` | Null-safe dashboard notifications |
| `GitBridge` | `git_bridge.py` | Git commit dispatch |
| `ProjectRunner` | `project_runner.py` | run_project / dry_run — no host back-ref (M3) |
| `ProjectRunnerCallables` | `project_runner_deps.py` | Injected execution callbacks (M3) |
| `ProjectRunState` | `project_runner_deps.py` | Shared mutable run metadata (M3) |
| `TaskExecutor` | `task_executor.py` | Single-task execution coordinator |
| `Evaluator` | `evaluator.py` | Score computation, CritiqueReport |
| `CritiqueCycle` | `critique_cycle.py` | Multi-round critique orchestration |
| `Decomposer` | `decomposer.py` | Project → Task decomposition |
| `ContextCompressor` | `context_compressor.py` | Context window management |
| `BudgetEnforcer` | `budget_enforcer.py` | Per-run budget tracking |

---

## M3 — ProjectRunner Decoupling

`ProjectRunner` used to hold `self._host = Orchestrator` and call 13+ private methods on it.  
After M3 the host reference is gone entirely.

```
BEFORE (cosmetic extraction):
  ProjectRunner._host._execute_all(...)
  ProjectRunner._host._topological_sort(tasks)
  ProjectRunner._host._project_id = project_id
  … (13 more calls)

AFTER (true decoupling):
  ProjectRunner._callables.execute_all(...)       ← ProjectRunnerCallables
  ProjectRunner._callables.topological_sort(tasks)
  ProjectRunner._run_state.project_id = project_id ← ProjectRunState
```

**ProjectRunnerCallables** — dataclass holding 8 async/sync callables injected by Orchestrator.  
**ProjectRunState** — dataclass holding `project_id`, `architecture_rules`, `entered`, `results`.  
Both objects are wired in `engine.py.__init__`; `_run_state.entered` is kept in sync by `__aenter__`/`__aexit__`.

---

## SkillOpt — Self-Improving Skill System

A text-space optimization loop where per-TaskType markdown skill documents evolve based on task trajectories. The worker model is frozen; only the skill document changes.

```
                    ┌─────────────────────────────────┐
                    │         SkillManager             │
                    │   (one SkillOptimizer / TaskType) │
                    └─────────────┬───────────────────┘
                                  │
          record_trajectory()     │     best_skill(task_type)
          ─────────────────►      │     ◄─────────────────────
                                  │
                    ┌─────────────▼───────────────────┐
                    │         SkillStore               │
                    │  trajectories.db  │  skills.db   │
                    └─────────────────────────────────┘
                                  │
               every epoch_size trajectories
                                  │
                    ┌─────────────▼───────────────────┐
                    │       SkillOptimizer.run_epoch() │
                    │                                  │
                    │  train/val split                 │
                    │  → optimizer LLM proposes patches│
                    │  → edit budget enforced (≤150t)  │
                    │  → val gate (must improve score) │
                    │  → accepted: save best_skill     │
                    │  → rejected: save negative feed  │
                    │  → every 5 epochs: slow update   │
                    └──────────────────────────────────┘
```

**Key design decisions:**
- `## Guidance` block is regex-protected — only slow-update rewrites it
- Epoch failures are logged at DEBUG; never surface to the main execution path
- Orphan task retained in `_background_tasks` (M1) — GC cannot cancel mid-flight
- Feature off by default (`ORCH_SKILL_OPTIMIZATION_ENABLED=false`)
- No fine-tuning; no re-evaluation LLM calls in validation (cheap proxy scoring)

**Starter skills** at `orchestrator/application/skills/` — 7 TaskTypes:
`code_generation`, `code_review`, `complex_reasoning`, `creative_writing`, `data_extraction`, `summarization`, `evaluation`

---

## Domain Ports (`domain/ports.py`)

| Protocol | Satisfied By | Null Adapter |
|----------|-------------|--------------|
| `CachePort` | `infrastructure/cache.py::DiskCache` | `NullCache` |
| `StatePort` | `infrastructure/state.py::StateManager` | `NullState` |
| `EventPort` | `unified_events/core.py::UnifiedEventBus` | `NullEventBus` |
| `ConfigPort` | `crosscutting/config.py::ConfigAdapter` | — |
| `LLMClient` | `infrastructure/llm_client.py::UnifiedClient` | — |
| `PlannerPort` | `engine_core/model_selector.py::ModelSelector` | — |
| `TelemetryPort` | `infrastructure/telemetry.py::TelemetryCollector` | — |
| `PolicyEnginePort` | `engine_core/policy.py::PolicyEngine` | — |
| `HookRegistryPort` | `unified_events/core.py::SyncHookRegistry` | `NullHookRegistry` ← M2 |
| `ValidatorPort` | `engine_core/validator.py::TaskValidator` | — |
| `SkillStorePort` | `application/skill_store.py::SkillStore` | `NullSkillStore` |

All null adapters live in `domain/ports.py` for zero-dependency testing.  
`NullHookRegistry` (M2) replaced the `type("HookRegistry", (), {...})()` dummy that was previously in `container.build()`.

---

## ServiceContainer (`engine_core/container.py`)

Wires all collaborators at startup via constructor injection.  
After M2: no structural dummy objects (`type("X", (), {})()`) remain — all optional services are either a real adapter or `None` with guarded callers.

Key wiring sequence:
1. Build infrastructure adapters (cache, state, event bus / hook registry)
2. Build domain services (model selector, policy engine, telemetry)
3. Build application services (decomposer, evaluator, pipeline)
4. Wire SkillManager if `skill_optimization_enabled`
5. Call `assert_healthy()` — raises `RuntimeError` on missing required services

**Typed fields (post-M2):**

| Field | Type |
|-------|------|
| `pipeline` | `Optional[TaskPipeline]` |
| `pipeline_runner` | `Optional[PipelineRunner]` |
| `project_planner` | `Optional[ProjectPlanner]` |
| `state_coordinator` | `Optional[StateCoordinator]` |
| `context_service` | `Optional[ContextService]` |
| `hook_registry` | `Optional[HookRegistryPort]` |
| `event_bus` | `Optional[EventPort]` |
| `semantic_cache` | `None` (guarded callers) |
| `adaptive_router` | `None` (guarded callers) |

---

## Module Layout — Canonical vs Shim

Ten root-level `orchestrator/*.py` files were duplicates of their `infrastructure/` counterparts.  
After M5, all ten are backward-compat shims; canonical code lives exclusively in `infrastructure/`.

| Root shim | Canonical location |
|-----------|--------------------|
| `orchestrator/state.py` | `infrastructure/state.py` |
| `orchestrator/cache.py` | `infrastructure/cache.py` |
| `orchestrator/telemetry.py` | `infrastructure/telemetry.py` |
| `orchestrator/tracing.py` | `infrastructure/tracing.py` |
| `orchestrator/audit.py` | `infrastructure/audit.py` |
| `orchestrator/bm25_search.py` | `infrastructure/bm25_search.py` |
| `orchestrator/caching.py` | `infrastructure/caching.py` |
| `orchestrator/cache_optimizer.py` | `infrastructure/cache_optimizer.py` |
| `orchestrator/semantic_cache.py` | `infrastructure/semantic_cache.py` |
| `orchestrator/token_optimizer.py` | `infrastructure/token_optimizer.py` |

> New code must import from `infrastructure/` directly. Shim deletion is deferred to a follow-up PR.

---

## 52 Models / 15 Providers

```
FREE:       owl-alpha, deepseek-v4-flash:free, nemotron-3-nano-omni:free,
            poolside/laguna-m.1:free, poolside/laguna-xs.2:free

ULTRA-LOW ($0.01–0.09):  ling-2.6-flash, granite-4.1-8b, deepseek-v4-flash

BUDGET ($0.10–0.50):     qwen-3.5-flash, qwen-3-coder-next, codestral-2508,
                         qwen-3.6-flash, gemini-2.5-flash

STANDARD ($0.50–2.00):   deepseek-reasoner, qwen-3.6-plus, kimi-k2.6,
                         gpt-5.4-nano, gemini-2.5-pro, grok-4.20

PREMIUM ($2.00+):        gpt-5, claude-sonnet-4.6, qwen-3.7-max, o3, sonar-pro
```

**Routing components:**
- `adaptive_router.py` — circuit breaker v2 (HEALTHY / DEGRADED / DISABLED)
- `outcome_router.py` — outcome-weighted router using production feedback
- `escalation.py` — auto-escalation to higher-capability models on quality failure

---

## 5-Layer Memory Architecture

| Layer | Scope | Storage |
|-------|-------|---------|
| ProjectWorkspace | Per-project, in-memory | dict |
| PersistentWorkspace | Cross-run crash recovery | SQLite (`state.db`) |
| SkillStore | Cross-run skill evolution | SQLite (`trajectories.db`, `skills.db`) |
| KnowledgeGraph | Relational task context | networkx (in-memory) |
| AgentCache | Response deduplication | Disk (SHA-256, TTL 1h) |

---

## Import Boundary Contracts

```
[importlinter:contract:domain-purity]
  orchestrator.domain  must NOT import  orchestrator.infrastructure
                                        orchestrator.application
                                        orchestrator.engine_core

[importlinter:contract:application-no-concrete-infra]
  orchestrator.application  must NOT import  orchestrator.infrastructure

[importlinter:contract:application-services-no-engine]
  orchestrator.application.{dashboard_bridge, git_bridge,
                             model_health_tracker, resumption_service,
                             project_runner, project_runner_deps}
                           must NOT import  orchestrator.engine

[importlinter:contract:engine-core-no-loose-infra]          ← NEW (M1+M4)
  orchestrator.engine_core.{pipeline, pipeline_runner,
                             project_planner, state_coordinator,
                             stages}
                           must NOT import  orchestrator.infrastructure
```

Run `lint-imports` to verify. Checked in CI on every push (763 files, 4 contracts).

---

## Feature Flags (`crosscutting/config.py::FeatureFlags`)

| Flag | Env Var | Default | Effect |
|------|---------|---------|--------|
| `skill_optimization_enabled` | `ORCH_SKILL_OPTIMIZATION_ENABLED` | `false` | SkillOpt trajectory collection + epoch optimization |
| `a2a_enabled` | `ORCH_A2A_ENABLED` | `false` | Agent-to-Agent protocol |
| `red_team_enabled` | `ORCH_RED_TEAM_ENABLED` | `false` | Adversarial quality checks |
| `tdd_enabled` | `ORCH_TDD_ENABLED` | `false` | Test-first task execution |
| `context_compression` | `ORCH_CONTEXT_COMPRESSION` | `true` | Token compression for long contexts |

---

## Architecture Compliance History

| Version | Score | Key gap closed |
|---------|-------|----------------|
| v6.0 (baseline) | 5.5 | — |
| M1 — Hygiene | 5.8 | Orphan asyncio task; swallowed exceptions; wildcard re-export |
| M2 — Container | 6.5 | Dummy fallbacks → null adapters; 7 core fields typed |
| M3 — ProjectRunner | 7.5 | `_host` back-reference eliminated; 13 private method calls removed |
| M4 — Stage Protocols | 7.8 | GenerateStage + CritiqueStage depend on `LLMClient` Protocol |
| M5 — Deduplication | 8.6 | 10 duplicate root modules → backward-compat shims |
| M6 — Tracker ownership | 8.9 | `ModelHealthTracker` owns its dicts; no shared mutable state |
| M7 — Immutable state | **9.2** | `ResumptionService` returns new `ProjectState` via `dataclasses.replace` |

---

## Key Metrics

| Metric | Value |
|--------|-------|
| **engine.py** | 2,672 lines (−49% from original) |
| **Application services** | 16 extracted classes (+ ProjectRunnerCallables/ProjectRunState) |
| **Domain protocols** | 11 typed Protocols |
| **Import contracts** | **4** (enforced in CI) |
| **Null adapters** | **12** (NullHookRegistry added in M2) |
| **Structural dummies** | **0** (all replaced by M2) |
| **`_host` back-references** | **0** (eliminated by M3) |
| **Root duplicate modules** | **0** live duplicates (10 shims pending deletion) |
| **Parallel tasks** | 3 (SQLite WAL + connection pool) |
| **Circuit breaker** | Trips after 3 consecutive failures |

---

*Last updated: 2026-05-29*  
*Maintainer: Georgios-Chrysovalantis Chatzivantsidis*  
*License: MIT*
