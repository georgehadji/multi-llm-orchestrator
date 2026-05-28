# Multi-LLM Orchestrator — Architecture Mindmap

> **Version:** v6.2.0 (2026-05-28)  
> **engine.py:** 2,643 lines (−50% from v5.0)  
> **Application services:** 15 extracted classes  
> **Import contracts:** 3 (domain purity, app-no-infra, app-services-no-engine)

---

## Hexagonal Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Multi-LLM Orchestrator v6.2               │
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
   models_skill.py       application/          api_clients.py
   domain/ports.py       engine_core/          state.py
                         agents/               cache.py
```

**Dependency rule:** each layer imports only from the layer to its left (inward).  
`import-linter` enforces this with 3 contracts; CI will fail if violated.

---

## Execution Flow

```
User: "Build a todo app"
        │
        ▼
   engine.py (Orchestrator)
        │
        ├─ decompose_project() → list[Task]
        │
        ├─ for each task:
        │     │
        │     ├─ best_skill(task.type) → skill_prefix   [SkillOpt]
        │     │
        │     ├─ PipelineContext(task, model, skill_prefix)
        │     │
        │     └─ TaskPipeline.run(ctx)
        │           │
        │           ▼ 7 stages (see below)
        │
        └─ record_trajectory(score, critique) → epoch may fire  [SkillOpt]
```

---

## Pipeline Stages (7)

```
GenerateStage
    │  injects skill_prefix as <skill>...</skill> XML tag in system prompt
    ▼
PersuasionDefenseStage   (claim extraction → NLI verification)
    ▼
CritiqueStage            (cross-model review, different provider)
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
| `ModelHealthTracker` | `model_health_tracker.py` | Circuit breaker record_success/record_failure |
| `ResumptionService` | `resumption_service.py` | resume_project logic |
| `DashboardBridge` | `dashboard_bridge.py` | Null-safe dashboard notifications |
| `GitBridge` | `git_bridge.py` | Git commit dispatch |
| `ProjectRunner` | `project_runner.py` | run_project / run_job / dry_run |
| `TaskExecutor` | `task_executor.py` | Single-task execution coordinator |
| `Evaluator` | `evaluator.py` | Score computation, CritiqueReport |
| `CritiqueCycle` | `critique_cycle.py` | Multi-round critique orchestration |
| `Decomposer` | `decomposer.py` | Project → Task decomposition |
| `ContextCompressor` | `context_compressor.py` | Context window management |
| `BudgetEnforcer` | `budget_enforcer.py` | Per-run budget tracking |
| `FallbackHandler` | `fallback_handler.py` | Model fallback chain management |

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
- Epoch failures are swallowed; never surface to the main execution path
- Feature off by default (`ORCH_SKILL_OPTIMIZATION_ENABLED=false`)
- No fine-tuning; no re-evaluation LLM calls in validation (cheap proxy scoring)

**Starter skills** at `orchestrator/application/skills/` — 7 TaskTypes:
`code_generation`, `code_review`, `complex_reasoning`, `creative_writing`, `data_extraction`, `summarization`, `evaluation`

---

## Domain Ports (`domain/ports.py`)

| Protocol | Satisfied By |
|----------|-------------|
| `CachePort` | `infrastructure/cache.py::DiskCache` |
| `StatePort` | `infrastructure/state.py::StateManager` |
| `EventPort` | `events/core.py::UnifiedEventBus` |
| `ConfigPort` | `crosscutting/config.py::ConfigAdapter` |
| `LLMClient` | `api_clients.py::UnifiedClient` |
| `PlannerPort` | `engine_core/model_selector.py::ModelSelector` |
| `TelemetryPort` | `engine_core/telemetry.py::TelemetryCollector` |
| `PolicyEnginePort` | `engine_core/policy.py::PolicyEngine` |
| `HookRegistryPort` | `events/core.py::SyncHookRegistry` |
| `ValidatorPort` | `engine_core/validator.py::TaskValidator` |
| `SkillStorePort` | `application/skill_store.py::SkillStore` |

Null adapters for all ports live in `domain/ports.py` for zero-dependency testing.

---

## ServiceContainer (`engine_core/container.py`)

Wires all collaborators at startup via constructor injection. Fields typed to their Protocol interfaces (not `Any`) so mypy catches wiring errors.

Key wiring sequence:
1. Build infrastructure adapters (cache, state, event bus)
2. Build domain services (model selector, policy engine, telemetry)
3. Build application services (decomposer, evaluator, pipeline)
4. Wire SkillManager if `skill_optimization_enabled`
5. Call `assert_healthy()` — raises `RuntimeError` on missing required services

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

[importlinter:contract:application-no-infra]
  orchestrator.application  must NOT import  orchestrator.infrastructure
                                              orchestrator.state
                                              orchestrator.cache

[importlinter:contract:application-services-no-engine]
  orchestrator.application  must NOT import  orchestrator.engine
```

Run `lint-imports` to verify. Checked in CI on every push.

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

## Key Metrics

| Metric | Value |
|--------|-------|
| **engine.py** | 2,643 lines (−50% from original) |
| **Application services** | 15 extracted classes |
| **Domain protocols** | 11 typed Protocols |
| **Import contracts** | 3 (enforced in CI) |
| **Null adapters** | 11 (one per port, in `domain/ports.py`) |
| **Parallel tasks** | 3 (SQLite WAL + connection pool) |
| **Circuit breaker** | Trips after 3 consecutive failures |

---

*Last updated: 2026-05-28*  
*Maintainer: Georgios-Chrysovalantis Chatzivantsidis*  
*License: MIT*
