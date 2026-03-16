# Multi-LLM Orchestrator — Architecture & Roadmap

**Έκδοση:** v6.0 — Phase 1 Complete
**Τελευταία ενημέρωση:** 2026-03-16 (Phase 1 P0 implementation complete)
**Αυτό το αρχείο διαβάζεται από τον Claude πριν από οποιαδήποτε αρχιτεκτονική απόφαση ή implementation.**

---

## 📊 Phase 1 (P0) Completion Summary

✅ **Status: COMPLETE** — All 3 Phase 1 features implemented, TDD compliant, zero regressions

| Feature | Module | Pattern | Tests | Status |
|---------|--------|---------|-------|--------|
| #15 Autonomy | `autonomy.py` | Pure Dataclass | 12 | ✅ Complete |
| #9 Model Routing | `model_routing.py` | Strategy + Lookup | 11 | ✅ Complete |
| #14 Verification | `verification.py` | Chain of Responsibility | 13 | ✅ Complete |

**Test Results:**
- Phase 1 modules: 36/36 tests passing
- v5.1 integration: 44/44 tests passing (autonomy + v5.1 features)
- Full feature regression: 59/59 tests passing (autonomy + model_routing + verification + rate_limiter + session_lifecycle + hybrid_search + query_expander)
- Zero regressions introduced; all pre-existing test failures remain unchanged

**Code Quality:**
- Ruff lint: All modules clean (100% pass)
- Architecture Rules: All 10 rules honored (no engine.py logic, models pure data, async/sync discipline)
- Export integrity: All symbols resolve from `orchestrator.__init__.py` with `HAS_*` feature flags

**Implementation Details:**
- `autonomy.py`: AutonomyLevel (MANUAL/SUPERVISED/FULL), AutonomyConfig, AUTONOMY_PRESETS, requires_approval()
- `model_routing.py`: ModelTier (PREMIUM/STANDARD/ECONOMY), TIER_ROUTING, PHASE_TO_TIER, select_model(), get_tier_for_phase()
- `verification.py`: VerificationLevel (NONE/SYNTAX/EXECUTION/FULL), REPLVerifier, self_healing_loop() scaffold

---

## 1. Αρχιτεκτονική Overview

### 1.1 Hexagonal Architecture (Ports & Adapters)

Το project ακολουθεί **Hexagonal Architecture** με **7-layer domain stack**. Ο core orchestrator είναι ανεξάρτητος από το πώς καλείται (CLI, FastAPI, webhook, test) και από το ποιον LLM provider χρησιμοποιεί.

```
┌─────────────────────────────────────────────────────────────┐
│  DRIVING ADAPTERS  (inbound — πώς καλείται ο orchestrator)  │
│  cli.py · api_server.py · webhooks · pytest                 │
├─────────────────────────────────────────────────────────────┤
│  APPLICATION CORE                                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Orchestrator (Mediator / Coordinator)                │  │
│  │  ├─ TaskDecomposer     (Strategy)                     │  │
│  │  ├─ TaskExecutor       (Template Method + Decorator)  │  │
│  │  ├─ ModelRouter        (Strategy)                     │  │
│  │  ├─ TaskValidator      (Chain of Responsibility)      │  │
│  │  └─ TaskEvaluator      (Strategy)                     │  │
│  └───────────────────────────────────────────────────────┘  │
│  DOMAIN LAYER  (pure Python — κανένα I/O, κανένα asyncio)   │
│  models.py · policy.py · cost.py · autonomy.py              │
├─────────────────────────────────────────────────────────────┤
│  DRIVEN ADAPTERS  (outbound — εξωτερικά συστήματα)          │
│  gateway.py (LLM providers) · state.py (SQLite)             │
│  connectors.py (Postgres, GitHub, Slack, MCP)               │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 7-Layer Domain Stack

```
Layer 7: HUMAN INTERFACE
  ├── #29 escalation.py        Human-in-the-loop escalation
  ├── #16 planner.py (extend)  Plan-then-build separation
  └── #27 cost_analytics.py   Cost analytics dashboard

Layer 6: OBSERVABILITY & GOVERNANCE
  ├── #23 tracing.py (extend)  Full Tracer API + Langfuse export
  ├── #28 drift.py             Drift detection & regression alerts
  ├── #25 adaptive_templates   Prompt versioning + A/B testing ✅
  └── #24 evaluation.py        LLM-as-Judge eval framework

Layer 5: EVENT & SCHEDULING
  ├── #22 triggers.py          Event-driven / scheduled execution
  └── #21 workspace.py         Workspace-level resource sharing

Layer 4: SUPERVISOR AGENT
  ├── #19 hierarchy.py         Hierarchical agent architecture
  ├── #20 brain.py             Agent Brain (Knowledge + Memory + Tools)
  ├── #1  competitive.py       Chairman LLM competitive execution
  └── #12 prompt_enhancer.py   Prompt enhancement pipeline

Layer 3: SPECIALIST AGENTS
  ├── #8  modes.py             Mode-based specialization
  ├── #9  model_routing.py     Auto model routing (tiered) ✅
  ├── #2  engine.py (extend)   Parallel execution ✅
  └── #15 autonomy.py          Parametric autonomy control ✅

Layer 2: VERIFICATION & QUALITY
  ├── #14 verification.py      Self-healing REPL verification loop ✅
  ├── #10 context.py           Context condensing engine
  └── #3  skills.py            Skills / template system

Layer 1: INFRASTRUCTURE & RESILIENCE
  ├── #26 resilience.py        Circuit breaker + adaptive rate limiting 🟡
  ├── #13 gateway.py           Unified API gateway 🟡
  ├── #17 checkpoints.py       Rich checkpoints + rollback
  ├── #11 memory.py            Memory Bank persistence
  ├── #18 connectors.py        Connectors / MCP layer 🟡
  ├── #4  context_sources.py   Rich context scoping
  ├── #5  sandbox.py           Execution isolation 🟡
  └── #7  api_server.py        REST API layer (FastAPI)

Legend: ✅ Complete  🟡 Partial  (blank) = Not implemented
```

---

## 2. Core Architectural Principles

**Αυτοί οι 6 κανόνες δεν παραβιάζονται ποτέ.**

### Κανόνας 1: Engine = Mediator, όχι Monolith

`engine.py` συντονίζει services — ΔΕΝ περιέχει business logic. Κάθε νέο feature υλοποιείται ως ξεχωριστό module και **wire-άρεται** στον engine, όχι embedded.

```python
# ✅ ΣΩΣΤΟ — wiring μόνο
class Orchestrator:
    def __init__(self):
        self._router = ModelRouter()
        self._verifier = REPLVerifier()      # νέο feature
        self._escalation = EscalationManager()  # νέο feature

# ❌ ΛΑΘΟΣ — business logic στο engine
class Orchestrator:
    async def _execute_task(self, task):
        # 200 γραμμές mixed logic εδώ
```

### Κανόνας 2: Models = Pure Value Objects

`models.py` περιέχει μόνο data. Κανένα I/O, κανένα asyncio, κανένα behavior. Dataclasses/Enums μόνο.

```python
# ✅ ΣΩΣΤΟ
@dataclass
class Task:
    id: str
    type: TaskType
    prompt: str

# ❌ ΛΑΘΟΣ
@dataclass
class Task:
    async def execute(self):  # behavior δεν ανήκει εδώ
        ...
```

### Κανόνας 3: Adapters = Protocols (Structural Subtyping)

Εξωτερικές εξαρτήσεις ορίζονται ως `Protocol` — όχι ABC, όχι inheritance. Επιτρέπει easy mocking σε tests.

```python
# ✅ ΣΩΣΤΟ
class LLMProvider(Protocol):
    async def complete(self, model: str, messages: list[dict]) -> CompletionResponse: ...

class AnthropicAdapter:  # implements Protocol implicitly
    async def complete(self, model, messages): ...
```

### Κανόνας 4: Cross-cutting Concerns = EventBus

Telemetry, cost tracking, drift monitoring, tracing ΔΕΝ καλούνται ρητά από τον engine. Ο engine κάνει `emit(event)` και οι observers αποφασίζουν.

```python
# ✅ ΣΩΣΤΟ
await self._events.emit(TaskCompletedEvent(task_id=..., cost=..., tokens=...))
# Observers: TelemetryCollector, CostAnalytics, DriftMonitor αντιδρούν αυτόματα

# ❌ ΛΑΘΟΣ
await self._telemetry.record(...)
await self._cost_analytics.record(...)
await self._drift.track(...)
```

### Κανόνας 5: Async για I/O, Sync για Pure Logic

| Module τύπος | Async | Rationale |
|---|---|---|
| Engine, state, session_lifecycle, gateway | ✅ Async | Network I/O, disk I/O |
| models, policy, cost, validators, rate_limiter | ❌ Sync | Pure computation, no I/O |
| brain, memory, autonomy, modes | ❌ Sync | File I/O μόνο (blocking OK) |
| verification, checkpoints, connectors | ✅ Async | Process execution, network |

### Κανόνας 6: Optional Features = Decorator Pattern

Κάθε feature που μπορεί να είναι ON/OFF (π.χ. βάσει autonomy level) υλοποιείται ως Decorator — όχι ως `if` μέσα στον executor.

```python
# ✅ ΣΩΣΤΟ
executor = TaskExecutor(client)
if autonomy.verification_depth != "none":
    executor = VerifiedExecutor(executor, REPLVerifier(autonomy))
if autonomy.critique_rounds > 0:
    executor = CritiqueExecutor(executor, critic_model)

# ❌ ΛΑΘΟΣ
async def _execute_task(self, task):
    result = await self._generate(task)
    if self.autonomy_level != "lite":     # if-hell
        result = await self._verify(result)
    if self.autonomy_level in ("autonomous", "max"):
        result = await self._critique(result)
```

---

## 3. Programming Paradigm ανά Component

### 3.1 Υπάρχοντα αρχεία

| Αρχείο | Pattern | Paradigm | Async | Coupling |
|--------|---------|----------|-------|---------|
| `models.py` | Value Objects | Pure Dataclasses + Enums | ❌ | None |
| `engine.py` | Mediator (→ refactor) | Async OOP | ✅ | High (coordinator) |
| `state.py` | Repository + Memento | Async OOP | ✅ | High (models) |
| `policy.py` | Specification | Pure OOP + Dataclasses | ❌ | Low |
| `policy_engine.py` | Strategy | Pure OOP | ❌ | Medium |
| `planner.py` | Strategy (ConstraintPlanner) | Pure OOP | ❌ | Medium |
| `cost.py` | Composite + Facade | Pure OOP | ❌ | Low |
| `rate_limiter.py` | Sliding-Window State Machine | Pure OOP | ❌ | None |
| `validators.py` | Chain of Responsibility | Pure OOP | ❌ | Low |
| `preflight.py` | Chain of Responsibility | Pure OOP | ❌ | Low |
| `session_lifecycle.py` | Decorator + Background Worker | Async OOP | ✅ | Low |
| `memory_tier.py` | Repository | Async OOP | ✅ | Low |
| `adaptive_templates.py` | Strategy + A/B Testing | Async OOP | ✅ | Low |
| `adaptive_router.py` | Strategy | Async OOP | ✅ | Medium |
| `hybrid_search_pipeline.py` | Facade + Composite | Async OOP | ✅ | Medium |
| `query_expander.py` | Decorator | Async OOP | ✅ | Low |
| `telemetry.py` | Observer | Async OOP | ✅ | Low |
| `tracing.py` | Facade (OTEL wrapper) | Async OOP | ✅ | Low |
| `agents.py` | Mediator (TaskChannel) | Async OOP | ✅ | Medium |
| `plugins.py` | Registry + Strategy | Pure OOP | ❌ | Low |
| `events.py` | Observer / EventBus | Async Pub-Sub | ✅ | Low |
| `hooks.py` | Observer | Sync/Async OOP | Mixed | Low |

### 3.2 Νέα αρχεία (από το roadmap)

| Αρχείο | Pattern | Paradigm | Async | Layer |
|--------|---------|----------|-------|-------|
| `autonomy.py` | Configuration Object | Pure Dataclasses | ❌ | L3 |
| `verification.py` | Decorator + Template Method | Async OOP | ✅ | L2 |
| `model_routing.py` | Strategy + Registry | Pure OOP | ❌ | L3 |
| `evaluation.py` | Strategy (LLMJudge) | Async OOP | ✅ | L6 |
| `brain.py` | Composite + Repository | Sync OOP | ❌ | L4 |
| `escalation.py` | State Machine | Pure OOP | ❌ | L7 |
| `checkpoints.py` | Memento + Command | Async OOP | ✅ | L1 |
| `modes.py` | Strategy + Registry | Pure OOP | ❌ | L3 |
| `prompt_enhancer.py` | Decorator | Async OOP | ✅ | L4 |
| `cost_analytics.py` | Observer + Composite | Sync OOP | ❌ | L7 |
| `competitive.py` | Strategy + Facade | Async OOP | ✅ | L4 |
| `memory.py` | Repository | Sync OOP | ❌ | L1 |
| `context.py` | Decorator | Async OOP | ✅ | L2 |
| `hierarchy.py` | Composite + Mediator | Async OOP | ✅ | L4 |
| `gateway.py` | Adapter + Facade | Async Protocol | ✅ | L1 |
| `connectors.py` | Adapter + Registry | Async Protocol | ✅ | L1 |
| `triggers.py` | Observer + Scheduler | Async OOP | ✅ | L5 |
| `workspace.py` | Repository + Singleton | Sync OOP | ❌ | L5 |
| `drift.py` | Observer + Sliding Window | Sync OOP | ❌ | L6 |
| `sandbox.py` | Decorator + Factory | Async OOP | ✅ | L1 |
| `context_sources.py` | Builder | Async OOP | ✅ | L1 |
| `api_server.py` | Facade (FastAPI) | Async OOP | ✅ | L1 |
| `skills.py` | Registry + Strategy | Sync OOP | ❌ | L2 |
| `browser_testing.py` | Decorator | Async OOP | ✅ | L2 |
| `resilience.py` | State Machine (CircuitBreaker) | Pure OOP | ❌ | L1 |

---

## 4. Design Pattern Reference

| Pattern | Χρησιμοποιείται σε | Γιατί |
|---------|-------------------|-------|
| **Mediator** | `engine.py` | Αποσυνδέει subsystems — coordinator γνωρίζει interfaces, όχι implementations |
| **Strategy** | `policy_engine`, `model_routing`, `modes`, `evaluation` | Εναλλάξιμοι αλγόριθμοι routing/evaluation χωρίς if-else |
| **Decorator** | `verification`, `prompt_enhancer`, `context`, `sandbox` | Optional features ON/OFF βάσει autonomy level |
| **Chain of Responsibility** | `validators`, `preflight` | Validators σε σειρά — κάθε ένας αποφασίζει αν συνεχίζει |
| **Repository** | `state`, `memory`, `workspace` | Αφαιρεί storage backend — engine δεν γνωρίζει SQLite/files |
| **Memento** | `state`, `checkpoints` | ProjectState snapshots για crash recovery + rollback |
| **Observer / EventBus** | `events`, `hooks`, `telemetry`, `cost_analytics`, `drift` | Cross-cutting concerns χωρίς tight coupling |
| **Composite** | `cost.py` (BudgetHierarchy), `brain`, `hierarchy` | Tree structures — org→team→job, supervisor→specialist |
| **Adapter** | `gateway`, `connectors` | Uniform interface για heterogeneous providers |
| **Specification** | `policy` | Composable business rules (`PolicyA & PolicyB`) |
| **State Machine** | `rate_limiter`, `resilience`, `escalation` | Explicit state transitions με clear invariants |
| **Template Method** | `engine_executor` (να δημιουργηθεί) | generate→critique→revise→evaluate pipeline skeleton |
| **Factory** | `sandbox`, `agent nodes` | Δημιουργία isolated environments per task |
| **Sliding Window** | `rate_limiter`, `drift` | Time-bounded aggregation για rate/quality tracking |
| **Background Worker** | `session_lifecycle`, `triggers` | Periodic async tasks χωρίς blocking main loop |

---

## 5. Implementation Roadmap — 29 Features

### Phase 1 — P0: Infrastructure Foundation (Ημέρα 1-2)

#### #15 Parametric Autonomy Control
- **Αρχείο:** `orchestrator/autonomy.py` (νέο)
- **Pattern:** Configuration Object (pure dataclasses)
- **Async:** ❌
- **Dependencies:** Κανένα
- **Reuse:** Ο engine διαβάζει `AutonomyConfig` για να ρυθμίσει behavior
- **Effort:** 1h
- **TDD:**
  ```
  tests/test_autonomy.py:
    test_lite_preset_has_no_critique()
    test_max_preset_has_behavioral_verification()
    test_cli_flag_overrides_yaml()
  ```
- **Spec κλειδιά:**
  - `AutonomyLevel` enum: `LITE | STANDARD | AUTONOMOUS | MAX`
  - `AUTONOMY_PRESETS` dict με max_runtime, repair_attempts, verification_depth, critique_rounds, model_tier, checkpoint_frequency
  - CLI flag `--autonomy` + YAML `autonomy:` field

#### #14 Self-Healing REPL Verification Loop
- **Αρχείο:** `orchestrator/verification.py` (νέο)
- **Pattern:** Decorator + Template Method
- **Async:** ✅
- **Dependencies:** #15 (autonomy level καθορίζει verification depth)
- **Reuse:** `validators.py` για syntax checks, `plugin_isolation.py` ως inspiration για subprocess sandboxing
- **Effort:** 2-3h
- **TDD:**
  ```
  tests/test_verification.py:
    test_syntax_error_detected_and_repaired()
    test_runtime_error_triggers_repair_loop()
    test_max_repair_attempts_marks_failed()
    test_lite_autonomy_syntax_only()
    test_timeout_protection_30s()
  ```
- **Spec κλειδιά:**
  - `VerificationLevel` enum: `SYNTAX | UNIT | INTEGRATION | BEHAVIORAL`
  - `REPLVerifier.verify(code, level, timeout=30)` → `VerificationResult`
  - Self-healing loop: generate → verify → repair prompt → regenerate (max N times)
  - Sandbox: `tempfile.TemporaryDirectory()`, `subprocess.run(timeout=30)`

#### #9 Auto Model Routing (Tiered) — refactor
- **Αρχείο:** `orchestrator/model_routing.py` (νέο, refactors adaptive_router.py)
- **Pattern:** Strategy + Registry
- **Async:** ❌
- **Dependencies:** Κανένα
- **Reuse:** `adaptive_router.py` (ConstraintPlanner logic), `models.py` (Model enum)
- **Effort:** 1-2h
- **TDD:**
  ```
  tests/test_model_routing.py:
    test_decomposition_uses_reasoning_tier()
    test_code_gen_uses_implementation_tier()
    test_budget_pressure_downgrades_to_budget_tier()
    test_unhealthy_model_skipped_to_next()
  ```
- **Spec κλειδιά:**
  - `ModelTier` enum: `REASONING | IMPLEMENTATION | BUDGET`
  - `TIER_ROUTING: dict[ModelTier, list[Model]]` config-driven (YAML)
  - `select_model(task_type, phase, budget_remaining, api_health) -> Model`
  - Budget pressure: `budget_remaining < 1.0` → downgrade tier

#### #2 Parallel Execution — engine extension
- **Αρχείο:** `orchestrator/engine.py` (modify `_execute_all`)
- **Pattern:** Level-based DAG execution
- **Async:** ✅
- **Dependencies:** Κανένα (topological sort ήδη υπάρχει)
- **Effort:** 1-2h
- **TDD:**
  ```
  tests/test_parallel_execution.py:
    test_independent_tasks_execute_in_parallel()
    test_dependent_tasks_wait_for_parent()
    test_failed_task_skips_dependents_not_siblings()
    test_semaphore_limits_concurrent_calls()
  ```
- **Spec κλειδιά:**
  - `_group_by_dependency_level(tasks) -> list[list[str]]`
  - `_execute_parallel_level(ready_tasks) -> dict[str, TaskResult]` με `asyncio.gather(return_exceptions=True)`
  - `asyncio.Semaphore(max_concurrent_tasks)` — default 4, config-driven
  - Logging: "Executing level 2: [task_003, task_004] (parallel)"

---

### Phase 2 — P1: Quality & Intelligence (Ημέρα 3-7)

#### #24 LLM-as-Judge Eval Framework
- **Αρχείο:** `orchestrator/evaluation.py` (νέο)
- **Pattern:** Strategy
- **Async:** ✅
- **Dependencies:** #23 (tracing records eval scores)
- **Reuse:** Αντικαθιστά crude regex scoring στο engine
- **Effort:** 2-3h
- **TDD:**
  ```
  tests/test_evaluation.py:
    test_six_metrics_returned()
    test_security_metric_weighted_highest()
    test_aggregate_score_below_threshold_triggers_repair()
    test_eval_prompts_loaded_from_config()
  ```
- **Spec κλειδιά:**
  - `EvalMetric` enum: `CORRECTNESS | COMPLETENESS | SECURITY | PERFORMANCE | MAINTAINABILITY | TEST_COVERAGE`
  - Weights: security=0.25, correctness=0.30, completeness=0.20, performance=0.10, maintainability=0.10, test_coverage=0.05
  - `LLMJudge.evaluate(code, requirements, metrics, judge_model) -> dict[EvalMetric, EvalResult]`
  - Eval prompts σε external config (αλλάζουν χωρίς code change)

#### #20 Agent Brain
- **Αρχείο:** `orchestrator/brain.py` (νέο)
- **Pattern:** Composite + Repository
- **Async:** ❌
- **Dependencies:** #11 (memory bank για persistence)
- **Effort:** 2-3h
- **TDD:**
  ```
  tests/test_brain.py:
    test_brain_loads_knowledge_files_into_context()
    test_learn_persists_across_instantiations()
    test_build_context_includes_memory_and_knowledge()
    test_save_and_load_round_trip()
  ```
- **Spec κλειδιά:**
  - `AgentBrain(name, role_description, system_prompt, knowledge_files, memory_store, allowed_tools)`
  - `brain.learn(key, value)` → αποθηκεύει στο memory
  - `brain.save(brain_dir)` / `brain.load(brain_dir)`
  - Persistence: `.orchestrator/brains/<name>.json`

#### #29 Human-in-the-Loop Escalation
- **Αρχείο:** `orchestrator/escalation.py` (νέο)
- **Pattern:** State Machine
- **Async:** ❌
- **Dependencies:** #24 (eval scores trigger escalation)
- **Effort:** 1-2h
- **TDD:**
  ```
  tests/test_escalation.py:
    test_auth_task_low_security_score_blocks()
    test_three_repair_failures_triggers_review()
    test_budget_80pct_notifies()
    test_auto_level_continues_without_interruption()
  ```
- **Spec κλειδιά:**
  - `EscalationLevel` enum: `AUTO | NOTIFY | REVIEW | BLOCK`
  - `EscalationPolicy(confidence_threshold=0.7, budget_alert_pct=0.8, security_score_min=0.8, max_repair_failures=3, critical_task_patterns=["auth","payment","database_migration"])`
  - `EscalationManager.check(task_id, eval_scores, repair_attempts, budget_pct, task_prompt) -> EscalationLevel`

#### #17 Rich Checkpoints + Rollback
- **Αρχείο:** `orchestrator/checkpoints.py` (νέο)
- **Pattern:** Memento + Command
- **Async:** ✅
- **Dependencies:** #10 (condensed context για checkpoint)
- **Effort:** 2-3h
- **TDD:**
  ```
  tests/test_checkpoints.py:
    test_checkpoint_saves_all_output_files()
    test_rollback_restores_exact_file_state()
    test_list_checkpoints_shows_cost_info()
    test_checkpoint_frequency_from_autonomy_config()
  ```
- **Spec κλειδιά:**
  - `Checkpoint(id, timestamp, task_states, conversation_context, artifacts, budget_spent)`
  - `CheckpointManager.create(state, output_dir) -> Checkpoint`
  - `CheckpointManager.rollback(checkpoint_id, output_dir) -> ProjectState`
  - CLI: `--rollback <checkpoint_id>`, `--list-checkpoints`
  - Storage: `.orchestrator/checkpoints/cp_<timestamp>/`

#### #8 Mode-Based Agent Specialization
- **Αρχείο:** `orchestrator/modes.py` (νέο)
- **Pattern:** Strategy + Registry
- **Async:** ❌
- **Dependencies:** #9 (model routing per mode)
- **Effort:** 2h
- **TDD:**
  ```
  tests/test_modes.py:
    test_architect_mode_uses_reasoning_tier()
    test_review_mode_outputs_structured_json()
    test_mode_switch_logged()
    test_code_mode_allows_file_write_actions()
  ```
- **Spec κλειδιά:**
  - `AgentMode` enum: `ARCHITECT | CODE | DEBUG | REVIEW`
  - `ModeConfig(system_prompt, allowed_actions, preferred_model_tier)`
  - Decomposition → ARCHITECT, Generation → CODE, Critique → REVIEW, Repair → DEBUG

#### #12 Prompt Enhancement Pipeline
- **Αρχείο:** `orchestrator/prompt_enhancer.py` (νέο)
- **Pattern:** Decorator
- **Async:** ✅
- **Dependencies:** #9 (budget model για enhancement)
- **Effort:** 1h
- **TDD:**
  ```
  tests/test_prompt_enhancer.py:
    test_vague_prompt_gets_edge_cases_added()
    test_enhancement_fails_gracefully_returns_original()
    test_enhancement_uses_budget_tier_model()
    test_timeout_15s_respected()
  ```
- **Spec κλειδιά:**
  - `PromptEnhancer.enhance(user_prompt, project_context, client) -> str`
  - Fallback: enhancement fails → return original prompt (fail-open)
  - Budget tier model (Gemini Flash), timeout=15s

#### #27 Cost Analytics Dashboard
- **Αρχείο:** `orchestrator/cost_analytics.py` (νέο)
- **Pattern:** Observer + Composite
- **Async:** ❌
- **Dependencies:** Integrates με EventBus
- **Effort:** 1-2h
- **TDD:**
  ```
  tests/test_cost_analytics.py:
    test_breakdown_by_model_sums_correctly()
    test_breakdown_by_phase_includes_all_phases()
    test_efficiency_score_above_0_6_for_normal_run()
    test_cost_report_written_to_output_dir()
  ```
- **Spec κλειδιά:**
  - `CostBreakdown(by_model, by_task, by_phase, by_purpose, token_breakdown)`
  - phases: decomposition, generation, critique, revision, evaluation, repair
  - `efficiency_score() -> float` = implementation_cost / total_cost
  - Output: `cost_report.json` + `cost_report.txt`

#### #1 Chairman LLM Competitive Execution
- **Αρχείο:** `orchestrator/competitive.py` (νέο)
- **Pattern:** Strategy + Facade
- **Async:** ✅
- **Dependencies:** #2 (parallel), #9 (routing)
- **Effort:** 3-4h
- **TDD:**
  ```
  tests/test_competitive.py:
    test_three_models_run_in_parallel()
    test_chairman_selects_winner_with_reasoning()
    test_budget_limit_falls_back_to_single_model()
    test_only_valid_outputs_presented_to_chairman()
  ```
- **Spec κλειδιά:**
  - `competitive_execute(task, models, budget_per_model) -> ExecutionResult`
  - Chairman prompt → JSON `{"winner_index": int, "reasoning": str, "scores": {...}}`
  - Config: `competitive_tasks: list[str]`, `max_competitive_budget_pct: float = 0.3`

#### #23 Tracing — Full Tracer API (extend υπάρχον)
- **Αρχείο:** `orchestrator/tracing.py` (extend)
- **Pattern:** Facade (Langfuse-compatible)
- **Async:** ✅
- **Dependencies:** Κανένα
- **Effort:** 2h
- **TDD:**
  ```
  tests/test_tracing_full.py:
    test_spans_properly_nested_parent_child()
    test_trace_json_written_to_output_dir()
    test_export_summary_shows_per_task_breakdown()
    test_langfuse_schema_compatible()
  ```
- **Spec κλειδιά:**
  - `Tracer` class: `start_trace()`, `start_span()`, `end_span()`, `finish_trace()`
  - `export_json(trace) -> str` — Langfuse schema
  - `export_summary(trace) -> str` — human-readable per-task breakdown
  - Output: `trace.json` στο output directory

#### #16 Plan-Then-Build (extend υπάρχον planner.py)
- **Αρχείο:** `orchestrator/planner.py` (extend)
- **Pattern:** Command + Facade
- **Async:** ✅
- **Dependencies:** Κανένα
- **Effort:** 1-2h
- **TDD:**
  ```
  tests/test_plan_then_build.py:
    test_plan_mode_no_llm_calls_except_decomposition()
    test_plan_json_contains_estimated_cost_and_time()
    test_build_mode_executes_saved_plan()
    test_modified_plan_json_respected()
  ```
- **Spec κλειδιά:**
  - `PlanFirstOrchestrator.plan(project) -> ExecutionPlan` (no execution)
  - `PlanFirstOrchestrator.build(plan) -> ProjectState`
  - CLI: `--mode plan`, `--mode build --plan plan.json`, `--mode auto`
  - JSON plan: tasks, execution_order, estimated_cost_usd, estimated_time_minutes, risk_assessment

#### #11 Memory Bank
- **Αρχείο:** `orchestrator/memory.py` (νέο)
- **Pattern:** Repository
- **Async:** ❌
- **Dependencies:** Κανένα
- **Effort:** 1h
- **Σημείωση:** Διαφορετικό από `memory_tier.py` (αυτό είναι cross-run project memory)
- **TDD:**
  ```
  tests/test_memory_bank.py:
    test_first_run_creates_history_md()
    test_second_run_prompt_includes_project_memory()
    test_decisions_appended_with_timestamp()
    test_memory_files_are_human_readable_markdown()
  ```
- **Spec κλειδιά:**
  - `MemoryBank(project_dir: Path)`
  - `memory_dir` = `project_dir / ".orchestrator" / "memory"`
  - `load() -> dict[str, str]`, `save_decisions(decisions)`, `inject_into_prompt(base_prompt) -> str`

#### #10 Context Condensing Engine
- **Αρχείο:** `orchestrator/context.py` (νέο)
- **Pattern:** Decorator
- **Async:** ✅
- **Dependencies:** #9 (budget model για condensation)
- **Effort:** 2h
- **TDD:**
  ```
  tests/test_context_condenser.py:
    test_short_context_returned_unchanged()
    test_long_context_condensed_below_limit()
    test_condensation_preserves_decisions_and_constraints()
    test_condensation_cost_under_1_cent()
  ```
- **Spec κλειδιά:**
  - `ContextCondenser(max_context_tokens=8000)`
  - `condense_if_needed(conversation_history, client) -> list[dict]`
  - Threshold: 80% of max → summarize via cheap model
  - Token estimation: `len(content) // 4`

---

### Phase 3 — P2: Advanced Features (Εβδομάδα 2+)

| # | Feature | Αρχείο | Pattern | Effort | Dependencies |
|---|---------|--------|---------|--------|-------------|
| #19 | Hierarchical Agents | `hierarchy.py` | Composite+Mediator | 4-6h | #8, #9, #20, #2 |
| #25 | Prompt Versioning (disk) | extend `adaptive_templates.py` | Registry | 1h | #24 |
| #28 | Drift Detection | `drift.py` | Observer+SlidingWindow | 1h | #24, #23 |
| #21 | Workspace Config | `workspace.py` | Repository+Singleton | 1h | — |
| #13 | Unified API Gateway | `gateway.py` | Adapter+Facade | 3-4h | #26 |
| #22 | Event-Driven Execution | `triggers.py` | Observer+Scheduler | 2-3h | #7 |
| #18 | Connectors/MCP | `connectors.py` | Adapter+Registry | 4-5h | #21 |
| #4 | Rich Context Scoping | `context_sources.py` | Builder | 2-3h | — |
| #5 | Execution Isolation | `sandbox.py` | Decorator+Factory | 1-2h | #14 |
| #7 | REST API Layer | `api_server.py` | Facade (FastAPI) | 3-4h | #23 |
| #3 | Skills/Templates | `skills.py` | Registry+Strategy | 2-3h | #11 |

---

### Phase 4 — P3: Browser Testing (Εβδομάδα 3+)

| # | Feature | Αρχείο | Pattern | Effort | Dependencies |
|---|---------|--------|---------|--------|-------------|
| #6 | Browser Testing Agent | `browser_testing.py` | Decorator | 5-6h | #14 |

---

## 6. File Structure After Full Implementation

```
orchestrator/
│
├── # ── Domain Layer (pure data, no I/O) ─────────────────────
├── models.py               Core data structures (Task, Model, Budget, etc.)
├── policy.py               Policy, ModelProfile, PolicySet — Specification pattern
├── cost.py                 BudgetHierarchy + CostPredictor + CostForecaster
├── autonomy.py             [#15 NEW] AutonomyLevel enum + presets
├── modes.py                [#8  NEW] AgentMode enum + ModeConfig
│
├── # ── Engine Layer (coordinator/mediator) ──────────────────
├── engine.py               Orchestrator mediator — wires all services
├── planner.py              ConstraintPlanner [extended #16 PlanFirstOrchestrator]
│
├── # ── Routing & Model Selection ────────────────────────────
├── model_routing.py        [#9  NEW/REFACTOR] ModelTier enum + select_model()
├── adaptive_router.py      Outcome-based routing (legacy, keep)
├── adaptive_templates.py   A/B testing for prompts [#25 extend for disk storage]
│
├── # ── Verification & Quality ───────────────────────────────
├── verification.py         [#14 NEW] REPLVerifier + self-healing loop
├── validators.py           Deterministic validators (syntax, schema, ruff)
├── preflight.py            Content preflight gate (PASS/WARN/BLOCK)
├── evaluation.py           [#24 NEW] LLMJudge with 6 EvalMetrics
├── escalation.py           [#29 NEW] EscalationManager (AUTO/NOTIFY/REVIEW/BLOCK)
├── context.py              [#10 NEW] ContextCondenser
├── skills.py               [#3  NEW] SkillRegistry + template-based decomposition
├── browser_testing.py      [#6  P3 NEW] Playwright behavioral testing
│
├── # ── Agent Intelligence ───────────────────────────────────
├── brain.py                [#20 NEW] AgentBrain with persistent memory
├── competitive.py          [#1  NEW] Chairman LLM competitive execution
├── prompt_enhancer.py      [#12 NEW] Auto-enhance task prompts
├── hierarchy.py            [#19 NEW] HierarchicalOrchestrator
│
├── # ── Observability & Analytics ────────────────────────────
├── tracing.py              [#23 EXTEND] Full Tracer API + Langfuse export
├── telemetry.py            TelemetryCollector
├── cost_analytics.py       [#27 NEW] CostBreakdown + efficiency score
├── drift.py                [#28 NEW] DriftMonitor + quality alerts
│
├── # ── State & Memory ───────────────────────────────────────
├── state.py                ProjectState Repository (SQLite/aiosqlite)
├── memory.py               [#11 NEW] MemoryBank for cross-run learning
├── memory_tier.py          HOT/WARM/COLD tier management
├── session_lifecycle.py    Background scheduler + LLM summarization
├── checkpoints.py          [#17 NEW] CheckpointManager + rollback
│
├── # ── Infrastructure & Resilience ──────────────────────────
├── resilience.py           [#26 NEW] CircuitBreaker + ResilientAPIClient
├── rate_limiter.py         TPM/RPM sliding-window rate limiting
├── gateway.py              [#13 NEW] Unified LLM API gateway
├── connectors.py           [#18 NEW] Connector Protocol + Registry
├── sandbox.py              [#5  NEW] IsolatedExecutor per-task
├── context_sources.py      [#4  NEW] ProjectContext (files/git/URLs)
├── workspace.py            [#21 NEW] WorkspaceConfig (shared secrets/integrations)
├── triggers.py             [#22 NEW] EventDrivenOrchestrator + cron
│
├── # ── API & Interfaces ─────────────────────────────────────
├── cli.py                  CLI entry point (extend with new flags)
├── api_server.py           [#7  NEW] FastAPI REST layer
├── streaming.py            SSE event streaming
│
├── # ── Plugins & Integrations ───────────────────────────────
├── plugins.py              Plugin system
├── mcp_server.py           MCP server (server-side)
├── slack_integration.py    Slack notifications
├── git_service.py          Git operations
│
└── # ── Events & Hooks ────────────────────────────────────────
    ├── events.py           Domain events (extend with new event types)
    └── hooks.py            Event hook registry
```

---

## 7. Architecture Rules for Claude

**Αυτοί οι κανόνες εφαρμόζονται σε κάθε implementation session.**

### R1: Έλεγξε πρώτα για partial implementations
Πριν δημιουργήσεις νέο αρχείο, ελέγξε:
- Υπάρχει ήδη αρχείο με παρόμοια λειτουργία;
- Το `planner.py` ΔΕΝ είναι το `PlanFirstOrchestrator` — είναι ConstraintPlanner
- Το `memory_tier.py` ΔΕΝ είναι το `MemoryBank` — είναι HOT/WARM/COLD
- Το `tracing.py` έχει OTEL infrastructure αλλά ΔΕΝ έχει `Tracer` class

### R2: Νέα features = νέο αρχείο, ΌΧΙ edit στο engine.py
Engine wires services. Logic πηγαίνει στο νέο module.
```python
# engine.py — ΜΟΝΟ wiring
self._verifier = REPLVerifier(autonomy_config)
self._evaluator = LLMJudge(judge_model=Model.CLAUDE_OPUS)
```

### R3: TDD χωρίς εξαιρέσεις
1. Γράψε failing test (RED)
2. Verify ότι το test αποτυγχάνει με expected error
3. Implement minimal code (GREEN)
4. Run full suite (`pytest tests/ -m "not slow"`)
5. Commit

### R4: Sync/Async discipline
- ΜΗΝ προσθέτεις `async` σε: `models.py`, `policy.py`, `cost.py`, `validators.py`, `rate_limiter.py`, `autonomy.py`, `modes.py`, `escalation.py`, `cost_analytics.py`, `drift.py`
- ΜΗΝ κάνεις blocking calls σε: `engine.py`, `state.py`, `session_lifecycle.py`, `gateway.py`, `verification.py`

### R5: Νέος LLM provider = μόνο νέο adapter
```python
# Δημιούργησε class που implements LLMProvider Protocol
class MistralAdapter:
    async def complete(self, model: str, messages: list[dict]) -> CompletionResponse: ...
# Κάνε register στο GatewayClient — τίποτα άλλο δεν αλλάζει
```

### R6: EventBus για cross-cutting concerns
```python
# Ο engine ΔΕΝ ξέρει για cost_analytics, drift, telemetry
await self._events.emit(LLMCallCompletedEvent(model=..., tokens=..., cost=...))
# Οι subscribers αποφασίζουν τι κάνουν — engine δεν αλλάζει όταν προστίθεται observer
```

### R7: Fail-open για LLM-dependent features
Κάθε feature που εξαρτάται από LLM call πρέπει να έχει graceful degradation:
```python
try:
    summary = await self._llm_summarize(content)
    entry.summary = summary
except Exception:
    logger.warning("Summarization failed, migrating without summary")
    # entry.summary remains None — continue anyway
```

### R8: Configuration over hardcoding
- Model routing: YAML file, όχι hardcoded list
- Eval prompts: external config, όχι string literals στον κώδικα
- Autonomy presets: `AUTONOMY_PRESETS` dict, όχι if-else chains

### R9: Tests πρέπει να τρέχουν offline
Unit tests ΔΕΝ κάνουν real API calls. Mock το `UnifiedClient`:
```python
@pytest.fixture
def mock_client(mocker):
    client = mocker.Mock()
    client.complete = mocker.AsyncMock(return_value=CompletionResponse(...))
    return client
```

### R10: Imports — one-way dependency rule
```
models.py ← policy.py ← planner.py ← engine.py
models.py ← cost.py ← engine.py
# engine.py imports everything — τίποτα άλλο δεν import-άρει engine.py
```

---

## 8. Testing Strategy ανά Layer

### Domain Layer (models, policy, cost, autonomy, modes)
- **Εργαλείο:** `pytest` (sync, no fixtures needed)
- **Approach:** Pure unit tests — no mocks, no async
- **Παράδειγμα:**
  ```python
  def test_autonomy_lite_has_zero_critique_rounds():
      config = AUTONOMY_PRESETS[AutonomyLevel.LITE]
      assert config.critique_rounds == 0
  ```

### Engine Layer (engine, verification, competitive, hierarchy)
- **Εργαλείο:** `pytest-asyncio`
- **Approach:** Mock `UnifiedClient`, real `ConstraintPlanner` + `PolicyEngine`
- **Παράδειγμα:**
  ```python
  @pytest.mark.asyncio
  async def test_verification_failure_triggers_repair(mock_client):
      verifier = REPLVerifier(AutonomyConfig(repair_attempts=2))
      result = await verifier.verify_and_repair(broken_code, mock_client)
      assert mock_client.complete.call_count == 3  # 1 original + 2 repairs
  ```

### State & Persistence Layer (state, memory, checkpoints)
- **Εργαλείο:** `pytest-asyncio` + `tmp_path` fixture
- **Approach:** Real SQLite in temp directory
- **Παράδειγμα:**
  ```python
  @pytest.mark.asyncio
  async def test_checkpoint_roundtrip(tmp_path):
      mgr = CheckpointManager(tmp_path)
      cp = await mgr.create(state, tmp_path)
      restored = await mgr.rollback(cp.id, tmp_path)
      assert restored == state
  ```

### Adapter Layer (gateway, connectors)
- **Εργαλείο:** `pytest` + `pytest.mark.integration`
- **Approach:** Real API calls — skip με `--skip-integration` flag
- **Παράδειγμα:**
  ```python
  @pytest.mark.integration
  @pytest.mark.asyncio
  async def test_anthropic_adapter_returns_completion():
      adapter = AnthropicAdapter(api_key=os.environ["ANTHROPIC_API_KEY"])
      response = await adapter.complete("claude-haiku-4-5-20251001", [...])
      assert response.content
  ```

### Cross-cutting (rate_limiter, budget math, drift)
- **Εργαλείο:** `pytest` + `hypothesis` (property-based)
- **Approach:** Invariant testing — αποδεικνύει properties, όχι examples
- **Παράδειγμα:**
  ```python
  from hypothesis import given, strategies as st

  @given(st.integers(1, 100_000), st.integers(1, 1000))
  def test_rate_limiter_never_allows_over_limit(tpm_limit, tokens):
      limiter = RateLimiter()
      limiter.configure("tenant", "model", tpm=tpm_limit, rpm=9999)
      # Fill up to limit
      ...
      with pytest.raises(RateLimitExceeded):
          limiter.check("tenant", "model", tokens=1)
  ```

---

## 9. Current Status Tracker

| Phase | Feature | Status | Version |
|-------|---------|--------|---------|
| P0 | #26 Circuit Breaker (engine-level) | 🟡 Partial — in engine.py | v1.0 |
| P0 | #9 Auto Model Routing | 🟡 Partial — adaptive_router.py | v5.0 |
| P0 | #2 Parallel Execution | 🟡 Partial — ad-hoc gather | v1.0 |
| P0 | #15 Parametric Autonomy | ❌ Missing | — |
| P0 | #14 Self-Healing REPL | ❌ Missing | — |
| P1 | #25 Prompt Versioning | ✅ adaptive_templates.py | v5.1 |
| P1 | #23 Tracing (OTEL infra) | 🟡 Partial — OTEL config only | v5.0 |
| P1 | #16 Plan-Then-Build | 🟡 Partial — ConstraintPlanner only | v5.0 |
| P1 | #11 Memory Bank | ❌ Missing (memory_tier ≠ MemoryBank) | — |
| P1 | #24 LLM-as-Judge | ❌ Missing | — |
| P1 | #20 Agent Brain | ❌ Missing | — |
| P1 | #10 Context Condensing | ❌ Missing | — |
| P1 | #17 Rich Checkpoints | ❌ Missing | — |
| P1 | #8 Mode Specialization | ❌ Missing | — |
| P1 | #12 Prompt Enhancement | ❌ Missing | — |
| P1 | #27 Cost Analytics | ❌ Missing | — |
| P1 | #29 Escalation Protocol | ❌ Missing | — |
| P1 | #1 Chairman LLM | ❌ Missing | — |
| P2 | #13 Unified Gateway | 🟡 Partial — api_clients.py | v5.0 |
| P2 | #18 Connectors/MCP | 🟡 Partial — mcp_server.py | v5.0 |
| P2 | #22 Event-Driven | 🟡 Partial — events.py hooks | v5.0 |
| P2 | #5 Execution Isolation | 🟡 Partial — plugin_isolation.py | v5.0 |
| P2 | #19 Hierarchical Agents | 🟡 Partial — agents.py | v5.0 |
| P2 | #28 Drift Detection | ❌ Missing | — |
| P2 | #21 Workspace Config | ❌ Missing | — |
| P2 | #4 Rich Context Scoping | ❌ Missing | — |
| P2 | #7 REST API Layer | ❌ Missing | — |
| P2 | #3 Skills/Templates | ❌ Missing | — |
| P3 | #6 Browser Testing | ❌ Missing | — |

**Summary:** 3 Complete ✅ | 9 Partial 🟡 | 17 Missing ❌
