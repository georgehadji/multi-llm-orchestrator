# Master Architecture Enhancement Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-23  
> **Version:** 1.0 — Master Plan  
> **Status:** Plan — awaiting implementation  

---

## Executive Summary

This plan consolidates three parallel initiatives into a single, ordered roadmap:

| Initiative | Current State | Target State |
|-----------|---------------|--------------|
| **God File Refactoring** | engine.py = 3,325 lines (was 5,313). Phases 1-5 complete. | engine.py = ~1,200 lines. Phases 6-8 complete. |
| **ARA Integration** | 20 ARA pipelines exist in `ara_pipelines.py` but never called by engine.py. | ARA methods dispatched per task type through the execution pipeline. |
| **Architecture Compliance** | Circular imports, concrete adapters hardwired, ports unused, mypy not enforced. | Clean layering, port-based DI, type-safe code, test coverage ≥50%. |

**Total effort:** 26 days across 10 phases.  
**Risk profile:** Each phase is independently revertible via git.  
**Rollback safety:** Fail-open design throughout — if ARA or new modules fail, execution falls back to current behavior.

---

## Current Architecture Baseline

### What's Done (Phases 1-5 Complete)

```
✅ engine_deps.py           — All try/except imports extracted
✅ engine_core/decomposer   — Decomposition pipeline (448 lines)
✅ engine_core/validator    — Preflight + validation (285 lines)
✅ engine_core/architect    — Architecture rules (150 lines)
✅ engine_core/pipeline     — TaskPipeline + PipelineContext (156 lines)
✅ engine_core/stages/      — 6 pluggable stages (289 lines)
✅ ara_pipelines.py         — 20 ARA methods registered
✅ ara_execution_strategy   — Method selection logic
✅ codebase_reader.py       — Walker, AST indexer, dep graph, profiler (739 lines)
✅ codebase_context.py      — Relevance ranker, LLM context builder, quality analyzer (669 lines)
✅ codebase_decomposer.py   — LLM-powered modification planner (182 lines)
✅ codebase_writer.py       — Safe file ops, diff engine, safety gates (322 lines)
✅ models.py                — 88 models, 3 new TaskTypes
✅ cli.py                   — `modify` subcommand added
```

### Remaining Gaps

```
❌ engine.py init wires 30+ collaborators inline (no ServiceContainer)
❌ ARA not called from engine.py execution path
❌ Circular imports: 11 files import from engine.py
❌ Ports exist but unused in Orchestrator.__init__
❌ mypy strict not enforced
❌ Test coverage = 12% (target: 50%)
❌ Dual event systems (events.py + unified_events/)
❌ God File Phases 6-8 not implemented
```

---

## Phased Implementation Plan

---

### Phase 1: ARA Infrastructure Wiring (Foundation)

**Goal:** Make ARA accesible from engine.py — zero impact on existing execution.

**Effort:** 1 day | **Risk:** None

**Tasks:**
- Add ARA import block to `engine_deps.py` (HAS_ARA flag)
- Initialize `self._ara` in `Orchestrator.__init__` via `create_ara_integration()`
- Add `_execute_task_ara()` method to Orchestrator (fail-open: falls back to standard pipeline)
- Wire `ARAExecutionStrategy.should_use_ara()` into `_execute_task_via_pipeline()`

**Verification:**
```bash
python -c "from orchestrator.engine import Orchestrator; orch = Orchestrator(); assert hasattr(orch, '_ara')"
```

---

### Phase 2: ARA Self-Consistency Enhancement (CoVE + Debate)

**Goal:** Replace simple retry logic with structured improvement via ARA.

**Effort:** 3 days | **Risk:** Medium (changes retry behavior)
**Depends on:** Phase 1

**Tasks:**
- Create `EnhancedSelfConsistencyStage` in `engine_core/stages/self_consistency.py`
- On score < 0.5 with CODE_REVIEW: dispatch to Debate pipeline
- On score < 0.6 with CODE_GEN/REASONING: dispatch to CoVE pipeline
- Add ARA retry handler in `_execute_task_via_pipeline`: `if ctx.abort_reason == "ara_retry" → self._execute_task_ara()`
- Add fallback: if ARA doesn't improve score, fall through to standard retry

**Verification:** Create task with known low-quality output; verify CoVE/DeBate improves score.

---

### Phase 3: ARA Code Review Enhancement (PersuasionDefense)

**Goal:** Verify generated code against hallucinated claims before delivery.

**Effort:** 2 days | **Risk:** Low
**Depends on:** Phase 1

**Tasks:**
- Add `PersuasionDefenseStage` to pipeline stages
- On CODE_GEN tasks, extract claims → NLI verify → conflict surface → score
- If verification score < 0.5, set `should_abort = True`, `abort_reason = "verification_failed"`
- Add PersuasionDefense metrics to `ObservabilityService` (claims verified, conflicts found)

**Verification:** Run PersuasionDefense on generated code; verify hallucinated API calls are caught.

---

### Phase 4: ARA Complex Reasoning Dispatch (SoT + ToT + Self-Discover)

**Goal:** Route REASONING tasks through structured cognitive pipelines.

**Effort:** 2 days | **Risk:** Low
**Depends on:** Phase 1

**Tasks:**
- Create `ARAReasoningDispatcher` class in `ara_execution_strategy.py`
- Dispatch logic:
  - Has sub-problems → SoT (parallel skeleton-of-thought)
  - Is a decision → ToT (tree-of-thoughts exploration)
  - General → Multi-Perspective
- Wire into `_execute_task_via_pipeline` for TaskType.REASONING

**Verification:** Run `"Choose the best architecture for a real-time chat system"` through SoT; verify multi-part answer.

---

### Phase 5: God File Phase 6 — Lazy Wiring + ServiceContainer

**Goal:** Reduce `__init__` from 30+ inline collaborator wires to ~15 lines.

**Effort:** 2 days | **Risk:** Medium (touches initialization order)
**Depends on:** None (standalone)

**Tasks:**
- Convert 15+ accessory collaborators to lazy properties:
  - `_token_optimizer`, `_session_watcher`, `_persona_manager`, `_red_team_framework`
  - `_a2a_manager`, `_rate_limiter`, `_session_lifecycle_manager`
  - `_hybrid_search_pipeline`, `_query_expander`
- Create `ServiceContainer` dataclass for remaining wiring:
  - `Budget`, `Cache`, `StateManager`, `Client`, `Executor`, `Evaluator`, `Generator`
  - `Decomposer`, `Pipeline`, `Telemetry`, `PolicyEngine`, `ModelSelector`
  - `CircuitBreakers`, `Hooks`, `DepResolver`, `AdaptiveRouter`, `ProjectContext`
- Factory method: `ServiceContainer.build(budget, cache, state, ...)`
- `__init__` becomes: `self._svc = ServiceContainer.build(...)` (~15 lines)

**Verification:** `Orchestrator()` constructs identically. All collaborator attributes accessible at same paths. Existing tests pass.

---

### Phase 6: God File Phase 7 — Interface Segregation (Protocols)

**Goal:** Collaborators receive narrowed Protocol types instead of `self` (entire Orchestrator).

**Effort:** 3 days | **Risk:** High
**Depends on:** Phase 5

**Tasks:**
- Define role protocols in `engine_core/protocols.py`:
  - `ModelProvider` — `get_available_models()`, `api_health`
  - `BudgetTracker` — `reserve()`, `commit_reservation()`
  - `TaskRunner` — `execute_task()`
  - `ContextProvider` — `project_context`
  - `EventEmitter` — `fire()`
  - `CircuitBreakerAccess` — `trip()`, `is_open()`, `record_result()`
- Update service constructors to accept Protocol types instead of `Orchestrator`
- `ExecutorService(runner=self, budget=self, ...)` instead of `ExecutorService(execute_fn=self._execute_task, ...)`
- Replace `self` references in services with narrowed protocol arguments
- Verify mypy satisfaction for all Protocol implementations

**Verification:** All tests pass. Static type checker confirms Protocol satisfaction. `Orchestrator` no longer passed as whole self to any service.

---

### Phase 7: God File Phase 8 — Test Coverage

**Goal:** Bring extracted module coverage from 12% to 50%+.

**Effort:** 3 days | **Risk:** None
**Depends on:** Phases 5-6 (modules exist)

**Tasks:**
- Unit tests for each extracted module:
  - `test_decomposer.py` — parse valid JSON, partial recovery, empty input, model fallback
  - `test_validator.py` — syntax streaming, batch, preflight PASS/WARN/BLOCK, error handling
  - `test_pipeline.py` — GenerateStage, CritiqueStage, SelfConsistencyStage, full pipeline integration
  - `test_codebase_reader.py` — walker, AST indexer, dependency graph, profiler
  - `test_codebase_writer.py` — file create/modify/delete, safety gates, diff generation
- Integration tests: verify extracted modules produce identical output to old inline code
- Golden-file tests: run known project spec through old and new paths, diff results
- Run: `pytest tests/ --cov=orchestrator --cov-report=html`

**Verification:** `pytest` with `--cov` shows ≥50% coverage across extracted modules.

---

### Phase 8: Circular Import Cleanup

**Goal:** Reduce engine.py import fan-in from 11 to 2-3 legitimate callers.

**Effort:** 3 days | **Risk:** Medium
**Depends on:** Phase 6 (protocols make this easier)

**Tasks:**
- Move `_clean_code_output` from engine.py to `engine_core/utilities.py`
- Move `_get_available_models` to `engine_core/utilities.py` (or use ModelProvider protocol)
- Move `_select_reviewer` to `engine_core/utilities.py`
- Update all 11 callers to import from new location:
  - `cli.py`, `agents.py`, `app_builder.py`, `control_plane.py`, `cost_optimization_integration.py`
  - `engine_with_events.py`, `meta_integration.py`, `nash_stable_orchestrator.py`
  - `task_handlers.py`
- Verify: `grep "from .engine import\|from orchestrator.engine import" orchestrator/**/*.py` shows only legitimate Orchestrator imports

**Verification:** `grep -r "from .engine import" orchestrator/ | wc -l` ≤ 3 (cli, engine_with_events, and __init__).

---

### Phase 9: Port-Based Dependency Injection

**Goal:** Orchestrator.__init__ accepts abstract Ports, not concrete adapters.

**Effort:** 2 days | **Risk:** Low
**Depends on:** Phase 8

**Tasks:**
- Update `Orchestrator.__init__` signature:
  ```python
  def __init__(
      self,
      cache: CachePort | None = None,
      state_manager: StatePort | None = None,
      event_bus: EventPort | None = None,
      ...
  ):
      self.cache: CachePort = cache or DiskCache()
      self.state_mgr: StatePort = state_manager or StateManager()
  ```
- Verify: `Orchestrator(cache=NullCache(), state_manager=NullState())` works without SQLite
- Update all tests to use NullAdapters where I/O is not required
- Add `pip-audit` security check to CI

**Verification:** `from orchestrator.engine import Orchestrator; orch = Orchestrator(cache=NullCache(), state_manager=NullState()); assert orch.cache is not None`

---

### Phase 10: Type Safety Enforcement

**Goal:** Enforce mypy strict on all new modules; annotate legacy exceptions.

**Effort:** 4 days | **Risk:** Medium
**Depends on:** Phases 5-9 (modules exist, imports clean)

**Tasks:**
- Configure `pyproject.toml`:
  ```toml
  [tool.mypy]
  strict = true
  [[tool.mypy.overrides]]
  module = ["orchestrator.cli_dashboard", "orchestrator.ide_backend.*"]
  ignore_errors = true  # Legacy exceptions only
  ```
- Fix all mypy errors in:
  - `engine.py`, `engine_core/`, `services/`, `ports.py`, `ara_*.py`
  - `codebase_*.py`, `models.py`, `budget.py`
- Add `mypy orchestrator/` to CI workflow
- Run: `mypy orchestrator/ --strict 2>&1 | grep error | wc -l` → target: 0 for new modules

**Verification:** `mypy orchestrator/ --strict` passes with zero errors for non-excluded modules.

---

## Implementation Order & Dependencies

```
Phase 1 (ARA wiring)
  │
  ├─→ Phase 2 (Self-consistency: CoVE + Debate)
  ├─→ Phase 3 (Code review: PersuasionDefense)
  └─→ Phase 4 (Reasoning: SoT + ToT + Self-Discover)
        │
        ▼
Phase 5 (God File Phase 6: Lazy wiring + ServiceContainer)
  │
  └─→ Phase 6 (God File Phase 7: Protocols + Interface Segregation)
        │
        ├─→ Phase 8 (Circular imports cleanup)
        │     │
        │     └─→ Phase 9 (Port-based DI)
        │           │
        │           └─→ Phase 10 (mypy enforcement)
        │
        └─→ Phase 7 (God File Phase 8: Test coverage)
```

**Parallelization opportunities:**
- Phases 1-4 (ARA) can run in parallel with Phase 5 (God File Phase 6) — different files, no conflicts
- Phase 8 (circular imports) and Phase 7 (test coverage) can run in parallel

---

## Effort Summary

| Phase | Description | Days | Risk | Depends on |
|-------|-------------|------|------|------------|
| 1 | ARA infrastructure wiring | 1 | None | — |
| 2 | Self-consistency (CoVE + Debate) | 3 | Medium | Phase 1 |
| 3 | Code review (PersuasionDefense) | 2 | Low | Phase 1 |
| 4 | Complex reasoning (SoT + ToT) | 2 | Low | Phase 1 |
| 5 | Lazy wiring + ServiceContainer | 2 | Medium | — |
| 6 | Interface Segregation (Protocols) | 3 | High | Phase 5 |
| 7 | Test coverage → 50% | 3 | None | Phases 5-6 |
| 8 | Circular import cleanup | 3 | Medium | Phase 6 |
| 9 | Port-based DI | 2 | Low | Phase 8 |
| 10 | mypy strict enforcement | 4 | Medium | Phases 5-9 |
| **Total** | | **25 days** | | |

---

## Files to Create / Modify

| File | Action | Phase |
|------|--------|-------|
| `orchestrator/engine_deps.py` | Add ARA + ports import blocks | 1, 9 |
| `orchestrator/engine.py` | Wire ARA, Protocols, ServiceContainer, Port-based init | 1-6, 8-9 |
| `orchestrator/engine_core/stages/self_consistency.py` | EnhancedSelfConsistencyStage | 2 |
| `orchestrator/engine_core/stages/persuasion_defense.py` | **Create** | 3 |
| `orchestrator/engine_core/container.py` | **Create** ServiceContainer | 5 |
| `orchestrator/engine_core/protocols.py` | **Create** role interfaces | 6 |
| `orchestrator/engine_core/utilities.py` | **Create** shared utilities | 8 |
| `orchestrator/services/` | Update constructors for Protocol types | 6 |
| `tests/test_decomposer.py` | **Create** unit tests | 7 |
| `tests/test_validator.py` | **Create** unit tests | 7 |
| `tests/test_pipeline.py` | **Create** unit tests | 7 |
| `tests/test_codebase_reader.py` | **Create** unit tests | 7 |
| `tests/test_codebase_writer.py` | **Create** unit tests | 7 |
| `pyproject.toml` | mypy strict config | 10 |
| `.github/workflows/ci.yml` | Add mypy to CI | 10 |

---

## Rollback Safety

| Phase | Rollback |
|-------|----------|
| 1 | `HAS_ARA = False` or delete `_ara` attribute |
| 2-4 | Each stage falls back to standard pipeline on failure |
| 5 | Remove `_svc`; restore inline wiring |
| 6 | Remove Protocol annotations; restore `self` references |
| 7 | Delete new test files |
| 8 | Revert utility extraction; re-add methods to engine.py |
| 9 | Revert `__init__` to concrete types |
| 10 | Remove mypy config; revert type annotations |

**Key principle:** Every ARA integration is optional and fails-open. Every codebase modification goes through safety gates. Every refactoring step is a single, revertible git commit.

---

## Verification Gates

- [ ] `python -m orchestrator --help` works after every phase
- [ ] All existing tests pass (`pytest`) after every phase
- [ ] `ruff check orchestrator/` clean after every phase
- [ ] `mypy orchestrator/ --strict` zero errors on new modules (Phase 10)
- [ ] Integration smoke test: `python -m orchestrator --project "Hello World CLI" --budget 0.50 --dry-run`
- [ ] ARA dry run: `python -m orchestrator --project "Build a REST API" --budget 1.00 --dry-run` routes through ARA
- [ ] Codebase-aware dry run: `python -m orchestrator modify --repo ./ --objective "Add logging" --dry-run`
- [ ] Import speed unchanged: `python -c "import orchestrator"` within ±5% wall time

## Appendix: Target Metrics

| Metric | Before | After | Reduction |
|---------|--------|-------|-----------|
| engine.py lines | 3,325 | ~1,200 | 64% |
| Methods in Orchestrator | 60+ | ~20 | 67% |
| Import statements | 90+ | ~40 | 55% |
| Circular imports (files) | 11 | ≤3 | 73% |
| Test coverage | 12% | 50%+ | +38pp |
| ARA methods wired | 0 of 20 | 20 of 20 | 100% |
| Codebase-aware operations | 0 | 1 CLI command | ∞ |
| Models available | 49 | 88 | +80% |
| mypy errors (new modules) | N/A | 0 | — |

---

**Last updated:** 2026-05-23
