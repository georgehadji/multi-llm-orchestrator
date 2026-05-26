# Engine.py Optimization Plan

> Generated: 2026-05-26
> Source: graphify analysis (14,091 nodes, 375 edges from Orchestrator, star topology)
> Status: Draft — not yet implemented

## Current State

```
orchestrator/engine.py: 5,272 lines · 97 methods · 1 class · 33 property accessors
Graph: 375 edges, 40+ direct module dependencies (star topology)
Community cohesion near 0.0 for top clusters — no clean module boundaries
```

## Phased Decomposition

| Phase | Lines Saved | Risk | Dependency | Description |
|-------|-----------|------|------------|-------------|
| **4: Validators** | ~100 | Low | None | Pure functions, no state — extract first |
| **1: Facade** | ~200 | Low | None | Mechanical extraction of 33 property accessors |
| **3: Dashboard** | ~120 | Low | None | Fire-and-forget notification methods |
| **2: Model Selection** | ~250 | Medium | Phase 1 | Needs Facade for api_health access |
| **5: ServiceCollection** | ~200 | High | Phase 1 | Needs Facade to exist first |
| **6: PipelineRunner** | ~400 | High | Phases 1-5 | Core loop extraction — everything else gone |

## Phase 1: Extract Property Accessors → OrchestratorFacade

**Target:** 33 thin wrapper methods (`a2a_manager()`, `reranker()`, `hybrid_search()`, etc.)

These methods return a reference to an optional sub-module. They exist because modules may not import (try/except ImportError) and callers need safe access. Replace 33 methods with one generic accessor on a dataclass.

**Lines saved:** ~200  
**Risk:** Low — purely mechanical, no logic changes.  
**New file:** `orchestrator/orchestration_facade.py`

```python
@dataclass
class OrchestratorFacade:
    """Safe accessors for optional subsystems."""
    a2a_manager: A2AManager | None = None
    bm25_search: BM25Search | None = None
    reranker: LLMReranker | None = None
    persona_manager: PersonaManager | None = None
    memory_manager: MemoryTierManager | None = None
    session_watcher: SessionWatcher | None = None
    preflight_validator: PreflightValidator | None = None
    token_optimizer: TokenOptimizer | None = None
    red_team: RedTeamFramework | None = None
    agent_safety: AgentSafetyMonitor | None = None
    accountability: AccountabilityTracker | None = None
    task_verifier: TaskVerifier | None = None
    hybrid_pipeline: HybridSearchPipeline | None = None
    knowledge_base: object | None = None
    rate_limiter: RateLimiter | None = None
    lifecycle_manager: SessionLifecycleManager | None = None
    
    def require(self, name: str):
        """Return subsystem or raise if not available."""
        ...
```

## Phase 2: Extract Model Selection → ModelSelector (complete)

**Target:** `_get_available_models()`, `_select_decomposition_model()`, `_get_fast_decomposition_model()`, `_get_cheapest_available()`, `_select_reviewer()`, `_get_fallback()`, `_get_next_tier_model()`, tier constants, `_tier_escalation_count`

A `ModelSelector` class already exists in `model_selector.py` but engine.py still contains ~250 lines of routing logic. The selector should own the complete decision: tier data, escalation tracking, and all selection methods.

**Lines saved:** ~250  
**Risk:** Medium — `_get_available_models` is called from multiple paths. Use delegation pattern.  
**Move to:** `orchestrator/model_selector.py` (expand existing module)

## Phase 3: Extract Dashboard Bridge → DashboardBridge

**Target:** `set_dashboard_integration()`, 4 notify methods, `_build_metrics_dict()`

Self-contained concern — 6 methods that publish events to the dashboard. Extract into a class that receives a reference to orchestrator state.

**Lines saved:** ~120  
**Risk:** Low — all dashboard methods are fire-and-forget.  
**New file:** `orchestrator/dashboard_bridge.py`

## Phase 4: Extract Validation Utilities → validators.py

**Target:** `_validate_syntax_streaming()`, `_validate_syntax_batch()`, `_extract_function_name()`, `_filter_validators_for_task()`

Standalone functions that live on Orchestrator only for access to `self._telemetry` and `self._profiles`. Convert to free functions with explicit parameters.

**Lines saved:** ~100  
**Risk:** Low — pure functions, no Orchestrator state dependency.  
**Move to:** `orchestrator/validators.py` (existing module)

## Phase 5: Slim __init__ → ServiceCollection

**Target:** ~300 lines of constructor wiring for 50+ sub-modules

Group subsystem initialization by concern and construct lazily via a ServiceCollection. Every subsystem is optional (try/except ImportError) so grouping them into named bundles reduces constructor noise without changing behavior.

**Lines saved:** ~200  
**Risk:** High — constructor touches 50+ modules simultaneously. Must be phased, not batched.  
**New file:** `orchestrator/service_collection.py`

## Phase 6: Extract Pipeline Orchestration → PipelineRunner

**Target:** `run_project()`, `run_job()`, `run_project_streaming()`, `dry_run()`, `_execute_all()`, `_warm_cache_for_level()`, `_build_system_prompt()`, `_build_project_context()`

The core loop: decompose → topological sort → execute by level → evaluate → checkpoint. The Orchestrator class remains as the public API that delegates to PipelineRunner.

**Lines saved:** ~400  
**Risk:** High — touches budget enforcement, state checkpointing, parallel execution, plateau detection. Must preserve all 5 invariants.  
**New file:** `orchestrator/pipeline_runner.py`

## Execution Order with Rationale

```
Phase 4: Validators      ← pure functions, no state, fastest win
Phase 1: Facade           ← mechanical, no logic change, enables Phase 2+5
Phase 3: Dashboard        ← fire-and-forget, isolated concern
Phase 2: Model Selection  ← expands existing module, depends on Facade
Phase 5: ServiceCollection← reorganizes constructor, depends on Facade
Phase 6: PipelineRunner   ← core loop, depends on everything else extracted
```

## Target End State

```
Before:
  engine.py                5,272 lines   97 methods   40+ dependencies

After:
  engine.py               ~2,500 lines   PipelineRunner delegation + public API
  model_selector.py         ~350 lines   (+250 from engine, existing expanded)
  validators.py             ~250 lines   (+100 from engine, existing expanded)
  orchestration_facade.py   ~150 lines   (new)
  dashboard_bridge.py       ~120 lines   (new)
  service_collection.py     ~200 lines   (new)
  pipeline_runner.py        ~450 lines   (new)
  
Graph edges from Orchestrator:  ~375 → ~120  (two-thirds reduction)
Community cohesion: expected improvement from single monolithic cluster to multiple isolated ones
```

## What NOT to Touch

- **`_execute_task()`** — core generate→critique→revise→evaluate loop. Too tightly wired.
- **`_record_failure()` / `_record_success()`** — already delegated to adaptive_router.
- **`_decompose()`** — already delegates to DecomposerService. Engine version is glue.
- **`__aexit__`** — just received BUG-004 fix. Do not touch cleanup code.

## Rollback Strategy

Each phase is a separate commit. `git revert <phase-commit>` cleanly reverses any phase.

- Phases 1-4 are independent — no shared state between extracted modules
- Phases 5-6 depend on Phase 1 only
- Every extraction follows existing pattern: `application/*.py` → `engine_core/*.py` → engine.py shim
- Every extracted module gets a backward-compat shim in the old location

## Invariants That Must Be Preserved

1. Cross-review always uses different provider than generator
2. Deterministic validators override LLM scores
3. Budget ceiling is never exceeded (checked mid-task per iteration)
4. State is checkpointed after each task
5. Plateau detection prevents runaway iteration

## Verification Gates Per Phase

After each phase:
1. `python -c "from orchestrator import Orchestrator"` — import cleanly
2. `pytest tests/test_bug_fixes_v2.py -v --no-cov` — 9 regression tests pass
3. `pytest tests/ -x --no-cov -m "not slow and not integration"` — unit test suite passes
4. Manual: graphify `--update` to verify edges reduced on Orchestrator node
