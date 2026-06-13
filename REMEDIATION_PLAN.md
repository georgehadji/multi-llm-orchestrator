# Architecture Remediation — Remaining Implementation Plan

> **Generated:** 2026-07-14  
> **Current score:** 7.6 / 10 (target 9.0+)  
> **Completed:** Sprints 1-5 (partial)  
> **Remaining:** 8 unaddressed audit findings + 11 deferred sprint items

---

## Phase 1 — Immediate Fixes (≤1 day)

### P1.1 — Apply queued asyncio.gather edits

**Status:** 6 edits queued for `/apply`  
**Files:** `agents/coordinator.py`, `ara_pipelines.py`×2, `infrastructure/bm25_search.py`, `infrastructure/reranker.py`×2  
**Action:** Run `/apply` to accept the pending multi_edit batch.  
**Verification:** `pytest -m unit -v`

### P1.2 — Fix 10 remaining orphan `asyncio.create_task` sites

**Risk:** Low — each fix is 2-3 lines adding task reference storage.  
**Files and approach:**

| File | Line | Fix |
|------|------|-----|
| `commands/server.py` | 173, 176, 386, 422 | Add `self._bg_tasks: set[asyncio.Task]` in `__init__`, store tasks with `add_done_callback(self._bg_tasks.discard)` |
| `kanban/dispatcher.py` | 113 | Store in `self._board._pending_tasks` or add `_bg_tasks` set |
| `monitoring.py` | 552 | Store in `self._pending` set |
| `infrastructure/monitoring.py` | 552 | Same pattern |
| `events/events_resilient.py` | 121 | Store in `self._pending_replications` set |
| `infrastructure/caching.py` | 625 | Store in `self._promotion_tasks` set |
| `analysis/performance.py` | 447 | Store in `self._cleanup_tasks` set |

**Verification:** `ruff check` all files, `pytest -m unit -v`

### P1.3 — Add `get_event_history()` to UnifiedEventBus

**Risk:** Low — additive method, no API break.  
**File:** `orchestrator/unified_events/core.py`  
**Action:** Add a `get_event_history(self, event_type=None, limit=100)` method that queries the SQLite event store. This matches the NashEventBus API surface, enabling `cli_nash.py` migration.  
**Verification:** `cli_nash.py` `_show_events` and `_follow_events` work when switched to UnifiedEventBus.

### P1.4 — Update 3 modules to use CachePathProvider

**Risk:** Low — drop-in replacement for `Path.home() / ".orchestrator_cache"`.  
**Files:**
- `orchestrator/infrastructure/state.py` — accept `paths: CachePathProvider | None` in `__init__`
- `orchestrator/infrastructure/cache.py` — same pattern
- `orchestrator/kanban/board.py` — same pattern

**Action:** Each module adds an optional `paths` parameter defaulting to `CachePathProvider()`. Replace `Path.home() / ".orchestrator_cache" / "xxx.db"` with `paths.db_name` or `paths.db("xxx")`.  
**Verification:** `pytest tests/ -k "state or cache or kanban" -v`

---

## Phase 2 — Event Bus Unification (3 days)

### P2.1 — Delete NashEventBus, migrate to UnifiedEventBus

**Risk:** Medium — `cli_nash.py` is the sole caller.

1. Verify `get_event_history()` works in UnifiedEventBus (done in P1.3)
2. Switch `cli_nash.py` imports from `.nash_events` → `.unified_events.core`
3. Add `.source` attribute fallback to `DomainEvent.from_dict()` or adapt the display loop
4. Convert `nash_events.py` to a backward-compat re-export shim
5. Delete `nash_events.py` after one release cycle

### P2.2 — Delete AgentMessageBus, complete agent bridge

**Risk:** Medium — 9 agent subclasses need `event_bus` injection.

1. Inject `event_bus` into `AgentOrchestrator` constructor (passes to all agents)
2. Remove `workspace.message_bus` attribute from `ProjectWorkspace`
3. Delete `orchestrator/workspace/message_bus.py`
4. Update `AgentBase.send_message()` — remove legacy fallback path
5. `pytest tests/ -k "agent" -v`

### P2.3 — Add Nash event projections to UnifiedEventBus

**Risk:** Low — additive.

1. Add projection handlers for Nash event types (KNOWLEDGE_GRAPH_UPDATED, TEMPLATE_SELECTED, etc.)
2. Wire Nash stability features to publish through UnifiedEventBus instead of directly calling projection code
3. `pytest tests/ -k "nash" -v`

---

## Phase 3 — God-Class Decomposition (4 days)

### P3.1 — Extract `__aexit__` cleanup into `ServiceContainer.shutdown()`

**Risk:** Medium — 7-step shutdown sequence, must preserve order.  
**File:** `orchestrator/engine.py` lines ~839-947 → `orchestrator/engine_core/container.py`

**Steps:**
1. Add `async def shutdown(self) -> None` method to `ServiceContainer`
2. Move each of the 7 cleanup steps from `Orchestrator.__aexit__`:
   - Close SemanticCache
   - Close SkillManager / wait for epoch tasks
   - Close EventBus
   - Close StateManager
   - Close DiskCache
   - Close TelemetryStore
   - Close A2A Manager
3. `Orchestrator.__aexit__` delegates to `self._container.shutdown()`
4. **Target:** `__aexit__` from 108L → ≤15L

### P3.2 — Slim `__init__` by moving lazy-imports to `ServiceContainer.build()`

**Risk:** Medium — 15+ `try/except ImportError` blocks must be preserved.  
**File:** `orchestrator/engine.py` lines ~446-638 → `orchestrator/engine_core/container.py`

**Steps:**
1. Move each lazy-import block from `Orchestrator.__init__` into `ServiceContainer.build()`
2. Container sets typed fields: `decomposer`, `executor`, `evaluator`, `critique_cycle`, `budget_enforcer`, etc.
3. `Orchestrator.__init__` accepts `container: ServiceContainer` and reads fields from it
4. **Target:** `__init__` from 192L → ≤60L

### P3.3 — Slim `_execute_task` by extracting skill/taste-skill logic

**Risk:** Low — pipeline already delegates to `TaskPipeline.run()`.  
**File:** `orchestrator/engine.py` lines ~2406-2519

**Steps:**
1. Move skill-prefix injection into `engine_core/pipeline.py::PipelineContext.build_system_prompt()`
2. Move taste-skill injection into same method (guarded by `flags.taste_skill_enabled`)
3. Move anti-slop validation into `engine_core/stages/validate_stage.py`
4. Move trajectory recording into `engine_core/pipeline.py::TaskPipeline.run()` post-execution hook
5. **Target:** `_execute_task` from 113L → ≤30L

### P3.4 — Verify engine.py line count

**Action:** `wc -l orchestrator/engine.py`  
**Target:** ≤1,000 lines (from current 2,227)

---

## Phase 4 — Dependency Cleanup (5 days)

### P4.1 — Wire DatabaseManager into all 18 SQLite modules

**Risk:** Low per module, cumulative medium.

| Module | DB Name | Migration |
|--------|---------|-----------|
| `infrastructure/state.py` | `state` | ✅ Already uses aiosqlite; accept `dbm: DatabaseManager` |
| `infrastructure/cache.py` | `cache` | ✅ Already uses aiosqlite |
| `infrastructure/cache_optimizer.py` | `cache_l2` | Accept `dbm` |
| `infrastructure/secure_cache.py` | `secure_cache` | Accept `dbm` |
| `infrastructure/caching.py` | `cache` | Convert from sqlite3 → aiosqlite via dbm |
| `infrastructure/bm25_search.py` | `bm25` | Accept `dbm` |
| `application/skill_store.py` | `trajectories`, `skills` | ✅ Already lazy-imports aiosqlite; accept `dbm` |
| `kanban/board.py` | `kanban` | Accept `dbm`; convert sqlite3 → aiosqlite |
| `pattern_learner/pattern_store.py` | `patterns` | Accept `dbm` |
| `state_mgmt/telemetry_store.py` | `telemetry` | Accept `dbm` |
| `events/async_event_store.py` | `events` | Accept `dbm` |
| `events/events_resilient.py` | `events` | Accept `dbm` |
| `unified_events/core.py` | `events` | Accept `dbm` |
| `workspace/persistent_workspace.py` | `workspace` | Accept `dbm` |
| `analysis/projections.py` | `model_performance` | Accept `dbm` |
| `cost.py` | `budget` | Accept `dbm` |
| `operations/canary_deployment.py` | `canary` | Accept `dbm` |
| `integrations/openrouter_ab_testing.py` | `ab_test` | Accept `dbm` |

**Pattern for each module:**
```python
def __init__(self, dbm: DatabaseManager | None = None, ...):
    self._dbm = dbm or DatabaseManager()
    self._db_path = str(self._dbm.db_path("name"))
```

**Verification:** Run full test suite after each module.

### P4.2 — Wire CachePathProvider into 21+ path references

**Risk:** Low — mechanical replacement.  
**Files:** `automations.py`, `costing/tracker.py`, `cost_tracker.py`, `gradual_rollout.py`, `hitl_workflow.py`, `learning/transfer_learning.py`, `memory_tier.py`, `meta/config.py`, `meta_orchestrator.py`, `restore_points.py`, `session_watcher.py`, plus any others found by grep.

**Verification:** `grep -r "Path.home()" orchestrator/ | grep "orchestrator_cache" | wc -l` — target: zero matches.

### P4.3 — Resolve 2 remaining duplicate pairs

**Files:** `architecture_rules.py` / `safety/architecture_rules.py`, `multi_platform_generator.py` / `generators/multi_platform_generator.py`  
**Action:** Diff the files, reconcile differences, pick one canonical location, convert the other to shim.  
**Risk:** Low — files are 147B and 56KB apart respectively; need manual merge.

---

## Phase 5 — Modularization (5 days)

### P5.1 — Break `cli.py` into subcommands

**Risk:** HIGH — affects all user-facing entry points. Requires TDD approach.

**Target structure:**
```
orchestrator/commands/
├── __init__.py
├── run.py          # run_project, run_job, resume, dry-run
├── dashboard.py    # dashboard launch
├── kanban.py       # kanban start, status, enqueue
├── config_cmd.py   # configuration management
├── chat.py         # interactive chat CLI
└── slash.py        # slash command REPL
```

**Steps:**
1. Create `orchestrator/commands/__init__.py` with a `CommandRegistry` class
2. Extract each command group into its own module
3. `cli.py` becomes a thin entry point (≤200 lines) that imports and registers commands
4. Existing tests that call `cli.py` functions must pass after extraction
5. **Target:** `cli.py` from 2,024L → ≤200L

### P5.2 — Remove 26 direct Orchestrator imports

**Risk:** HIGH — 2 modules use lazy imports to break circular dependencies.

**Strategy — replace direct imports with injected dependencies:**

| Caller | What it needs | Inject instead |
|--------|--------------|----------------|
| `cli.py` | `Orchestrator` | `OrchestratorFactory.create()` |
| `dashboard_mission_control.py` | `Orchestrator` | `OrchestratorFactory` |
| `nash_stable_orchestrator.py` | `Orchestrator` (inheritance) | Use composition with injected `ProjectRunner` |
| `app_builder.py` | `Orchestrator` | `ProjectRunner` + `TaskDecomposer` |
| `control_plane.py` | `Orchestrator` | `ProjectRunner` + `BudgetEnforcer` |
| `agents.py` | `Orchestrator` | `AgentPool` already wraps it; inject via factory |
| `gateway/run.py` | `Orchestrator` | `OrchestratorFactory` |
| `dashboard_core/chat_view.py` | `Orchestrator` | `ConversationAgent` (already in application/) |
| `kanban/dispatcher.py` | `Orchestrator` | `ProjectRunner` + `TaskExecutor` |
| `meta_integration.py` | `Orchestrator` | TYPE_CHECKING only — move import to TYPE_CHECKING block |
| `cost_optimization_integration.py` | `Orchestrator` | `BudgetHierarchy` + `CostPredictor` |

**Verification:** `grep -r "from .engine import Orchestrator" orchestrator/ | grep -v "engine.py\|chat_cli.py\|__init__.py\|tests/" | wc -l` — target: zero matches.

---

## Phase 6 — Observability & Quality (2 days)

### P6.1 — Centralize remaining `os.getenv` calls

**Risk:** Low — mechanical replacement.  
**Action:** Replace `os.getenv("ORCH_XXX")` → `FeatureFlags.env_str("XXX")` or `settings.xxx` where a field already exists.  
**Files:** ~140 remaining sites (down from 146 after context_compressor fix).  
**Verification:** `grep -r "os\.getenv\|os\.environ" orchestrator/application/ orchestrator/domain/ | wc -l` — target: zero matches in non-infrastructure layers.

### P6.2 — Verify coverage at 15%

**Action:** `pytest --cov=orchestrator --cov-fail-under=15 --cov-report=term-missing`  
**If coverage insufficient:** Add focused tests for newly extracted modules from Sprints 3-5:
- `engine_core/decomposer.py` — decomposition logic
- `infrastructure/database_manager.py` — schema initialization
- `infrastructure/path_provider.py` — path construction
- `application/context_compressor.py` — compression flow

### P6.3 — Adopt structlog in engine.py

**Risk:** Low — additive, backward-compatible.  
**Action:** Replace `logging.getLogger` in `engine.py` with `structlog.get_logger()`.  
**File:** `orchestrator/engine.py`  
**Verification:** Log output unchanged in format; structured fields available when `ORCH_LOG_FORMAT=json`.

---

## Execution Order (Dependency Graph)

```
P1.1 ──► P1.2 ──► P1.3 ──► P1.4
                              │
                              ▼
                P2.1 ──► P2.2 ──► P2.3
                              │
                              ▼
                P3.1 ──► P3.2 ──► P3.3 ──► P3.4
                              │
                              ▼
                P5.2 ◄── P4.1 ──► P4.2 ──► P4.3
                  │           │
                  ▼           ▼
                P5.1       P6.1 ──► P6.2 ──► P6.3
```

**Critical path:** P1.1 → P2.2 → P3.4 → P4.1 → P5.2 (estimated 12 working days)  
**Parallelizable:** P1.4 + P2.1, P4.2 + P4.3 + P6.1, P5.1 + P6.3

---

## Expected Score After All Phases

| Dimension | Current | After | Δ |
|-----------|---------|-------|---|
| Concurrency safety | 8 | **9** | +1 |
| Event architecture | 5 | **9** | +4 |
| Modularity/cohesion | 7 | **9** | +2 |
| Dependency management | 7 | **9** | +2 |
| Layer discipline | 8 | **9** | +1 |
| Port/adapter hygiene | 8 | **9** | +1 |
| Observability | 7 | **8** | +1 |
| Testability | 8 | **8** | — |
| **Weighted total** | **7.6** | **9.0** | **+1.4** |

---

## Risk Register

| Risk | Phase | Likelihood | Mitigation |
|------|-------|-----------|------------|
| CLI decomposition breaks user workflows | P5.1 | Medium | TDD: extract + test each subcommand before deleting original |
| Orchestrator import removal breaks circular deps | P5.2 | High | Lazy imports preserved; `OrchestratorFactory` with `TYPE_CHECKING` guards |
| DatabaseManager wiring introduces connection leaks | P4.1 | Medium | Each module migration tested independently; `aiosqlite.connect` always paired with `close()` |
| NashEventBus deletion breaks nash stability features | P2.1 | Medium | Suture: 2-release deprecation cycle; shim in release 1, delete in release 2 |
| Coverage at 15% cannot be reached without new tests | P6.2 | Medium | Accept 12% if needed; raise ratchet incrementally in follow-up PRs |
