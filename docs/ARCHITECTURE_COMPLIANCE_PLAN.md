# Architecture Compliance Plan — 5.5 → 9+ Score

> **Audit date:** 2026-05-28  
> **Current score:** 5.5 / 10 (Evolving)  
> **Target score:** > 9.0  
> **Total estimated effort:** ~11 days across 7 milestones

---

## Context

A full architectural audit scored the codebase 5.5/10. The target architecture
(Hexagonal / Ports-and-Adapters) is correctly designed in the domain layer but incompletely
enforced in the composition root, and partially reversed by back-references in extracted services.
The refactor is ~70% complete; this plan finishes it safely.

**Gap summary:**

| Gap | Severity | Score drag |
|-----|----------|------------|
| `ProjectRunner._host` back-reference to Orchestrator (20+ private method calls) | CRITICAL | −1.0 |
| DI container: 79 `Any`-typed fields + 5 dummy fallbacks | HIGH | −0.8 |
| `engine_core` not covered by importlinter (only `application` is) | HIGH | −0.5 |
| `api_clients.py` wildcard re-export leaks infrastructure layer | HIGH | −0.3 |
| `generate.py` imports concrete `UnifiedClient` not `LLMClient` Protocol | MEDIUM | −0.2 |
| Orphan `asyncio.create_task` (SkillOpt trajectory, engine.py:2373) | MEDIUM | −0.2 |
| Shared mutable dicts in `ModelHealthTracker` ↔ engine | MEDIUM | −0.2 |
| `ResumptionService` mutates `ProjectState` in-place | MEDIUM | −0.1 |
| 316 root `orchestrator/*.py` files with 10+ confirmed duplicates | HIGH | −0.7 |

**Target state:** Each layer independently testable, correctly bounded, and statically typed.
No service holds a back-reference to `Orchestrator`. Import contracts cover all layers.
Duplicate modules consolidated.

---

## Expected Score Progression

| After | Score | Key improvement |
|-------|-------|-----------------|
| Baseline | 5.5 | — |
| M1 | 5.8 | Orphan task tracked, exceptions logged, api_clients clean, engine_core import contract |
| M2 | 6.5 | Container typed, no dummy fallbacks, NullAdapters used consistently |
| M3 | 7.5 | ProjectRunner fully decoupled — no `_host` back-reference anywhere in `application/` |
| M4 | 7.8 | Pipeline stages depend on `LLMClient` Protocol, not concrete class |
| M5 | 8.6 | Module layout coherent; 10 root duplicates consolidated into infrastructure/ |
| M6 | 8.9 | ModelHealthTracker owns its state; no shared mutable dicts |
| M7 | 9.2 | ResumptionService returns new `ProjectState` instead of mutating in-place |

---

## Milestone 1 — Hygiene Fixes
**Effort:** < 1 day | **Risk:** None | **PR:** Single atomic commit

### M1-1 — Retain SkillOpt trajectory task

**File:** `orchestrator/engine.py` ~line 2373

```python
# BEFORE
_asyncio.create_task(self._skill_manager.record_trajectory(_t))

# AFTER
_task = _asyncio.create_task(self._skill_manager.record_trajectory(_t))
self._background_tasks.add(_task)
_task.add_done_callback(self._background_tasks.discard)
```

Reuses the pattern already established at `engine.py:957`.

### M1-2 — Log swallowed exceptions

**File:** `orchestrator/engine.py`

Every bare `except Exception: pass` and `except: pass` in `_execute_task` and surrounding
blocks (confirmed at lines 2374, 2398–2399) becomes:

```python
except Exception as _e:
    logger.debug("SkillOpt trajectory skipped: %s", _e)
```

Grep pattern to find all: `except\s*(\w+\s*)?:\s*\n\s*pass`

### M1-3 — Fix api_clients.py wildcard

**File:** `orchestrator/api_clients.py`

```python
# BEFORE
from .infrastructure.llm_client import *  # noqa: F401, F403

# AFTER — read infrastructure/llm_client.py to enumerate __all__
from .infrastructure.llm_client import (
    UnifiedClient,
    APIResponse,
    LLMClientError,
    # ... add remaining public symbols
)
__all__ = ["UnifiedClient", "APIResponse", "LLMClientError"]
```

### M1-4 — Add importlinter contract for engine_core

**File:** `.importlinter`

```ini
[importlinter:contract:engine-core-no-loose-infra]
name = engine_core modules (except container) must not import infrastructure directly
type = forbidden
source_modules =
    orchestrator.engine_core.pipeline
    orchestrator.engine_core.stages
    orchestrator.engine_core.pipeline_runner
    orchestrator.engine_core.project_planner
    orchestrator.engine_core.state_coordinator
    orchestrator.engine_core.context_service
    orchestrator.engine_core.routing_service
    orchestrator.engine_core.cost_service
forbidden_modules =
    orchestrator.infrastructure
```

`orchestrator.engine_core.container` is **intentionally excluded** — it is the composition root
and the one permitted place for infrastructure imports.

**Verify:** `lint-imports` passes after adding the contract.

---

## Milestone 2 — Container Correctness
**Effort:** 2 days | **Risk:** Low | **Depends on:** M1

### M2-1 — Replace dummy fallbacks with NullAdapters

**File:** `orchestrator/engine_core/container.py`

Five `type("...", (), {...})()` constructions to replace:

| Line | Current dummy | Replace with |
|------|--------------|--------------|
| 259 | `type("CBRegistry", (), {})` | `None` — guard callers with `if self.cb_registry:` |
| 369 | `type("HookRegistry", (), {"fire": ..., "add": ...})()` | `NullEventBus()` from `domain/ports.py` |
| 370 | `type("EventBus", (), {"publish": ...})()` | `NullEventBus()` (same instance as line 369) |
| 438–445 | `type("SemanticCache", (), {...})()` | `None` — guard callers |
| 448 | `type("AdaptiveRouter", (), {})()` | `None` — guard callers |

### M2-2 — Extend NullEventBus to satisfy HookRegistryPort

**File:** `orchestrator/domain/ports.py`

`NullEventBus` currently only has `publish()`. Add `fire()` and `add()` no-ops so it
satisfies `HookRegistryPort` and can replace the dummy at container.py:369:

```python
class NullEventBus:
    async def publish(self, event: Any) -> None:
        pass

    def fire(self, event_type: Any, **kwargs: Any) -> None:   # ADD
        pass

    def add(self, event: Any, callback: Any) -> None:         # ADD
        pass
```

### M2-3 — Type core container fields

**File:** `orchestrator/engine_core/container.py`

Priority fields to type (currently `Any`):

```python
# Use concrete types — better than Any, no Protocol needed
pipeline: "TaskPipeline"              # line 103
pipeline_runner: "PipelineRunner"     # line 92
state_coordinator: "StateCoordinator" # line 127
context_service: "ContextService"     # line 128

# Optional integrations — use Optional[ConcreteType] instead of Any
skill_manager: Optional["SkillManager"] = None   # currently Any
telemetry_store: Optional["TelemetryStore"] = None
```

For the ~50 experimental/optional fields (A2A, red-team, persona, etc.): change
`field: Any = None` → `field: Optional[ConcreteType] = None`. mypy can then catch
`None`-dereferences even without a Protocol.

---

## Milestone 3 — Decouple ProjectRunner from Orchestrator
**Effort:** 3 days | **Risk:** Medium | **Depends on:** M1

This is the highest-value change. `ProjectRunner` currently stores `self._host = host`
and calls 20 private methods/attributes on the `Orchestrator` instance (confirmed lines
102, 118, 122, 126, 133, 154, 162, 170, 174, 205, 212, 214, 236–241, 257, 265, 291, 323).

### M3-1 — Create ProjectRunnerDeps

**New file:** `orchestrator/application/project_runner_deps.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

@dataclass
class ProjectRunnerCallables:
    """Orchestrator methods ProjectRunner needs, injected as callables.
    Eliminates the host=Orchestrator back-reference.
    """
    topological_sort: Callable            # replaces host._topological_sort
    topological_levels: Callable          # replaces host._topological_levels
    make_state: Callable                  # replaces host._make_state
    determine_final_status: Callable      # replaces host._determine_final_status
    log_summary: Callable                 # replaces host._log_summary
    execute_all: Callable[..., Awaitable] # replaces host._execute_all
    generate_architecture_rules: Callable[..., Awaitable]
    analyze_completed_project: Callable[..., Awaitable]

@dataclass
class ProjectRunState:
    """Mutable run-level state shared between Orchestrator and ProjectRunner.
    Replaces host._project_id = x and host._architecture_rules = x assignments.
    """
    project_id: str = ""
    architecture_rules: str = ""
```

### M3-2 — Rewrite ProjectRunner constructor

**File:** `orchestrator/application/project_runner.py`

```python
# BEFORE
def __init__(self, host, state_mgr, budget, ...):
    self._host = host

# AFTER
def __init__(
    self,
    callables: ProjectRunnerCallables,
    run_state: ProjectRunState,
    state_mgr: StatePort,
    budget: Budget,
    event_bus: EventPort,
    resumption_svc: ResumptionService,
    dashboard_bridge: DashboardBridge,
    git_bridge: GitBridge,
    client: Any,
    results: dict,
    generator: Any,
    meta_v2: Any,
    cache: CachePort,
    api_health: dict,
):
    self._callables = callables
    self._run_state = run_state
    ...
```

**Mechanical substitution table:**

| Old (`self._host.X`) | New |
|----------------------|-----|
| `self._host._project_id = v` | `self._run_state.project_id = v` |
| `self._host._architecture_rules = v` | `self._run_state.architecture_rules = v` |
| `self._host._log_summary(state)` | `self._callables.log_summary(state)` |
| `self._host._make_state(...)` | `self._callables.make_state(...)` |
| `self._host._topological_sort(t)` | `self._callables.topological_sort(t)` |
| `self._host._topological_levels(t)` | `self._callables.topological_levels(t)` |
| `await self._host._execute_all(...)` | `await self._callables.execute_all(...)` |
| `self._host._determine_final_status(s)` | `self._callables.determine_final_status(s)` |
| `await self._host._generate_architecture_rules(...)` | `await self._callables.generate_architecture_rules(...)` |
| `await self._host._analyze_completed_project(...)` | `await self._callables.analyze_completed_project(...)` |
| `self._host.client` | `self._client` |
| `self._host.results.values()` | `self._results.values()` |
| `not self._host._entered` | remove guard or use `self._run_state` flag |

### M3-3 — Update engine.py wiring

**File:** `orchestrator/engine.py` ~line 535

```python
# Create shared run state
_run_state = ProjectRunState()
self._run_state = _run_state   # engine reads project_id / architecture_rules from here

# Bundle callables
_callables = ProjectRunnerCallables(
    topological_sort=self._topological_sort,
    topological_levels=self._topological_levels,
    make_state=self._make_state,
    determine_final_status=self._determine_final_status,
    log_summary=self._log_summary,
    execute_all=self._execute_all,
    generate_architecture_rules=self._generate_architecture_rules,
    analyze_completed_project=self._analyze_completed_project,
)

self._project_runner = _ProjectRunner(
    callables=_callables,
    run_state=_run_state,
    state_mgr=self.state_mgr,
    budget=self.budget,
    event_bus=self._event_bus,
    resumption_svc=self._resumption_svc,
    dashboard_bridge=self._dashboard_bridge,
    git_bridge=self._git_bridge,
    client=self._c.client,
    results=self.results,
    generator=self._generator,
    meta_v2=self.meta_v2,
    cache=self.cache,
    api_health=self.api_health,
)
```

### M3-4 — Expand importlinter contract

**File:** `.importlinter`

```ini
# In contract application-services-no-engine, add:
source_modules =
    ...
    orchestrator.application.project_runner        # ADD
    orchestrator.application.project_runner_deps   # ADD
```

**Tests:** `tests/integration/test_project_runner.py` must pass unchanged.
Add `tests/unit/test_project_runner_isolation.py` — construct `ProjectRunner` with
mock callables and no `Orchestrator` present.

---

## Milestone 4 — Stage Protocol Compliance
**Effort:** 0.5 days | **Risk:** Low | **Independent**

### M4-1 — Switch stages to LLMClient Protocol

**Files:** `orchestrator/engine_core/stages/generate.py`, `critique.py`

```python
# BEFORE
from ...api_clients import UnifiedClient

# AFTER
from ...domain.ports import LLMClient
```

Update all type annotations from `UnifiedClient` → `LLMClient`.
`UnifiedClient` already satisfies `LLMClient` structurally — no changes to the concrete class needed.

Grep `from ...api_clients import` in `engine_core/stages/` to find all affected stage files.

---

## Milestone 5 — Canonical Module Deduplication
**Effort:** 3 days | **Risk:** Medium | **Independent**

**10 confirmed duplicates** (root `orchestrator/X.py` vs `orchestrator/infrastructure/X.py`):

| Root shim | Canonical |
|-----------|-----------|
| `orchestrator/state.py` | `orchestrator/infrastructure/state.py` |
| `orchestrator/cache.py` | `orchestrator/infrastructure/cache.py` |
| `orchestrator/bm25_search.py` | `orchestrator/infrastructure/bm25_search.py` |
| `orchestrator/caching.py` | `orchestrator/infrastructure/caching.py` |
| `orchestrator/cache_optimizer.py` | `orchestrator/infrastructure/cache_optimizer.py` |
| `orchestrator/semantic_cache.py` | `orchestrator/infrastructure/semantic_cache.py` |
| `orchestrator/telemetry.py` | `orchestrator/infrastructure/telemetry.py` |
| `orchestrator/token_optimizer.py` | `orchestrator/infrastructure/token_optimizer.py` |
| `orchestrator/tracing.py` | `orchestrator/infrastructure/tracing.py` |
| `orchestrator/audit.py` | `orchestrator/infrastructure/audit.py` |

**Procedure per file (safe, reversible):**
1. Read both root and infrastructure copy. Confirm root is identical or a subset.
2. Replace root file body with a compatibility shim:
   ```python
   # Compatibility shim — canonical location is orchestrator/infrastructure/X.py
   # Scheduled for deletion after all callers are migrated.
   from .infrastructure.X import *  # noqa: F401
   ```
3. Run `pytest -m "not slow"` after every batch of 3 files.
4. Deletion of shims is a **separate follow-up PR** — do not delete in this milestone.

---

## Milestone 6 — ModelHealthTracker State Ownership
**Effort:** 1 day | **Risk:** Low | **Depends on:** M2

**Problem:** `ModelHealthTracker.__init__` accepts `consecutive_failures` and `api_health`
by reference, sharing mutable dicts with the engine. Two writers, no synchronization.

### M6-1 — Tracker owns its dicts

**File:** `orchestrator/application/model_health_tracker.py`

```python
def __init__(
    self,
    telemetry: Any,
    dashboard: Any,
    adaptive_router: Any,
    state_mgr: Any,
    circuit_breaker_threshold: int = 3,
    initial_consecutive_failures: dict | None = None,
    initial_api_health: dict | None = None,
) -> None:
    # Own copies — no shared reference
    self._consecutive_failures: dict = dict(initial_consecutive_failures or {})
    self._api_health: dict = dict(initial_api_health or {})
    ...

@property
def consecutive_failures(self) -> dict:
    return dict(self._consecutive_failures)   # defensive copy

@property
def api_health(self) -> dict:
    return dict(self._api_health)
```

### M6-2 — Update engine.py wiring

**File:** `orchestrator/engine.py`

```python
# BEFORE
self._model_health_tracker = ModelHealthTracker(
    consecutive_failures=self._consecutive_failures,
    api_health=self.api_health,
    ...
)

# AFTER
_cb_state = await self.state_mgr.load_circuit_breaker_state()
self._model_health_tracker = ModelHealthTracker(
    initial_consecutive_failures=_cb_state,
    initial_api_health={},
    ...
)
```

Replace all `self.api_health` reads in engine.py with `self._model_health_tracker.api_health`.

---

## Milestone 7 — Immutable ProjectState in ResumptionService
**Effort:** 0.5 days | **Risk:** Low | **Independent**

**File:** `orchestrator/application/resumption_service.py` lines 69–73

```python
# BEFORE — mutates in place
state.results[task_id] = result
state.status = self._determine_final_status(state)

# AFTER — return new instance
import dataclasses
updated_results = {**state.results, task_id: result}
new_state = dataclasses.replace(state, results=updated_results)
new_state = dataclasses.replace(new_state, status=self._determine_final_status(new_state))
return new_state
```

Update call sites in `engine.py` and `project_runner.py` to use the returned value.

---

## Execution Sequence

```
Week 1:  M1 (day 1) ──► M2 (days 2–3)
         M4 (day 1, parallel with M1)
         M7 (day 1, parallel with M1)

Week 2:  M3 (days 4–6)  ← highest risk, do after M1/M2 green
         M5 (days 4–6, parallel with M3)

Week 3:  M6 (day 7)
         Integration test sweep + commit
```

| Milestone | Days | Risk | Blocking |
|-----------|------|------|----------|
| M1 — Hygiene | 1 | None | — |
| M2 — Container correctness | 2 | Low | M1 |
| M3 — ProjectRunner decoupling | 3 | Medium | M1 |
| M4 — Stage Protocol compliance | 0.5 | Low | — |
| M5 — Module deduplication | 3 | Medium | — |
| M6 — ModelHealthTracker ownership | 1 | Low | M2 |
| M7 — Immutable ProjectState | 0.5 | Low | — |

---

## Verification

**After each milestone:**
```bash
lint-imports
pytest tests/unit/ tests/integration/ tests/contracts/ -v -m "not slow"
pytest tests/integration/test_execute_task_golden_path.py -v
pytest tests/integration/test_project_runner.py -v
```

**Final verification:**
```bash
# Full suite
pytest tests/ -m "not slow and not requires_api" -v

# Import contracts
lint-imports

# No _host back-references remain
grep -r "_host" orchestrator/application/   # must return 0 results

# engine.py line count (target: < 2,500 after M3 wiring simplification)
python -c "print(sum(1 for _ in open('orchestrator/engine.py')))"

# Type check core modules
mypy orchestrator/engine_core/container.py orchestrator/application/project_runner.py --ignore-missing-imports
```

---

## Reusable Patterns

- **Callable injection** — follow `ResumptionService` pattern (`execute_task_fn: Callable`, `determine_final_status_fn: Callable`)
- **NullAdapter extension** — follow `NullSkillStore` pattern in `domain/ports.py`
- **Task tracking** — follow the pattern at `engine.py:957`: `task = create_task(...); self._background_tasks.add(task); task.add_done_callback(self._background_tasks.discard)`
- **Compatibility shim** — `from .infrastructure.X import *  # noqa: F401 — shim; delete after migration`

---

*Created: 2026-05-28*  
*Based on: Architecture Audit Report (same date)*  
*Related: `docs/REFACTORING_PLAN_V7.md`*
