# Refactoring Plan V7 — Safe, Incremental Architecture Recovery

**Date:** 2026-05-28  
**Auditor / Author:** Principal Engineer (Architecture Governance Audit)  
**Baseline audit score:** 3.5 / 10  
**Target score after Phases 0–4:** ≥ 6.5 / 10  
**Predecessor:** `docs/REFACTORING_PLAN_V6.md` (Phase 1 complete, Phase 2 ~95% done)

---

## Context

The codebase was audited on 2026-05-28. Key findings:

- `engine.py` is a **3 037-line God Object** containing 20+ distinct concerns.  
- `ServiceContainer` has **40+ `Any`-typed fields** — type safety is nominal.  
- `domain/ports.py` Protocols are correctly defined but **engine.py bypasses them**, importing concrete adapters directly.  
- A **legacy `orchestrator/pipeline_runner.py`** duplicates `engine_core/pipeline_runner.py`.  
- `UnifiedEventBus` is assigned to **both `event_bus` and `hook_registry`** — async/sync role confusion.  
- `WeakSet` background task tracking causes **telemetry data loss** under GC pressure.  
- **Circuit breaker state** resets on every `Orchestrator` instantiation — protection is lost across runs.  
- `fail_under = 0` means **regressions are invisible**.  
- 30+ **`try/except ImportError`** blocks at module level make the dependency graph unresolvable.

V6 completed Event System Consolidation (Phase 1) and Domain Service extraction (Phase 2).
This plan picks up from Phase 3 and adds the reliability / boundary work V6 did not address.

---

## Guiding Principles

1. **Test first.** Write tests for a class before extracting it from `engine.py`.  
2. **One extraction per PR.** Never bundle two Phase 3 extractions.  
3. **Green CI before merge.** No PR lands with failing tests or a coverage regression.  
4. **Delegation, not deletion.** After extraction, `engine.py` keeps a 1-line delegation method for one sprint before the old body is removed. This allows callers to adapt gradually.  
5. **No feature additions during this refactor.** New ideas go in `docs/BACKLOG.md`.

---

## Do-Not-Touch List (scope guard)

Do **not** modify these files during Phases 0–4:

| File / Path | Reason |
|---|---|
| `orchestrator/domain/ports.py` (existing Protocols) | Contract tests depend on exact signatures |
| `orchestrator/engine_core/stages/` | Already well-designed; leave pipeline stages alone |
| `orchestrator/application/critique_cycle.py` | Already extracted cleanly |
| `orchestrator/application/evaluator.py` | Already extracted cleanly |
| `tests/contracts/` | Validate port contracts — must not change during extraction |
| `orchestrator/infrastructure/nexusscope/` | Separate profiling concern; isolated |

---

## Phase 0 — Safety Net
**Duration:** 1–2 days | **Risk:** None | **Prerequisite:** none

Establish the test harness before touching any code. Every subsequent phase relies on these.

### P0-1: Establish coverage baseline

```bash
pytest tests/ -m "not slow and not requires_api" \
  --cov=orchestrator --cov-report=term-missing
```

Record the percentage. Set `fail_under` in `pyproject.toml` to `(baseline − 2)`.
From this point, coverage must never drop below that floor.

### P0-2: Add golden-path integration test

**New file:** `tests/integration/test_execute_task_golden_path.py`

- Use `NullCache`, `NullState`, `mock_client` from `tests/conftest.py`.
- Assert that `Orchestrator._execute_task(task)` on a single `TaskType.CODE_GEN` task
  returns a `TaskResult` with `status == TaskStatus.COMPLETED`.
- This is the **canary test**: if it turns red during any phase, stop and diagnose before continuing.

### P0-3: Add resume integration test

**New file:** `tests/integration/test_resume_golden_path.py`

- Pre-populate `NullState` with a `ProjectState` having `status = ProjectStatus.PARTIAL_SUCCESS`
  and one task marked `FAILED`.
- Assert `run_project(...)` re-enters `_resume_project` and the final state is
  `ProjectStatus.SUCCESS` or `ProjectStatus.PARTIAL_SUCCESS` (not `SYSTEM_FAILURE`).

### Verification

```bash
pytest tests/integration/ -v   # all green
```

---

## Phase 1 — Dead Code Removal & Reliability Fixes
**Duration:** 2–3 days | **Risk:** Low

### P1-1: Delete legacy `orchestrator/pipeline_runner.py`

`orchestrator/pipeline_runner.py` (198 lines) is a wrapper with a back-reference to
`Orchestrator`. The canonical implementation is `orchestrator/engine_core/pipeline_runner.py`
(86 lines, clean callback injection). The legacy file is not used in the live execution path.

Steps:
1. `grep -r "from .pipeline_runner\|from orchestrator.pipeline_runner" orchestrator/ --include="*.py"`
2. If nothing imports it from the main path → **delete the file**.
3. If something imports it → redirect that import to `orchestrator.engine_core.pipeline_runner`
   and then delete.

### P1-2: Fix `UnifiedEventBus` dual-role

**Problem:** `container.py` lines 322–327 assign a single `UnifiedEventBus` object to both
`event_bus` (async publish) and `hook_registry` (sync fire). These are incompatible roles.

**Fix in `orchestrator/unified_events/core.py`:**

```python
class SyncHookRegistry:
    """Synchronous hook callbacks keyed by EventType."""
    def __init__(self) -> None:
        self._hooks: dict[EventType, list[Callable]] = defaultdict(list)

    def register(self, event_type: EventType, callback: Callable) -> None:
        self._hooks[event_type].append(callback)

    def fire(self, event_type: EventType, **kwargs: Any) -> None:
        for cb in self._hooks.get(event_type, []):
            try:
                cb(**kwargs)
            except Exception as exc:
                logger.warning("Hook %s raised: %s", cb, exc)


class UnifiedEventBus:
    def __init__(self) -> None:
        self.sync_hooks = SyncHookRegistry()   # ← new
        self._subscribers: list[asyncio.Queue] = []

    async def publish(self, event: Any) -> None:
        # fire sync hooks on publish
        if hasattr(event, "event_type"):
            self.sync_hooks.fire(event.event_type, event=event)
        for q in self._subscribers:
            await q.put(event)
```

**Fix in `orchestrator/engine_core/container.py`:**

```python
event_bus = UnifiedEventBus()
hook_registry = event_bus.sync_hooks   # SyncHookRegistry, not the bus itself
```

### P1-3: Fix telemetry data-loss (WeakSet → strong set)

**Problem:** `engine.py` line 461 uses `weakref.WeakSet` for background tasks. Tasks with no
other strong reference are garbage-collected before completion, silently dropping telemetry writes.

**Fix in `engine.py`:**

```python
# BEFORE
import weakref
self._background_tasks: weakref.WeakSet = weakref.WeakSet()

# AFTER
self._background_tasks: set[asyncio.Task] = set()
```

Replace the `_cleanup_task` done-callback:

```python
def _cleanup_task_callback(self, task: asyncio.Task) -> None:
    self._background_tasks.discard(task)
    if not task.cancelled() and task.exception() is not None:
        logger.warning("Background telemetry task failed: %s", task.exception())
```

Remove all `gc.collect()` calls from `_cleanup_background_tasks` and `_start_periodic_cleanup`.

### P1-4: Persist circuit breaker state

**Problem:** `self._consecutive_failures: dict[Model, int]` lives only in memory. It resets
on every `Orchestrator` instantiation, so a model that failed 3 times in the last run gets a
clean slate in the next run.

**Fix — extend `StatePort` in `orchestrator/domain/ports.py`:**

```python
@runtime_checkable
class StatePort(Protocol):
    async def save_project(self, project_id: str, state: ProjectState) -> None: ...
    async def load_project(self, project_id: str) -> ProjectState | None: ...
    async def save_checkpoint(self, project_id: str, task_id: str, state: ProjectState) -> None: ...
    async def save_circuit_breaker_state(self, state: dict[str, int]) -> None: ...   # NEW
    async def load_circuit_breaker_state(self) -> dict[str, int]: ...               # NEW
    async def close(self) -> None: ...
```

Implement in `orchestrator/infrastructure/state.py` using a dedicated `circuit_breaker`
table in the existing SQLite database.

In `engine.py`:
- `_record_failure`: after incrementing, call `await self.state_mgr.save_circuit_breaker_state(...)`
- `__init__` (or `__aenter__`): load state via `await self.state_mgr.load_circuit_breaker_state()`

Update `NullState` in `domain/ports.py` to implement the two new methods (in-memory dict).

### P1-5: Lock in coverage baseline

Edit `pyproject.toml`:

```toml
[tool.coverage.report]
fail_under = <number from P0-1>   # e.g. 42
```

### Verification

```bash
pytest tests/ -m "not slow and not requires_api" -v    # all green, coverage ≥ baseline
pytest tests/contracts/ -v                              # port contracts pass
```

---

## Phase 2 — Type the DI Container
**Duration:** 3–4 days | **Risk:** Low-Medium

### P2-1: Add missing Protocols to `orchestrator/domain/ports.py`

Append after the existing Protocol definitions:

```python
class PlannerPort(Protocol):
    """Model selection and availability queries."""
    def select_model(self, task_type: TaskType, policy_set: Any) -> Model: ...
    def available_models(self, task_type: TaskType) -> list[Model]: ...

class TelemetryPort(Protocol):
    """Records per-model performance and cost metrics."""
    def record_call(
        self, model: Model, latency_ms: float, cost_usd: float, success: bool = True
    ) -> None: ...
    def error_rate(self, model: Model) -> float: ...

class HookRegistryPort(Protocol):
    """Synchronous hook dispatch."""
    def fire(self, event_type: Any, **kwargs: Any) -> None: ...

class ValidatorPort(Protocol):
    """Task output validation."""
    async def validate(self, task: Any, output: str) -> bool: ...
```

### P2-2: Replace `Any` in `ServiceContainer` with Protocols

In `orchestrator/engine_core/container.py`, change the field types:

| Field | Before | After |
|---|---|---|
| `cache` | `Any` | `CachePort` |
| `state_mgr` | `Any` | `StatePort` |
| `event_bus` | `Any` | `EventPort` |
| `hook_registry` | `Any` | `HookRegistryPort` |
| `selector` | `Any` | `PlannerPort` |
| `validator` | `Any` | `ValidatorPort` |
| `telemetry` | `TelemetryCollector` | `TelemetryPort` |

Group remaining optional `Any` fields with a clear comment:

```python
# ── Optional accessory services ── None when not configured ──────────────────
session_watcher: Any = None
persona_manager: Any = None
memory_manager: Any = None
bm25_search: Any = None
# ... (keep full list)
```

### P2-3: Add `Orchestrator.assert_healthy()`

Add the method and call it from `__aenter__`:

```python
def assert_healthy(self) -> None:
    """Assert all required services are wired. Log absent optional services."""
    required = {
        "cache": self.cache,
        "state_mgr": self.state_mgr,
        "client": self.client,
        "telemetry": self._telemetry,
        "policy_engine": self._policy_engine,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise RuntimeError(f"Orchestrator missing required services: {missing}")

    optional = {
        "a2a_manager": self._a2a_manager,
        "red_team": self._red_team,
        "tdd_generator": self._tdd_generator,
        # ... list all optional properties
    }
    absent = [k for k, v in optional.items() if v is None]
    if absent:
        logger.info("Optional services not configured (feature-flagged off): %s", absent)

async def __aenter__(self) -> Orchestrator:
    self.assert_healthy()   # ← add this line
    self._entered = True
    ...
```

### Verification

```bash
mypy orchestrator/domain/ports.py orchestrator/engine_core/container.py \
     --ignore-missing-imports --no-strict-optional
# expect: 0 new errors vs pre-Phase-2 baseline

pytest tests/contracts/ -v     # port contracts still pass
```

---

## Phase 3 — Decompose `engine.py` God Object
**Duration:** 2–3 weeks | **Risk:** Medium

**Prerequisite:** Phase 0 canary tests must be green before starting any extraction.

Each sub-task follows this two-commit pattern:

```
Commit A: add orchestrator/application/<new_service>.py + tests/unit/test_<new_service>.py
Commit B: replace engine.py method body with 1-line delegation; delete old body after 1 sprint
```

### P3-0: Extract `_clean_code_output` (start here — no dependencies, zero risk)

**Target:** `orchestrator/output/code_cleaner.py`

```python
def clean_code_output(text: str, task_type: TaskType) -> str:
    """Strip markdown fences and placeholder comments from LLM code output."""
    ...  # move exact body from engine.py lines 319–356
```

In `engine.py`, replace the function body with:
```python
from .output.code_cleaner import clean_code_output as _clean_code_output
```

Test: `tests/unit/test_code_cleaner.py` — parametrized over fence styles and placeholder patterns.

### P3-1: Extract `_gather_dependency_context` → `DependencyContextService`

**Method location:** `engine.py` ~lines 2784–2869 (86 lines)

**Target:** `orchestrator/application/dependency_context_service.py`

```python
class DependencyContextService:
    def __init__(
        self,
        results: dict[str, TaskResult],
        compressor: ContextCompressor | None,
        truncation_limit: int,
    ) -> None: ...

    async def build(self, task: Task) -> str:
        """Return dependency context string for task prompt injection."""
        ...
```

**Test:** `tests/unit/test_dependency_context_service.py`
- Empty results → empty string
- Single dependency → context contains dependency output
- Context exceeding limit → calls compressor (mock)

### P3-2: Extract `_record_success` / `_record_failure` → `ModelHealthTracker`

**Methods:** `engine.py` ~lines 2594–2675 (79 lines combined)

**Target:** `orchestrator/application/model_health_tracker.py`

```python
class ModelHealthTracker:
    def __init__(
        self,
        telemetry: TelemetryPort,
        consecutive_failures: dict[Model, int],
        api_health: dict[Model, bool],
        state_mgr: StatePort,
        dashboard: Any = None,
        rate_limiter: Any = None,
        circuit_breaker_threshold: int = 3,
    ) -> None: ...

    async def record_success(self, model: Model, response: APIResponse) -> None: ...
    async def record_failure(self, model: Model, error: Exception) -> None: ...
    def is_healthy(self, model: Model) -> bool: ...
```

**Test:** `tests/unit/test_model_health_tracker.py`
- 3 consecutive failures → `is_healthy(model) == False`
- Success after failures → counter resets
- Permanent HTTP 401 → model marked unhealthy immediately (no retries)

### P3-3: Extract `_resume_project` → `ResumptionService`

**Method:** `engine.py` ~lines 2897–2936 (39 lines)

**Target:** `orchestrator/application/resumption_service.py`

```python
class ResumptionService:
    def __init__(
        self,
        state_mgr: StatePort,
        budget: Budget,
        execute_all_fn: Callable[..., Awaitable[ProjectState]],
    ) -> None: ...

    async def resume(self, existing: ProjectState) -> ProjectState:
        """Restore budget, re-execute failed/pending tasks, return updated state."""
        ...
```

**Test:** `tests/integration/test_resumption_service.py`
- Uses `NullState` with a pre-populated `PARTIAL_SUCCESS` state.
- Asserts resumed state includes results from the previous run + newly completed tasks.

### P3-4: Extract dashboard notifications → `DashboardBridge`

**Methods:** `_notify_dashboard_project_start`, `_notify_dashboard_task_start`,
`_notify_dashboard_task_progress`, `_notify_dashboard_task_complete`

**Target:** `orchestrator/application/dashboard_bridge.py`

```python
class DashboardBridge:
    def __init__(self, integration: Any = None) -> None:
        self._integration = integration

    def on_project_start(self, project_id: str, state: Any, rules: Any) -> None: ...
    def on_task_start(self, task_id: str, task: Any, model: Any) -> None: ...
    def on_task_progress(self, iteration: int, score: float) -> None: ...
    def on_task_complete(self, task_id: str, status: str) -> None: ...
```

All methods are null-safe (no-op when `self._integration is None`).

### P3-5: Extract git notifications → `GitBridge`

**Target:** `orchestrator/application/git_bridge.py`

```python
class GitBridge:
    def __init__(self, integration: Any = None) -> None:
        self._git = integration

    def commit_project_completion(
        self, project_name: str, total_tasks: int, total_cost: float, elapsed_time: float
    ) -> str | None: ...
```

### P3-6: Extract OpenRouter helpers → `OpenRouterParams`

**Methods:** `_get_openrouter_call_params`, `_record_optimization_metrics`

**Target:** `orchestrator/application/openrouter_params.py`

```python
class OpenRouterParams:
    def __init__(self, opts: Any, canary: Any = None) -> None: ...

    def build(
        self,
        task_type: TaskType,
        primary_model: Model,
        available_models: list[Model] | None,
        project_id: str,
    ) -> dict[str, Any]: ...

    def record_metrics(self, project_id: str, optimization: str, metrics: dict) -> None: ...
```

### P3-7: Extract `run_project` / `run_job` / `dry_run` → `ProjectRunner`

This is the largest extraction and must be done last (depends on P3-1 through P3-6 being complete).

**Methods:** `run_project` (~160 lines), `run_job` (~45 lines), `dry_run` (~100 lines),
`run_project_streaming` (~30 lines)

**Target:** `orchestrator/application/project_runner.py`

```python
@dataclass
class ProjectRunnerDeps:
    decomposer: Any
    pipeline_runner: Any           # engine_core.pipeline_runner.PipelineRunner
    state_mgr: StatePort
    budget: Budget
    event_bus: EventPort
    hook_registry: HookRegistryPort
    telemetry_store: Any
    budget_hierarchy: Any
    architect: Any
    resumption_svc: ResumptionService
    dashboard_bridge: DashboardBridge
    git_bridge: GitBridge
    openrouter_params: OpenRouterParams
    meta_v2: Any = None


class ProjectRunner:
    def __init__(self, deps: ProjectRunnerDeps) -> None: ...

    async def run_project(
        self,
        description: str,
        criteria: str,
        project_id: str = "",
        app_profile: Any = None,
        analyze_on_complete: bool = False,
        output_dir: Any = None,
    ) -> ProjectState: ...

    async def run_job(self, spec: JobSpec) -> ProjectState: ...

    async def dry_run(self, description: str, criteria: str) -> Any: ...

    async def run_project_streaming(
        self, description: str, criteria: str, project_id: str = ""
    ): ...
```

After extraction, `Orchestrator.run_project` becomes:

```python
async def run_project(self, description, criteria, **kwargs) -> ProjectState:
    return await self._project_runner.run_project(description, criteria, **kwargs)
```

**Test:** `tests/integration/test_project_runner.py`
- Full golden-path run via `NullState` / `mock_client`
- Budget exhaustion halts execution
- Resume path enters `ResumptionService`

### Target state after Phase 3

`engine.py` should reach **≤ 500 lines** containing only:
- `Orchestrator.__init__` (wires container, assigns shortcuts)
- `__aenter__` / `__aexit__` / `close`
- `run_project` / `run_job` / `dry_run` / `run_project_streaming` (3–5 line delegation shells)
- `_execute_task` (wires to pipeline, ~50 lines)
- Public property accessors for optional services
- `_build_metrics_dict` / `export_metrics`

### Verification after each P3-x step

```bash
pytest tests/integration/test_execute_task_golden_path.py -v   # must stay green
pytest tests/integration/test_resume_golden_path.py -v         # must stay green
pytest tests/ -m "not slow and not requires_api" --cov=orchestrator  # coverage ≥ floor
```

---

## Phase 4 — Enforce Architectural Boundaries
**Duration:** 1 week | **Risk:** Low

### P4-1: Add `import-linter`

Add to `pyproject.toml` dev extras:
```toml
[project.optional-dependencies]
dev = [
    ...
    "import-linter>=2.0",
]
```

Create `.importlinter` in the project root:

```ini
[importlinter]
root_package = orchestrator

[importlinter:contract:domain-purity]
name = Domain layer must not import application or infrastructure
type = forbidden
source_modules =
    orchestrator.domain
forbidden_modules =
    orchestrator.infrastructure
    orchestrator.application
    orchestrator.engine_core
    orchestrator.engine

[importlinter:contract:application-uses-ports]
name = Application layer must not import concrete infrastructure adapters
type = forbidden
source_modules =
    orchestrator.application
forbidden_modules =
    orchestrator.infrastructure
    orchestrator.state
    orchestrator.cache
    orchestrator.semantic_cache
```

Add to `.github/workflows/ci.yml` in the **Lint** job:

```yaml
- name: Check import boundaries
  run: lint-imports
```

### P4-2: Replace `try/except ImportError` with feature flags

`engine.py` has 30+ module-level `try/except ImportError` blocks producing `None` fallbacks.
`crosscutting/config.py` already defines `FeatureFlags` backed by `ORCH_*` env vars.

**Strategy for each block:**

1. Add a flag to `FeatureFlags` if not present (e.g., `a2a_enabled: bool = True`).
2. Move the import inside `ServiceContainer.build()`, gated by the flag:
   ```python
   if flags.a2a_enabled:
       from ..a2a_protocol import A2AManager, AgentCard
       container.a2a_manager = A2AManager()
   ```
3. Remove the `try/except` from `engine.py`'s module level.

Priority order (highest coupling risk first):
1. A2A protocol (`a2a_enabled`)
2. Red team framework (`red_team_enabled`)
3. TDD / Diff generators (`tdd_enabled`, `diff_gen_enabled`)
4. Session watcher, persona manager, memory tier
5. Cost optimization tier (batch client, speculative gen, streaming validator)

### P4-3: Audit and archive orphaned packages

Packages to audit — likely empty stubs or single-file experiments:

```
graphify-out   hitl        kanban      ux          builders
connectors     ci          runtime     product     prompting
skills         nash        ux
```

For each package:
```bash
# Check if anything outside tests imports it
grep -r "from orchestrator.<pkg>\|import orchestrator.<pkg>" \
  orchestrator/ --include="*.py" | grep -v __pycache__
```

- **No external imports → delete** (confirm no active tests first).
- **Test-only imports → move to `tests/fixtures/` or delete test**.
- **Active imports → leave in place, document in `BACKLOG.md`**.

### P4-4: Unify port systems — delete `engine_core/protocols.py`

`engine_core/protocols.py` defines `ModelProvider`, `BudgetTracker`, `TaskRunner`,
`ProjectStateService`, `EventPublisher`, `HookManager`, `TelemetryService` — overlapping
with `domain/ports.py`.

Steps:
1. Compare each Protocol in `engine_core/protocols.py` against `domain/ports.py`.
2. For any Protocol not yet in `domain/ports.py`, merge it in (Phase 2 may cover most).
3. Update all imports from `engine_core.protocols` to `domain.ports`.
4. Delete `orchestrator/engine_core/protocols.py`.

### Verification

```bash
lint-imports                                          # all contracts pass
mypy orchestrator/domain/ orchestrator/application/ \
     orchestrator/engine_core/container.py \
     --ignore-missing-imports --no-strict-optional   # no new errors
pytest tests/ -m "not slow and not requires_api" -v  # all green
```

---

## Phase 5 — Horizontal Scaling Preparation *(future sprint)*
**Duration:** TBD | **Risk:** High  
**Prerequisite:** Phases 0–4 complete and stable for ≥ 2 weeks.

### P5-1: `TaskQueuePort` abstraction

```python
# orchestrator/domain/ports.py
class TaskQueuePort(Protocol):
    async def enqueue(self, task: Task, project_id: str) -> str: ...
    async def dequeue(self) -> tuple[Task, str] | None: ...
    async def ack(self, task_id: str) -> None: ...
    async def nack(self, task_id: str, reason: str) -> None: ...
```

Adapters:
- `InProcessTaskQueue` — `asyncio.Queue` (current behavior, zero migration cost)
- `RedisTaskQueue` — `redis.asyncio` (future, enables worker fan-out)

### P5-2: SQLite write serialization

Concurrent tasks currently share one `aiosqlite` connection pool. Add an
`asyncio.Semaphore(1)` around all write operations in `StateManager`, or introduce
a dedicated write-worker coroutine that serializes writes through a queue.

---

## Files to Create / Modify

| Action | File | Phase |
|---|---|---|
| Create | `tests/integration/test_execute_task_golden_path.py` | P0 |
| Create | `tests/integration/test_resume_golden_path.py` | P0 |
| Modify | `pyproject.toml` — `fail_under` | P0 |
| Delete | `orchestrator/pipeline_runner.py` | P1 |
| Modify | `orchestrator/unified_events/core.py` — `SyncHookRegistry` | P1 |
| Modify | `orchestrator/engine.py` — `WeakSet → set` | P1 |
| Modify | `orchestrator/domain/ports.py` — `StatePort` circuit breaker methods | P1 |
| Modify | `orchestrator/infrastructure/state.py` — circuit breaker persistence | P1 |
| Modify | `orchestrator/engine_core/container.py` — `hook_registry = event_bus.sync_hooks` | P1 |
| Modify | `orchestrator/domain/ports.py` — new Protocols | P2 |
| Modify | `orchestrator/engine_core/container.py` — typed fields | P2 |
| Modify | `orchestrator/engine.py` — `assert_healthy()` | P2 |
| Create | `orchestrator/output/code_cleaner.py` | P3-0 |
| Create | `tests/unit/test_code_cleaner.py` | P3-0 |
| Create | `orchestrator/application/dependency_context_service.py` | P3-1 |
| Create | `tests/unit/test_dependency_context_service.py` | P3-1 |
| Create | `orchestrator/application/model_health_tracker.py` | P3-2 |
| Create | `tests/unit/test_model_health_tracker.py` | P3-2 |
| Create | `orchestrator/application/resumption_service.py` | P3-3 |
| Create | `tests/integration/test_resumption_service.py` | P3-3 |
| Create | `orchestrator/application/dashboard_bridge.py` | P3-4 |
| Create | `orchestrator/application/git_bridge.py` | P3-5 |
| Create | `orchestrator/application/openrouter_params.py` | P3-6 |
| Create | `orchestrator/application/project_runner.py` | P3-7 |
| Create | `tests/integration/test_project_runner.py` | P3-7 |
| Create | `.importlinter` | P4 |
| Modify | `.github/workflows/ci.yml` — `lint-imports` step | P4 |
| Modify | `orchestrator/crosscutting/config.py` — new feature flags | P4 |
| Delete | orphaned packages (after audit) | P4 |
| Delete | `orchestrator/engine_core/protocols.py` | P4 |

---

## End-to-End Verification (after all phases)

```bash
# 1. Coverage floor holds
pytest tests/ -m "not slow and not requires_api" --cov=orchestrator --cov-fail-under=<baseline>

# 2. Import boundaries enforced
lint-imports

# 3. Type safety in core layers
mypy orchestrator/domain/ orchestrator/application/ \
     orchestrator/engine_core/container.py \
     --ignore-missing-imports --no-strict-optional

# 4. God Object dismantled
python -c "
import subprocess, sys
lines = int(subprocess.check_output(['wc', '-l', 'orchestrator/engine.py']).split()[0])
print(f'engine.py: {lines} lines')
sys.exit(0 if lines <= 500 else 1)
"

# 5. Integration suite green
pytest tests/integration/ -v

# 6. Contract tests green
pytest tests/contracts/ -v
```

Expected architecture score after completion: **≥ 6.5 / 10**
