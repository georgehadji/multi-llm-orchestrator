# Refactoring Plan — REFACTORING01

**Author:** Architecture Governance Audit (2026-05-27)
**Auditor Role:** Principal Engineer / Senior Software Architect
**Codebase:** Multi-LLM Orchestrator v6.0.0
**Branch:** master
**Scope:** Full static analysis — `orchestrator/` (351 files, ~183K LOC) + `tests/` (38 files)

---

## Overall Architecture Score: 4.5 / 10

| Dimension | Score | Grade |
|---|---|---|
| Intended vs Actual Architecture | 3.5 / 10 | F |
| Layer Separation (Hexagonal) | 4 / 10 | D |
| Code Cohesion | 2 / 10 | F |
| Testability | 4 / 10 | D |
| Observability | 4.5 / 10 | D |
| Resilience | 6 / 10 | C |
| Scalability | 5 / 10 | C |
| Security | 7 / 10 | B |
| **Overall** | **4.5 / 10** | **D+** |

**Architectural Maturity:** Level 2 — "Structured Chaos"
The system has aspirational architecture (good docstrings, declared intentions, some extracted
modules) but implementation has severely drifted from the stated design. New features were
appended as flat files rather than integrated into the intended structure.

---

## Part I — Audit Findings

### 1.1 The Three Unbreakable Rules — Compliance Status

| Rule | Stated | Status | Evidence |
|---|---|---|---|
| `engine.py` = Mediator only | Wire services, no business logic | VIOLATED | 5,036 lines, 104 methods, 1,247-line `_execute_task()` |
| `models.py` = Pure data | No I/O, no asyncio | VIOLATED | `import asyncio`, 4 async methods on `Budget` |
| TDD — test first always | 80%+ coverage enforced | NOT ENFORCED | `fail_under = 12` in `pyproject.toml` |

### 1.2 Intended vs Actual Architecture

**Stated target:**
```
Hexagonal (Ports & Adapters)
  Driving side: cli.py, api_server.py
  Core:         engine.py (Mediator) + engine_core/ (pipeline stages)
  Driven side:  api_clients.py, state.py, cache.py
```

**Actual implementation:**
- `engine.py` is a God Object with 104 methods and 79 import dependencies
- `engine_core/container.py` (ServiceContainer) was built as the DI factory — but `engine.py`
  does not use it
- `application/` layer imports infrastructure directly, violating Dependency Inversion
- 315 of ~351 files sit in a flat, unnamespaced root directory
- Two parallel pipeline implementations exist: `engine_core/stages/` (clean, unused live)
  and `engine.py._execute_task()` (1,247 lines, the real path)
- Two separate port/protocol systems: `domain/ports.py` and `engine_core/protocols.py`
  (not unified)

### 1.3 God Object Evidence — engine.py

| Metric | Value | Healthy Target |
|---|---|---|
| Lines of code | 5,036 | < 400 (Mediator) |
| Methods on Orchestrator | 104 | < 15 |
| Import dependencies | 79 | < 12 |
| Attributes in `__init__` | 37 | < 10 |
| Largest single method (`_execute_task`) | 1,247 lines | < 50 |
| Second largest (`_execute_all`) | 320 lines | < 50 |
| `__init__` constructor | 289 lines | < 30 |

### 1.4 Confirmed Architectural Violations

**Circular dependencies:**
```
agents       <-> engine            (CRITICAL)
engine       <-> meta_integration  (HIGH)
models       <-> policy            (HIGH — domain/policy layer inversion)
autonomous_debugger <-> output_organizer  (MEDIUM)
```

**Application layer importing infrastructure (Dependency Inversion violation):**
```
application/context_compressor.py  imports api_clients
application/critique_cycle.py      imports api_clients
application/evaluator.py           imports api_clients
application/task_executor.py       imports api_clients
```
These should depend on the `ModelProvider` Protocol, not the concrete `UnifiedClient`.

**models.py async violation:**
```python
# models.py — SHOULD NOT EXIST HERE
async def charge(self, amount: float, phase: str = "generation"): ...
async def reserve(self, amount: float) -> bool: ...
async def commit_reservation(self, reserved_amount, actual_amount, phase): ...
async def release_reservation(self, amount: float): ...
```
`Budget` with async behavior belongs in `application/budget_enforcer.py`.

**Dead ServiceContainer:**
```python
# engine_core/container.py — defines full DI factory (360 lines, well-designed)
class ServiceContainer: ...

# engine.py — does NOT use it
def __init__(self, budget, cache, state_manager, ...):  # 289 lines
    self.budget = budget or Budget()     # wired manually
    self.cache = cache or DiskCache()    # wired manually
    self.client = UnifiedClient(...)     # wired manually
    # ... 34 more attributes wired manually
```

**asyncio.run() inside async context (deadlock risk):**
```
cli_nash.py L169: asyncio.run() inside async def _create_backup()
cli_nash.py L225: asyncio.run() inside async def _restore_backup()
```

### 1.5 Flat File Sprawl

```
orchestrator/
  315 flat .py files  (90% of all files)
   33 subdirectories  (10%)
```

Natural groupings visible from file prefixes (candidates for sub-packages):

| Prefix group | Count | Files |
|---|---|---|
| `codebase_*` | 7 | analyzer, context, decomposer, profile, reader, understanding, writer |
| `app_*` | 6 | assembler, builder, detector, store_assets, store_validator, verifier |
| `context_*` | 6 | compressor, condensing, dedup, sources, window, multi |
| `meta_*` | 6 | config, integration, monitoring, orchestrator, planner, store |
| `project_*` | 6 | analyzer, assembler, context, copier, profiler, scaffold |
| `git_*` | 5 | hooks, integration, integration_example, service, sync |
| `nash_*` | 5 | auto_tuning, events, infrastructure_v2, monitor, stable_orchestrator |
| `command_*` | 4 | center, center_integration, center_server, registry |
| `task_*` | 4 | factory, handlers, schemas, verifier |
| `model_*` | 3 | registry, routing, selector |
| `ara_*` | 3 | execution_strategy, integration, pipelines |
| `prompt_*` | 3 | builder, compressor, enhancer |
| `security_*` | 3 | review, templates, validator |
| `output_*` | 3 | organizer, writer, writer_trimmed |
| `cost_*` | 3 | analytics, optimization_integration, tracker |

### 1.6 Error Handling

```
Total swallowed exceptions (except: pass / except X: pass): 117
Files with swallowed exceptions:                             51
Occurrences in engine.py alone:                              9
```

Notable affected files: `engine.py`, `cli.py`, `agents/coordinator.py`,
`application/critique_cycle.py`, `infrastructure/state.py`, `infrastructure/cache.py`.

### 1.7 Observability

```
Total .py files:               497
Using stdlib logging:          253  (50%)
Using print() statements:       79  (15%)
Using structlog:                 1  (<1%)
```

No unified observability standard is enforced. Telemetry files exist (`telemetry.py`,
`telemetry_store.py`, `tracing.py`) but are inconsistently adopted. `application/observability.py`
and `services/observability.py` appear to be parallel implementations.

### 1.8 Test Coverage

```
fail_under = 12    (current enforced minimum in pyproject.toml)
Comment:  "ratchet: current baseline; raise as more tests are added (target: 50)"
Stated requirement in CLAUDE.md: 80%
```

`_execute_task()` — the most critical 1,247-line method in the entire codebase — has no isolated
unit test. The `engine_core/stages/` (the clean pipeline implementation) is tested, but the live
execution path in `engine.py` is not.

### 1.9 Concurrency Model

- Max parallel tasks: **1** (serial execution forced due to SQLite locking)
- SQLite is a single-writer bottleneck — this is the root cause
- `asyncio.run()` called inside async context in `cli_nash.py` — deadlock risk
- `WeakSet` for background task tracking: correct pattern
- Two `asyncio.Lock()` + one `asyncio.Semaphore` in `engine.py` — appropriate

### 1.10 AI Orchestrator Specific

**Two parallel systems, not integrated:**
- `agents/coordinator.py` (`AgentOrchestrator`) — the declared multi-agent system
- `engine.py` (`Orchestrator._execute_task()`) — the actual execution path

These two systems do not appear to call each other. The 9-agent system in `agents/` may be
aspirational/partially implemented while `engine.py` handles all actual work.

**Pipeline duplication:**
- `engine_core/stages/` — clean, composable, tested (`TaskPipeline`)
- `engine.py._execute_task()` — inline reimplementation (1,247 lines, the live path)

**Anti-patterns detected:**

| Anti-Pattern | Location | Impact |
|---|---|---|
| God Object | `engine.py` | Single point of failure for all changes |
| Hidden Monolith | All agents share one process/loop/SQLite | Cannot scale horizontally |
| Flat Namespace Explosion | 315 flat .py files | No encapsulation between domains |
| Parallel Competing Implementations | `engine_core/stages/` vs inline pipeline | Confusion — which path runs? |
| Dead Refactoring | `ServiceContainer` defined, never consumed | Cost without benefit |
| Anemic Domain Violation | `Budget` async methods in `models.py` | Domain logic in wrong layer |
| Infrastructure Leakage | `application/` imports `api_clients` | Application untestable without LLM |
| Silent Failure Propagation | 117 `except: pass` | Errors swallowed, debugging is impossible |
| Two Port Systems | `domain/ports.py` + `engine_core/protocols.py` | Confusion on which to implement against |
| Configuration Sprawl | 5 config files, no hierarchy | No single source of truth |
| Temporal Coupling | `max_parallel_tasks=1` due to SQLite lock | Hidden performance ceiling |
| asyncio misuse | `asyncio.run()` inside async in `cli_nash.py` | Production deadlock risk |
| Global Mutable Singletons | `container.py`, `dashboard_core/core.py`, etc. | Not testable, not thread-safe |

---

## Part II — Refactoring Plan

> **How to read this plan:**
> Each phase is independently deliverable and tests-first. No phase breaks production.
> Each task has a size estimate (S = <1 day, M = 2-3 days, L = 1 week, XL = 2 weeks).

---

## Phase 0 — Zero-Risk Immediate Fixes

**Target:** 1 week. No architectural changes, no regressions. Pure correctness fixes.

---

### TASK-001 — Fix `models.py` async violation [S]

**Problem:** `models.py` imports `asyncio` and contains 4 async methods on `Budget`. Rule 2
states models.py = pure data.

**What to do:**
1. Move the `Budget` class out of `models.py` into `orchestrator/budget.py` (a dedicated file,
   since it has async behavior it is not a pure model).
2. In `models.py`, keep only a lightweight `BudgetSnapshot` dataclass (frozen, no async) for
   serialization purposes if needed.
3. Update all imports that currently pull `Budget` from `models.py`.

**Files to change:**
- `orchestrator/models.py` — remove `Budget` class and `asyncio` import
- `orchestrator/budget.py` — ensure `Budget` lives here (may already exist, verify)
- All files importing `Budget` from `models` — update import path

**Acceptance criteria:**
- Running the following passes without output:
  ```
  python -c "import ast; t=ast.parse(open('orchestrator/models.py').read());
  assert not any(isinstance(n, ast.AsyncFunctionDef) for n in ast.walk(t))"
  ```
- All existing tests pass

---

### TASK-002 — Fix `asyncio.run()` inside async context [S]

**Problem:** `cli_nash.py` lines 169 and 225 call `asyncio.run()` inside `async def` functions.
This causes a `RuntimeError("This event loop is already running")` and is a deadlock risk.

**What to do:**
1. In `async def _create_backup()` (L169): replace `asyncio.run(coro())` with `await coro()`.
2. In `async def _restore_backup()` (L225): same fix.
3. If these functions are also called from sync contexts, create a sync wrapper at the call site
   that uses `asyncio.run()`, not inside the async def.

**Files to change:**
- `orchestrator/cli_nash.py` — lines 169 and 225

**Acceptance criteria:**
- No `asyncio.run(` call exists inside any `async def` in `cli_nash.py`

---

### TASK-003 — Replace swallowed exceptions in `engine.py` [S]

**Problem:** 9 occurrences of `except: pass` or `except X: pass` in the most critical file.
Silent failures make production debugging impossible.

**What to do:**
For each `except: pass` in `engine.py`, replace with:
```python
except SomeSpecificError as e:
    logger.warning("Context: what was being attempted. Error: %s", e)
```
Where the exception is truly expected and ignorable (e.g., optional feature not installed):
```python
except ImportError:
    pass  # optional feature unavailable — intentional
```

**Files to change:**
- `orchestrator/engine.py` — 9 locations

**Acceptance criteria:**
- `grep -n "except:\s*$" orchestrator/engine.py` returns no results
- All existing tests pass

---

### TASK-004 — Merge `architecture_rules_fixed.py` [S]

**Problem:** `architecture_rules_fixed.py` is a stale patch file alongside `architecture_rules.py`.
The `_fixed` suffix indicates an incomplete fix that was never integrated.

**What to do:**
1. Diff `architecture_rules.py` vs `architecture_rules_fixed.py`.
2. Cherry-pick any improvements from `_fixed` into the canonical `architecture_rules.py`.
3. Delete `architecture_rules_fixed.py`.
4. Update any import referencing `architecture_rules_fixed`.

**Files to change:**
- `orchestrator/architecture_rules.py` — absorb any unique logic
- `orchestrator/architecture_rules_fixed.py` — delete

**Acceptance criteria:**
- `orchestrator/architecture_rules_fixed.py` does not exist

---

### TASK-005 — Remove `output_writer_trimmed.py` duplicate [S]

**Problem:** `output_writer_trimmed.py` is a copy of `output_writer.py`. Parallel implementations
drift apart and introduce subtle bugs.

**What to do:**
1. Diff both files. Merge any unique logic from `_trimmed` into `output_writer.py`.
2. Delete `output_writer_trimmed.py`.
3. Update any imports pointing to the trimmed version.

**Files to change:**
- `orchestrator/output_writer.py` — absorb unique logic
- `orchestrator/output_writer_trimmed.py` — delete

---

### TASK-006 — Raise coverage baseline [M]

**Problem:** `fail_under = 12` is below any defensible engineering standard. The stated target in
CLAUDE.md is 80%. The codebase's most critical method (`_execute_task`, 1,247 lines) has no
isolated unit test.

**What to do:**
1. Run `pytest tests/ --cov=orchestrator --cov-report=term-missing` to establish the true baseline.
2. Write unit tests for the 5 most critical untested paths:
   - `orchestrator/budget.py` — `reserve()`, `commit_reservation()`, `release_reservation()`
   - `orchestrator/validators.py` — `all_validators_pass()`
   - `orchestrator/model_selector.py` — `ModelSelector.select()`
   - `orchestrator/engine_core/stages/generate.py` — `GenerateStage.run()`
   - `orchestrator/engine_core/stages/evaluate.py` — `EvaluateStage.run()`
3. Set `fail_under = 25` in `pyproject.toml`.
   Add comment: `# ratchet: raise 5% per sprint, target 80%`

**Files to change:**
- `pyproject.toml` — `fail_under = 25`
- `tests/test_budget.py` — extend
- `tests/test_model_selector.py` — new
- `tests/test_pipeline_stages.py` — new (or extend `test_pipeline.py`)

**Acceptance criteria:**
- `pytest tests/ --cov=orchestrator -q` reports >= 25% without failures

---

## Phase 1 — High-Impact Structural Fixes

**Target:** 2–6 weeks. Fixes the most damaging architectural violations. Zero behavior change.

---

### TASK-101 — Wire `ServiceContainer` into `engine.py` [L]

**Problem:** `engine_core/container.py` defines a `ServiceContainer` that centralizes DI for 30+
collaborators. `engine.py.__init__()` is 289 lines that manually wire all these same dependencies.
The container is dead code from an incomplete migration.

**What to do:**
1. Complete the `ServiceContainer.build()` factory method so it constructs all services currently
   wired in `Orchestrator.__init__()`.
2. Refactor `Orchestrator.__init__()` to accept a `ServiceContainer`:
   ```python
   def __init__(self, container: ServiceContainer | None = None, **legacy_kwargs):
       if container is None:
           container = ServiceContainer.build(**legacy_kwargs)
       self._c = container
       self.budget = container.budget
       self.client = container.client
       # etc. — simple attribute assignment only
   ```
3. The goal: `Orchestrator.__init__()` should be under 30 lines.
4. Keep `ServiceContainer.build(**legacy_kwargs)` so existing call sites work without changes.

**Files to change:**
- `orchestrator/engine_core/container.py` — complete `build()` factory
- `orchestrator/engine.py` — replace 289-line `__init__` with container-based init

**Test first (write before touching engine.py):**
```python
def test_orchestrator_uses_service_container():
    container = ServiceContainer.build(budget=Budget(max_usd=1.0))
    orch = Orchestrator(container=container)
    assert orch.budget is container.budget
    assert orch.client is container.client
```

**Acceptance criteria:**
- `Orchestrator.__init__()` < 30 lines
- `ServiceContainer.build()` creates a fully working Orchestrator
- All existing tests pass

---

### TASK-102 — Extract `_execute_task()` into `TaskPipeline` [XL]

**Problem:** `_execute_task()` is 1,247 lines — the largest method in the system. It
inline-reimplements the 7-stage pipeline that already exists cleanly in `engine_core/stages/`.
Two pipeline implementations exist; only one runs. The inline version has no unit tests.

**What to do:**
1. Map each logical section of `_execute_task()` to the corresponding stage in `engine_core/stages/`:
   - Generate section     → `GenerateStage`
   - Critique section     → `CritiqueStage`
   - Evaluate section     → `EvaluateStage`
   - Validate section     → `ValidateStage`
   - Preflight section    → `PreflightStage`
   - Self-consistency     → `EnhancedSelfConsistencyStage`
   - Persuasion defense   → `PersuasionDefenseStage`
2. Extend `PipelineContext` in `engine_core/pipeline.py` with all fields the stages need
   (budget, client, model, task, circuit breaker state).
3. Replace the body of `_execute_task()` with:
   ```python
   async def _execute_task(self, task: Task, ...) -> TaskResult:
       context = PipelineContext(task=task, budget=self.budget, client=self.client, ...)
       pipeline = TaskPipeline(stages=self._build_stages())
       return await pipeline.run(context)
   ```
4. The old `_execute_task()` body must be deleted entirely — do NOT keep it commented out.

This is the highest-value single refactoring in the entire codebase. It converts 1,247 lines of
untestable monolithic code into 7 independently testable stages.

**Files to change:**
- `orchestrator/engine.py` — replace `_execute_task()` body with pipeline delegation
- `orchestrator/engine_core/pipeline.py` — extend `PipelineContext`
- `orchestrator/engine_core/stages/*.py` — each stage must be production-ready (fill any gaps)
- `tests/test_pipeline.py` — integration test covering the full pipeline end-to-end

**Test first (write ALL of these before touching engine.py):**
```python
async def test_full_pipeline_executes_all_stages(mock_client):
    """Pipeline runs all 7 stages in order and returns a TaskResult."""
    ...

async def test_pipeline_retries_on_low_score(mock_client):
    """Self-consistency stage triggers retry when score < 0.7."""
    ...

async def test_pipeline_respects_budget(mock_client):
    """Pipeline halts when budget is exceeded mid-execution."""
    ...

async def test_pipeline_falls_back_on_model_failure(mock_client):
    """Pipeline uses fallback model when primary model circuit-breaks."""
    ...
```

**Acceptance criteria:**
- `_execute_task()` < 30 lines
- All stages independently testable with mocks (no live LLM required)
- All existing integration tests pass
- Coverage on `engine_core/stages/` >= 70%

---

### TASK-103 — Fix application-layer infrastructure leakage [M]

**Problem:** 4 files in `application/` import `api_clients.UnifiedClient` (infrastructure). This
breaks Dependency Inversion and requires a live LLM connection to run any application-layer test.

**Files affected:**
- `application/context_compressor.py`
- `application/critique_cycle.py`
- `application/evaluator.py`
- `application/task_executor.py`

**What to do:**
1. In each file, replace `from ..api_clients import UnifiedClient` with:
   `from ..engine_core.protocols import ModelProvider` (or `from ..domain.ports import ModelProvider`
   after TASK-105 is complete).
2. Change method signatures to accept `ModelProvider` (the Protocol) not `UnifiedClient`.
3. The concrete `UnifiedClient` is injected by `ServiceContainer` at startup — the application
   layer never sees it.

**Test first:**
```python
async def test_critique_cycle_with_mock_provider():
    """CritiqueCycle works with any ModelProvider — no real LLM needed."""
    mock_provider = MockModelProvider()  # implements ModelProvider Protocol
    cycle = CritiqueCycle(model_provider=mock_provider)
    result = await cycle.run(task=some_task)
    assert result is not None
    mock_provider.call_model.assert_called_once()
```

**Acceptance criteria:**
- No file in `application/` imports from `api_clients`
- All 4 files pass tests using `MockModelProvider` (no real LLM)

---

### TASK-104 — Break `agents <-> engine` circular import [M]

**Problem:** `agents` imports from `engine`, and `engine` imports from `agents`. This circular
dependency is masked by Python's import system but prevents isolation, testing, and future service
extraction.

**What to do:**
1. Define a `TaskDispatch` protocol in `domain/ports.py`:
   ```python
   class TaskDispatch(Protocol):
       async def submit(self, task: Task) -> TaskResult: ...
   ```
2. In `agents/`, replace `from ..engine import Orchestrator` with:
   `from ..domain.ports import TaskDispatch`
3. In `engine.py`, remove any import of `agents`. Use a lazy import or event-based dispatch
   for any reverse dependency.
4. Wire the concrete `Orchestrator` as the `TaskDispatch` implementation in `ServiceContainer`.

**Acceptance criteria:**
- `python -c "from orchestrator import agents"` and
  `python -c "from orchestrator import engine"` both import without pulling the other
- Static analysis confirms the cycle is broken

---

### TASK-105 — Consolidate port definitions [S]

**Problem:** Two separate port/protocol systems:
- `domain/ports.py` — `CachePort`, `StatePort`, `EventPort` (+ null implementations)
- `engine_core/protocols.py` — `ModelProvider`, `BudgetTracker`, `TaskRunner`, `ContextProvider`,
  `EventEmitter`, `CircuitBreakerAccess`, `TelemetryRecorder`, `LoggingProvider`

**What to do:**
1. Move all 8 protocols from `engine_core/protocols.py` into `domain/ports.py`.
2. Either delete `engine_core/protocols.py` or keep it as a pure re-export shim:
   ```python
   from ..domain.ports import ModelProvider, BudgetTracker, ...  # noqa — backwards compat
   ```
3. Update all imports across the codebase.

**Acceptance criteria:**
- `domain/ports.py` is the single file that defines all ports/protocols
- `engine_core/protocols.py` either deleted or is a pure re-export (no new definitions)

---

### TASK-106 — Consolidate configuration [M]

**Problem:** 5 config files with overlapping responsibilities and no clear hierarchy:
```
config.py           (8,199 bytes)
config_as_code.py   (8,851 bytes)
config_sync.py      (3,686 bytes)
crosscutting/config.py (2,452 bytes)
nexus_search/config.py (2,012 bytes)
```

**What to do:**
1. Define a single `RuntimeConfig` dataclass in `orchestrator/config.py` as the canonical root.
2. `crosscutting/config.py` — make it a feature-flag subset that reads from `RuntimeConfig`.
3. `nexus_search/config.py` — subsystem config scoped from `RuntimeConfig`; keeps its location.
4. `config_as_code.py` — policy-as-code DSL, a different concern; keep separate but document it.
5. `config_sync.py` — if it syncs config to an external store, it belongs in `infrastructure/`.

**Acceptance criteria:**
- Single `RuntimeConfig` in `config.py` is the root source of truth
- No config value defined in more than one file
- `config_sync.py` either deleted or moved to `infrastructure/`

---

## Phase 2 — Structural Reorganization

**Target:** 6–12 weeks. Converts flat file sprawl into proper domain packages. Largest body of work.

---

### TASK-201 — Create domain sub-packages from flat file groups [XL]

**Problem:** 315 flat files with no namespace boundaries make the codebase ungovernable and
onboarding prohibitively expensive.

**Migration strategy:** Create sub-packages one group at a time. Each group becomes a package with
a clean `__init__.py` that defines the public API.

**Target structure:**
```
orchestrator/
├── codebase/            # 7 files — codebase_*.py
│   ├── __init__.py      # exports: CodebaseReader, ASTIndexer, DependencyGraph
│   ├── reader.py        # <- codebase_reader.py
│   ├── analyzer.py      # <- codebase_analyzer.py
│   ├── context.py       # <- codebase_context.py
│   ├── decomposer.py    # <- codebase_decomposer.py
│   ├── profile.py       # <- codebase_profile.py
│   ├── understanding.py # <- codebase_understanding.py
│   └── writer.py        # <- codebase_writer.py
│
├── appbuilder/          # 6 files — app_*.py
│   ├── __init__.py
│   ├── assembler.py, builder.py, detector.py, verifier.py, ...
│
├── context_mgmt/        # 6 files — context_*.py
│   ├── __init__.py      # (avoid name "context" — collides with Python builtins context)
│   ├── compressor.py, condensing.py, dedup.py, sources.py, ...
│
├── meta/                # 6 files — meta_*.py
│   ├── __init__.py
│   ├── config.py, integration.py, monitoring.py, orchestrator.py, ...
│
├── project_mgmt/        # 6 files — project_*.py
│   ├── __init__.py
│   ├── analyzer.py, assembler.py, context.py, copier.py, ...
│
├── vcs/                 # 5 files — git_*.py
│   ├── __init__.py
│   ├── integration.py, hooks.py, service.py, sync.py, ...
│
├── reasoning/           # ara_*.py (3) + brain.py + brainstorming.py
│   ├── __init__.py
│   ├── ara_pipelines.py, ara_execution_strategy.py, ara_integration.py
│   ├── brain.py, brainstorming.py
│
├── routing/             # model_*.py (3) + routing-adjacent files
│   ├── __init__.py
│   ├── registry.py, routing.py, selector.py
│
├── prompting/           # prompt_*.py (3)
│   ├── __init__.py
│   ├── builder.py, compressor.py, enhancer.py
│
├── output/              # output_*.py (deduplicated to 2 files)
│   ├── __init__.py
│   ├── organizer.py, writer.py
│
└── interfaces/          # CLI, API, MCP — the driving side
    ├── __init__.py
    ├── cli.py, api_server.py, mcp_server.py, ...
```

**Migration procedure (repeat per package):**
1. Create `orchestrator/<new_package>/` directory.
2. Create `__init__.py` that re-exports the old public names (backwards compatibility shim).
3. Copy files into the package, renaming to remove the prefix
   (e.g., `codebase_reader.py` → `codebase/reader.py`).
4. Update internal imports within the package to use relative imports.
5. In the old flat location, create a backwards-compat shim:
   `from .codebase.reader import *  # noqa — backwards compat, remove after next release`
6. After one release cycle, delete the shims.

**Acceptance criteria (per package):**
- All internal imports within the package use relative imports
- External imports still work via the backwards-compat shims
- No test regressions

---

### TASK-202 — Replace SQLite single-writer bottleneck [L]

**Problem:** `max_parallel_tasks=1` is forced by SQLite write-lock contention. A comment in the
codebase states: `# FIX: Serial execution to avoid SQLite lock`. This artificially serializes all
task execution and creates a hidden performance ceiling.

**What to do:**
1. Migrate `StateManager` to use `aiosqlite` with WAL mode enabled at connection:
   ```python
   async with aiosqlite.connect(db_path) as db:
       await db.execute("PRAGMA journal_mode=WAL")
       await db.execute("PRAGMA synchronous=NORMAL")
       await db.execute("PRAGMA cache_size=10000")
   ```
2. Implement a connection pool (max 5 concurrent writers, queue for excess).
3. Remove the `max_parallel_tasks=1` workaround — raise default to 3.
4. Add a concurrency stress test.

**Files to change:**
- `orchestrator/state.py` or `orchestrator/infrastructure/state.py`
- `orchestrator/engine.py` — remove `max_parallel_tasks=1` comment and the workaround

**Test first:**
```python
async def test_concurrent_state_writes():
    """10 concurrent writes with WAL should all succeed without data loss."""
    mgr = StateManager()
    tasks = [mgr.save(ProjectState(id=f"t{i}")) for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert all(r is not None for r in results)
    # Verify all 10 are readable
    for i in range(10):
        assert await mgr.load(f"t{i}") is not None
```

**Acceptance criteria:**
- `max_parallel_tasks` default is 3 (not 1)
- Concurrency stress test with 10 parallel writes passes

---

### TASK-203 — Unify the two pipeline systems [L]

**Problem:** After TASK-102, `_execute_task()` will delegate to `TaskPipeline`. This task ensures
no code paths in `engine.py` bypass the pipeline and that both systems are fully merged.

**What to do:**
1. After TASK-102 is merged, grep for any remaining inline generate/critique/evaluate logic in
   `engine.py` that bypasses a pipeline stage.
2. Remove any such code — it must go through a stage.
3. Verify the live pipeline covers all 7 documented stages including the fallback chain.
4. Delete any test fixtures that mock the inline pipeline logic (they should now mock stage objects).

**Acceptance criteria:**
- Only one pipeline implementation exists in the codebase
- `_execute_task()` calls `TaskPipeline.run()` exclusively
- No stage logic lives directly in `engine.py`

---

### TASK-204 — Unify event bus systems [M]

**Problem:** Two overlapping event systems:
- `workspace/message_bus.py` — workspace-level pub/sub (file writes, decisions)
- `hooks.py` (`EventType`/`HookRegistry`) — orchestrator-level hooks (task completion, failures)

**What to do:**
1. Audit all publishers and subscribers in both systems.
2. Define a unified `OrchestratorEvent` hierarchy in `domain/events.py`:
   ```python
   @dataclass(frozen=True)
   class OrchestratorEvent:
       event_type: str
       timestamp: float
       payload: dict

   class TaskCompletedEvent(OrchestratorEvent): ...
   class FileWrittenEvent(OrchestratorEvent): ...
   class DecisionRecordedEvent(OrchestratorEvent): ...
   ```
3. Implement one `EventBus` in `infrastructure/event_bus.py` that backs both.
4. Migrate workspace events to use the unified bus.
5. Migrate hook events to use the unified bus.
6. Delete `workspace/message_bus.py` and `hooks.py` (or keep as thin adapters during migration).

**Acceptance criteria:**
- One `EventBus` implementation
- All current event publishers and subscribers still function
- `workspace/message_bus.py` and `hooks.py` either deleted or pure re-export shims

---

## Phase 3 — Long-Term Architecture Evolution

**Target:** 3–6 months. Positions the system for team development and horizontal scaling.

---

### TASK-301 — Distributed tracing end-to-end [L]

**What to do:**
1. Wire `tracing.py` (`Tracer`) through the `ServiceContainer` as a `LoggingProvider` protocol
   implementation.
2. Every `ModelProvider.call_model()` invocation emits a trace span containing:
   `task_id`, `model`, `tokens_in`, `tokens_out`, `latency_ms`, `cost_usd`
3. Every pipeline stage transition emits a stage span.
4. Export to OpenTelemetry-compatible format (OTLP stdout for dev, configurable for prod).

**Acceptance criteria:**
- Every LLM call produces a trace span
- Traces are queryable (local Jaeger container for dev)
- No LLM call is uninstrumented

---

### TASK-302 — Standardize logging [M]

**Problem:** 50% stdlib logging, 15% print(), <1% structlog, 0% consistent format.

**What to do:**
1. Choose one standard: `structlog` with structured JSON for production,
   human-readable console format for development.
2. Add `configure_logging(env: str)` in `orchestrator/__init__.py`, called at all entry points.
3. Convert all `print()` statements to `logger.info()` or `logger.debug()`.
4. Add a pre-commit hook: `grep -r "^\s*print(" orchestrator/ --include="*.py"` must return empty.

**Acceptance criteria:**
- Zero `print()` calls in `orchestrator/` (excluding tests)
- One logging configuration point in `__init__.py`
- Logs emit structured JSON in production mode

---

### TASK-303 — Contract tests for all ports [M]

**What to do:**
For each Protocol in `domain/ports.py`, write a contract test base class:
```python
class CachePortContract:
    """Any CachePort implementation must pass this contract."""

    @pytest.fixture
    def cache(self) -> CachePort:
        raise NotImplementedError

    async def test_get_returns_none_for_missing_key(self, cache):
        assert await cache.get("nonexistent") is None

    async def test_set_then_get_returns_value(self, cache):
        await cache.set("k", "v")
        assert await cache.get("k") == "v"

    async def test_ttl_expires_entry(self, cache):
        await cache.set("k", "v", ttl_seconds=0.01)
        await asyncio.sleep(0.02)
        assert await cache.get("k") is None
```

Then each adapter: `class TestDiskCache(CachePortContract): cache = DiskCache()`

**Acceptance criteria:**
- Every concrete adapter (DiskCache, SemanticCache, NullCache, StateManager, etc.) passes
  its corresponding contract test suite
- New adapters cannot merge without a passing contract test

---

### TASK-304 — Prepare for horizontal scaling [XL]

**What to do:**
1. After TASK-202 (async SQLite) is stable, introduce an optional message queue as a task
   dispatcher (Redis Pub/Sub or NATS as the first implementation).
2. The `TaskDispatch` port from TASK-104 abstracts the dispatch layer — swapping in a queue
   requires only a new `TaskDispatch` adapter, no changes to agents or engine.
3. Design an `AgentWorker` process that reads `AgentTask` from a queue, executes the task,
   and publishes the result — enabling each agent to run in its own process.
4. This makes the system genuinely distributed: separate workers for Developer, Reviewer,
   Tester, DevOps, etc. The `AgentOrchestrator` becomes a coordinator publishing to the queue.

**Depends on:** TASK-104, TASK-202, TASK-204

---

## Summary Table

| Task | Phase | Size | Priority | Risk | Depends On |
|---|---|---|---|---|---|
| TASK-001: Fix models.py async | P0 | S | CRITICAL | Low | — |
| TASK-002: Fix asyncio.run() | P0 | S | CRITICAL | Low | — |
| TASK-003: Fix swallowed exceptions in engine.py | P0 | S | HIGH | Low | — |
| TASK-004: Merge architecture_rules_fixed | P0 | S | MEDIUM | Low | — |
| TASK-005: Remove output_writer_trimmed | P0 | S | MEDIUM | Low | — |
| TASK-006: Raise coverage baseline to 25% | P0 | M | CRITICAL | Low | — |
| TASK-101: Wire ServiceContainer | P1 | L | CRITICAL | Medium | TASK-001, TASK-006 |
| TASK-102: Extract _execute_task() into pipeline | P1 | XL | CRITICAL | High | TASK-101 |
| TASK-103: Fix app layer infra leakage | P1 | M | HIGH | Low | TASK-105 |
| TASK-104: Break agents <-> engine circular import | P1 | M | HIGH | Medium | TASK-105 |
| TASK-105: Consolidate port definitions | P1 | S | HIGH | Low | — |
| TASK-106: Consolidate config files | P1 | M | MEDIUM | Low | — |
| TASK-201: Domain sub-packages from flat files | P2 | XL | HIGH | Medium | TASK-102 |
| TASK-202: Async-safe SQLite persistence | P2 | L | HIGH | Medium | TASK-101 |
| TASK-203: Unify pipeline systems | P2 | L | HIGH | Low | TASK-102 |
| TASK-204: Unify event bus | P2 | M | MEDIUM | Low | TASK-105 |
| TASK-301: Distributed tracing end-to-end | P3 | L | MEDIUM | Low | TASK-201 |
| TASK-302: Standardize logging | P3 | M | MEDIUM | Low | TASK-201 |
| TASK-303: Contract tests for all ports | P3 | M | HIGH | Low | TASK-201 |
| TASK-304: Horizontal scaling preparation | P3 | XL | LOW | High | TASK-202, TASK-204 |

---

## Target Architecture (End State)

```
orchestrator/
├── domain/              # Pure domain — models, ports, exceptions (ALREADY CLEAN)
│   ├── models.py        # Immutable dataclasses only — NO async, NO I/O
│   ├── ports.py         # All protocol definitions (CachePort, ModelProvider, etc.)
│   ├── events.py        # Domain event hierarchy
│   └── exceptions.py
│
├── application/         # Use cases — depends only on domain/ports (IN PROGRESS)
│   ├── budget_enforcer.py    # Budget with async behavior lives here
│   ├── critique_cycle.py     # Uses ModelProvider protocol, not UnifiedClient
│   ├── evaluator.py
│   ├── task_executor.py
│   └── ...
│
├── infrastructure/      # Concrete adapters (ALREADY CLEAN STRUCTURE)
│   ├── cache.py         # DiskCache implements CachePort
│   ├── llm_client.py    # UnifiedClient implements ModelProvider
│   ├── state.py         # StateManager implements StatePort
│   └── event_bus.py     # Concrete EventBus (unified after TASK-204)
│
├── engine/              # Thin mediator only (TARGET — needs surgery)
│   ├── engine.py        # Orchestrator: < 400 lines, wires ServiceContainer, runs pipeline
│   ├── container.py     # ServiceContainer.build() — the DI factory
│   └── pipeline_builder.py  # Constructs the 7-stage TaskPipeline
│
├── pipeline/            # 7-stage pipeline (ALREADY GOOD — promote to top-level)
│   ├── pipeline.py      # TaskPipeline, PipelineContext
│   └── stages/
│       ├── generate.py, critique.py, evaluate.py
│       ├── validate.py, preflight.py
│       ├── self_consistency.py, persuasion_defense.py
│
├── agents/              # Agent hierarchy (MOSTLY CLEAN)
│   ├── base.py, coordinator.py
│   ├── developer.py, reviewer.py, devops.py
│   ├── researcher.py, user.py, product_manager.py, qc.py
│
├── reasoning/           # ARA methods + Brain (EXTRACTED from flat)
│   ├── ara_pipelines.py, ara_execution_strategy.py
│   └── brain.py
│
├── codebase/            # Codebase analysis (EXTRACTED from flat)
├── routing/             # Model registry, selector, routing table (EXTRACTED from flat)
├── prompting/           # Prompt builder, compressor, enhancer (EXTRACTED from flat)
├── vcs/                 # Git integration (EXTRACTED from flat)
├── project_mgmt/        # Sprint, milestone, progress (EXTRACTED from flat)
├── appbuilder/          # App builder tools (EXTRACTED from flat)
│
├── interfaces/          # The driving side — entry points
│   ├── cli.py
│   ├── api_server.py
│   └── mcp_server.py
│
└── tests/               # Mirror the package structure, 80% coverage enforced
    ├── domain/
    ├── application/
    ├── infrastructure/
    ├── pipeline/
    └── ...
```

---

## Success Metrics

| Metric | Current | P0 Target | P1 Target | P2 Target |
|---|---|---|---|---|
| `engine.py` LOC | 5,036 | 5,036 | < 400 | < 400 |
| `_execute_task()` LOC | 1,247 | 1,247 | < 30 | < 30 |
| Flat files at root | 315 | 310 | 310 | < 50 |
| Test `fail_under` | 12% | 25% | 40% | 60% |
| Swallowed exceptions | 117 | < 108 | < 50 | < 10 |
| Circular imports | 4 | 4 | 0 | 0 |
| App layer infra leaks | 4 files | 4 | 0 | 0 |
| Port systems | 2 | 2 | 1 | 1 |
| Config files | 5 | 5 | 2 | 2 |
| `ServiceContainer` used | No | No | Yes | Yes |
| Max parallel tasks | 1 | 1 | 1 | 3 |

---

## Confidence Assessment

### Verified Findings (empirically measured)
- `engine.py`: 5,036 lines, 104 methods, 79 import deps, 37 `__init__` attributes
- `_execute_task()`: 1,247 lines
- `models.py` async violation: `asyncio` import + 4 async methods
- `ServiceContainer` defined in `engine_core/container.py` but not used by `engine.py`
- 315 flat .py files at top level
- Circular imports: `agents <-> engine`, `engine <-> meta_integration`, `models <-> policy`
- Application-layer imports `api_clients`: 4 files confirmed
- 117 swallowed exceptions across 51 files
- `fail_under = 12` in `pyproject.toml`
- `asyncio.run()` inside async context in `cli_nash.py` lines 169 and 225
- `domain/`: clean — 5 files, no infrastructure imports
- `infrastructure/`: 3 files, correct dependency direction

### HYPOTHESIS (patterns observed, not fully runtime-traced)
- The 9-agent system in `agents/` may not be connected to the main execution pipeline —
  runtime tracing required to confirm
- `KnowledgeGraph` (networkx in-memory) may lose data on restart if not persisted to disk
- MAP-Elites ARA method may lack timeout protection for its 3-generation loop
- `multi_tenant_gateway.py` isolation guarantees are untested

### Areas Requiring Further Investigation
- Runtime behavior of `MessageBus` under concurrent load
- Whether `canary_deployment.py` and `gradual_rollout.py` are in active production use
- Multi-tenant isolation guarantees in `multi_tenant_gateway.py`
- Actual API key management flow end-to-end

---

## Final Assessment

The AI Orchestrator has excellent architectural ambition. The stated design — hexagonal
architecture, 7-stage pipeline, protocol-first ports, DI container — is sound. Several modules
(`domain/`, `infrastructure/`, `engine_core/stages/`, `agents/`) demonstrate what the codebase
looks like when the architecture is respected.

The fundamental problem is that the refactoring was never completed, and organic feature growth
continued into the flat namespace while the structured subsystems were left half-wired. The result
is two parallel systems (one clean and correct, one monolithic and live), 315 ungoverned flat
files, and a God Object at the center that absorbed logic intended for the extracted layers.

**This is fixable.** The foundations are sound and the target architecture is well-understood. The
work is organizational and surgical, not conceptual. The `ServiceContainer` already exists. The
pipeline stages already exist. The domain layer is already clean. The refactoring plan above
completes what was already started.

*Next review checkpoint: After Phase 0 completion (TASK-001 through TASK-006).*

---

*Report generated: 2026-05-27*
*Auditor: Principal Engineer — Architecture Governance*
