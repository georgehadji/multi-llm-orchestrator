# Architectural Remediation Plan — Multi-LLM Orchestrator v7.0

**Target:** Transform the v6.0 monolith-with-folders into a Clean Architecture / Hexagonal system
**Based on:** Architectural Audit 2026-05-25 (composite score 5.20/10)
**Author:** Architectural Reaper
**Date:** 2026-05-25

---

## 0. Target Architecture

### Clean Architecture Layers (inside-out)

```
┌─────────────────────────────────────────────────────────────┐
│ ENTRYPOINTS (CLI, Gateway, IDE Backend, MCP Server)         │
│  orchestrator/entrypoints/{cli,gateway,ide,mcp}/            │
└────────────────────────┬────────────────────────────────────┘
                         │ depends on
┌────────────────────────▼────────────────────────────────────┐
│ APPLICATION LAYER (use cases, orchestration)                │
│  orchestrator/application/{orchestrator,decomposer,         │
│    critique_cycle, task_executor, batch_runner}             │
│  DEPENDS ON: domain ports (Protocols), NOT on infra         │
└────────────────────────┬────────────────────────────────────┘
                         │ depends on
┌────────────────────────▼────────────────────────────────────┐
│ DOMAIN LAYER (pure business logic, zero dependencies)       │
│  orchestrator/domain/                                       │
│    models.py          — enums, dataclasses, cost tables     │
│    budget.py          — Budget value object                 │
│    task_factory.py    — Task creation logic                 │
│    routing.py         — ROUTING_TABLE, FALLBACK_CHAIN       │
│    validators.py      — deterministic validation            │
│    ports.py           — CachePort, StatePort, EventPort     │
│    exceptions.py      — error hierarchy                     │
│  ZERO imports from: infrastructure, application, entrypoints│
│  ALLOWED imports: stdlib, pydantic, typing                  │
└────────────────────────┬────────────────────────────────────┘
                         │ implements
┌────────────────────────▼────────────────────────────────────┐
│ INFRASTRUCTURE LAYER (adapters, external I/O)               │
│  orchestrator/infrastructure/                               │
│    llm_client.py      — UnifiedClient (OpenRouter adapter)  │
│    cache.py           — DiskCache implements CachePort      │
│    state.py           — StateManager implements StatePort   │
│    circuit_breaker.py — CircuitBreaker (pure infra)         │
│    telemetry.py       — TelemetryCollector                  │
│    tracing.py         — OpenTelemetry adapter               │
│    nexus_search/      — Web search adapter                  │
│  IMPLEMENTS: domain ports (Protocols)                       │
│  DEPENDS ON: domain layer ONLY                              │
└─────────────────────────────────────────────────────────────┘
```

### Cross-Cutting Concerns (used by all layers)

```
orchestrator/crosscutting/
  config.py           — FeatureFlags, RuntimeConfig classes
  logging_config.py   — Structured logging setup
  hooks.py            — Event hooks (Observer pattern)
  events.py           — Unified event bus
```

### Key Pattern: Dependency Rule

```
entrypoints → application → domain ← infrastructure
                                ↑
                         crosscutting
```

The domain layer has **zero** imports from any other layer. Infrastructure implements domain ports. Application orchestrates using domain types + ports. Entrypoints wire everything together.

---

## 1. Phase 1: Break the Circular Import (Week 1, Days 1-2)

### Problem

`models.py:18` has `from .budget import Budget` — a domain-data module importing from a domain-logic module. This creates a circular dependency chain that forced the `__getattr__` lazy-load workaround in `__init__.py`.

### Solution: Invert the dependency — Budget becomes a pure domain type

#### Step 1.1 — Move `Budget` into `models.py`

The `Budget` class in `budget.py` is ~120 lines of value-object logic (reserve/commit/release, time tracking). It has no imports from `models.py`. Move it:

```python
# models.py — add after existing dataclasses
@dataclass
class Budget:
    """Async budget tracker with atomic reserve pattern."""
    max_usd: float = 8.0
    max_time_seconds: float = 10800.0
    spent_usd: float = 0.0
    _start_time: float | None = None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def reserve(self, estimated_cost: float) -> bool: ...
    async def commit(self, actual_cost: float) -> None: ...
    async def release(self, amount: float) -> None: ...
    @property
    def remaining_usd(self) -> float: ...
    def time_remaining(self) -> bool: ...
    def elapsed_seconds(self) -> float: ...
```

#### Step 1.2 — `budget.py` becomes a re-export shim

```python
# budget.py — backward-compat re-export
from .domain.models import Budget  # noqa: F401
```

#### Step 1.3 — Remove the circular import

Delete `from .budget import Budget` from `models.py` line 18. It's now defined in the same file.

#### Step 1.4 — Remove `__getattr__` lazy-load

Once the circular import is broken, the 23-symbol lazy-load map in `__init__.py` can be replaced with direct imports. This eliminates import-time overhead and makes IDE tooling work correctly.

#### Verification gate

```bash
python -c "from orchestrator import Orchestrator, Budget, Model, Task; print('No circular import')"
```

---

## 2. Phase 2: Establish Domain Layer (Week 1, Days 3-5)

### Problem

Domain types (`models.py`, `budget.py`, `validators.py`, `exceptions.py`, `ports.py`) are scattered in the root `orchestrator/` directory alongside infrastructure code (`cache.py`, `state.py`, `api_clients.py`). There's no layer separation.

### Solution: Create `orchestrator/domain/` package

#### Step 2.1 — Move pure-domain files

```
orchestrator/domain/
  __init__.py          # Re-exports all public symbols
  models.py            # ← from orchestrator/models.py
  budget.py            # ← from orchestrator/budget.py (now a shim)
  task_factory.py      # ← from orchestrator/task_factory.py
  validators.py        # ← from orchestrator/validators.py
  exceptions.py        # ← from orchestrator/exceptions.py
  ports.py             # ← from orchestrator/ports.py
  model_registry.py    # ← from orchestrator/model_registry.py
  routing.py            # NEW: ROUTING_TABLE, FALLBACK_CHAIN, COST_TABLE extracted from models.py
```

#### Step 2.2 — Split `models.py` (731 lines → ~300 lines of pure data)

Extract routing tables into `domain/routing.py`:

```python
# domain/routing.py — pure data, no behavior
ROUTING_TABLE: dict[TaskType, list[Model]] = {...}
FALLBACK_CHAIN: dict[Model, Model] = {...}
COST_TABLE: dict[Model, dict[str, float]] = {...}
DEFAULT_THRESHOLDS: dict[TaskType, float] = {...}
MAX_OUTPUT_TOKENS: dict[TaskType, int] = {...}
MODEL_MAX_TOKENS: dict[Model, int] = {...}
```

#### Step 2.3 — Backward-compat shims in `orchestrator/`

```python
# orchestrator/models.py — backward compat
from .domain.models import *  # noqa: F401, F403
from .domain.routing import *  # noqa: F401, F403
```

#### Verification gate

```bash
python -c "from orchestrator.domain.models import Task, TaskResult, Model; print('Domain layer clean')"
# ZERO imports from orchestrator infrastructure/application/entrypoints
```

---

## 3. Phase 3: Extract Infrastructure Layer (Week 2, Days 1-3)

### Problem

Infrastructure adapters (`cache.py`, `state.py`, `api_clients.py`, `circuit_breaker.py`) live in the root and are imported directly by application code — violating the dependency rule.

### Solution: Create `orchestrator/infrastructure/` package

#### Step 3.1 — Move infrastructure adapters

```
orchestrator/infrastructure/
  __init__.py
  llm_client.py         # ← from orchestrator/api_clients.py (UnifiedClient)
  cache.py              # ← from orchestrator/cache.py (DiskCache)
  state.py              # ← from orchestrator/state.py (StateManager)
  circuit_breaker.py    # ← from orchestrator/circuit_breaker.py
  resilience.py         # ← from orchestrator/resilience.py
  retry_utils.py        # ← from orchestrator/retry_utils.py
  telemetry.py          # ← from orchestrator/telemetry.py
  telemetry_store.py    # ← from orchestrator/telemetry_store.py
  tracing.py            # ← from orchestrator/tracing.py
  semantic_cache.py     # ← from orchestrator/semantic_cache.py
  cache_optimizer.py    # ← from orchestrator/cache_optimizer.py
  concurrency.py        # ← from orchestrator/concurrency_controller.py
```

#### Step 3.2 — Wire ports to adapters via explicit registration

Each infrastructure adapter explicitly declares which port it implements:

```python
# infrastructure/cache.py
from ..domain.ports import CachePort

class DiskCache:  # implicitly satisfies CachePort via structural subtyping
    """SQLite-backed response cache."""
    async def get(self, ...) -> Any | None: ...
    async def put(self, ...) -> None: ...
    async def close(self) -> None: ...
```

#### Step 3.3 — Application layer depends on ports, NOT adapters

```python
# application/orchestrator.py
from ..domain.ports import CachePort, StatePort, EventPort

class Orchestrator:
    def __init__(
        self,
        cache: CachePort,          # Protocol, not DiskCache
        state_manager: StatePort,  # Protocol, not StateManager
        event_bus: EventPort,      # Protocol, not ProjectEventBus
        ...
    ):
        self.cache = cache
        self.state_mgr = state_manager
```

#### Verification gate

```bash
# Zero imports from infrastructure in domain/
grep -r "from.*infrastructure" orchestrator/domain/ && echo "FAIL" || echo "PASS"
# Zero imports from application in infrastructure/
grep -r "from.*application" orchestrator/infrastructure/ && echo "FAIL" || echo "PASS"
```

---

## 4. Phase 4: Decompose the God Class (Week 2, Days 3-5 + Week 3)

### Problem

`engine.py` is 5,251 lines with a single `Orchestrator.__init__` that instantiates 50+ components. Tasks execute inline. There's a parallel `engine_core/core.py` duplicating the same pattern.

### Solution: Extract application services, eliminate duplication

#### Step 4.1 — Create `orchestrator/application/` package

```
orchestrator/application/
  __init__.py
  orchestrator.py       # Thin facade (~200 lines): wires services, delegates
  task_executor.py      # Single-task execution with critique cycle
  decomposer.py         # Project → task DAG decomposition
  critique_cycle.py     # Generate → critique → revise loop
  fallback_handler.py   # Model health, circuit breaker, failover
  budget_enforcer.py    # Budget monitoring, mid-task enforcement
  dependency_resolver.py # DAG resolution, topological sort, context building
  batch_runner.py       # Parallel task execution (from delegation/)
  context_compressor.py # LLM summarization for dependency context
  pattern_learner/      # Pattern extraction, storage, curation, injection
  memory/               # Memory manager, consolidation
```

#### Step 4.2 — `Orchestrator` becomes a thin facade

```python
# application/orchestrator.py — ~200 lines
@dataclass
class OrchestratorConfig:
    max_concurrency: int = 3
    max_parallel_tasks: int = 1
    context_truncation_limit: int = 40000

class Orchestrator:
    """Thin facade. Wires services. Delegates all work."""

    def __init__(
        self,
        cache: CachePort,
        state_manager: StatePort,
        llm_client: "UnifiedClient",
        config: OrchestratorConfig = field(default_factory=OrchestratorConfig),
        budget: Budget | None = None,
    ):
        self.budget = budget or Budget()
        self._config = config
        self._cache = cache
        self._state_mgr = state_manager
        self._client = llm_client

        # Services — each is independently testable
        self._decomposer = DecomposerService(client=self._client)
        self._resolver = DependencyResolver(
            limit=config.context_truncation_limit,
            compressor=ContextCompressor(client=self._client),
        )
        self._executor = TaskExecutor(
            client=self._client,
            critique=CritiqueCycle(client=self._client),
            fallback=FallbackHandler(client=self._client),
            budget_enforcer=BudgetEnforcer(self.budget),
            guardrails=ToolCallGuardrailController(),
        )
        self._batch_runner = BatchRunner(max_concurrent=config.max_parallel_tasks)

        self.results: dict[str, TaskResult] = {}
        self._results_lock = asyncio.Lock()

    async def run_project(
        self, description: str, criteria: str
    ) -> ProjectState:
        tasks = await self._decomposer.decompose(description, criteria)
        order = self._resolver.topological_sort(tasks)

        for level in self._resolver.group_by_level(order):
            results = await self._batch_runner.run_level(
                level, tasks, self._executor, self._resolver
            )
            async with self._results_lock:
                self.results.update(results)

        return self._build_state(tasks, description, criteria)
```

#### Step 4.3 — Delete `engine_core/core.py`

The `OrchestratorCore` class duplicates the same pattern. All its logic has been extracted into `application/` services. Remove the file. Update any imports.

#### Step 4.4 — Delete `engine_with_events.py` and `engine_deps.py`

Parallel engine variants. The new `application/orchestrator.py` subsumes their functionality.

#### Step 4.5 — Consolidate `services/` into `application/`

- `services/executor.py` → `application/task_executor.py`
- `services/evaluator.py` → `application/critique_cycle.py`
- `services/generator.py` → `application/decomposer.py`
- `services/observability.py` → `infrastructure/telemetry.py`

Delete the `services/` directory.

#### Verification gate

```bash
# orchestrator.py ≤ 300 lines
wc -l orchestrator/application/orchestrator.py
# engine.py is a backward-compat re-export shim ≤ 30 lines
wc -l orchestrator/engine.py
```

---

## 5. Phase 5: Centralize Configuration (Week 3, Days 1-2)

### Problem

13+ files call `os.environ.get()` directly. Feature flags are scattered. `config.py` centralizes magic numbers but not environment-derived configuration.

### Solution: Single `FeatureFlags` class + `Settings` pydantic model

#### Step 5.1 — Create `crosscutting/config.py`

```python
# crosscutting/config.py
from pydantic_settings import BaseSettings

class FeatureFlags(BaseSettings):
    """All feature flags — single source of truth."""
    context_compression: bool = False
    pattern_injection: bool = False
    batch_parallelism: bool = False
    plugin_sandbox: bool = True
    audit_log: bool = True

    model_config = {"env_prefix": "ORCH_", "env_file": ".env"}

class OrchestratorSettings(BaseSettings):
    """Runtime settings."""
    max_concurrency: int = 3
    max_parallel_tasks: int = 3
    default_budget_usd: float = 10.0
    default_timeout_seconds: int = 10800
    rate_limit_per_minute: int = 60
    context_truncation_limit: int = 40000

    model_config = {"env_prefix": "ORCH_", "env_file": ".env"}

# Singleton — created once at startup
flags = FeatureFlags()
settings = OrchestratorSettings()
```

#### Step 5.2 — Replace all `os.environ.get()` calls with `config.flags`

```python
# Before (engine.py):
enabled=os.environ.get("ORCH_CONTEXT_COMPRESSION", "").lower() == "true",

# After:
from ..crosscutting.config import flags
enabled=flags.context_compression,
```

#### Step 5.3 — Fix the crash-risk bare access

```python
# Before (issue_tracking.py):
api_key = os.environ["LINEAR_API_KEY"]  # CRASHES if unset

# After:
api_key = os.environ.get("LINEAR_API_KEY")
if not api_key:
    logger.warning("LINEAR_API_KEY not set — issue tracking disabled")
    return
```

#### Verification gate

```bash
# Zero bare os.environ access outside crosscutting/config.py
grep -r "os\.environ\[" orchestrator/ --include="*.py" \
  | grep -v crosscutting/config.py \
  | grep -v "__init__" \
  && echo "FAIL: scattered env access" || echo "PASS"
```

---

## 6. Phase 6: Pattern Consistency (Week 3, Days 3-5)

### Repository Pattern

`StateManager` and `TelemetryStore` are already SQLite-backed stores. Formalize them as Repositories:

```python
# domain/ports.py
class ProjectRepository(Protocol):
    async def save(self, project_id: str, state: ProjectState) -> None: ...
    async def load(self, project_id: str) -> ProjectState | None: ...
    async def save_checkpoint(self, project_id: str, task_id: str, state: ProjectState) -> None: ...

class PatternRepository(Protocol):
    async def insert(self, pattern: ExtractedPattern, ...) -> bool: ...
    async def find_similar(self, task_type: str, prompt: str, limit: int) -> list[dict]: ...
    async def archive_stale(self, days: int) -> int: ...
    async def get_active_by_type(self) -> dict[str, list[dict]]: ...
```

### Strategy Pattern

Model selection is already a strategy. Formalize it:

```python
# domain/routing.py
class ModelSelectionStrategy(Protocol):
    def select(self, task_type: TaskType, available: list[Model]) -> Model: ...

class CostOptimizedStrategy: ...
class QualityFirstStrategy: ...
class AdaptiveStrategy: ...
```

### Observer Pattern (Event Bus)

Use the unified event bus consistently across all services:

```python
self._event_bus.publish(TaskStartedEvent(task_id=..., model=...))
self._event_bus.publish(TaskCompletedEvent(task_id=..., score=...))
self._event_bus.publish(BudgetWarningEvent(remaining=...))
```

### Factory Pattern

`TaskFactory.create()` already exists. Extend it:

```python
# domain/task_factory.py
class TaskFactory:
    @staticmethod
    def create(id: str, task_type: TaskType, prompt: str, **kwargs) -> Task:
        defaults = {
            "acceptance_threshold": DEFAULT_THRESHOLDS[task_type],
            "max_output_tokens": MAX_OUTPUT_TOKENS[task_type],
            "max_iterations": get_max_iterations(task_type),
        }
        return Task(id=id, type=task_type, prompt=prompt, **{**defaults, **kwargs})
```

### Adapter Pattern

Already in place via `UnifiedClient` wrapping OpenRouter. Maintain in `infrastructure/llm_client.py`.

### Circuit Breaker Pattern

Already implemented. Move to `infrastructure/circuit_breaker.py` with no logic changes.

---

## 7. Phase 7: Eliminate Anti-Patterns (Week 4, Days 1-2)

### 7.1 — Remove Duplicate Files

| File | Action |
|------|--------|
| `architecture_rules_fixed.py` | Delete — merge fixes into `architecture_rules.py` |
| `output_writer_trimmed.py` | Delete — merge into `output_writer.py` |
| `state_fix_bug001.py` | Delete — fix applied to `state.py` |
| `engine_with_events.py` | Delete — subsumed by `application/orchestrator.py` |
| `engine_deps.py` | Delete — no longer needed after layer separation |
| `engine_core/core.py` | Delete — duplicate orchestration |

### 7.2 — Eliminate Singletons

```python
# BEFORE (nash_infrastructure_v2.py):
_instance = None
def get_instance():
    global _instance
    if _instance is None:
        _instance = AsyncIOManager()
    return _instance

# AFTER: Inject via constructor
class Orchestrator:
    def __init__(self, io_manager: AsyncIOManager, ...):
        self._io = io_manager
```

Apply to all 5 singletons: `AsyncIOManager`, `MetricsRegistry`, `CapabilityLogger`, `CommandCenterServer`, `GlobalConcurrencyController`.

### 7.3 — Replace Bare Exception Catches

```python
# BEFORE:
except Exception:
    pass

# AFTER:
except (ValueError, KeyError, ConnectionError) as exc:
    logger.warning("Expected failure in %s: %s", context, exc)
```

Systematic rule: every bare `except Exception:` must either log the exception or be replaced with specific types from `domain/exceptions.py`.

### 7.4 — Standardize Error Handling

```python
# domain/exceptions.py — all application errors
class OrchestratorError(Exception): ...
class ModelUnavailableError(OrchestratorError): ...
class BudgetExceededError(OrchestratorError): ...
class TaskTimeoutError(OrchestratorError): ...
class ValidationFailedError(OrchestratorError): ...
class CircuitBreakerOpenError(OrchestratorError): ...
```

---

## 8. Phase 8: Test Architecture (Week 4, Days 3-5)

### Problem

12% coverage. 25 ignored test files. Tests import concrete classes, not ports.

### Solution

#### Step 8.1 — Test pyramid

```
tests/
  unit/                  # Fast, no I/O, no network
    domain/              # Tests for models, budget, validators, routing tables
    application/         # Tests for services with mocked ports
  integration/           # Real SQLite, mocked LLM
    infrastructure/      # DiskCache, StateManager, CircuitBreaker
  e2e/                   # Full pipeline with live (or recorded) API
    test_full_run.py
```

#### Step 8.2 — Use NullAdapters for unit tests

```python
# tests/unit/application/test_task_executor.py
async def test_execute_task_with_mocks():
    cache = NullCache()
    state = NullState()
    client = MockUnifiedClient()

    executor = TaskExecutor(
        client=client,
        critique=CritiqueCycle(client=client),
        fallback=FallbackHandler(client=client),
    )

    result = await executor.execute(task, ...)
    assert result.score > 0.7
```

#### Step 8.3 — Un-ignore and fix existing tests

25 tests are ignored by `pyproject.toml`. Triage:
- Tests that pass: un-ignore immediately
- Tests with minor import errors: fix the import paths
- Tests testing deleted features: archive them

#### Step 8.4 — Coverage ratchet

```toml
# pyproject.toml
[tool.coverage.report]
fail_under = 25  # Phase 1 target (up from 12)
# Phase 2: raise to 40
# Phase 3: raise to 50
```

---

## 9. SOLID Principles Enforcement

### Single Responsibility
- **BEFORE**: `engine.py` handles decomposition + execution + critique + budget + caching + patterns + memory + ARA
- **AFTER**: Each `application/` service does exactly one thing. `Orchestrator` only wires and delegates.

### Open/Closed
- `ModelSelectionStrategy` open for extension (new strategies), closed for modification
- Validators in `domain/validators.py` addable without changing `TaskExecutor`

### Liskov Substitution
- `NullCache`, `NullState`, `NullEventBus` must be fully substitutable for their real counterparts
- Any `CachePort` implementation must behave identically

### Interface Segregation
- `ports.py` already has focused Protocols (CachePort ~3 methods, StatePort ~3 methods)
- No "God Protocol" with 20 methods

### Dependency Inversion
- Application → domain ports (abstractions)
- Infrastructure → implements domain ports
- Entrypoints → wire concrete implementations into abstractions

---

## 10. Migration Sequence & Risk Matrix

| Phase | Step | Risk | Rollback | Depends On |
|-------|------|------|----------|------------|
| 1 | Move Budget into models.py | LOW — backward-compat shim | Revert shim | — |
| 1 | Remove __getattr__ lazy-load | LOW — direct imports | Re-add __getattr__ | Step 1.1 |
| 2 | Create domain/ package | LOW — backward-compat shims | Delete shims | Phase 1 |
| 2 | Split models → models + routing | LOW — re-exports | Revert split | Phase 1 |
| 3 | Create infrastructure/ package | MEDIUM — import path changes | Revert moves | Phase 2 |
| 3 | Application depends on ports | MEDIUM — constructor changes | Revert to concrete | Phase 2 |
| 4 | Extract application services | HIGH — engine.py hot path | Keep engine.py shim | Phase 3 |
| 4 | Delete engine_core/core.py | MEDIUM — check consumers | Restore from git | Phase 4.1 |
| 5 | Centralize FeatureFlags | LOW — additive change | Keep old env reads | — |
| 6 | Formalize Repository/Strategy | LOW — additive | Remove Protocol defs | Phase 3 |
| 7 | Delete duplicate files | LOW — git history | Restore from git | — |
| 7 | Eliminate singletons | MEDIUM — constructor changes | Inject as before | Phase 3 |
| 8 | Test pyramid + coverage ratchet | LOW — additive | Lower fail_under | Phase 4 |

---

## 11. Verification Gates

Each phase must pass these gates before proceeding:

```bash
# Phase 1: No circular import
python -c "from orchestrator.domain.models import Budget, Task, Model; print('OK')"

# Phase 2: Domain has zero infra imports
grep -r "from.*infrastructure\|import.*api_clients\|import.*DiskCache" \
  orchestrator/domain/ && exit 1 || echo "Domain clean"

# Phase 3: Application depends on ports, not adapters
grep -r "DiskCache\|StateManager" orchestrator/application/ \
  && exit 1 || echo "Application clean"

# Phase 4: orchestrator.py ≤ 300 lines
test $(wc -l < orchestrator/application/orchestrator.py) -le 300 || exit 1

# Phase 5: No bare os.environ outside config
grep -r "os\.environ\[" orchestrator/ --include="*.py" \
  | grep -v config.py | grep -v __init__ \
  && exit 1 || echo "Config centralized"

# Phase 7: No *_fixed.py files
find orchestrator/ -name "*_fixed.py" -o -name "*_v2.py" \
  | test $(wc -l) -eq 0 || exit 1

# Phase 8: Coverage ≥ 25%
pytest --cov=orchestrator --cov-report=term --cov-fail-under=25
```

---

## 12. Key Programming Paradigms Adopted

| Paradigm | Where | Why |
|----------|-------|-----|
| **Dependency Injection** | All constructors | Testability, loose coupling |
| **Protocol-based polymorphism** | `ports.py` | Structural subtyping — no ABC boilerplate |
| **Repository pattern** | `StateManager`, `PatternStore` | Abstract persistence behind domain interfaces |
| **Strategy pattern** | `ModelSelectionStrategy` | Pluggable model selection algorithms |
| **Observer pattern** | `EventPort` / unified event bus | Decoupled progress reporting |
| **Factory pattern** | `TaskFactory.create()` | Consistent task construction with routing defaults |
| **Adapter pattern** | `UnifiedClient` | Single interface for multiple LLM providers |
| **Circuit Breaker** | `CircuitBreaker` | Fail-fast on provider degradation |
| **Facade pattern** | `application/orchestrator.py` | Thin orchestration facade |
| **Value Object** | `Budget`, `Task`, `TaskResult` | Immutable domain concepts |
| **Null Object** | `NullCache`, `NullState` | Test doubles without mocking frameworks |

---

## 13. Completion Criteria

The remediation is complete when:

1. **Layer dependency rule** enforced: domain ← infrastructure, application → domain ports, entrypoints → application
2. **`engine.py`** ≤ 30 lines (backward-compat re-export)
3. **`application/orchestrator.py`** ≤ 300 lines, delegates to 6+ focused services
4. **Zero circular imports** — `__getattr__` lazy-load removed from `__init__.py`
5. **All feature flags** read from `crosscutting/config.FeatureFlags`, not `os.environ`
6. **Zero singleton patterns** — all state injected via constructors
7. **Zero `*_fixed.py` / `*_v2.py`** duplicate files
8. **All bare `except Exception:`** replaced with specific types or logged
9. **Coverage ≥ 25%** (ratchet to 50%)
10. **All 6 AI Orchestrator criteria** from the audit pass at PASS level

---

## Appendix A: Current vs Target File Map

| Current File | Target File | Action |
|-------------|------------|--------|
| `engine.py` (5251L) | `application/orchestrator.py` (~200L) | Extract, slim, shim |
| `engine_core/core.py` | DELETE | Duplicate, merge into application |
| `engine_with_events.py` | DELETE | Parallel variant |
| `engine_deps.py` | DELETE | Parallel import surface |
| `models.py` (731L) | `domain/models.py` (~300L) + `domain/routing.py` (~200L) | Split, move |
| `budget.py` | `domain/models.py` (Budget class inline) | Merge into models |
| `api_clients.py` (691L) | `infrastructure/llm_client.py` | Move, rename |
| `cache.py` | `infrastructure/cache.py` | Move |
| `state.py` | `infrastructure/state.py` | Move |
| `circuit_breaker.py` | `infrastructure/circuit_breaker.py` | Move |
| `resilience.py` | `infrastructure/resilience.py` | Move |
| `retry_utils.py` | `infrastructure/retry_utils.py` | Move |
| `ports.py` | `domain/ports.py` | Move |
| `validators.py` | `domain/validators.py` | Move |
| `exceptions.py` | `domain/exceptions.py` | Move |
| `config.py` | `crosscutting/config.py` | Move, extend |
| `services/` | DELETE (merge into `application/`) | Consolidate |
| `architecture_rules_fixed.py` | DELETE (merge into `architecture_rules.py`) | Deduplicate |
| `output_writer_trimmed.py` | DELETE (merge into `output_writer.py`) | Deduplicate |
| `state_fix_bug001.py` | DELETE (fix applied to `state.py`) | Deduplicate |

## Appendix B: Audit Score Trajectory

| Dimension | Before (v6.0) | After Phase 4 | After Phase 8 |
|-----------|---------------|---------------|---------------|
| Pattern Consistency | 5/10 | 7/10 | 8/10 |
| Boundary Integrity | 4/10 | 7/10 | 9/10 |
| Observability | 6/10 | 6/10 | 7/10 |
| Resilience | 7/10 | 7/10 | 8/10 |
| Testability | 4/10 | 5/10 | 7/10 |
| Security Posture | 6/10 | 7/10 | 7/10 |
| **Composite** | **5.20** | **6.55** | **7.75** |
