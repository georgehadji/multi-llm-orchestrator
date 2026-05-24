# God File Refactoring Plan — `orchestrator/engine.py`

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-23  
> **Version:** 1.0  
> **Status:** Plan — awaiting implementation

---

## Executive Summary

`orchestrator/engine.py` is the central nervous system of the Multi-LLM Orchestrator. It implements the full generate → critique → revise → evaluate pipeline, model routing, budget enforcement, circuit breaking, dashboard notifications, telemetry, context management, state persistence, preflight checks, architecture rule generation, event hooks, and TDD-first generation — all in a single 5,313-line class with 97 methods.

This plan decomposes the monolith into a coherent module structure using the **Strangler Fig** pattern — each extraction is an independent, revertible, testable change. No big-bang rewrites.

### Current metrics

| Metric | Value |
|---------|-------|
| Total lines | 5,313 |
| Methods in `Orchestrator` class | 97 |
| Top-level functions | 105 |
| Import statements | 122 |
| Distinct concern categories | 18 |
| Largest method (`_execute_task`) | 1,254 lines (24%) |
| Top 5 methods | 2,286 lines (43%) |
| Methods <10 lines | 41 (42% of methods) |
| "Misc utilities" bucket methods | 40 (41% of methods) |

### Target metrics

| Metric | Before | After | Reduction |
|---------|--------|-------|-----------|
| engine.py lines | 5,313 | ~1,800 | 66% |
| Methods in Orchestrator | 97 | ~25 | 74% |
| Import statements | 122 | ~40 | 67% |
| Largest method | 1,254 | ~200 | 84% |
| Concern categories | 18 | ~5 | 72% |

---

## Architecture of the Problem

### Concern sprawl (pre-refactoring)

```
Lifecycle (init/enter/exit)       5 meth    436 lines (  8%)
Public API (run/execute)          4 meth    358 lines (  7%)
Decomposition + parsing           6 meth    685 lines ( 13%)
Task execution core               4 meth  1,589 lines ( 30%)  ← worst
Model/Provider routing            7 meth    156 lines (  3%)
Evaluation + scoring              1 meth     52 lines (  1%)
Budget + cost                     2 meth     32 lines (  1%)
Dashboard integration             5 meth     33 lines (  1%)
Context management                2 meth    136 lines (  3%)
State persistence                 2 meth     57 lines (  1%)
Telemetry + metrics               5 meth    110 lines (  2%)
Event hooks                       3 meth     33 lines (  1%)
Validation + testing              5 meth    137 lines (  3%)
Cleanup + background tasks        4 meth    120 lines (  2%)
Preflight checks                  3 meth    147 lines (  3%)
Architecture rules                1 meth     96 lines (  2%)
Optimization helpers              7 meth    126 lines (  2%)
Misc utilities                   40 meth    619 lines ( 12%)
```

### Root causes

1. **`_execute_task()` is 1,254 lines.** It runs the full generate → critique → revise → evaluate loop inline, plus model selection, circuit breaker logic, budget checks, plateau detection, deterministic validation, self-consistency retries, preflight checks, and test validation. This is an 8-stage pipeline implemented as a single method.

2. **`__init__()` wires 30+ collaborators.** Every optional feature (TokenOptimizer, SessionWatcher, PersonaManager, RedTeamFramework, HybridSearchPipeline, A2AManager, RateLimiter, etc.) is instantiated unconditionally, even when unused.

3. **122 imports, 180 lines of try/except gating.** The import block alone is a dependency graph that should live in its own module.

4. **40 methods in "misc utilities."** No single concern — methods scattered across the class without co-location.

---

## Strategy: Strangler Fig + Protocol Narrowing

We use the **Strangler Fig** pattern: extract one concern at a time into a new module, keep the old Orchestrator delegating to it, then delete the old methods once callers migrate. Each extraction is a standalone PR.

We complement this with **Interface Segregation via Protocols**: instead of passing `self` (the entire Orchestrator) to collaborators, we define narrow role interfaces.

### Strangler Fig in practice

```
Phase 1:  engine.py imports from engine_deps.py
Phase 2:  engine.py delegates _decompose() to Decomposer class
Phase 3:  engine.py delegates validation to TaskValidator class
Phase 4:  engine.py delegates architecture rules to Architect class
Phase 5:  engine.py delegates _execute_task to TaskPipeline
Phase 6:  engine.py __init__ delegates to ServiceContainer
Phase 7:  Collaborators receive Protocols instead of Orchestrator
```

At each step, the old code remains in engine.py until the new module is verified. Then we delete the old methods. Each step is a single, revertible commit.

---

## Phased Implementation Plan

---

### Phase 1: Extract Import Graph → `engine_deps.py`

**Target:** 180 lines of try/except imports  
**Risk:** None (pure relocation)  
**New file:** `orchestrator/engine_deps.py`  
**Effort:** 1 day

Move ALL optional/try-except imports and `HAS_*` flags into a dedicated module. engine.py imports from it.

```python
# orchestrator/engine_deps.py
"""All optional dependency imports for the Orchestrator engine.

Every try/except import block and corresponding HAS_* flag lives here.
engine.py does `from .engine_deps import *` and gets everything.
"""

# Test validation
try:
    from .test_validator import TestValidator, validate_and_generate_test
    HAS_TEST_VALIDATOR = True
except ImportError:
    HAS_TEST_VALIDATOR = False
    TestValidator = None
    validate_and_generate_test = None

# Code validation
try:
    from .code_validator import validate_code, extract_code_from_llm_response
    HAS_CODE_VALIDATOR = True
except ImportError:
    HAS_CODE_VALIDATOR = False
    validate_code = None
    extract_code_from_llm_response = None

# ... all other try/except blocks (cost_optimization, hooks, memory_tier,
#     persona, planner, preflight, rate_limiter, red_team, reranker,
#     session_lifecycle, session_watcher, task_verifier, telemetry,
#     telemetry_store, token_optimizer, tracing, test_first_generator,
#     diff_generator, context_condensing, context_dedup, context_truncator,
#     test_fixer, pre_submission_testing)
```

**Engine change:** Replace 180 lines of imports with:

```python
from .engine_deps import *
from .project_context import ProjectContext
```

**Verification:** `python -m orchestrator --help` still works. All `HAS_*` flags accessible.

---

### Phase 2: Extract Decomposition → `engine_core/decomposer.py`

**Target:** `_decompose`, `_parse_decomposition`, `_try_parse_partial_json_array`, `_select_decomposition_model`, `_get_fast_decomposition_model`  
**Lines extracted:** ~685  
**Risk:** Low (decomposition has a stable interface: returns `dict[str, Task]`)  
**New file:** `orchestrator/engine_core/decomposer.py`  
**Effort:** 2 days

```python
# orchestrator/engine_core/decomposer.py
from dataclasses import dataclass
from typing import Optional
from ..api_clients import UnifiedClient
from ..models import Model, Task, TaskType
from ..resilience import ResiliencePolicy
from ..project_context import ProjectContext
from ..tracing import Tracer


@dataclass
class DecomposerConfig:
    max_retries: int = 2
    timeout_seconds: int = 120


class Decomposer:
    """Project decomposition into atomic tasks.

    Handles the full decomposition lifecycle: model selection, prompt
    construction, LLM call, JSON parsing, and partial-result recovery.
    """

    def __init__(
        self,
        client: UnifiedClient,
        tracer: Tracer | None = None,
        config: DecomposerConfig | None = None,
    ) -> None:
        self._client = client
        self._tracer = tracer
        self._config = config or DecomposerConfig()

    async def decompose(
        self,
        project: str,
        criteria: str,
        app_profile=None,
        project_context: ProjectContext | None = None,
        policy: ResiliencePolicy | None = None,
    ) -> dict[str, Task]:
        """Break project into ordered task dict. Never raises.

        Returns empty dict on irrecoverable failure.
        """
        valid_types = [t.value for t in TaskType]
        # ... decomposition logic from current _decompose()
        return {}
```

**Engine change:** Add `self._decomposer = Decomposer(client=self.client, tracer=_tracer)` in `__init__`. Update `_decompose()` to delegate. Once verified, delete the ~685 lines of inline methods from engine.py and have `self._generator` point directly to `self._decomposer.decompose`.

**Verification:** `_decompose()` returns identical `dict[str, Task]`. GeneratorService continues to work. Dry-run produces same task list.

---

### Phase 3: Extract Preflight + Validation → `engine_core/validator.py`

**Target:** `_run_preflight_check`, `_validate_syntax_streaming`, `_validate_syntax_batch`, `_filter_validators_for_task`, test validation integration  
**Lines extracted:** ~284  
**Risk:** Low (validation is side-effect-free, returns booleans/scores)  
**New file:** `orchestrator/engine_core/validator.py`  
**Effort:** 1 day

```python
@dataclass
class ValidationResult:
    passed: bool
    score: float
    errors: list[str]


class TaskValidator:
    """Deterministic + preflight validation for task outputs."""

    def __init__(self, preflight=None):
        self._preflight = preflight

    async def validate(
        self, task, output, score, model
    ) -> ValidationResult:
        """Run all validation gates on a task output."""
        ...

    def syntax_check(self, code: str) -> bool:
        """Compile-check Python code for syntax errors."""
        ...

    def _filter_validators(self, task) -> list:
        """Return validators applicable to this task type."""
        ...
```

**Verification:** Existing tests for validation still pass. Output quality unchanged.

---

### Phase 4: Extract Architecture Rules → `engine_core/architect.py`

**Target:** `_generate_architecture_rules`  
**Lines extracted:** ~96  
**Risk:** None (isolated method, called once per project)  
**New file:** `orchestrator/engine_core/architect.py`  
**Effort:** 0.5 day

Move the architecture rule generation into the existing `engine_core/` package alongside `critique_cycle.py` and `task_executor.py`.

**Verification:** Architecture rules output identical.

---

### Phase 5: Convert `_execute_task` → `TaskPipeline` with Stages

**Target:** `_execute_task` (1,254 lines), `_execute_all` (328 lines)  
**Lines extracted:** ~1,582  
**Risk:** Medium (core execution path)  
**New files:** 8 (pipeline + 7 stages)  
**Effort:** 4 days

This is the highest-value extraction. The 1,254-line method is an 8-stage pipeline implemented as one function. We decompose it into pluggable stages.

**New module structure:**

```
orchestrator/engine_core/
├── pipeline.py          # TaskPipeline + PipelineContext
├── stages/
│   ├── __init__.py
│   ├── generate.py      # GenerateStage
│   ├── critique.py      # CritiqueStage
│   ├── revise.py        # ReviseStage
│   ├── evaluate.py      # EvaluateStage
│   ├── validate.py      # ValidateStage (deterministic)
│   ├── preflight.py     # PreflightStage
│   └── self_consistency.py  # SelfConsistencyStage
```

**Core abstraction:**

```python
@dataclass
class PipelineContext:
    """Mutable state carried through pipeline stages."""
    task: Task
    attempt: int = 0
    output: str = ""
    score: float = 0.0
    critique: str = ""
    model: Model | None = None
    reviewer_model: Model | None = None
    tokens_used: dict
    cost_usd: float = 0.0
    should_abort: bool = False
    abort_reason: str = ""
    attempt_history: list


class PipelineStage(Protocol):
    """A single stage in the task execution pipeline."""

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Transform the context. Set ctx.should_abort to stop."""
        ...


class TaskPipeline:
    """Composable task execution pipeline."""

    def __init__(self, stages: list[PipelineStage]):
        self._stages = stages

    async def run(self, ctx: PipelineContext) -> PipelineContext:
        for stage in self._stages:
            ctx = await stage.process(ctx)
            if ctx.should_abort:
                break
        return ctx
```

**Example stage implementations:**

```python
class GenerateStage:
    def __init__(self, client, budget, selector):
        self._client = client
        self._budget = budget
        self._selector = selector

    async def process(self, ctx):
        model = ctx.task.preferred_model or self._selector.select(ctx.task.type)
        response = await self._client.call(
            model=model,
            prompt=ctx.task.prompt,
            max_tokens=ctx.task.max_output_tokens,
        )
        ctx.output = response.text
        ctx.model = model
        ctx.cost_usd += response.cost_usd
        ctx.tokens_used["input"] = response.usage.input_tokens
        ctx.tokens_used["output"] = response.usage.output_tokens
        return ctx


class SelfConsistencyStage:
    def __init__(self, max_attempts=2, quality_threshold=0.70):
        self._max_attempts = max_attempts
        self._threshold = quality_threshold

    async def process(self, ctx):
        if ctx.score >= self._threshold:
            return ctx
        if ctx.attempt >= self._max_attempts:
            return ctx
        ctx.attempt += 1
        ctx.task.revision_context = ctx.critique
        ctx.task.preferred_model = FALLBACK_CHAIN.get(ctx.model, ctx.model)
        ctx.should_abort = True
        ctx.abort_reason = "retry_for_quality"
        return ctx
```

**Engine change:** `_execute_task` shrinks to ~60 lines:

```python
async def _execute_task(self, task, policy=None):
    ctx = PipelineContext(task=task)
    while True:
        ctx = await self._pipeline.run(ctx)
        if ctx.abort_reason != "retry_for_quality":
            break
    return TaskResult(task_id=task.id, output=ctx.output, ...)
```

**Verification:** All existing integration tests pass. Output quality unchanged.

---

### Phase 6: Split `__init__` — Lazy Wiring + Service Container

**Target:** `__init__` (266 lines, 30+ collaborators)  
**Lines extracted:** ~200  
**Risk:** Medium (touches initialization order)  
**New file:** `orchestrator/engine_core/container.py`  
**Effort:** 2 days

**Strategy: Lazy accessory wiring**

These collaborators are wired unconditionally but used only in specific flows:
TokenOptimizer, SessionWatcher, PersonaManager, RedTeamFramework,
A2AManager, RateLimiter, SessionLifecycleManager, HybridSearchPipeline,
QueryExpander.

They get property-based lazy initialization.

**Group remaining wiring into a `ServiceContainer`:**

```python
# orchestrator/engine_core/container.py
@dataclass
class ServiceContainer:
    budget: Budget
    cache: DiskCache
    state_mgr: StateManager
    client: UnifiedClient
    executor: ExecutorService
    evaluator: EvaluatorService
    generator: GeneratorService
    decomposer: Decomposer
    pipeline: TaskPipeline
    project_context: ProjectContext
    # ... remaining wired collaborators

    @classmethod
    def build(cls, **kwargs):
        """Factory method that wires everything."""
        ...
```

**Verification:** `Orchestrator()` constructs identically.

---

### Phase 7: Interface Segregation — Narrow Protocols

**Target:** Replace `self` references with narrow Protocol types  
**Risk:** High (touches every collaborator's constructor)  
**New file:** `orchestrator/engine_core/protocols.py`  
**Effort:** 3 days

Define role interfaces so collaborators only see what they need:

```python
class ModelProvider(Protocol):
    def get_available_models(self, task_type: TaskType) -> list[Model]: ...
    @property
    def api_health(self) -> dict[Model, bool]: ...

class BudgetTracker(Protocol):
    async def reserve(self, amount: float, phase: str) -> str: ...
    async def commit_reservation(self, reservation_id: str, actual: float, phase: str): ...

class TaskRunner(Protocol):
    async def execute_task(self, task: Task, policy=None) -> TaskResult: ...

class ContextProvider(Protocol):
    @property
    def project_context(self) -> ProjectContext: ...
    @property
    def context_truncation_limit(self) -> int: ...
```

Then collaborators accept protocols instead of `Orchestrator`:

```python
# Before
self._executor = ExecutorService(execute_fn=self._execute_task, ...)

# After
self._executor = ExecutorService(
    runner=self,          # implements TaskRunner
    budget=self,          # implements BudgetTracker
)
```

---

### Phase 8: Test Coverage for Extracted Modules

**Target:** Bring coverage from ~12% to 50%+  
**Effort:** 3 days

For each extracted module: unit tests (mocked dependencies), integration tests (vs old inline code), golden-file tests (diff outputs).

---

## Implementation Order & Effort

| Phase | Enhancement | Effort | Risk | Lines Saved |
|-------|-------------|--------|------|-------------|
| 1 | Extract imports → `engine_deps.py` | 1 day | None | 180 |
| 2 | Extract Decomposition → `decomposer.py` | 2 days | Low | 685 |
| 3 | Extract Preflight/Validation → `validator.py` | 1 day | Low | 284 |
| 4 | Extract Architecture Rules → `architect.py` | 0.5 day | None | 96 |
| 5 | `_execute_task` → TaskPipeline stages | 4 days | Medium | 1,582 |
| 6 | `__init__` → lazy wiring + container | 2 days | Medium | 200 |
| 7 | Interface Segregation (Protocols) | 3 days | High | qualitative |
| 8 | Test coverage for extracted modules | 3 days | None | N/A |

**Total:** ~16.5 days

**Phases 1-4 (low-risk):** 4.5 days → engine.py drops to ~4,100 lines (23% reduction)  
**Phases 1-5 (core extraction):** 8.5 days → engine.py drops to ~2,500 lines (53% reduction)  
**All phases:** 16.5 days → engine.py drops to ~1,800 lines (66% reduction)

---

## Rollback Safety

Each phase is independently revertible:

| Phase | Rollback mechanism |
|-------|-------------------|
| 1 | Revert `engine_deps.py` import; copy imports back |
| 2 | Revert `Decomposer` delegation; restore inline `_decompose` |
| 3 | Revert `TaskValidator` delegation; restore inline methods |
| 4 | Revert `Architect` delegation; restore inline method |
| 5 | Revert pipeline; restore `_execute_task` from git |
| 6 | Revert lazy properties; restore `__init__` from git |
| 7 | Remove Protocol annotations; restore `self` references |

---

## Verification Gates (per phase)

- [ ] `python -m orchestrator --help` works
- [ ] All existing tests pass (`pytest`)
- [ ] `ruff check orchestrator/` clean
- [ ] `mypy orchestrator/` clean (or no new errors)
- [ ] Integration smoke test: `python -m orchestrator --project "Hello World CLI" --budget 0.50 --dry-run`
- [ ] Import speed unchanged: `python -c "import orchestrator"` within ±5% wall time

---

## Appendix A: Files to Create

| File | Purpose | Phase |
|------|---------|-------|
| `orchestrator/engine_deps.py` | All optional imports and HAS_* flags | 1 |
| `orchestrator/engine_core/decomposer.py` | Decomposition pipeline | 2 |
| `orchestrator/engine_core/validator.py` | Preflight + validation logic | 3 |
| `orchestrator/engine_core/architect.py` | Architecture rule generation | 4 |
| `orchestrator/engine_core/pipeline.py` | TaskPipeline + PipelineContext | 5 |
| `orchestrator/engine_core/stages/__init__.py` | Stage exports | 5 |
| `orchestrator/engine_core/stages/generate.py` | GenerateStage | 5 |
| `orchestrator/engine_core/stages/critique.py` | CritiqueStage | 5 |
| `orchestrator/engine_core/stages/revise.py` | ReviseStage | 5 |
| `orchestrator/engine_core/stages/evaluate.py` | EvaluateStage | 5 |
| `orchestrator/engine_core/stages/validate.py` | ValidateStage | 5 |
| `orchestrator/engine_core/stages/preflight.py` | PreflightStage | 5 |
| `orchestrator/engine_core/stages/self_consistency.py` | SelfConsistencyStage | 5 |
| `orchestrator/engine_core/container.py` | ServiceContainer | 6 |
| `orchestrator/engine_core/protocols.py` | Role Protocols | 7 |

## Appendix B: Files Modified

| File | Changes | Phase |
|------|---------|-------|
| `orchestrator/engine.py` | Remove extracted methods; add delegation | 1-7 |
| `orchestrator/services/generator.py` | Point decompose_fn at Decomposer | 2 |
| `orchestrator/services/executor.py` | Accept TaskRunner protocol | 7 |
| `orchestrator/services/evaluator.py` | Accept ModelProvider protocol | 7 |

## Appendix C: Graph Verification Baseline

After all phases complete, re-run graphify:

- [ ] Community count: 267 → < 80 (concerns co-located in modules)
- [ ] Isolated nodes: 3,057 → < 1,000 (module boundaries create clear ownership)
- [ ] EXTRACTED edges: 39% → > 70% (Protocols create explicit edges)
- [ ] `engine.py` method count: 97 → ~25
- [ ] `engine.py` line count: 5,313 → ~1,800

---

**Last updated:** 2026-05-23
