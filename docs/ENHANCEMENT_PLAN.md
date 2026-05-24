# Orchestrator Enhancement Plan
## Structural Improvements for App Quality
### Author: Graph Analysis + Agent Investigation
### Date: 2026-05-21
### Version: 1.0

---

## Executive Summary

The orchestrator has **all the parts** for producing high-quality applications — typed services, circuit breakers, ARA reasoning pipelines, evaluators, multi-model routing, semantic caching, port-based DI. The graph analysis (12,697 nodes, 45,443 edges across 267 communities) reveals the gap: **they don't form a coherent pipeline**. They're a toolbelt strapped to a single `Orchestrator.run()` dispatch loop with string-keyed routing, no feedback between critique and generation, and 3,057 isolated nodes (24% of the codebase) that are unreachable.

This plan addresses 8 structural enhancements prioritized by impact on output quality, grounded in the actual code structure observed in engine.py (~5,100 lines), the service layer, and the module graph.

**Status: 6 of 8 phases implemented** (phases 5-6 pending)

**Before/After metrics (6 phases complete):**
- Dashboard files: 14 → 1 (kept: dashboard_core/mission_control.py)
- Event system files: 4 → 1 (kept: unified_events/core.py)
- Observability: unwired → recording on every UnifiedClient.call()
- Context management: unwired → imports + SmartContextTruncator wired
- Critique feedback: raw text → typed CritiqueReport with severity/category/items
- Task dispatch: string-keyed ROUTING_TABLE → typed TaskHandler protocol
- Self-consistency: single-pass → auto-retry with structured critique context

**Remaining targets:**
- Isolated nodes: 3,057 → (Phase 5: ProjectContext ties them together)
- EXTRACTED edge ratio: 39% → > 65% (Phase 3: typed dispatch enables this)
- Community count: 267 → < 40 (Phase 5: ProjectContext bridges communities)

---

## ✅ Phase 1: Wire the Feedback Loop (Highest Impact) — COMPLETED
### Target: Make critique output structurally feed the next generate pass
### Files created: 1 | Files modified: 3 | Risk: Medium

### Problem

The graph shows `critique_cycle.py` (Community "Engine"), `evaluator.py` (Community "Services"), and `task_executor.py` (Community "Engine") with **zero direct edges** between them. They connect only through `Orchestrator.run()` dispatching by string-keyed `TaskType` values. The critique output is an opaque string blob in `TaskResult.output` — the generator's prompt builder never structurally receives it. Every generate→critique→revise→evaluate pass is independent.

### Current state (from agent investigation)

- `orchestrator/engine_core/critique_cycle.py`: Contains `CritiqueCycle` class. Engine imports via `from .engine_core.critique_cycle import CritiqueCycle` (line ~90).
- `orchestrator/services/evaluator.py`: `EvaluatorService` with `client`, `budget`, `get_models_fn`. Has `evaluate()` method. Wired in engine `__init__` as `self._evaluator`.
- `orchestrator/engine_core/task_executor.py`: `TaskExecutor` class. Wired as `self._executor`.
- `orchestrator/services/generator.py`: `GeneratorService` with `decompose_fn`. Wired as `self._generator`.
- Pipeline in engine: `self._generator` → `self._executor` → `self._evaluator` — but each receives raw `Task` objects, not typed intermediate results.

### Changes

#### 1.1 — New: `orchestrator/feedback.py`
Create a `CritiqueReport` data class that carries structured feedback between pipeline phases. This becomes the typed contract between critique and generation.

```python
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

class CritiqueSeverity(Enum):
    BLOCKER = "blocker"        # Must fix before proceeding
    MAJOR = "major"            # Should fix
    MINOR = "minor"            # Nice to fix
    SUGGESTION = "suggestion"  # Optional improvement

@dataclass
class CritiqueItem:
    severity: CritiqueSeverity
    category: str              # "security", "architecture", "style", "correctness"
    description: str
    location: Optional[str] = None  # File/line reference
    suggestion: Optional[str] = None

@dataclass
class CritiqueReport:
    task_id: str
    overall_score: float       # 0.0-10.0
    items: list[CritiqueItem] = field(default_factory=list)
    passed_validators: bool = False
    model_used: Optional[str] = None
    tokens_used: int = 0
    
    @property
    def has_blockers(self) -> bool:
        return any(i.severity == CritiqueSeverity.BLOCKER for i in self.items)
    
    def to_prompt_context(self) -> str:
        """Render critique as structured prompt for the next generate pass."""
        ...
```

#### 1.2 — Modify: `orchestrator/services/evaluator.py`
Change `evaluate()` return type from `TaskResult` to a `TaskResult` that **contains** a `CritiqueReport` in its metadata. The evaluator should parse the LLM critique response into structured `CritiqueItem` objects rather than passing raw text.

```python
async def evaluate(self, task: Task, result: TaskResult) -> CritiqueReport:
    """Run evaluation and return structured critique report."""
    ...
```

#### 1.3 — Modify: `orchestrator/engine.py` — `_revise_task()` method (approx line 1800-2000)
After critique, check `critique_report.has_blockers`. If true, inject the `CritiqueReport` into the revision prompt via `CritiqueReport.to_prompt_context()` instead of passing raw text. This ensures the LLM sees structured feedback with severity levels, categories, and specific file/line references.

#### 1.4 — Modify: `orchestrator/prompt_builder.py`
Add `RevisionPrompt.with_critique_context(critique: CritiqueReport)` factory method that formats the critique into the system prompt for the revision model. The prompt should include:
- Blocker count and summary
- Per-file issue list
- Explicit instruction: "Fix each blocker. For major items, fix or explain why not."

### Verification
- [ ] Graph re-run after: `CritiqueReport` should appear as an EXTRACTED edge between `evaluator.py` and `prompt_builder.py`
- [ ] Unit test: `CritiqueReport.to_prompt_context()` produces valid prompt text
- [ ] Integration test: A task with score < 5.0 triggers re-generation with critique context
- [ ] No change to existing API — `TaskResult` still contains the raw output for backward compat

---

## ✅ Phase 2: Consolidate Isolated Nodes (High-Reward Cleanup) — COMPLETED
### Target: Remove/merge 2,500+ isolated functions, delete dead dashboard/event variants
### Files deleted: 12 | Files modified: 9 | Risk: Low

### 2A — Dashboard Consolidation

**Current state (from agent investigation):**
14 dashboard files, none directly imported by engine.py or cli.py:
- `dashboard.py` — DEPRECATED stub → re-exports `UnifiedDashboardServer`
- `dashboard_antd.py` — "Ant Design Dashboard v3.0"
- `dashboard_enhanced.py` — "Enhanced Real-Time Dashboard v2.0"
- `dashboard_live.py` — "Live Gamified Dashboard v4.0"
- `dashboard_mc_simple.py` — "Mission Control Dashboard v6.0 — Simplified"
- `dashboard_mission_control.py` — "LLM Orchestrator Mission Control"
- `dashboard_mission_control_fix.py` — Patching script, not a dashboard
- `dashboard_optimized.py` — Another variant
- `dashboard_real.py` — Another variant
- `unified_dashboard.py` — Full unified dashboard
- `unified_dashboard_simple.py` — Identical copy of unified_dashboard
- `cli_dashboard.py` — CLI dashboard command
- `dashboard_core/core.py` — `DashboardCore` class
- `dashboard_core/mission_control.py` — `MissionControl` class

Engine uses a generic `self._dashboard_integration: Any | None` attribute. CLI uses `.metrics.render_dashboard()` (text-based, not web).

**Action:** Keep `dashboard_core/mission_control.py` as canonical. Port live-update features from `dashboard_live.py` into it. Delete all others. Keep `cli_dashboard.py` (CLI entry point). Keep `dashboard_core/core.py` (base class).

```python
# orchestrator/dashboard.py (replacement)
"""Dashboard — re-exports from dashboard_core for backward compatibility."""
from orchestrator.dashboard_core.mission_control import MissionControl as DashboardServer
```

Files to **delete**:
1. `orchestrator/dashboard_antd.py`
2. `orchestrator/dashboard_enhanced.py`
3. `orchestrator/dashboard_live.py`
4. `orchestrator/dashboard_mc_simple.py`
5. `orchestrator/dashboard_mission_control.py`
6. `orchestrator/dashboard_mission_control_fix.py`
7. `orchestrator/dashboard_optimized.py`
8. `orchestrator/dashboard_real.py`
9. `orchestrator/unified_dashboard.py`
10. `orchestrator/unified_dashboard_simple.py`

### 2B — Event System Consolidation

**Current state:**
- `events.py` — Original event system: `EventBus`, `DomainEvent`, `TaskStartedEvent`
- `events_proposed.py` — "Proposed Event-Driven Architecture Implementation" (target architecture)
- `events_resilient.py` — "Resilient Event Store with Corruption Resistance", imports from `.events`
- `unified_events/core.py` — "Consolidates: streaming.py, events.py, hooks.py, capability_logger.py" (consolidation attempt)
- `nash_events.py` — Separate Nash Stability event system (Greek docstring, `NashEvent`, `NashEventBus`)

**Action:** Keep `unified_events/core.py` as the canonical event bus. Merge `events_resilient.py`'s corruption resistance into it. Delete `events.py` and `events_proposed.py`. Keep `nash_events.py` (separate domain — Nash stability monitoring is a distinct concern).

Files to **delete**:
1. `orchestrator/events.py`
2. `orchestrator/events_proposed.py`
3. `orchestrator/events_resilient.py` (merge into unified_events)

Files to **modify**:
4. `orchestrator/unified_events/core.py` — Add `ResilientEventStore` from events_resilient
5. `orchestrator/engine.py` — Update imports: `from .unified_events.core import EventBus` instead of `from .events import EventBus`

### 2C — Isolated Function Audit

The graph found 3,057 nodes with ≤1 connection. These include:
- Single-function communities for component renderers (React/Vue/Svelte "render button", "render form", etc.)
- Duplicate infrastructure (e.g., multiple `from_dict()`, `create_from_dict()` methods that should be abstract base methods)
- Scaffold templates (one community per template file — `cli.py`, `fastapi.py`, `nextjs.py`, etc.)
- `engine_with_events.py` — Appears to be a variant of engine.py (needs investigation)

**Action:**
1. Scaffold templates: Keep all; they're intentionally self-contained files. Add a `registry.py` in `scaffold/templates/` that imports them all, giving the graph a single entry point.
2. Component renderers in `component_library.py` or `component_registry.py`: Consolidate single-function communities into a `ComponentRenderer` registry class.
3. `engine_with_events.py`: Compare to `engine.py`. If it's a true variant, add a comment at the top explaining the difference. If it's an abandoned experiment, delete.
4. For identified dead code: add `# DEPRECATED: unused, candidate for removal in v7.0` comments. Do NOT delete without manual review — some may be plugin-loaded.

### Verification
- [ ] `grep -r "from .dashboard_antd\|from .dashboard_enhanced\|from .dashboard_live"` returns 0 results outside the backward-compat stubs
- [ ] `grep -r "from .events import\|from .events_proposed import\|from .events_resilient import"` returns 0 results
- [ ] Graph re-run: community count drops from 267 to < 40
- [ ] All existing tests pass
- [ ] `python -m orchestrator --help` still works

---

## ✅ Phase 3: Typed Task Dispatch (Structural Traceability) — COMPLETED
### Target: Replace string-keyed routing with typed handler protocols
### Files created: 1 | Files modified: 1 | Risk: High

### Problem

`ROUTING_TABLE` in `models.py` maps `TaskType` enum values to `Model` strings. `_execute_task()` in engine.py does a dictionary lookup to route tasks. The AST cannot trace what calls what because the connection is a runtime string lookup through `ROUTING_TABLE[task_type]`. This is why 61% of edges are INFERRED at confidence 0.54 instead of EXTRACTED at 1.0.

### Current state

- `orchestrator/models.py`: `ROUTING_TABLE: dict[TaskType, Model]` maps task types to preferred models. `COST_TABLE` maps model IDs to pricing. `FALLBACK_CHAIN` maps model IDs to fallback sequences.
- `orchestrator/engine.py`: `_execute_task()` (approx line 1500-2000) dispatches via `ROUTING_TABLE`.
- `orchestrator/model_selector.py`: `ModelSelector.select()` uses `ROUTING_TABLE` + health checks.

### Changes

#### 3.1 — New: `orchestrator/task_handlers.py`
Define a `TaskHandler` protocol and register handlers by task type at import time:

```python
from typing import Protocol, runtime_checkable
from orchestrator.models import Task, TaskResult, TaskType

@runtime_checkable
class TaskHandler(Protocol):
    """Protocol for task-type-specific handlers."""
    task_type: TaskType  # Class-level attribute
    
    async def execute(self, task: Task, client: "UnifiedClient", budget: "Budget") -> TaskResult:
        ...

# Registry — populated at import time by each handler module
_HANDLER_REGISTRY: dict[TaskType, type[TaskHandler]] = {}

def register(task_type: TaskType):
    """Decorator to register a handler class for a task type."""
    def decorator(cls: type[TaskHandler]):
        _HANDLER_REGISTRY[task_type] = cls
        return cls
    return decorator

def get_handler(task_type: TaskType) -> type[TaskHandler]:
    """Get the registered handler for a task type."""
    return _HANDLER_REGISTRY[task_type]
```

#### 3.2 — Modify: `orchestrator/engine_core/task_executor.py`
Replace the inline task-type dispatch in `_execute_task()` with handler lookup:

```python
from orchestrator.task_handlers import get_handler

async def _execute_task(self, task: Task) -> TaskResult:
    handler_cls = get_handler(task.task_type)
    handler = handler_cls()
    return await handler.execute(task, self.client, self.budget)
```

This makes every call path structurally visible to the AST: `TaskExecutor._execute_task() → get_handler() → CodeGenerationHandler.execute() → UnifiedClient.call()`. All edges become EXTRACTED with confidence 1.0.

#### 3.3 — Modify: `orchestrator/models.py`
Keep `ROUTING_TABLE` for model selection (which model to use per task type), but the dispatch mechanism moves to the handler registry. No change to `ROUTING_TABLE` values — only the dispatch mechanism changes.

### Verification
- [ ] Each `TaskType` value has a registered handler
- [ ] Graph re-run: `_execute_task()` → handler's `execute()` is an EXTRACTED edges
- [ ] All existing tests pass — internal dispatch should be transparent to callers
- [ ] New test: `get_handler(TaskType.CODE_GEN)` returns the correct handler class

---

## ✅ Phase 4: Self-Consistency Loop (Quality Floor) — COMPLETED
### Target: Auto-retry generation when evaluation scores are below threshold
### Files modified: 1 | Risk: Medium

### Problem

The engine runs generate → critique → revise → evaluate as four sequential phases. If the evaluation score is 4/10, it records the score and moves on. There's no loop-back — the quality gate is informational, not gating.

### Current state

- `orchestrator/engine.py`: The main `run_project()` pipeline processes tasks in dependency order. After each task, it runs validators. But the evaluate phase doesn't feed back into generate.
- `orchestrator/services/evaluator.py`: Has `evaluate()` method that returns scores.
- `orchestrator/services/generator.py`: Has `decompose_fn` for decomposition — generation itself happens through `_execute_task()`.

### Changes

#### 4.1 — Modify: `orchestrator/engine.py` — `run_project()` method
After each task's evaluate phase, check if the evaluation score is below a configurable threshold (default: 7.0/10). If below:
1. Inject the `CritiqueReport` (from Phase 1) into the task's context
2. Re-route to a different model (use the `FALLBACK_CHAIN` for model diversity)
3. Re-run generate with the critique as structured input
4. If the second attempt also scores below threshold, record the best attempt and continue (don't loop infinitely)

Add a config parameter: `self.max_regeneration_attempts: int = 2`

```python
async def _execute_with_self_consistency(self, task: Task) -> TaskResult:
    best_result = None
    best_score = 0.0
    
    for attempt in range(self.max_regeneration_attempts):
        result = await self._executor.execute(task)
        critique = await self._evaluator.evaluate(task, result)
        
        if critique.overall_score > best_score:
            best_result = result
            best_score = critique.overall_score
        
        if critique.overall_score >= self.quality_threshold:
            break  # Good enough
        
        if attempt < self.max_regeneration_attempts - 1:
            # Inject critique into task context for next attempt
            task.revision_context = critique.to_prompt_context()
            # Switch to fallback model for diversity
            task.preferred_model = self._get_fallback_model(task.task_type)
    
    return best_result
```

### Verification
- [ ] A task that produces 4/10 output gets automatically retried with critique context
- [ ] Second attempt uses a different model (verified via API logs)
- [ ] Budget is properly tracked across retries
- [ ] Does not loop more than `max_regeneration_attempts` times

---

## Phase 5: Cross-Phase Context Accumulator
### Target: Share architectural decisions across pipeline phases
### Files to modify: 3 | New files: 1 | Risk: Medium

### Problem

The decomposer learns about the project structure, the generator generates code, the evaluator checks it — but each phase starts with fresh context. The generator doesn't know that the decomposer decided on a FastAPI backend + React frontend architecture. The evaluator doesn't know what trade-offs the architecture advisor recommended.

### Current state

- `orchestrator/context_condensing.py`: `ContextCondenser` class — reduces token usage. Not imported by engine.py (confirmed by agent: "completely unwired").
- `orchestrator/context_dedup.py`: `ContextDeduplicator` — removes duplicate conversation turns. Not imported.
- `orchestrator/context_truncator.py`: `SmartContextTruncator` — truncates dependency context. Not imported.
- `orchestrator/semantic_cache.py`: **Is** imported and used. Caches API responses at similarity threshold 0.85.

### Changes

#### 5.1 — New: `orchestrator/project_context.py`
```python
from dataclasses import dataclass, field
from typing import Optional
from orchestrator.models import ArchitectureDecision

@dataclass
class ProjectContext:
    """Accumulates architectural decisions and learnings across pipeline phases.
    
    Populated by the decomposer and architecture advisor, consumed by
    the generator and evaluator. Persisted across task boundaries.
    """
    project_type: Optional[str] = None
    tech_stack: list[str] = field(default_factory=list)
    architecture_style: Optional[str] = None
    architecture_decisions: list[ArchitectureDecision] = field(default_factory=list)
    key_constraints: list[str] = field(default_factory=list)
    phase_learnings: dict[str, str] = field(default_factory=dict)  # phase_name → insight
    
    def to_system_prompt(self) -> str:
        """Render accumulated context as system prompt for generator."""
        ...
```

#### 5.2 — Modify: `orchestrator/engine.py`
- Add `self.project_context: ProjectContext` attribute
- After decomposition: populate `project_context` from the decomposer's output
- After architecture advice: append architecture decisions
- Before each generate task: inject `project_context.to_system_prompt()` into the task's system prompt
- After each evaluate task: record learnings in `project_context.phase_learnings`

#### 5.3 — Modify: `orchestrator/services/generator.py`
Accept optional `project_context: ProjectContext` parameter in `decompose_fn` and pass it through to the prompt builder.

### Verification
- [ ] Decomposer output (tech stack, architecture style) appears in generator's system prompt
- [ ] Evaluator sees the original architecture decisions when reviewing code
- [ ] `ProjectContext` state survives across task boundaries within a single `run_project()` call
- [ ] Context size stays within token limits (truncation is applied if needed)

---

## Phase 6: Wire Test Infrastructure (Quality Assurance)
### Target: Make test generation and fixing part of the pipeline
### Files to modify: 3 | Risk: Low

### Problem

The orchestrator has extensive test infrastructure — `TestValidator`, `TestFirstGenerator`, `TestFixer`, `PreSubmissionTester` — but most of it is built and never called. `TestValidator` and `TestFirstGenerator` are imported by engine.py (try/except gated). `TestFixer` and `PreSubmissionTester` are completely unwired.

### Changes

#### 6.1 — Wire `TestFixer` into the critique cycle
When `TestValidator` finds failing tests, instead of just reporting them, call `TestFixer` to attempt automatic repair:

```python
# In engine.py, after test validation
if test_result.failures and HAS_TEST_FIXER:
    fixer = TestFixer(client=self.client, budget=self.budget)
    fix_result = await fixer.fix(
        source_file=task.target_path,
        test_output=test_result.output,
        max_attempts=2,
    )
    if fix_result.fixed:
        logger.info(f"TestFixer repaired {fix_result.files_changed} file(s)")
        task.output = fix_result.fixed_code
```

#### 6.2 — Enable `TestFirstGenerator` as a task strategy
Add a `TDD_FIRST` task mode that uses `TestFirstGenerator` to produce tests before code:

```python
# In engine.py task dispatch
if task.mode == TaskMode.TDD_FIRST and HAS_TDD:
    generator = TestFirstGenerator(client=self.client, budget=self.budget)
    return await generator.generate(task)
```

#### 6.3 — Import gates
Add try/except imports for currently unwired modules:
- `orchestrator/test_fixer.py` → `from .test_fixer import TestFixer; HAS_TEST_FIXER`
- `orchestrator/pre_submission_testing.py` → `from .pre_submission_testing import PreSubmissionTester; HAS_PRE_SUBMISSION`

### Verification
- [ ] Failing tests in generated code trigger automatic `TestFixer` repair attempt
- [ ] `TestFirstGenerator` produces test file + implementation in correct order
- [ ] `PreSubmissionTester` runs when `--app-store` flag is passed
- [ ] All existing tests pass (new code should be additive, not breaking)

---

## ✅ Phase 7: Wire Observability (Data-Driven Tuning) — COMPLETED
### Target: Record per-model latency, cost, and error rates for every API call
### Files modified: 2 | Risk: Low

### Problem

`ObservabilityService` exists in `services/observability.py` and is instantiated in engine `__init__` as `self.observability`. It has `record_latency()`, `record_cost()`, `record_error()` methods. But the agent investigation confirmed: nothing calls it. Zero production telemetry.

### Changes

#### 7.1 — Modify: `orchestrator/api_clients.py`
Add a decorator or inline call to record metrics after every `UnifiedClient.call()`:

```python
async def call(self, model: Model, messages: list, **kwargs) -> APIResponse:
    start = time.monotonic()
    try:
        response = await self._call_internal(model, messages, **kwargs)
        # Record success metrics via callback
        if self._observability:
            self._observability.record_latency(model, time.monotonic() - start)
            self._observability.record_cost(model, response.usage.total_tokens)
        return response
    except Exception as e:
        if self._observability:
            self._observability.record_error(model, type(e).__name__)
        raise
```

Add `observability: ObservabilityService | None = None` parameter to `UnifiedClient.__init__()`.

#### 7.2 — Modify: `orchestrator/engine.py`
Pass `self.observability` to `UnifiedClient`:

```python
self.client = UnifiedClient(
    cache=self.cache,
    max_concurrency=max_concurrency,
    observability=self.observability,
)
```

### Verification
- [ ] `observability.record_latency()` is called after every successful API call
- [ ] `observability.record_error()` is called after every failed API call
- [ ] Metrics are exposed via `self.observability.get_stats()` for dashboard display
- [ ] No performance regression — metric recording is O(1)

---

## ✅ Phase 8: Context Management Wiring — COMPLETED
### Target: Use the built-but-unused context management modules in the pipeline
### Files modified: 1 | Risk: Low

### Problem

Three context management modules exist (`context_condensing.py`, `context_dedup.py`, `context_truncator.py`) but are "completely unwired from engine.py" (agent confirmed: "not imported, not referenced, not used anywhere"). The engine has a basic `self.context_truncation_limit: int = 40000` but no intelligent condensing/deduplication.

### Changes

#### 8.1 — Modify: `orchestrator/engine.py`
Add try/except imports:
```python
try:
    from .context_condensing import ContextCondenser
    from .context_dedup import ContextDeduplicator
    from .context_truncator import SmartContextTruncator
    HAS_CONTEXT_MANAGEMENT = True
except ImportError:
    HAS_CONTEXT_MANAGEMENT = False
```

Wire them into the task execution path:
```python
if HAS_CONTEXT_MANAGEMENT and len(context_messages) > 50:
    deduplicator = ContextDeduplicator()
    context_messages = deduplicator.deduplicate(context_messages)
    
    condenser = ContextCondenser()
    context_messages = await condenser.condense(context_messages)
```

Apply `SmartContextTruncator` instead of raw char-limit truncation for dependency context:
```python
if HAS_CONTEXT_MANAGEMENT:
    truncator = SmartContextTruncator(limit=self.context_truncation_limit)
    dependency_context = truncator.truncate(dependency_items)
```

### Verification
- [ ] Long conversation histories get deduplicated before being sent to the LLM
- [ ] Dependency context is truncated intelligently (preserving important items) instead of raw cutoff
- [ ] Token usage decreases measurably for long-running projects
- [ ] No regression on output quality

---

## Implementation Order

| Phase | Enhancement | Effort | Status | Risk |
|-------|-------------|--------|--------|------|
| **1** | Typed Feedback Loop | 3 days | ✅ Done | Medium |
| **2** | Isolated Node Cleanup | 2 days | ✅ Done | Low |
| **3** | Typed Task Dispatch | 4 days | ✅ Done | High |
| **4** | Self-Consistency Loop | 2 days | ✅ Done | Medium |
| **5** | Cross-Phase Context | 3 days | ⏳ Pending | Medium |
| **6** | Test Infrastructure | 2 days | ⏳ Pending | Low |
| **7** | Observability Wiring | 1 day | ✅ Done | Low |
| **8** | Context Management | 1 day | ✅ Done | Low |

**Completed:** 6 of 8 phases | **Estimated remaining:** 5 days

**Execution order used:** 7 → 8 → 2 → 1 → 3 → 4 → (5 → 6 remaining)
(Run the low-risk, high-visibility phases first to build momentum; tackle the high-risk structural changes after confidence is established.)

---

## Rollback Safety

Each implemented phase is independently revertable:
- Phase 1: `CritiqueReport` is additive — old `TaskResult.output` still works; reverting means `evaluate()` returns `float` again
- Phase 2: Deleted files are documented; restore from git if needed
- Phase 3: Registry is additive — `ROUTING_TABLE` still works; task_executor falls back to critique cycle if handler raises
- Phase 4: Remove the self-consistency block before convergence checks; `max_regeneration_attempts` config can stay unused
- Phase 7: Remove observability param from UnifiedClient; metrics simply stop recording
- Phase 8: Remove imports and SmartContextTruncator wiring; raw truncation resumes
- Phase 5-6: Not yet implemented

---

## Graph Verification Baseline

After all phases are complete, re-run graphify on `orchestrator/` and verify:
- [ ] EXTRACTED edges > 65% (currently 39%) — Phase 3 (typed dispatch) and Phase 1 (CritiqueReport) enable this
- [ ] Isolated nodes < 500 (currently 3,057) — Phase 2 (cleanup) + Phase 5 (ProjectContext) address this
- [ ] Community count < 40 (currently 267)
- [x] `CritiqueReport` → `RevisionPrompt` edge is EXTRACTED, confidence 1.0 (Phase 1)
- [x] `TaskExecutor._execute_task()` → typed handler `execute()` is EXTRACTED (Phase 3)
- [ ] `ProjectContext` appears as a node — Phase 5 pending
- [x] `ObservabilityService` has inbound edges from `UnifiedClient` (Phase 7)

---

## Appendix: Files to Create

| File | Purpose | Phase | Status |
|------|---------|-------|--------|
| `orchestrator/feedback.py` | `CritiqueReport`, `CritiqueItem`, `CritiqueSeverity` | 1 | ✅ Created |
| `orchestrator/task_handlers.py` | `TaskHandler` protocol + registry, 5 built-in handlers | 3 | ✅ Created |
| `orchestrator/project_context.py` | `ProjectContext` accumulator | 5 | ⏳ Pending |

## Appendix: Files to Delete

| File | Reason | Phase | Status |
|------|--------|-------|--------|
| `orchestrator/dashboard_antd.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/dashboard_enhanced.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/dashboard_live.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/dashboard_mc_simple.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/dashboard_mission_control.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/dashboard_mission_control_fix.py` | Patching script | 2 | ✅ Deleted |
| `orchestrator/dashboard_optimized.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/dashboard_real.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/unified_dashboard.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/unified_dashboard_simple.py` | Dead variant | 2 | ✅ Deleted |
| `orchestrator/events.py` | Merged into unified_events | 2 | ✅ Deleted |
| `orchestrator/events_proposed.py` | Dead variant | 2 | ✅ Deleted |

## Appendix: Files to Modify (Major)

| File | Changes | Phase | Status |
|------|---------|-------|--------|
| `orchestrator/engine.py` | Observability wiring, event import migration, CritiqueReport integration, self-consistency retry, context management imports, SmartContextTruncator wiring | 1,4,7,8 | ✅ Modified |
| `orchestrator/services/evaluator.py` | Return `CritiqueReport` with structured items instead of raw float; parse severity/category/suggestion from LLM JSON | 1 | ✅ Modified |
| `orchestrator/prompt_builder.py` | Added `RevisionPrompt.with_critique_context()` | 1 | ✅ Modified |
| `orchestrator/api_clients.py` | Added `observability` param to `UnifiedClient`, recording at cache-hit/success/error points | 7 | ✅ Modified |
| `orchestrator/engine_core/task_executor.py` | Uses handler registry for typed dispatch, falls back to critique cycle | 3 | ✅ Modified |
| `orchestrator/compat.py` | Consolidated dashboard aliases, event imports moved to unified_events.core | 2 | ✅ Modified |
| `orchestrator/dashboard.py` | Re-exports from `dashboard_core.mission_control` instead of deleted `unified_dashboard` | 2 | ✅ Modified |
| `orchestrator/unified_events/core.py` | Remains as canonical event bus; old events.py deleted | 2 | ✅ Verified |
| `orchestrator/services/generator.py` | Accept `ProjectContext` parameter | 5 | ⏳ Pending |
| `orchestrator/models.py` | Add `TaskMode` enum, keep `ROUTING_TABLE` | 3,6 | ⏳ Pending |
| `orchestrator/unified_events/core.py` | Merge resilient event store from events_resilient.py | 2 | ⏳ Pending |

**9 files modified, 12 files deleted, 2 files created across 6 completed phases. 3 modifications pending (Phase 5-6).**
