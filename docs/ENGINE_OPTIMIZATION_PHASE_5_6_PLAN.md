# Engine.py Optimization — Phases 5 & 6 Implementation Plan

> Generated: 2026-05-26
> Status: **Draft — NOT IMPLEMENTED**
> Dependencies: Phases 1-4 complete (see `docs/ENGINE_OPTIMIZATION_PLAN.md`)

## Pre-Flight: Current State

```
engine.py: 5,030 lines (was 5,272 — Phases 1-4 saved 242 lines)
Constructor: L371–L670  (~300 lines, 50+ subsystems)
Pipeline:    L1512–L2520 (~400 lines, run_project/run_job/dry_run/_execute_all/_decompose)
Core loop:   L2883–L3800 (_execute_task — ~200 lines, most tightly-wired method)
```

## Phase 5: ServiceCollection

### Goal

Move the constructor's 50+ subsystem initializations out of `Orchestrator.__init__()` into a `ServiceCollection` builder that constructs subsystems grouped by concern. The Orchestrator still owns all state; the collection is just responsible for wiring.

### Strategy: Incremental Slice Extraction

Do NOT extract all 50 at once. Extract them in **4 independent slices**, testing after each:

| Slice | Subsystems | Lines | Risk |
|-------|-----------|-------|------|
| **A: Safety** | `_task_verifier`, `_accountability`, `_agent_safety`, `_red_team`, `_tool_guardrails` | ~15 | None — all unconditionally constructed, no imports can fail |
| **B: Integration** | `_token_optimizer`, `_preflight_validator`, `_session_watcher`, `_persona_manager` | ~20 | None — unconditionally constructed |
| **C: Search** | `_memory_manager`, `_bm25_search`, `_reranker`, `_knowledge_base`, `_hybrid_pipeline`, `_rate_limiter`, `_lifecycle_manager` | ~35 | Low — `get_bm25_search()` and `get_reranker()` have side effects but are imported unconditionally |
| **D: Learning** | `_telemetry_store`, `_memory_provider_mgr`, `_pattern_store`, `_pattern_extractor`, `_pattern_injector`, `_pattern_curator`, `_batch_runner`, `_batch_guard` | ~40 | Medium — all behind try/except ImportError guards |

### Slice A Example (Safety)

```python
# New file: orchestrator/service_collection.py

@dataclass
class SafetyServices:
    task_verifier: TaskVerifier
    accountability: AccountabilityTracker
    agent_safety: AgentSafetyMonitor
    red_team: RedTeamFramework
    tool_guardrails: ToolCallGuardrailController

    @classmethod
    def build(cls) -> "SafetyServices":
        return cls(
            task_verifier=TaskVerifier(),
            accountability=AccountabilityTracker(),
            agent_safety=AgentSafetyMonitor(),
            red_team=RedTeamFramework(),
            tool_guardrails=ToolCallGuardrailController(),
        )
```

```python
# In engine.py __init__, replace:
#   self._task_verifier: TaskVerifier = TaskVerifier()
#   self._accountability: AccountabilityTracker = AccountabilityTracker()
#   ...5 lines...
# With:
safety = SafetyServices.build()
self._task_verifier = safety.task_verifier
self._accountability = safety.accountability
self._agent_safety = safety.agent_safety
self._red_team = safety.red_team
self._tool_guardrails = safety.tool_guardrails
```

### Slice C/D Pattern — Handle Optional Imports

For slices behind try/except ImportError guards, the builder falls back gracefully:

```python
@dataclass
class LearningServices:
    telemetry_store: TelemetryStore | None
    memory_provider_mgr: MemoryManager | None
    pattern_store: PatternStore | None
    # ...

    @classmethod
    def build(cls, telemetry_store: TelemetryStore | None = None) -> "LearningServices":
        try:
            from orchestrator.telemetry_store import TelemetryStore as TS
            from orchestrator.memory.memory_manager import MemoryManager as MM
            # ...
        except (ImportError, TimeoutError):
            return cls(telemetry_store=None, memory_provider_mgr=None, ...)
        return cls(
            telemetry_store=telemetry_store or TS(),
            memory_provider_mgr=MM(),
            # ...
        )
```

### Verification Per Slice

```bash
python -c "from orchestrator import Orchestrator; o = Orchestrator(); print(o._task_verifier)"
pytest tests/test_bug_fixes_v2.py -v --no-cov
pytest tests/ -x --no-cov -m "not slow and not integration"
```

### Phase 5 Target

```
Before: __init__ is ~300 lines, 50+ sequential assignments
After:  __init__ is ~100 lines, 4 builder calls + wiring
        service_collection.py: ~200 lines (4 dataclasses)
```

---

## Phase 6: PipelineRunner

### Goal

Extract `run_project()`, `run_job()`, `run_project_streaming()`, `dry_run()`, `_execute_all()`, `_warm_cache_for_level()`, `_build_system_prompt()`, `_build_project_context()` into a `PipelineRunner` class. The `Orchestrator` keeps the same public API but delegates to the runner.

### Critical Constraint

`_execute_task()` (L2883) **stays in Orchestrator**. It's the core generate→critique→revise→evaluate loop that touches budget, telemetry, circuit breaker, and state checkpointing simultaneously. Extracting it would require threading all those references through a protocol — too risky.

### Strategy: Start with the Easy Methods

Extract in dependency order, each method verified standalone:

| Step | Methods | Risk |
|------|---------|------|
| **A: Context** | `_build_system_prompt`, `_build_project_context` | None — pure functions, no Orchestrator state |
| **B: Cache warm** | `_warm_cache_for_level` | Low — only touches `self.cache`, `self.client` |
| **C: Execute all** | `_execute_all` | Medium — iterates tasks, calls `_execute_task` |
| **D: Entry points** | `run_project`, `run_job`, `run_project_streaming`, `dry_run` | High — public API surface |

### Architecture

```python
# New file: orchestrator/pipeline_runner.py

class PipelineRunner:
    """Executes the generate->critique->revise->evaluate pipeline."""

    def __init__(self, orch: "Orchestrator"):
        self._orch = orch  # Back-reference for _execute_task, state, budget

    def build_system_prompt(self, task_type: str = "") -> str:
        ...

    def build_project_context(self) -> str:
        ...

    async def warm_cache_for_level(self, tasks, runnable) -> None:
        ...

    async def execute_all(self, tasks, execution_order) -> None:
        ...

    async def run_project(self, description, criteria) -> ProjectState:
        ...

    async def run_job(self, spec: JobSpec) -> ProjectState:
        ...
```

### Orchestrator Becomes a Facade

```python
class Orchestrator:
    def __init__(self, ...):
        # ... existing constructor ...
        self._pipeline = PipelineRunner(self)

    async def run_project(self, desc, criteria):
        return await self._pipeline.run_project(desc, criteria)

    async def run_job(self, spec):
        return await self._pipeline.run_job(spec)
```

### Step A Details (Context Methods)

Both methods are pure functions that happen to be methods. `_build_system_prompt` reads `self._profiles.quality_mode()` and `_build_project_context` iterates `self.results`. Move them as-is, pass parameters explicitly:

```python
def build_system_prompt(quality_mode: str, task_type: str = "") -> str:
    # Formerly self._build_system_prompt(task_type)
    ...

def build_project_context(results: dict, profile_cache) -> str:
    # Formerly self._build_project_context()
    ...
```

### Step C Details (_execute_all)

This method iterates `execution_order`, calls `_execute_task()`, and runs evaluations. It depends on:
- `self._task_guard` (concurrency)
- `self._execute_task` (core loop — stays on Orchestrator)
- `self.budget` (budget checks)
- `self._selector` (model selection)
- `self.results` (shared state)

The runner receives these as back-references through `self._orch`:

```python
async def execute_all(self, tasks: dict, execution_order: list) -> None:
    orch = self._orch
    guard = orch._task_guard
    
    for level in self._topological_levels(execution_order, tasks):
        async with guard:
            results = await asyncio.gather(*[
                orch._execute_task(tasks[tid], ...) for tid in level
            ])
            for r in results:
                orch.results[r.task_id] = r
```

### Step D Details (Entry Points)

`run_project` is the public API entry. It calls:
1. `_decompose()` — already delegated to `DecomposerService`
2. `_execute_all()` — moved to PipelineRunner
3. State checkpointing — uses `self.state_mgr` (public attribute, accessible)
4. Post-project analysis — uses `self._analyze_on_complete`, `self._make_state`

### Verification Per Step

```bash
# After each step:
python -c "from orchestrator import Orchestrator; o = Orchestrator(); print(type(o._pipeline))"
pytest tests/test_bug_fixes_v2.py -v --no-cov
pytest tests/ -x --no-cov -m "not slow and not integration"

# After Step D (entry points), also run integration tests:
pytest tests/integration/ -v --no-cov -m "not requires_api"
```

### Phase 6 Target

```
Before: engine.py ~5,030 lines with pipeline methods inline
After:  engine.py ~4,600 lines (PipelineRunner delegation)
        pipeline_runner.py: ~450 lines
```

---

## Combined Safety Rules

### What MUST NOT Change

1. `_execute_task()` stays on Orchestrator — it's the most tightly-wired method
2. `_decompose()` stays as-is — already delegates to DecomposerService
3. `__aexit__` and `close()` — cleanup code, regression-tested
4. `_record_success` / `_record_failure` — already thin delegations
5. `_get_active_policies` / `_should_exit_early` — small, stable

### Rollback Per Step

Each step is one commit:
```
[Phase 5-A] Extract SafetyServices
[Phase 5-B] Extract IntegrationServices
[Phase 5-C] Extract SearchServices
[Phase 5-D] Extract LearningServices
[Phase 6-A] Extract context methods -> PipelineRunner
[Phase 6-B] Extract warm_cache -> PipelineRunner
[Phase 6-C] Extract execute_all -> PipelineRunner
[Phase 6-D] Extract entry points -> PipelineRunner
```

`git revert <commit>` cleanly reverses any step. Steps are independent within each phase — you can revert Phase 5-C (Search) without affecting Safety or Integration.

### Invariants Checked Per Commit

After each commit, verify:
1. `from orchestrator import Orchestrator` — clean import
2. 9 regression tests pass
3. Unit test suite passes (excluding slow/integration)
4. `Orchestrator()` constructs without exception

### Execution Order

```
Phase 5-A: SafetyServices      [5 min, 0 risk]
Phase 5-B: IntegrationServices  [5 min, 0 risk]
Phase 5-C: SearchServices       [10 min, low]
Phase 5-D: LearningServices     [15 min, medium]
------------------------------------------------
Phase 6-A: Context methods      [10 min, 0 risk]
Phase 6-B: Cache warm           [5 min, low]
Phase 6-C: Execute all          [20 min, medium — touches concurrency]
Phase 6-D: Entry points         [30 min, high — public API surface]
```

### Final State Target

```
engine.py:                  5,030 → ~4,200 lines  (-830, 16% total reduction)
service_collection.py:        ~200 lines  (new)
pipeline_runner.py:           ~450 lines  (new)

Total code: ~830 lines moved to properly-scoped modules
Graph edges from Orchestrator: ~375 → ~80 (79% reduction from original)
```
