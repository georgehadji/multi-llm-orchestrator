# Engine Decomposition Plan

## Overview
Split `orchestrator/engine.py` (4,268 lines) into 6 focused modules following Single Responsibility Principle.

## Target Structure

```
orchestrator/
├── engine.py (NEW - Facade, ~200 lines)
├── engine_core/
│   ├── __init__.py
│   ├── core.py              # Main orchestration loop (~800 lines)
│   ├── task_executor.py     # Task execution logic (~700 lines)
│   ├── critique_cycle.py    # Generate→critique→revise pipeline (~600 lines)
│   ├── fallback_handler.py  # Cross-provider fallback & circuit breaker (~500 lines)
│   ├── budget_enforcer.py   # Budget monitoring & enforcement (~400 lines)
│   └── dependency_resolver.py # DAG resolution & topological sort (~400 lines)
```

## Module Responsibilities

### 1. `engine_core/core.py` (~800 lines)
**Responsibility:** Main orchestration control loop, public API

**Key Methods:**
- `run_project()` - Main entry point
- `run_project_streaming()` - Streaming execution
- `run_job()` - Policy-driven execution
- `__init__()` - Component initialization
- `_decompose_project()` - Spec → tasks
- `_determine_final_status()` - Project status evaluation

**Dependencies:**
- All other engine_core modules
- Minimal direct business logic

### 2. `engine_core/task_executor.py` (~700 lines)
**Responsibility:** Execute individual tasks with validation

**Key Methods:**
- `_execute_task()` - Core task execution
- `_execute_with_retry()` - Retry logic
- `_validate_task_output()` - Output validation
- `_build_task_prompt()` - Prompt construction
- `_extract_code()` - Code extraction from LLM response

**Dependencies:**
- `critique_cycle.py` for generate-critique loops
- `budget_enforcer.py` for budget checks
- `fallback_handler.py` for model selection

### 3. `engine_core/critique_cycle.py` (~600 lines)
**Responsibility:** Generate → critique → revise pipeline

**Key Methods:**
- `_run_critique_cycle()` - Main cycle orchestration
- `_generate_with_model()` - LLM generation
- `_critique_output()` - Cross-model review
- `_revise_with_feedback()` - Revision logic
- `_build_delta_prompt()` - Feedback prompt construction
- `_validate_syntax_streaming()` - Early syntax validation

**Dependencies:**
- `fallback_handler.py` for model routing
- API clients for LLM calls

### 4. `engine_core/fallback_handler.py` (~500 lines)
**Responsibility:** Circuit breaker, model health, fallback chains

**Key Methods:**
- `_get_available_models()` - Get healthy models for task
- `_select_reviewer()` - Cross-provider reviewer selection
- `_record_model_failure()` - Circuit breaker logic
- `_record_model_success()` - Reset failure counters
- `_is_model_healthy()` - Health check

**State:**
- `api_health: dict[Model, bool]`
- `_consecutive_failures: dict[Model, int]`
- `_adaptive_router: AdaptiveRouter`

**Dependencies:**
- `adaptive_router.py` (existing)
- Models for Model enum

### 5. `engine_core/budget_enforcer.py` (~400 lines)
**Responsibility:** Budget monitoring, enforcement, phase partitions

**Key Methods:**
- `_check_budget()` - Mid-task budget check
- `_enforce_phase_partition()` - Budget phase limits
- `_predict_task_cost()` - Cost estimation
- `_record_task_cost()` - Cost tracking
- `_warn_budget_threshold()` - Warning at thresholds

**State:**
- `budget: Budget`
- `_budget_hierarchy: BudgetHierarchy`
- `_cost_predictor: CostPredictor`

**Dependencies:**
- `models.py` for Budget class
- `cost.py` for budget hierarchy

### 6. `engine_core/dependency_resolver.py` (~400 lines)
**Responsibility:** DAG resolution, topological sort, dependency tracking

**Key Methods:**
- `_build_dependency_graph()` - Parse task dependencies
- `_topological_sort()` - Execution ordering
- `_get_ready_tasks()` - Tasks with satisfied dependencies
- `_mark_task_complete()` - Update dependency state
- `_get_dependency_context()` - Build context from dependencies

**State:**
- `dependency_graph: dict[str, list[str]]`
- `execution_order: list[str]`

**Dependencies:**
- `models.py` for Task class
- `planner.py` for ConstraintPlanner

## Migration Strategy

### Phase 1: Extract Helper Functions (Day 1-2)
1. Move all `_build_*` methods → appropriate modules
2. Move all `_validate_*` methods → appropriate modules
3. Move all `_extract_*` methods → appropriate modules

### Phase 2: Extract Core Logic (Day 3-4)
1. Extract `fallback_handler.py` (most isolated)
2. Extract `budget_enforcer.py` (clear boundaries)
3. Extract `dependency_resolver.py` (self-contained)

### Phase 3: Extract Complex Logic (Day 5-6)
1. Extract `critique_cycle.py` (needs careful refactoring)
2. Extract `task_executor.py` (orchestrates other modules)

### Phase 4: Create Facade (Day 7)
1. Create new `engine.py` as facade
2. Update all imports in codebase
3. Run tests to verify functionality

## Testing Strategy

### Unit Tests (Per Module)
- Each module tested in isolation with mocks
- Target: 85% coverage per module

### Integration Tests
- Test module interactions
- Focus on: executor ↔ critique ↔ fallback

### Regression Tests
- Run existing `test_engine_e2e.py` unchanged
- Verify identical behavior

## Import Structure

```python
# orchestrator/engine.py (new facade)
from .engine_core import Orchestrator

__all__ = ['Orchestrator']

# orchestrator/engine_core/__init__.py
from .core import Orchestrator

__all__ = ['Orchestrator']

# Internal imports (within engine_core/)
from .core import OrchestratorCore
from .task_executor import TaskExecutor
from .critique_cycle import CritiqueCycle
from .fallback_handler import FallbackHandler
from .budget_enforcer import BudgetEnforcer
from .dependency_resolver import DependencyResolver
```

## Benefits

1. **Testability:** Each module can be tested in isolation
2. **Maintainability:** Clear boundaries, easier to understand
3. **Parallel Development:** Multiple devs can work on different modules
4. **Type Safety:** Smaller files = better mypy coverage
5. **Performance:** Can optimize hot paths in isolation

## Risks & Mitigation

| Risk | Mitigation |
|------|------------|
| Circular imports | Careful dependency ordering, use TYPE_CHECKING |
| Breaking changes | Maintain same public API in facade |
| Test failures | Run full test suite after each extraction |
| Performance regression | Benchmark before/after |

## Success Criteria

- [ ] All 170+ tests pass
- [ ] Coverage ≥ 70% (up from current ~60%)
- [ ] No circular import errors
- [ ] mypy passes with < 10 errors
- [ ] No performance regression (< 5% latency increase)
