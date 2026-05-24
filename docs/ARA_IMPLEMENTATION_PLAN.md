# ARA Integration Implementation Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-23  
> **Version:** 1.0  
> **Status:** Plan — awaiting implementation  

---

## Executive Summary

The AI Orchestrator has 20 ARA (Advanced Reasoning & Analysis) pipeline methods fully implemented and registered in `PipelineFactory`, but **none are wired into the core execution path** (`engine.py`). This plan integrates them strategically — each method at the pipeline stage where it adds maximum value.

---

## Architecture Overview

### Current State

```
engine.py
  _execute_task()
    ├─→ _pipeline.run()
    │     ├─→ GenerateStage       (single model, single pass)
    │     ├─→ CritiqueStage       (single cross-model review)
    │     ├─→ EvaluateStage       (single evaluator)
    │     ├─→ ValidateStage       (deterministic checks)
    │     ├─→ PreflightStage      (quality gate)
    │     └─→ SelfConsistencyStage (retry with fallback model)
    │
    └─→ ARA methods exist in ara_pipelines.py but are NEVER called
```

### Target State

```
engine.py
  _execute_task()
    ├─→ ARAExecutionStrategy.select(task, budget)
    │     │
    │     ├─ [CODE_GEN, RISK=HIGH] → PersuasionDefense + MultiPerspective review
    │     ├─ [REASONING, COMPLEX]  → SoT + ToT pipeline path  
    │     ├─ [EVALUATE, CRITICAL]  → Jury pipeline path
    │     └─ [default]             → Standard TaskPipeline (current path)
    │
    ├─→ _pipeline.run()   (standard path, unchanged)
    │
    └─→ _ara.execute()    (ARA path, new)
          ├─→ Method selection (rule-based + LLM)
          ├─→ Pipeline dispatch
          └─→ Result integration
```

---

## Phased Implementation Plan

---

### Phase 1: ARA Infrastructure Wiring (Foundation)

**Target:** Make ARA accessible from engine.py with zero impact on existing execution.

**Effort:** 1 day  
**Risk:** None (additive change only)

#### 1.1 Add ARA initialization to `Orchestrator.__init__`

```python
# In engine.py __init__, after self._pipeline init:

# Phase 6+: ARA reasoning pipeline integration
try:
    from .ara_integration import create_ara_integration
    self._ara = create_ara_integration(
        client=self.client,
        cache=self.cache,
        telemetry=self._telemetry,
        enabled=True,
        auto_select=True,
    )
    HAS_ARA = True
except ImportError:
    self._ara = None
    HAS_ARA = False
```

#### 1.2 Add ARA execution method to Orchestrator

```python
async def _execute_task_ara(self, task: Task, policy=None) -> TaskResult:
    """Execute a task through ARA reasoning pipeline."""
    if self._ara is None or not self._ara.config["enabled"]:
        return await self._execute_task(task, policy)  # fallback
    
    context = ""  # Could inject dependency context
    result = await self._ara.execute_task_with_pipeline(
        task=task, context=context,
    )
    return result
```

#### 1.3 Add ARA to `engine_deps.py` (optional import)

```python
# In engine_deps.py:
try:
    from .ara_integration import create_ara_integration, ARAPipelineIntegration
    HAS_ARA = True
except ImportError:
    HAS_ARA = False
    create_ara_integration = None
    ARAPipelineIntegration = None
```

**Verification:** `python -c "from orchestrator.engine import Orchestrator; assert hasattr(Orchestrator(), '_ara')"` — True if available.

---

### Phase 2: ARA Execution Strategy (Method Selector Wiring)

**Target:** Create an `ARAExecutionStrategy` class that decides whether to use ARA and which method.

**Effort:** 2 days  
**Risk:** Low  
**New file:** `orchestrator/ara_execution_strategy.py`

#### 2.1 ARAExecutionStrategy class

```python
@dataclass
class ARAStrategyConfig:
    """Configuration for ARA execution strategy."""
    enabled: bool = True
    # Thresholds
    complexity_threshold: float = 0.7     # Use ARA when complexity > this
    quality_deficit_threshold: float = 0.3  # Use ARA after score deficit
    budget_fraction_ara: float = 0.3       # Max budget fraction for ARA
    # Per-task-type method defaults
    default_methods: dict[TaskType, ReasoningMethod] = {
        TaskType.CODE_GEN: ReasoningMethod.PERSUASION_DEFENSE,
        TaskType.REASONING: ReasoningMethod.SOT,
        TaskType.EVALUATE: ReasoningMethod.JURY,
        TaskType.CODE_REVIEW: ReasoningMethod.MULTI_PERSPECTIVE,
    }

class ARAExecutionStrategy:
    """Decides when and how to use ARA for task execution."""
    
    def __init__(self, ara_integration, config=None):
        self._ara = ara_integration
        self._config = config or ARAStrategyConfig()
        self._selection = MethodSelector()
    
    def should_use_ara(self, task: Task, current_score: float = 0.0) -> bool:
        """Decide if ARA should be used for this task."""
        if not self._config.enabled or self._ara is None:
            return False
        
        # Always use for explicitly configured task types
        if task.type in self._config.default_methods:
            return True
        
        # Use for self-consistency retry when quality is low
        if current_score > 0 and current_score < self._config.quality_deficit_threshold:
            return True
        
        return False
    
    def select_method(self, task: Task) -> ReasoningMethod:
        """Select the best ARA method for this task."""
        return self._ara.select_method_for_task(task).method
```

#### 2.2 Wire into `_execute_task_via_pipeline`

```python
async def _execute_task_via_pipeline(self, task: Task, policy=None) -> TaskResult:
    """Execute task using pipeline, optionally escalated to ARA."""
    
    # Check if this task qualifies for ARA execution
    if hasattr(self, '_ara_strategy') and self._ara_strategy.should_use_ara(task):
        try:
            result = await self._execute_task_ara(task, policy)
            if result.score >= 0.7:  # ARA produced good output
                return result
        except Exception as e:
            logger.warning("ARA execution failed, falling back to pipeline: %s", e)
    
    # Standard pipeline execution (current path)
    ctx = PipelineContext(task=task)
    while True:
        ctx = await self._pipeline.run(ctx)
        if ctx.abort_reason == "retry_for_quality":
            ctx.reset_for_retry()
            continue
        break
    return ctx.to_task_result(...)
```

**Verification:** Run a project with `--dry-run`. Verify that ARA-eligible tasks are routed through `_execute_task_ara`.

---

### Phase 3: Self-Consistency Enhancement (CoVE + Debate + Jury)

**Target:** Replace the simple retry logic with structured improvement.

**Effort:** 3 days  
**Risk:** Medium (changes retry behavior)

#### 3.1 Enhanced Self-Consistency Stage

```python
class EnhancedSelfConsistencyStage:
    """Self-consistency with ARA-powered improvement."""
    
    def __init__(self, ara_strategy, max_attempts=2):
        self._ara = ara_strategy
        self._max_attempts = max_attempts
    
    async def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.score >= 0.7 or ctx.attempt >= self._max_attempts:
            return ctx
        
        ctx.attempt += 1
        task_type = ctx.task.type
        
        if task_type == TaskType.CODE_REVIEW and ctx.score < 0.5:
            # Use Debate for low-quality code review
            ctx.abort_reason = "ara_debate"
            ctx.ara_method = ReasoningMethod.DEBATE
        elif task_type in (TaskType.CODE_GEN, TaskType.REASONING) and ctx.score < 0.6:
            # Use CoVE for fact-heavy generation
            ctx.abort_reason = "ara_cove"
            ctx.ara_method = ReasoningMethod.COVE
        else:
            # Standard fallback retry
            ctx.abort_reason = "retry_for_quality"
            ctx.task.preferred_model = FALLBACK_CHAIN.get(ctx.model, ctx.model)
        
        ctx.should_abort = True
        return ctx
```

#### 3.2 Wire ARA retry into pipeline dispatcher

In `_execute_task_via_pipeline`:

```python
if ctx.abort_reason == "ara_cove" or ctx.abort_reason == "ara_debate":
    # Run ARA with the critique as context
    task_copy = ctx.task
    task_copy.context = ctx.critique[:2000]
    ara_result = await self._execute_task_ara(task_copy)
    if ara_result.score > ctx.score:
        return ara_result  # ARA improvement succeeded
    # Fall through to standard retry
```

**Verification:** Create a task with known low-quality output, verify CoVE improves the output score.

---

### Phase 4: Code Review Enhancement (PersuasionDefense)

**Target:** Run PersuasionDefense on ALL code generation outputs before final delivery.

**Effort:** 2 days  
**Risk:** Medium (adds processing to every code task)

#### 4.1 PersuasionDefense Integration Stage

```python
class PersuasionDefenseStage:
    """Post-generation claim verification for code tasks."""
    
    def __init__(self, ara_integration):
        self._ara = ara_integration
    
    async def process(self, ctx: PipelineContext) -> PipelineContext:
        if ctx.task.type != TaskType.CODE_GEN:
            return ctx
        if not ctx.output:
            return ctx
        
        # Run PersuasionDefense on the output
        verification_task = Task(
            id=f"{ctx.task.id}_verify",
            type=TaskType.EVALUATE,
            prompt=ctx.output[:4000],
            max_output_tokens=1000,
        )
        
        result = await self._ara.execute_task_with_pipeline(
            task=verification_task,
            method=ReasoningMethod.PERSUASION_DEFENSE,
        )
        
        ctx.metadata["persuasion_defense"] = {
            "claims": result.metadata.get("claims", 0),
            "verified": result.metadata.get("verified", 0),
            "conflicts": result.metadata.get("conflicts", 0),
        }
        
        if result.score < 0.5:
            ctx.should_abort = True
            ctx.abort_reason = "verification_failed"
        
        return ctx
```

**Verification:** Run PersuasionDefense on sample generated code. Verify it catches unsupported claims.

---

### Phase 5: Complex Reasoning Enhancement (SoT + ToT + Self-Discover)

**Target:** Route REASONING tasks through structured pipelines.

**Effort:** 2 days  
**Risk:** Low (only affects REASONING task type)

#### 5.1 Reasoning Dispatcher

```python
class ARAReasoningDispatcher:
    """Dispatch REASONING tasks to optimal ARA method."""
    
    async def execute(self, task: Task, ara_integration) -> TaskResult:
        prompt_len = len(task.prompt)
        has_subproblems = any(kw in task.prompt.lower() for kw in 
            ["multiple", "several", "complex", "trade-off", "choose"])
        
        if has_subproblems:
            method = ReasoningMethod.SOT
        elif "decision" in task.prompt.lower() or "choose" in task.prompt.lower():
            method = ReasoningMethod.TOT
        else:
            method = ReasoningMethod.MULTI_PERSPECTIVE
        
        return await ara_integration.execute_task_with_pipeline(
            task=task, method=method,
        )
```

**Integration:** Wire into `_execute_task_via_pipeline` — when task type is REASONING and complexity is MEDIUM+, dispatch to ARA.

**Verification:** Run a complex reasoning task through SoT, verify it produces a more structured answer than single-pass generation.

---

### Phase 6: Decomposition Enhancement (Multi-Perspective)

**Target:** Use Multi-Perspective for richer project decomposition.

**Effort:** 1 day  
**Risk:** Low (additive — old decomposition path preserved)

#### 6.1 Integration

```python
async def _decompose(self, project, criteria, ...):
    # Standard decomposition (unchanged)
    tasks = await self._decomposer.decompose(project, criteria, ...)
    
    # Enhance with Multi-Perspective if ARA available
    if hasattr(self, '_ara') and self._ara and self._ara.config["enabled"]:
        try:
            perspective_task = Task(
                id="decomp_perspectives",
                type=TaskType.REASONING,
                prompt=f"Review this task breakdown for completeness:\n{tasks}",
                max_output_tokens=2000,
            )
            enhanced = await self._ara.execute_task_with_pipeline(
                task=perspective_task,
                method=ReasoningMethod.MULTI_PERSPECTIVE,
            )
            # The output includes perspective analysis — log it
            logger.info("Multi-Perspective decomposition analysis: %s", 
                       enhanced.output[:500])
        except Exception:
            pass  # Fail-open: standard decomposition is fine
    
    return tasks
```

**Verification:** Run decomposition with project `"Build an e-commerce platform with inventory and payments"`. Verify Multi-Perspective catches cross-cutting concerns.

---

### Phase 7: Budget-Aware Method Escalation

**Target:** Allocate reasoning budget intelligently across tasks.

**Effort:** 2 days  
**Risk:** Medium (affects budget tracking)

#### 7.1 MethodBudget class

```python
class MethodBudget:
    """Tracks and enforces reasoning method cost multipliers."""
    
    # Estimated cost multipliers for each method
    METHOD_COSTS = {
        ReasoningMethod.MULTI_PERSPECTIVE: 4.0,
        ReasoningMethod.ITERATIVE: 3.0,
        ReasoningMethod.DEBATE: 4.0,
        ReasoningMethod.RESEARCH: 3.0,
        ReasoningMethod.JURY: 5.0,
        ReasoningMethod.COVE: 2.0,
        ReasoningMethod.SOT: 3.0,
        ReasoningMethod.TOT: 5.0,
        ReasoningMethod.PERSUASION_DEFENSE: 3.0,
        # ... all 20 methods
    }
    
    def can_afford(self, method: ReasoningMethod, remaining_budget: float, 
                   estimated_cost_per_call: float) -> bool:
        multiplier = self.METHOD_COSTS.get(method, 1.0)
        estimated_total = estimated_cost_per_call * multiplier
        return estimated_total <= remaining_budget * 0.3  # Max 30% for this task
```

#### 7.2 Integration

```python
# In _execute_task_via_pipeline:
if hasattr(self, '_method_budget'):
    if not self._method_budget.can_afford(selected_method, ...):
        selected_method = ReasoningMethod.ITERATIVE  # Cheaper fallback
```

**Verification:** Run a project with budget $1.00. Verify that expensive methods are downgraded when budget runs low.

---

### Phase 8: Pre-Mortem Injection

**Target:** Pre-Mortem analysis before code generation for high-risk tasks.

**Effort:** 1 day  
**Risk:** Low

#### 8.1 Implementation

```python
async def _inject_pre_mortem_context(self, task: Task) -> str:
    """Run pre-mortem analysis and inject findings into task context."""
    if not hasattr(self, '_ara') or not self._ara:
        return ""
    
    pm_task = Task(
        id=f"{task.id}_pre_mortem",
        type=TaskType.REASONING,
        prompt=f"Assume this module has already failed in production:\n{task.prompt}\n\nWhy did it fail? What were the root causes?",
        max_output_tokens=1000,
    )
    
    result = await self._ara.execute_task_with_pipeline(
        task=pm_task,
        method=ReasoningMethod.PRE_MORTEM,
    )
    
    return f"\n\n[Pre-Mortem Analysis - failures to avoid]\n{result.output}"
```

**Verification:** Run Pre-Mortem on a task, verify the output contains actionable failure-prevention advice.

---

### Phase 9: Testing, Monitoring, Documentation

**Effort:** 2 days  
**Risk:** None

#### 9.1 Tests

- Test ARAExecutionStrategy correctly selects methods per task type
- Test self-consistency with CoVE improves scores on known-bad outputs
- Test PersuasionDefense catches hallucinated claims
- Test budget escalation downgrades methods when budget is low
- Test fallback path: ARA failure → standard pipeline executes correctly

#### 9.2 Monitoring

- Log ARA method used per task
- Track ARA execution success rate vs standard pipeline
- Track ARA cost overhead vs output quality improvement
- Expose metrics via `ObservabilityService`

#### 9.3 Documentation

- Update `AGENTS.md` with ARA integration details
- Add ARA configuration env vars to `.env.example`
- Document how to enable/disable ARA per task type

---

## Implementation Order & Effort Summary

| Phase | Enhancement | Days | Risk | Dependencies |
|-------|-------------|------|------|--------------|
| 1 | ARA infrastructure wiring | 1 | None | — |
| 2 | ARA execution strategy | 2 | Low | Phase 1 |
| 3 | Self-consistency (CoVE + Debate) | 3 | Medium | Phase 2 |
| 4 | Code review (PersuasionDefense) | 2 | Medium | Phase 2 |
| 5 | Complex reasoning (SoT + ToT) | 2 | Low | Phase 2 |
| 6 | Decomposition (Multi-Perspective) | 1 | Low | Phase 2 |
| 7 | Budget-aware escalation | 2 | Medium | Phase 5 |
| 8 | Pre-Mortem injection | 1 | Low | Phase 2 |
| 9 | Testing + monitoring + docs | 2 | None | All above |
| **Total** | | **16 days** | | |

---

## Files to Create / Modify

| File | Action | Phase |
|------|--------|-------|
| `orchestrator/ara_execution_strategy.py` | **Create** | 2 |
| `orchestrator/engine.py` | Modify (`__init__`, `_execute_task_via_pipeline`) | 1-8 |
| `orchestrator/engine_core/stages/` | Modify / Create ARA stages | 3-5 |
| `orchestrator/engine_deps.py` | Add `HAS_ARA` import block | 1 |
| `tests/test_ara_integration.py` | **Create** | 9 |
| `docs/ARA_INTEGRATION_ANALYSIS.md` | Exists | — |
| `.env.example` | Add ARA env vars | 9 |

---

## Rollback Safety

| Phase | Rollback |
|-------|----------|
| 1 | Set `HAS_ARA=False` via import or feature flag |
| 2 | `ARAExecutionStrategy.enabled = False` |
| 3-9 | Each `ARAReasoningDispatcher` check falls back to standard execution on failure |
| All | Remove `_ara` attribute from Orchestrator; old execution paths unchanged |

**Key principle:** Every ARA integration is optional and fails-open. If ARA fails or is disabled, the orchestrator falls back to the existing generation→critique→revise→evaluate loop.

---

## Verification Gates

- [ ] `python -m orchestrator --help` works
- [ ] All existing tests pass with `--no-cov`
- [ ] All new ARA integration tests pass
- [ ] Dry run with ARA enabled produces correct output
- [ ] Dry run with ARA disabled produces identical output to before
- [ ] Budget tracking still accurate with ARA cost multipliers
- [ ] No regression in execution time for standard tasks

---

## Appendix A: Environment Variables

```bash
# ARA Pipeline Configuration
ARA_ENABLED=true                    # Enable/disable ARA pipelines
ARA_AUTO_SELECT=true                # Auto-select method per task
ARA_MAX_COST_MULTIPLIER=5.0         # Maximum cost multiplier allowed
ARA_MAX_TIME_MULTIPLIER=2.0         # Maximum time multiplier allowed
ARA_METHOD_BUDGET_FRACTION=0.3      # Max fraction of budget for ARA
```

## Appendix B: Method → Task Type Default Mapping

| Method | Default Task Type | Rationale |
|--------|-------------------|-----------|
| CoVE | CODE_GEN, REASONING | Factual accuracy improvement |
| Debate | CODE_REVIEW, EVALUATE | Multi-perspective quality assessment |
| Jury | EVALUATE, CRITICAL tasks | Multi-model consensus |
| SoT | REASONING | Structured multi-part problems |
| ToT | REASONING (decision) | Strategic decision problems |
| Multi-Perspective | CODE_GEN, DECOMPOSITION | Comprehensive analysis |
| PersuasionDefense | CODE_GEN, CODE_REVIEW | Hallucination detection |
| Pre-Mortem | CODE_GEN | Failure anticipation |
| Iterative | Any (budget-constrained) | Cheap self-improvement |
| Brainstorming | WRITING | Creative ideation |
| Research | DATA_EXTRACT | Web-aware research |

---

**Last updated:** 2026-05-23
