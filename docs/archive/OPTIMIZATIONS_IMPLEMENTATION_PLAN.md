# Optimizations Implementation Plan

**Date:** 2026-03-26  
**Source:** `Optimizations.md`  
**Status:** Phase 1 In Progress  

---

## 📊 CURRENT STATE ASSESSMENT

### Existing Implementations (✅ Complete)

All 12 optimizations from `Optimizations.md` are **already implemented** in `orchestrator/cost_optimization/`:

| Tier | Optimization | File | Status |
|------|--------------|------|--------|
| **Tier 1** | Prompt Caching | `prompt_cache.py` | ✅ Implemented |
| **Tier 1** | Batch API | `batch_client.py` | ✅ Implemented |
| **Tier 1** | Output Token Limits | `token_budget.py` | ✅ Implemented |
| **Tier 2** | Model Cascading | `model_cascading.py` | ✅ Implemented |
| **Tier 2** | Speculative Generation | `speculative_gen.py` | ✅ Implemented |
| **Tier 2** | Streaming Validation | `streaming_validator.py` | ✅ Implemented |
| **Tier 3** | Auto Eval Dataset | `tier3_quality.py` | ✅ Implemented |
| **Tier 3** | Structured Output | `structured_output.py` | ✅ Implemented |
| **Tier 3** | Adaptive Temperature | `tier3_quality.py` | ✅ Implemented |
| **Tier 3** | Dependency Context | `dependency_context.py` | ✅ Implemented |
| **Tier 4** | GitHub Auto-Push | `github_push.py` | ✅ Implemented |
| **Tier 4** | Docker Sandbox | `docker_sandbox.py` + `code_executor.py` | ✅ Implemented |

### Integration Gaps (⚠️ Work Required)

| Gap | Description | Priority | Effort |
|-----|-------------|----------|--------|
| **G1** | Cache warming before parallel calls | 🔴 HIGH | 2h |
| **G2** | Model cascading in `_execute_task()` | 🔴 HIGH | 3h |
| **G3** | Dependency context injection | 🔴 HIGH | 2h |
| **G4** | Batch API phase routing | 🟡 MEDIUM | 2h |
| **G5** | Output token limits per phase | 🟡 MEDIUM | 1h |
| **G6** | Speculative generation for critical tasks | 🟡 MEDIUM | 3h |
| **G7** | Streaming validation integration | 🟡 MEDIUM | 2h |
| **G8** | Adaptive temperature in retry logic | 🟡 MEDIUM | 2h |
| **G9** | Auto eval dataset wiring | 🟡 MEDIUM | 2h |

**Total Effort:** 19 hours (~3 working days)

---

## 🎯 PHASE 1: ENGINE INTEGRATION (Priority: HIGH)

### Goal
Wire existing optimizations into main `engine.py` execution path.

### Integration Points

#### 1. Cache Warming Before Parallel Execution

**Location:** `engine.py:_execute_parallel_level()` (around line 1850)

**Current Code:**
```python
async def _execute_parallel_level(self, level, tasks, ...):
    # Directly executes tasks in parallel
    level_results = await asyncio.gather(
        *(_run_one(tid) for tid in runnable),
        return_exceptions=True,
    )
```

**Required Change:**
```python
async def _execute_parallel_level(self, level, tasks, ...):
    # OPTIMIZATION: Warm cache before parallel execution
    if self.optim_config.enable_prompt_caching and self.optim_config.cache_warming_enabled:
        await self._warm_cache_for_level(level, tasks)
    
    # Then execute in parallel
    level_results = await asyncio.gather(...)
```

**Implementation:**
```python
async def _warm_cache_for_level(self, level: List[str], tasks: Dict[str, Task]) -> None:
    """Proactively warm cache with system prompt + project context."""
    from orchestrator.cost_optimization import warm_prompt_cache
    
    # Get system prompt and project context
    system_prompt = self._build_system_prompt()
    project_context = self._build_project_context()
    
    # Warm cache with a single call
    await warm_prompt_cache(
        client=self.client,
        model="claude-sonnet-4.6",  # Use mid-tier model for warming
        system_prompt=system_prompt,
        project_context=project_context,
    )
    logger.info("Cache warmed for parallel execution")
```

**Expected Impact:** Prevents cache miss storm when firing parallel requests.

---

#### 2. Model Cascading in `_execute_task()`

**Location:** `engine.py:_execute_task()` (around line 2100)

**Current Code:**
```python
for iteration in range(task.max_iterations):
    # GENERATE
    gen_response = await self.client.call(
        primary, full_prompt,
        system=system_prompt,
        max_tokens=effective_max_tokens,
        temperature=gen_temperature,
        timeout=gen_timeout,
    )
```

**Required Change:**
```python
for iteration in range(task.max_iterations):
    # OPTIMIZATION: Use model cascading if enabled
    if self.optim_config.enable_cascading and task.type in self.optim_config.cascade_chains:
        from orchestrator.cost_optimization import cascading_generate
        gen_response = await cascading_generate(
            client=self.client,
            prompt=full_prompt,
            system=system_prompt,
            task_type=task.type,
            cascade_chain=self.optim_config.cascade_chains[task.type],
            max_tokens=effective_max_tokens,
        )
    else:
        # Fallback to standard call
        gen_response = await self.client.call(...)
```

**Expected Impact:** 40-60% cost reduction per task.

---

#### 3. Dependency Context Injection

**Location:** `engine.py:_gather_dependency_context()` (around line 2300)

**Current Code:**
```python
def _gather_dependency_context(self, dependencies: List[str]) -> str:
    """Gather outputs from dependency tasks."""
    context_parts = []
    for dep_id in dependencies:
        if dep_id in self.results:
            result = self.results[dep_id]
            context_parts.append(f"## {dep_id}\n{result.output[:self.context_truncation_limit]}")
    return "\n\n".join(context_parts) if context_parts else ""
```

**Required Change:**
```python
def _gather_dependency_context(self, dependencies: List[str], task: Task) -> str:
    """
    Gather outputs from dependency tasks with intelligent injection.
    
    OPTIMIZATION: Use DependencyContextInjector for smart context building.
    """
    if self.optim_config.enable_dependency_context:
        from orchestrator.cost_optimization import inject_dependency_context
        return await inject_dependency_context(
            task_prompt=task.prompt,
            task_type=task.type,
            completed_tasks={dep_id: self.results[dep_id] for dep_id in dependencies if dep_id in self.results},
            dependencies=task.dependencies,
        )
    
    # Fallback to simple concatenation
    context_parts = []
    for dep_id in dependencies:
        if dep_id in self.results:
            result = self.results[dep_id]
            context_parts.append(f"## {dep_id}\n{result.output[:self.context_truncation_limit]}")
    return "\n\n".join(context_parts) if context_parts else ""
```

**Expected Impact:** 30-50% fewer repair cycles from duplicate definitions.

---

#### 4. Batch API Phase Routing

**Location:** `engine.py:_execute_task()` (evaluation/critique phases)

**Current Code:**
```python
# CRITIQUE
critique_response = await self.client.call(
    reviewer, critique_prompt,
    max_tokens=800,
    timeout=60,
)
```

**Required Change:**
```python
# OPTIMIZATION: Use batch API for non-critical phases
from orchestrator.cost_optimization import OptimizationPhase, batch_call

phase = self._get_phase_for_task(task)
if self.optim_config.enable_batch_api and phase in self.optim_config.batch_phases:
    critique_response = await batch_call(
        client=self.client,
        model=reviewer,
        prompt=critique_prompt,
        phase=phase,
        max_tokens=800,
    )
else:
    critique_response = await self.client.call(...)
```

**Configuration:**
```python
# In OptimizationConfig
batch_phases: List[OptimizationPhase] = [
    OptimizationPhase.EVALUATION,
    OptimizationPhase.PROMPT_ENHANCEMENT,
    OptimizationPhase.CONDENSING,
    OptimizationPhase.CRITIQUE,  # Can be batched if latency acceptable
]
```

**Expected Impact:** 50% discount on batched phases.

---

#### 5. Output Token Limits Per Phase

**Location:** `engine.py:_execute_task()` (max_tokens parameter)

**Current Code:**
```python
effective_max_tokens = task.max_output_tokens  # Same for all tasks
```

**Required Change:**
```python
# OPTIMIZATION: Apply phase-specific output token limits
if self.optim_config.enable_token_budget:
    from orchestrator.cost_optimization import get_token_limit
    phase = self._get_phase_for_task(task)
    phase_limit = self.optim_config.output_token_limits.get(phase.value, 4000)
    effective_max_tokens = min(effective_max_tokens, phase_limit)
```

**Configuration:**
```python
# In OptimizationConfig
output_token_limits: Dict[str, int] = {
    "decomposition": 2000,      # Task list, structured JSON
    "generation": 4000,          # Code output
    "critique": 800,             # Score + brief reasoning
    "evaluation": 500,           # Score only
    "prompt_enhancement": 500,   # Enhanced prompt text
    "condensing": 1000,          # Summary
}
```

**Expected Impact:** 15-25% output cost reduction.

---

## 🛠️ IMPLEMENTATION STEPS

### Step 1: Add Optimization Config to Engine

**File:** `orchestrator/engine.py`

**Add to `__init__()`:**
```python
from .cost_optimization import OptimizationConfig, get_optimization_config

# Optimization configuration
self.optim_config: OptimizationConfig = get_optimization_config()

# Initialize optimization components (lazy imports to avoid circular deps)
self._prompt_cacher = None
self._model_cascader = None
self._dependency_injector = None
self._eval_dataset_builder = EvalDatasetBuilder()
```

---

### Step 2: Implement Cache Warming

**File:** `orchestrator/engine.py`

**Add method:**
```python
async def _warm_cache_for_level(self, level: List[str], tasks: Dict[str, Task]) -> None:
    """
    OPTIMIZATION: Proactively warm cache before parallel execution.
    
    Prevents cache miss storm when firing parallel requests.
    """
    if not self.optim_config.enable_prompt_caching or not self.optim_config.cache_warming_enabled:
        return
    
    from orchestrator.cost_optimization import warm_prompt_cache
    
    try:
        # Get system prompt and project context
        system_prompt = self._build_system_prompt()
        project_context = self._build_project_context()
        
        # Warm cache with a single call
        await warm_prompt_cache(
            client=self.client,
            model="claude-sonnet-4.6",  # Use mid-tier model for warming
            system_prompt=system_prompt,
            project_context=project_context,
        )
        logger.info("Cache warmed for parallel execution level %d", level)
    except Exception as e:
        logger.warning(f"Cache warming failed: {e}")
```

**Modify `_execute_parallel_level()`:**
```python
# Before executing parallel tasks
if self.optim_config.enable_prompt_caching and self.optim_config.cache_warming_enabled:
    await self._warm_cache_for_level(level, tasks)
```

---

### Step 3: Integrate Model Cascading

**File:** `orchestrator/engine.py`

**Modify `_execute_task()` generation section:**
```python
# ── GENERATE ──
try:
    # OPTIMIZATION: Use model cascading if enabled
    if (self.optim_config.enable_cascading and 
        task.type.value in self.optim_config.cascade_chains):
        
        from orchestrator.cost_optimization import cascading_generate
        
        cascade_chain = self.optim_config.cascade_chains[task.type.value]
        logger.info(f"  {task.id}: using model cascading")
        
        gen_response = await cascading_generate(
            client=self.client,
            prompt=full_prompt,
            system=system_prompt,
            task_type=task.type,
            cascade_chain=cascade_chain,
            max_tokens=effective_max_tokens,
            timeout=gen_timeout,
        )
    else:
        # Standard single-model generation
        logger.info(f"  {task.id}: calling {primary.value} for generation")
        gen_response = await self.client.call(
            primary, full_prompt,
            system=system_prompt,
            max_tokens=effective_max_tokens,
            temperature=gen_temperature,
            timeout=gen_timeout,
        )
```

---

### Step 4: Integrate Dependency Context

**File:** `orchestrator/engine.py`

**Modify `_gather_dependency_context()`:**
```python
async def _gather_dependency_context(self, task: Task) -> str:
    """
    Gather outputs from dependency tasks with intelligent injection.
    
    OPTIMIZATION: Use DependencyContextInjector for smart context building.
    """
    if not task.dependencies:
        return ""
    
    # Check if dependency context optimization is enabled
    if self.optim_config.enable_dependency_context:
        from orchestrator.cost_optimization import inject_dependency_context
        
        completed_tasks = {
            dep_id: self.results[dep_id]
            for dep_id in task.dependencies
            if dep_id in self.results and self.results[dep_id].success
        }
        
        if completed_tasks:
            try:
                context = await inject_dependency_context(
                    task_prompt=task.prompt,
                    task_type=task.type,
                    completed_tasks=completed_tasks,
                    dependencies=task.dependencies,
                )
                # Return enhanced prompt (context already injected)
                return context
            except Exception as e:
                logger.warning(f"Dependency context injection failed: {e}")
    
    # Fallback to simple concatenation
    context_parts = []
    for dep_id in task.dependencies:
        if dep_id in self.results and self.results[dep_id].success:
            result = self.results[dep_id]
            context_parts.append(
                f"## Already implemented: {dep_id}\n"
                f"```python\n{result.output[:self.context_truncation_limit]}\n```"
            )
    
    if context_parts:
        return (
            f"{task.prompt}\n\n"
            f"## Context: Previously generated code\n"
            f"{''.join(context_parts)}\n\n"
            f"IMPORTANT: Import from and reference these existing modules. "
            f"Do NOT redefine classes/functions that already exist."
        )
    
    return task.prompt
```

**Update call site in `_execute_task()`:**
```python
# OLD
context = self._gather_dependency_context(task.dependencies)

# NEW
context = await self._gather_dependency_context(task)
```

---

## 📈 EXPECTED IMPACT

| Optimization | Before | After | Reduction |
|--------------|--------|-------|-----------|
| **Input tokens** | $1.00/project | $0.15/project | 85% ↓ |
| **Output tokens** | $0.80/project | $0.35/project | 56% ↓ |
| **Model selection** | $0.20/project | $0.08/project | 60% ↓ |
| **Repair cycles** | 2.5 cycles/project | 1.2 cycles/project | 52% ↓ |
| **TOTAL** | ~$2.00/project | ~$0.58/project | **71% ↓** |

---

## ✅ VERIFICATION CHECKLIST

After implementation:

- [ ] Run cost comparison test (before/after)
- [ ] Verify cache warming reduces cache misses
- [ ] Verify cascading exits early ≥40% of time
- [ ] Verify dependency context reduces duplicate definitions
- [ ] Verify batch API used for evaluation/critique
- [ ] Verify output token limits enforced per phase
- [ ] No regression in code quality scores
- [ ] No increase in failed tasks

---

**Next Step:** Proceed with implementation or request modifications to plan.
