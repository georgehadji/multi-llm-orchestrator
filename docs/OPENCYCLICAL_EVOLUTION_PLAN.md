# OpenEvolve Integration Enhancement Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-25  
> **Version:** 1.0  
> **Status:** Plan  

---

## Executive Summary

OpenEvolve demonstrated that LLMs can evolve populations of code variants through MAP-Elites quality-diversity algorithms, island migration patterns, and diff-based mutations. The AI Orchestrator already has 7 of the 8 pieces needed (population generation via ARA, evaluation via evaluator, scoring via critique, mutation via ModifyFile, parallelism via agent dispatch). This plan adds the **missing coordination layer**: MAP-Elites selection, island migration, and diff-based mutation strategies.

**Total effort:** 4 days. **Risk:** Low — all additions are additive.

---

## What OpenEvolve Has (7 of 8 pieces)

| Component | OpenEvolve | AI Orchestrator | Status |
|-----------|-----------|----------------|--------|
| **Code population** | MAP-Elites grid initialization | GPT creates code from spec | ✅ Different approach |
| **Mutation engine** | LLM-powered diff-based mutations | CodebaseWriter.ModifyFile | ✅ Full rewrite only |
| **Evaluation** | Unit tests + performance benchmarks | EvaluatorService + QCAgent | ✅ |
| **Selection** | MAP-Elites grid (per-feature best) | Score-based single selection | ❌ Single dimension |
| **Migration** | Ring topology island model | AgentOrchestrator parallel dispatch | ❌ No sharing |
| **Prompt strategies** | 6 prompt templates for mutations | 20 ARA methods | ✅ More coverage |
| **Cascaded eval** | 3-stage evaluation pipeline | CritiqueStage + EvaluateStage | ✅ |
| **Quality-diversity** | Feature dimensions grid | None | ❌ Missing |

**Missing pieces:** 2 (MAP-Elites selection, island migration)

---

## Enhancement 1: MAP-Elites ARA Pipeline (ARA Method #21)

**Target:** Add population-based optimization for optimization-heavy tasks.  
**Effort:** 2 days. **File:** `orchestrator/ara_pipelines.py`

### Where It Fits

```
Task: "Optimize this sorting algorithm for speed" (REASONING, optimization)
  │
  ▼
ARAMethodSelector picks MAP_ELITES
  │
  ▼
MAPElitesPipeline:
  ├─ Initialize: generate 9 variants (3×3 grid: complexity × performance)
  ├─ Evaluate: each variant runs through EvaluatorService
  ├─ Grid: keep best per cell
  ├─ Mutate: top-3 variants diff-mutated by LLM
  ├─ Repeat: 3 generations
  └─ Return: best-performing variant across grid
```

### MAP-Elites Grid

| Complexity ↓ / Performance → | Low | Medium | High |
|------|-----|--------|------|
| **Low** | O(n) naive | O(n log n) merge | O(n log n) heap |
| **Medium** | O(n²) bubble | O(n²) optimized | O(n²) early-exit |
| **High** | O(2^n) brute | O(2^n) branch-bound | O(2^n) memoized |

Each cell stores the **best program** for that complexity-performance pair. After evolution, the **Pareto frontier** (best performance at each complexity) is returned.

### Implementation

```python
class MAPElitesPipeline(BasePipeline):
    """MAP-Elites quality-diversity evolutionary pipeline.

    Generates a population of code variants across a feature grid,
    evaluates each, keeps the best per cell, and mutates elite candidates.
    """
    
    def __init__(self, grid_rows=3, grid_cols=3, generations=3):
        self.grid = [[None] * grid_cols for _ in range(grid_rows)]
        self.generations = generations
    
    async def execute(self, task: Task) -> TaskResult:
        # 1. Initialize population
        variants = await self._initialize(task, self.grid_rows * self.grid_cols)
        
        for gen in range(self.generations):
            # 2. Evaluate each variant
            scored = await self._evaluate(variants)
            
            # 3. Place in grid
            for v, features in scored:
                row = self._complexity_bin(features["complexity"])
                col = self._performance_bin(features["performance"])
                if self.grid[row][col] is None or features["score"] > self.grid[row][col]["score"]:
                    self.grid[row][col] = {"code": v, **features}
            
            # 4. Select elites (top cells)
            elites = self._select_elites(top_n=3)
            
            # 5. Mutate elites
            variants = await self._mutate(elites)
        
        # 6. Return best across grid
        return self._best_variant()
```

### Selection Logic

```python
def _select_elites(self, top_n=3):
    """Select from elite + diverse + exploratory pools."""
    all_cells = [cell for row in self.grid for cell in row if cell]
    
    # 20% elite (highest scorers)
    elite_count = max(1, len(all_cells) // 5)
    elites = sorted(all_cells, key=lambda c: c["score"], reverse=True)[:elite_count]
    
    # 30% diverse (novel features)
    diverse = self._select_novel(all_cells, count=len(all_cells) // 3)
    
    # 50% exploratory (random grid cells)
    import random
    exploratory = random.sample(all_cells, min(len(all_cells) - elite_count - len(diverse), 
                                                 len(all_cells) // 2))
    
    return elites + diverse + exploratory
```

### Integration

```python
# In ara_pipelines.py PipelineFactory
def _register_pipelines():
    _pipelines[ReasoningMethod.MAP_ELITES] = MAPElitesPipeline

# In ara_execution_strategy.py
def select_method(self, task: Task) -> ReasoningMethod:
    if any(kw in task.prompt.lower() for kw in 
           ["optimize", "improve speed", "faster", "performance", "evolution"]):
        return ReasoningMethod.MAP_ELITES
```

---

## Enhancement 2: Diff-Based Mutation Strategy

**Target:** Mutate code via diffs instead of full rewrites — saves ~70% token cost.  
**Effort:** 0.5 day. **File:** `orchestrator/codebase_writer.py` + `models.py`

### Where It Fits

```
Task: ModifyFile with modification_strategy="diff"
  │
  ▼
CodebaseWriter.apply():
  ├─ Read original file content
  ├─ Send OG + prompt to LLM: "Generate a unified diff"
  ├─ Apply diff via unified_diff (not full file write)
  └─ Safety gate: validate syntax of result
```

### Implementation

```python
# In models.py Task dataclass
modification_strategy: str = "replace"  # "replace", "diff", "insert", "patch"

# In codebase_writer.py CodebaseWriter.apply()
if task.modification_strategy == "diff" and target.exists():
    original = self._files.read_file(target)
    # Generate diff through LLM
    diff = await self._generate_diff(task.prompt, original)
    # Apply diff
    patched = self._apply_patch(original, diff)
    self._files.modify_file(target, patched, strategy="replace")
```

**Token savings:** Full rewrite of 200-line file = ~3000 tokens. Diff of 20-line change = ~500 tokens. **83% reduction.**

---

## Enhancement 3: Island Migration for Multi-Agent

**Target:** Multiple DeveloperAgents share best solutions during execution.  
**Effort:** 1 day. **File:** `orchestrator/agents/coordinator.py`

### Where It Fits

```
AgentOrchestrator.execute_goal(island_migration=True)
  │
  ├─ Spawn 3 DeveloperAgents (island-1, island-2, island-3)
  │
  ├─ Every 5 task completions → migration event
  │   ├─ Each agent shares best solution via message bus
  │   ├─ Recipient merges best pieces via ARA Merge
  │   └─ Continue with improved code
  │
  └─ Final result: best across 3 islands
```

### Implementation

```python
class AgentOrchestrator:
    async def execute_goal(
        self, goal: str, island_migration: bool = False, islands: int = 3
    ):
        if island_migration:
            return await self._island_execution(goal, islands)
        return await self._standard_execution(goal)
    
    async def _island_execution(self, goal: str, islands: int = 3) -> dict:
        """Execute with island model — parallel agents sharing best results."""
        migration_frequency = 5  # every 5 steps
        task_counter = 0
        
        for tasks_batch in self._batched_tasks:
            # Parallel island execution
            results = await asyncio.gather(*[
                agent.handle_task(task) for agent in self.islands
            ])
            
            task_counter += 1
            if task_counter % migration_frequency == 0:
                # Migration event
                await self._migrate_best_solutions()
        
        return self._best_across_islands()
    
    async def _migrate_best_solutions(self):
        """Ring topology: island-N shares best with island-N+1."""
        for i in range(len(self.islands)):
            sender = self.islands[i]
            receiver = self.islands[(i + 1) % len(self.islands)]
            
            best = sender.get_best_solution()
            await receiver.merge(best)
```

**Ring topology migration pattern:**
```
Island 1 ──→ Island 2 ──→ Island 3 ──→ Island 1 (cycle)
```

---

## Implementation Order & Effort

| # | Enhancement | Days | Risk | Files |
|---|------------|------|------|-------|
| 1 | MAP-Elites ARA Pipeline (#21) | 2 | Low | `ara_pipelines.py`, `ara_execution_strategy.py` |
| 2 | Diff-based mutation strategy | 0.5 | Low | `models.py`, `codebase_writer.py` |
| 3 | Island migration for multi-agent | 1 | Low | `agents/coordinator.py` |
| **Total** | | **3.5 days** | | |

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| MAP-Elites grid exhausting budget | Cost multiplier = `grid_rows × grid_cols × generations`. Default: 3×3×3 = 27 LLM calls. Cap at 5×5×3 = 75. |
| Diff application corrupting files | Safety gate: validate syntax after applying diff. If broken, rollback. |
| Island migration causing merge conflicts | Merge via ARA Debate: two agents argue merge approach, third judges. |

---

## Verification Gates

- [ ] `python -m orchestrator --project "Optimize bubble sort" --ara=map_elites` — generates 9 variants, evolves 3 generations, returns best
- [ ] Diff mutation: 200-line file, 20-line change → diff generated and applied correctly
- [ ] Island migration: 3 agents → migration at step 5 → recipient has merged solution
- [ ] All existing core tests pass (test_god_file_refactoring.py)
- [ ] Budget remaining after MAP-Elites run is within expected range

---

**Last updated:** 2026-05-25
