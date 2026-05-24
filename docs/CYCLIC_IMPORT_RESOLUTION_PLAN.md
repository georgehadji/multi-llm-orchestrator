# Cyclic Import Resolution Plan — AI Orchestrator

> **Author:** Georgios-Chrysovalantis Chatzivantsidis  
> **Date:** 2026-05-23  
> **Version:** 1.0  
> **Status:** Plan — awaiting implementation  

---

## Executive Summary

The orchestrator has **7 circular import chains** across 332 modules and 772 edges. All 7 chains are resolvable without structural rewrites using three safe techniques: TYPE_CHECKING guards, function-level lazy imports, and utility extraction.

---

## Cyclic Import Inventory

### Chain 1: `output_organizer ↔ autonomous_debugger` (2 nodes)

**Severity:** LOW  
**Evidence:** Both modules import each other at module level  
**Impact:** Import fails if either is loaded before the other; currently works because one is imported first due to call order  
**Resolution:** Guard one direction with `TYPE_CHECKING` + `if TYPE_CHECKING:` block

### Chain 2: `engine ↔ meta_integration` (2 nodes)

**Severity:** MEDIUM  
**Evidence:** `meta_integration.py` imports `Orchestrator` from `engine` at module level; `engine.py` imports from `meta_integration`  
**Impact:** If either module is top-level imported, Python hits an infinite loop  
**Resolution:** Move `meta_integration` import in `engine.py` to function-level (already in `__init__` body, but check for module-level imports)

### Chain 3: `resume_detector ↔ state` (2 nodes)

**Severity:** LOW  
**Evidence:** Both modules import each other's classes at module level  
**Impact:** Similar to Chain 1  
**Resolution:** TYPE_CHECKING guard in one direction

### Chain 4: `policy ↔ models` (2 nodes)

**Severity:** DOCUMENTED (known, handled)  
**Evidence:** `policy.py` uses `from .models import ...` at module level  
**Impact:** Already works via lazy-loading in `__init__.py` and import order  
**Resolution:** Remove `from .models import` from `policy.py` module level; use function-level imports or TYPE_CHECKING

### Chain 5: `models ↔ budget` (2 nodes)

**Severity:** LOW (data-only, unlikely to cause runtime failures)  
**Evidence:** `models.py` imports `Budget` for type annotations; `budget.py` imports from `models` for configuration  
**Impact:** Both modules are domain-layer; circular is benign if only types are imported  
**Resolution:** Already uses `TYPE_CHECKING` in `budget.py`. Verify both directions use guards.

### Chain 6: `engine → engine_deps → agents → engine` (3 nodes)

**Severity:** HIGH  
**Evidence:** `agents.py` imports `Orchestrator` from `engine.py` at module level. `engine_deps.py` imports from agents. `engine.py` imports from `engine_deps.py`.  
**Impact:** This is the most dangerous cycle — it involves the central `engine.py` module  
**Resolution:** Move `from .engine import Orchestrator` in `agents.py` to function-level lazy import (only when agent is actually used)

### Chain 7: `policy → budget → models → policy` (3 nodes)

**Severity:** MEDIUM (mostly domain data, but 3 hops)  
**Evidence:** `policy.py` → `budget.py` → `models.py` → `policy.py`  
**Impact:** Breaks if any of the three modules is imported first without the others available  
**Resolution:** Remove `policy` import from `models.py` (check if it's used at all); `policy.py` should only import types from `models.py` via TYPE_CHECKING

---

## Resolution Strategy

Three safe techniques, ordered by least to most invasive:

### Technique A: TYPE_CHECKING Guard (zero-risk)

```python
# Before (module level)
from .engine import Orchestrator

# After (TYPE_CHECKING guard)
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .engine import Orchestrator

# In method body (function level):
def my_func(self) -> Any:
    from .engine import Orchestrator  # Lazy import, no cycle
    orch = Orchestrator()
```

**When to use:** When the import is only used for type annotations, or when the class is instantiated inside method bodies (not at module level).

### Technique B: Function-Level Lazy Import (low-risk)

```python
class AgentOrchestrator:
    def run(self):
        from .engine import Orchestrator  # Import when needed, not at module load
        orch = Orchestrator()
```

**When to use:** When the import IS used at runtime but only inside specific methods, not in class definitions or module-level code.

### Technique C: Utility Extraction (moderate-risk)

```python
# Move shared function to a new utility module that both modules import from
# Before: engine.py has _get_available_models, meta_integration imports it from engine
# After: Both import _get_available_models from engine_core.utilities
```

**When to use:** When BOTH modules need the same function/class at module level. Extract to a neutral third module.

---

## Detailed Fix Per Chain

### Fix 1: `output_organizer ↔ autonomous_debugger`

**File:** `orchestrator/autonomous_debugger.py`  
**Change:** Move the import of `output_organizer` classes to function level (inside `debug_output()` method)  
**Risk:** None — the import is only used inside one method  

```diff
 # autonomomous_debugger.py
-from .output_organizer import organize_project_output
 
 async def debug_output(self, project_state):
+    from .output_organizer import organize_project_output
     result = organize_project_output(project_state)
```

### Fix 2: `engine ↔ meta_integration`

**File:** `orchestrator/meta_integration.py`  
**Change:** Replace module-level `from .engine import Orchestrator` with TYPE_CHECKING guard. The only runtime usage is in `initialize_meta_optimization()` which receives `orchestrator` as parameter.  

```diff
 # meta_integration.py
-from .engine import Orchestrator
+from __future__ import annotations
+from typing import TYPE_CHECKING
+if TYPE_CHECKING:
+    from .engine import Orchestrator

 def initialize_meta_optimization(orchestrator, ...):
-    # orchestrator is already an Orchestrator instance passed in
+    # orchestrator is already an Orchestrator instance (passed as parameter)
     ...
```

**File:** `orchestrator/engine.py`  
**Change:** Verify that `meta_integration` import in `__init__` is inside the method body (it already is — no change needed).  

### Fix 3: `resume_detector ↔ state`

**File:** `orchestrator/resume_detector.py`  
**Change:** Move `from .state import StateManager` to function-level import  

```diff
 # resume_detector.py
-from .state import StateManager
 
 async def detect_resume_state(self):
+    from .state import StateManager
     state_mgr = StateManager()
```

### Fix 4: `policy ↔ models`

**File:** `orchestrator/policy.py`  
**Change:** Replace module-level `from .models import ...` with TYPE_CHECKING guard  

```diff
 # policy.py
-from .models import TaskType, Model, ProjectState, TaskResult
+from __future__ import annotations
+from typing import TYPE_CHECKING
+if TYPE_CHECKING:
+    from .models import TaskType, Model, ProjectState, TaskResult

 class JobSpec:
-    def __init__(self, task_type: TaskType, ...):
+    def __init__(self, task_type: "TaskType", ...):
         ...
```

### Fix 5: `models ↔ budget`

**Status:** Already handled.  
**Verify:** Both files use `TYPE_CHECKING` guards. No changes needed unless `budget.py` still has module-level import.

### Fix 6: `engine → engine_deps → agents → engine` (3 nodes)

**File:** `orchestrator/agents.py`  
**Change:** Move `from .engine import Orchestrator` to function-level import inside `AgentOrchestrator.run()`  

```diff
 # agents.py
-from .engine import Orchestrator
 
 class AgentOrchestrator:
     async def run(self, job_spec):
+        from .engine import Orchestrator
         orchestrator = Orchestrator(...)
```

**File:** `orchestrator/engine_deps.py`  
**Change:** Check if `agents` import is inside a `try/except` block (it already is — no change needed).  

### Fix 7: `policy → budget → models → policy` (3 nodes)

**File:** `orchestrator/models.py`  
**Change:** Check if `policy` is imported at module level in `models.py`. If yes, move to TYPE_CHECKING or remove.  
**Likely:** `models.py` does NOT import `policy` — the cycle is `policy → budget → models → policy`. The final hop (`models → policy`) may not exist; if it doesn't, the cycle is actually 3 nodes with 2 edges, not a true cycle.  

---

## Implementation Order

| Step | Chain | File Changed | Technique | Risk |
|------|-------|-------------|-----------|------|
| 1 | Chain 1 | `autonomous_debugger.py` | A (function-level) | None |
| 2 | Chain 2 | `meta_integration.py` | A (TYPE_CHECKING) | None |
| 3 | Chain 3 | `resume_detector.py` | A (function-level) | None |
| 4 | Chain 4 | `policy.py` | A (TYPE_CHECKING) | Low |
| 5 | Chain 6 | `agents.py` | B (lazy import) | Low |
| 6 | Chain 5 | Verify — may already be resolved | — | None |
| 7 | Chain 7 | Verify — may be false positive | — | None |

**Total effort:** 1 day (6 file edits, all zero-risk or low-risk)  
**Dependencies:** None — all steps are independent  
**Parallelization:** All steps can be done in one commit  

---

## Verification Gates

After each fix:

1. `python -c "import orchestrator"` — import succeeds without ImportError
2. `python -c "from orchestrator.engine import Orchestrator"` — main class importable
3. `pytest tests/test_god_file_refactoring.py -q --no-cov` — 40 tests pass
4. `pytest tests/test_decomposer.py tests/test_validator.py tests/test_pipeline.py -q --no-cov` — 32 tests pass

After all fixes:

5. `python -c "import orchestrator; print('OK')"` — full import succeeds
6. Re-run cyclic import analysis script — target: **0 cycles**
7. `python -m orchestrator --help` — CLI help works
8. All 72+ tests pass

---

## Rollback Plan

Each fix is a single, independent edit. Revert any file via `git checkout -- <file>`.

**No file is modified structurally** — only import statements change. Module behavior is identical at runtime because all changes use deferred imports (the imported module is loaded on first use, which was the same previous import time).

---

## Risk Assessment

| Risk | Likelihood | Impact |
|------|-----------|--------|
| Deferred import times out at runtime | Very Low | Import is already inside function bodies; no change to call timing |
| TYPE_CHECKING breaks class signature inspection | Very Low | `__future__ import annotations` makes all annotations strings by default; forward references already work |
| Function-level import causes runtime delay | None | Python caches imports; first call is same cost as module-level import |
| Fix breaks an untested code path | Low | All fixes touch only import statements; 72 tests cover the core paths |

---

## Appendix A: Full Import Graph Statistics

```
Total modules: 332
Total import edges: 772
Cyclic chains: 7
  - 2-node cycles: 5
  - 3-node cycles: 2

Files importing from engine.py: 8
  - cli.py (3 imports)
  - agents.py (1 import)
  - app_builder.py (1 import)
  - control_plane.py (1 import)
  - cost_optimization_integration.py (1 import)
  - engine_with_events.py (1 import)
  - meta_integration.py (1 import)
  - nash_stable_orchestrator.py (1 import)
```

## Appendix B: Target State

```
Cyclic chains: 0 (all resolved)
Files importing from engine.py: 8 (unchanged — they need it)
  All imports are either:
    - TYPE_CHECKING guarded (forward references only)
    - Function-level lazy (runtime import only when used)
```

---

**Last updated:** 2026-05-23
