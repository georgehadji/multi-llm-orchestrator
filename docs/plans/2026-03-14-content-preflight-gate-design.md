# Content Preflight Gate — Design Document

**Date**: 2026-03-14
**Feature**: Content Preflight Validation Gate (PASS/ENRICH/WARN/BLOCK)
**Status**: Approved, ready for implementation

---

## Context

`orchestrator/preflight.py` already exists with full PASS/ENRICH/WARN/BLOCK logic and `PreflightValidator` is already initialized in `engine.py` (line 265). However, it is **never called** in the execution pipeline. This feature wires it in as a post-loop delivery gate.

---

## Architecture

Post-loop preflight pass inserted in `_execute_task()` after the iteration loop selects `best_output`:

```
_execute_task() — existing iteration loop
  └─ generate → critique → revise → evaluate (iterations 1-4)
  └─ best_output selected

[NEW] post-loop preflight pass
  └─ _run_preflight_check(task, best_output, best_score)
       ├─ PASS   → deliver as-is
       ├─ WARN   → log warning + score × 0.85 → deliver
       ├─ ENRICH → 1 extra LLM call (enrich reason as critique) → deliver
       └─ BLOCK  → 1 extra LLM call (block reason as critique)
                     ├─ recovered → deliver
                     └─ still BLOCK → TaskStatus.DEGRADED, score=0
```

---

## Files Changed

| File | Change |
|------|--------|
| `orchestrator/models.py` | +2 fields on `TaskResult`: `preflight_result`, `preflight_passed` |
| `orchestrator/engine.py` | +1 method `_run_preflight_check()` + 1 call site after loop |
| `orchestrator/hooks.py` | +1 event type `PREFLIGHT_CHECK` |

**No new files created.** `orchestrator/preflight.py` already exists.

---

## Interface Changes

### TaskResult (models.py)
```python
@dataclass
class TaskResult:
    # ... existing fields unchanged ...
    preflight_result: Optional[PreflightResult] = None   # full check details
    preflight_passed: bool = True                         # quick check flag
```

### _run_preflight_check (engine.py)
```python
async def _run_preflight_check(
    self, task: Task, output: str, score: float, primary: Model
) -> tuple[str, float, Optional[PreflightResult]]:
    """Post-loop preflight gate. Returns (final_output, final_score, preflight_result)."""
```

---

## Behavior by Action

| Action | Behavior |
|--------|----------|
| PASS | Deliver unchanged, score unchanged |
| WARN | Deliver unchanged, score × 0.85, fire PREFLIGHT_CHECK hook, log warning |
| ENRICH | 1 extra LLM call with enrich reason as critique, deliver revised output |
| BLOCK | 1 extra LLM call with block reason as critique → if still BLOCK: DEGRADED |

---

## Error Handling

- `PreflightValidator` exception → fail-open (treat as PASS), log error
- Extra LLM call fails (BLOCK/ENRICH retry) → use original `best_output` + WARN
- Backward compatible: existing `TaskResult` without preflight → `preflight_passed=True`, `preflight_result=None`

---

## Test Plan

**File**: `tests/test_preflight_integration.py` (5 tests)

| Test | Scenario | Expected |
|------|----------|---------|
| `test_pass` | Clean output | Delivered unchanged, score unchanged |
| `test_warn` | Suspicious output | Delivered, score × 0.85, hook fired |
| `test_enrich_recovers` | Missing context → retry succeeds | Revised output delivered |
| `test_block_recovers` | Blocked → retry passes preflight | Revised output delivered |
| `test_block_degraded` | Blocked → retry still blocked | `TaskStatus.DEGRADED`, score=0 |
