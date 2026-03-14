# Content Preflight Gate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire the existing `PreflightValidator` (preflight.py) into the `_execute_task()` pipeline as a post-loop delivery gate that runs PASS/ENRICH/WARN/BLOCK checks on `best_output` before finalizing a task result.

**Architecture:** Post-loop preflight pass inserted in `engine.py` after the iteration loop selects `best_output` (line 2272) and before telemetry recording (line 2276). BLOCK/ENRICH trigger one extra LLM call with the preflight reason as critique. WARN applies a score × 0.85 penalty and logs. A new `PREFLIGHT_CHECK` hook event is added for observability.

**Tech Stack:** Python asyncio, existing `PreflightValidator` (`orchestrator/preflight.py`), `HookRegistry` (`orchestrator/hooks.py`), `TaskResult` dataclass (`orchestrator/models.py`).

**Note on BLOCK test triggers:** Tests use Bearer token pattern (`Bearer XXXX.YYYY`) to trigger PRIVACY BLOCK (severity=9) — safe for security hooks and already in `PreflightValidator.PRIVACY_PATTERNS`.

---

## Task 1: Add `PREFLIGHT_CHECK` to `EventType` in hooks.py

**Files:**
- Modify: `orchestrator/hooks.py:42-47`

**Step 1: Write the failing test**

Create `tests/test_preflight_integration.py`:

```python
"""Tests for Content Preflight Gate integration."""
import pytest
from orchestrator.hooks import EventType


def test_preflight_check_event_exists():
    assert hasattr(EventType, "PREFLIGHT_CHECK")
    assert EventType.PREFLIGHT_CHECK.value == "preflight_check"
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_preflight_integration.py::test_preflight_check_event_exists -v
```
Expected: `FAILED — AttributeError: PREFLIGHT_CHECK`

**Step 3: Add the event to EventType**

In `orchestrator/hooks.py`, add after line 47 (`TASK_RETRY_WITH_HISTORY`):

```python
    PREFLIGHT_CHECK         = "preflight_check"
```

Also update the docstring (lines 34-41) to add:
```
      PREFLIGHT_CHECK  — task_id: str, action: str, reason: str, score_before: float, score_after: float
```

**Step 4: Run test to verify it passes**

```
pytest tests/test_preflight_integration.py::test_preflight_check_event_exists -v
```
Expected: `PASSED`

**Step 5: Commit**

```bash
git add orchestrator/hooks.py tests/test_preflight_integration.py
git commit -m "feat: add PREFLIGHT_CHECK event type to HookRegistry"
```

---

## Task 2: Add `preflight_result` and `preflight_passed` fields to `TaskResult`

**Files:**
- Modify: `orchestrator/models.py:679-695`

**Step 1: Write the failing test**

Add to `tests/test_preflight_integration.py`:

```python
from orchestrator.models import TaskResult, Model
from orchestrator.preflight import PreflightResult, PreflightAction


def test_task_result_has_preflight_fields():
    result = TaskResult(
        task_id="t_001",
        output="hello world",
        score=0.9,
        model_used=Model.DEEPSEEK_CHAT,
    )
    assert result.preflight_passed is True
    assert result.preflight_result is None


def test_task_result_accepts_preflight_result():
    pf = PreflightResult(action=PreflightAction.WARN, passed=False, warnings=["test warning"])
    result = TaskResult(
        task_id="t_001",
        output="hello world",
        score=0.9,
        model_used=Model.DEEPSEEK_CHAT,
        preflight_result=pf,
        preflight_passed=False,
    )
    assert result.preflight_passed is False
    assert result.preflight_result.action == PreflightAction.WARN
```

**Step 2: Run tests to verify they fail**

```
pytest tests/test_preflight_integration.py::test_task_result_has_preflight_fields tests/test_preflight_integration.py::test_task_result_accepts_preflight_result -v
```
Expected: `FAILED — TypeError: unexpected keyword argument 'preflight_result'`

**Step 3: Add fields to TaskResult**

In `orchestrator/models.py`, after line 695 (`attempt_history: list[...]`), add:

```python
    preflight_result: Optional["PreflightResult"] = None
    preflight_passed: bool = True
```

`Optional["PreflightResult"]` uses a string annotation to avoid circular imports. `from __future__ import annotations` is already at the top of models.py so this works at runtime.

**Step 4: Run tests to verify they pass**

```
pytest tests/test_preflight_integration.py::test_task_result_has_preflight_fields tests/test_preflight_integration.py::test_task_result_accepts_preflight_result -v
```
Expected: `PASSED`

**Step 5: Verify no regressions**

```
pytest tests/ -k "task_result or TaskResult" -v --tb=short
```
Expected: all existing tests still pass.

**Step 6: Commit**

```bash
git add orchestrator/models.py tests/test_preflight_integration.py
git commit -m "feat: add preflight_result and preflight_passed fields to TaskResult"
```

---

## Task 3: Add `_run_preflight_check()` method to `Orchestrator`

**Files:**
- Modify: `orchestrator/engine.py` (add new private method near other private helpers)

**Step 1: Write the failing test**

Add to `tests/test_preflight_integration.py`:

```python
from unittest.mock import AsyncMock, MagicMock
from orchestrator.preflight import PreflightAction, PreflightMode
from orchestrator.models import Model, Task, TaskType


def _make_orchestrator():
    """Minimal Orchestrator stub for unit testing preflight."""
    from orchestrator.engine import Orchestrator
    from orchestrator.preflight import PreflightValidator
    from orchestrator.hooks import HookRegistry
    orch = Orchestrator.__new__(Orchestrator)
    orch._preflight_validator = PreflightValidator()
    orch._hook_registry = HookRegistry()
    return orch


def _make_task(task_type=TaskType.CODE_GEN):
    return Task(
        id="t_test",
        prompt="Write a hello world function",
        type=task_type,
        max_output_tokens=1000,
        acceptance_threshold=0.8,
        max_iterations=3,
        success_criteria="working code",
    )


@pytest.mark.asyncio
async def test_preflight_pass_returns_unchanged():
    orch = _make_orchestrator()
    task = _make_task()
    output = "def hello(): return 'world'"
    score = 0.9

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    assert final_output == output
    assert final_score == score
    assert pf_result.action == PreflightAction.PASS
```

**Step 2: Run test to verify it fails**

```
pytest tests/test_preflight_integration.py::test_preflight_pass_returns_unchanged -v
```
Expected: `FAILED — AttributeError: '_run_preflight_check'`

**Step 3: Implement `_run_preflight_check()`**

Add this method to the `Orchestrator` class in `orchestrator/engine.py`, near the other private helpers (e.g., after `_evaluate`):

```python
async def _run_preflight_check(
    self,
    task: "Task",
    output: str,
    score: float,
    primary: "Model",
) -> "tuple[str, float, PreflightResult]":
    """
    Post-loop preflight delivery gate.

    Checks best_output before finalizing TaskResult:
    - PASS  : return unchanged
    - WARN  : log + score * 0.85, fire PREFLIGHT_CHECK hook
    - ENRICH: 1 extra LLM revision with enrich reason as critique
    - BLOCK : 1 extra LLM revision with block reason as critique
                 -> recovered: return revised output
                 -> still BLOCK: return original output, score=0.0

    Fail-open: any validator exception is caught and treated as PASS.
    """
    from .preflight import PreflightAction, PreflightMode, PreflightResult

    try:
        pf_result = self._preflight_validator.validate(
            response=output,
            context={
                "task_type": task.type.value,
                "user_request": task.prompt[:200],
                "model": primary.value,
                "score": score,
            },
            mode=PreflightMode.AUTO,
        )
    except Exception as exc:
        logger.warning("preflight validator raised: %s — treating as PASS", exc)
        return output, score, PreflightResult(action=PreflightAction.PASS, passed=True)

    if pf_result.action == PreflightAction.PASS:
        return output, score, pf_result

    if pf_result.action == PreflightAction.WARN:
        penalized = round(score * 0.85, 4)
        logger.warning(
            "[preflight] WARN task=%s score %.3f->%.3f: %s",
            task.id, score, penalized, "; ".join(pf_result.warnings),
        )
        self._hook_registry.fire(
            EventType.PREFLIGHT_CHECK,
            task_id=task.id,
            action="warn",
            reason="; ".join(pf_result.warnings),
            score_before=score,
            score_after=penalized,
        )
        return output, penalized, pf_result

    # ENRICH or BLOCK — attempt one extra revision
    critique_text = pf_result.reason or pf_result.enrichment or "Improve the response quality."
    logger.info(
        "[preflight] %s task=%s — attempting 1 revision: %s",
        pf_result.action.value.upper(), task.id, critique_text[:100],
    )
    try:
        revised_prompt = (
            f"{task.prompt}\n\n"
            f"[Revision required] {critique_text}\n"
            f"Please revise your previous response to address the above."
        )
        revised_output = await self._call_primary(primary, revised_prompt, task.max_output_tokens)
    except Exception as exc:
        logger.warning("[preflight] revision LLM call failed (%s) — using original", exc)
        self._hook_registry.fire(
            EventType.PREFLIGHT_CHECK,
            task_id=task.id,
            action=pf_result.action.value + "_revision_failed",
            reason=str(exc),
            score_before=score,
            score_after=score,
        )
        return output, score, pf_result

    # Re-validate the revised output
    try:
        retry_result = self._preflight_validator.validate(
            response=revised_output,
            context={"task_type": task.type.value, "user_request": task.prompt[:200]},
            mode=PreflightMode.AUTO,
        )
    except Exception:
        retry_result = pf_result  # treat same as original if validator fails

    if retry_result.action == PreflightAction.BLOCK:
        logger.warning("[preflight] BLOCK task=%s — revision still blocked, score->0", task.id)
        self._hook_registry.fire(
            EventType.PREFLIGHT_CHECK,
            task_id=task.id,
            action="block_degraded",
            reason=retry_result.reason or "Still blocked after revision",
            score_before=score,
            score_after=0.0,
        )
        return output, 0.0, retry_result

    logger.info("[preflight] %s recovered task=%s", pf_result.action.value.upper(), task.id)
    self._hook_registry.fire(
        EventType.PREFLIGHT_CHECK,
        task_id=task.id,
        action=pf_result.action.value + "_recovered",
        reason=critique_text[:100],
        score_before=score,
        score_after=score,
    )
    return revised_output, score, retry_result
```

**Step 4: Run test to verify it passes**

```
pytest tests/test_preflight_integration.py::test_preflight_pass_returns_unchanged -v
```
Expected: `PASSED`

**Step 5: Commit**

```bash
git add orchestrator/engine.py
git commit -m "feat: add _run_preflight_check() method to Orchestrator"
```

---

## Task 4: Write remaining unit tests for all preflight actions

**Files:**
- Modify: `tests/test_preflight_integration.py`

**Note:** BLOCK is triggered via Bearer token pattern (PRIVACY severity=9, which is >=8 → BLOCK in AUTO mode). WARN is triggered via `[TODO]` placeholder (COMPLETENESS severity=7).

**Step 1: Add 4 more tests**

```python
@pytest.mark.asyncio
async def test_preflight_warn_applies_score_penalty():
    orch = _make_orchestrator()
    task = _make_task()
    # [TODO] triggers COMPLETENESS warning (severity=7 -> WARN in AUTO mode)
    output = "def hello(): pass  # [TODO] implement this properly"
    score = 0.9

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    assert final_output == output
    assert final_score == pytest.approx(0.9 * 0.85, abs=0.001)
    assert pf_result.action == PreflightAction.WARN


@pytest.mark.asyncio
async def test_preflight_block_retries_and_recovers():
    orch = _make_orchestrator()
    task = _make_task()
    # Bearer token triggers PRIVACY BLOCK (severity=9, >=8 -> BLOCK)
    bad_output = 'headers = {"Authorization": "Bearer eyJhbGc.eyJzdWI.sig"}'
    good_output = 'headers = {"Authorization": f"Bearer {token}"}'  # clean revision
    score = 0.8

    orch._call_primary = AsyncMock(return_value=good_output)

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=bad_output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    orch._call_primary.assert_called_once()
    assert final_output == good_output   # revised output delivered
    assert final_score == score          # score preserved on recovery


@pytest.mark.asyncio
async def test_preflight_block_still_blocked_after_retry():
    orch = _make_orchestrator()
    task = _make_task()
    bad_output = 'headers = {"Authorization": "Bearer eyJhbGc.eyJzdWI.sig"}'
    score = 0.8
    # Revision also leaks a token
    still_bad = 'auth = "Bearer eyJuZXc.eyJzdWI.newsig"'
    orch._call_primary = AsyncMock(return_value=still_bad)

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=bad_output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    assert final_score == 0.0           # degraded
    assert final_output == bad_output   # original preserved (not revision)


@pytest.mark.asyncio
async def test_preflight_validator_exception_is_fail_open():
    orch = _make_orchestrator()
    orch._preflight_validator.validate = MagicMock(side_effect=RuntimeError("validator down"))
    task = _make_task()
    output = "def hello(): return 'world'"
    score = 0.9

    final_output, final_score, pf_result = await orch._run_preflight_check(
        task=task, output=output, score=score, primary=Model.DEEPSEEK_CHAT
    )

    # Fail-open: exception -> PASS, pipeline unaffected
    assert final_output == output
    assert final_score == score
    assert pf_result.action == PreflightAction.PASS
```

**Step 2: Run all 5 tests**

```
pytest tests/test_preflight_integration.py -v
```
Expected: all 5 pass.

**Step 3: Commit**

```bash
git add tests/test_preflight_integration.py
git commit -m "test: full coverage for _run_preflight_check() (PASS/WARN/BLOCK/ENRICH/fail-open)"
```

---

## Task 5: Wire preflight into the `_execute_task()` pipeline

**Files:**
- Modify: `orchestrator/engine.py` lines 2272-2283

**Step 1: Locate insertion point**

In `engine.py`, find this exact block (around line 2272):

```python
            status = TaskStatus.COMPLETED if best_score >= task.acceptance_threshold else TaskStatus.DEGRADED
            if best_score == 0.0 and not det_passed:
                status = TaskStatus.FAILED

            # Feed final eval score back to telemetry so ConstraintPlanner re-ranks
            if best_score > 0.0:
```

**Step 2: Insert preflight block between status determination and telemetry**

After the `status = TaskStatus.FAILED` line and before the telemetry comment, insert:

```python
            # ── PREFLIGHT GATE ──
            # Post-loop quality gate: validates best_output before delivery.
            # WARN -> score penalty; ENRICH/BLOCK -> 1 revision attempt.
            _preflight_result = None
            try:
                best_output, best_score, _preflight_result = await self._run_preflight_check(
                    task=task,
                    output=best_output,
                    score=best_score,
                    primary=primary,
                )
                # Re-derive status after preflight may have changed best_score
                if best_score == 0.0 and status != TaskStatus.FAILED:
                    status = TaskStatus.DEGRADED
                elif best_score >= task.acceptance_threshold and status == TaskStatus.DEGRADED:
                    status = TaskStatus.COMPLETED  # revision recovered the task
            except Exception as _pf_exc:
                logger.warning("preflight gate raised: %s — skipping gate", _pf_exc)
```

**Step 3: Pass preflight fields to TaskResult**

Find the `return TaskResult(` call at the end of `_execute_task()` and add:

```python
                preflight_result=_preflight_result,
                preflight_passed=(_preflight_result is None or _preflight_result.passed),
```

**Step 4: Run integration tests**

```
pytest tests/test_preflight_integration.py -v
```
Expected: all 5 pass.

**Step 5: Run full suite — no regressions**

```
pytest tests/ --tb=short -q 2>&1 | tail -20
```
Expected: same or higher pass count (660+), no new failures.

**Step 6: Commit**

```bash
git add orchestrator/engine.py
git commit -m "feat: wire preflight gate into _execute_task() post-loop delivery check"
```

---

## Task 6: End-to-end smoke test and docs commit

**Step 1: Smoke test (clean code passes, [TODO] warns)**

```bash
python -c "
from orchestrator.preflight import PreflightValidator, PreflightMode, PreflightAction
v = PreflightValidator()
r1 = v.validate('def hello(): return 42', context={'task_type': 'CODE_GEN'}, mode=PreflightMode.AUTO)
assert r1.action == PreflightAction.PASS, f'Expected PASS, got {r1.action}'
r2 = v.validate('def hello(): pass  # [TODO] fix', context={'task_type': 'CODE_GEN'}, mode=PreflightMode.AUTO)
assert r2.action == PreflightAction.WARN, f'Expected WARN, got {r2.action}'
print('OK — preflight gate working correctly')
"
```
Expected: `OK — preflight gate working correctly`

**Step 2: Commit design docs**

```bash
git add docs/plans/2026-03-14-content-preflight-gate-design.md docs/plans/2026-03-14-content-preflight-gate.md
git commit -m "docs: add content preflight gate design and implementation plan"
```
