---
name: tdd-feature-dev
description: TDD workflow for implementing new features or bug fixes in the orchestrator. Use when starting any implementation task.
---

# TDD Feature Development — AI Orchestrator

## Workflow

### 1. Create isolated worktree (for features, not for quick fixes)
```bash
git worktree add .claude/worktrees/<feature-name> -b <feature-name>
cd .claude/worktrees/<feature-name>
```

### 2. RED — Write failing test first
```python
# tests/test_<module>.py
def test_new_behavior():
    from orchestrator.<module> import NewClass
    obj = NewClass()
    assert obj.method() == expected  # Will fail: ImportError
```

Run to confirm failure:
```bash
python -m pytest tests/test_<module>.py::test_new_behavior -v --no-cov
```
Expected: `ImportError` or `AssertionError`. If it passes, the test is wrong.

### 3. GREEN — Minimal implementation
Write the smallest code that makes the test pass. No over-engineering.

Run to confirm pass:
```bash
python -m pytest tests/test_<module>.py -v --no-cov
```

### 4. Commit (atomic, per test group)
```bash
git add orchestrator/<module>.py tests/test_<module>.py
git commit -m "feat: <what it does in one line>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
```

### 5. Verify no regressions
```bash
python -m pytest tests/ -v --no-cov -m "not slow" -x
```

### 6. Merge back (after all tests pass)
```bash
cd ../..
git merge --no-ff <feature-name>
git worktree remove .claude/worktrees/<feature-name>
```

## Module Conventions

**New module pattern** (`orchestrator/<module>.py`):
```python
"""
<ModuleName> — <One-line description>
"""
from __future__ import annotations
from .log_config import get_logger
logger = get_logger(__name__)
```

**Export from `orchestrator/__init__.py`**:
```python
from .<module> import ClassName
```
Add `"ClassName"` to `__all__`.

**Test file convention** (`tests/test_<module>.py`):
- Use `tmp_path` pytest fixture for any file I/O
- Use `async def` for async tests (AUTO mode active)
- Use `AsyncMock` / `MagicMock` from `unittest.mock`
- Keep tests fast: mock external APIs, no real network calls

## Commit Message Format
```
<type>: <subject in present tense, 50 chars max>

<optional body: why, not what>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
```
Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
