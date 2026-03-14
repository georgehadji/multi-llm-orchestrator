---
name: run-tests
description: Run tests for the AI Orchestrator project. Use when implementing or verifying any feature.
---

# Running Tests — AI Orchestrator

## Standard Commands

**Single test file (fastest feedback, no coverage):**
```bash
python -m pytest tests/test_<module_name>.py -v --no-cov
```

**After adding a new feature (verify no regressions):**
```bash
python -m pytest tests/test_<module_name>.py tests/test_<related_module>.py -v --no-cov
```

**Full suite with coverage (before committing):**
```bash
python -m pytest tests/ -v --cov=orchestrator --cov-report=term-missing -m "not slow"
```

**CI-equivalent (GitHub Actions threshold: 70%):**
```bash
python -m pytest tests/ --cov=orchestrator --cov-report=term-missing --cov-fail-under=70 -m "not slow"
```

## TDD Pattern

1. Write test first → run it → confirm FAIL (ImportError or AssertionError)
2. Write minimal implementation → run it → confirm PASS
3. Add to full suite → confirm no regressions

## Async Tests

The project uses `pytest-asyncio` with `asyncio_mode = "auto"`. Mark async tests as:
```python
async def test_something():  # No decorator needed with AUTO mode
    result = await some_coroutine()
    assert result == expected
```

## Coverage Threshold

- **Target**: ≥ 70% (enforced in CI)
- If coverage drops, check `coverage_html/index.html` for uncovered lines
- Use `--no-cov` during TDD red/green cycles for speed
