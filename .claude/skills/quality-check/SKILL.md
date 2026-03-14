---
name: quality-check
description: Run all code quality checks (format, lint, type-check) before committing. Use before any git commit.
---

# Quality Check — AI Orchestrator

Run in this order before committing:

## 1. Format (Black)
```bash
black orchestrator/ tests/
```
Or check without modifying:
```bash
black --check orchestrator/ tests/
```

## 2. Lint (Ruff)
```bash
python -m ruff check --fix orchestrator/ tests/
```
Or check without auto-fixing:
```bash
python -m ruff check orchestrator/ tests/
```

## 3. Type Check (MyPy)
```bash
mypy orchestrator/ --ignore-missing-imports
```

## All at once (Makefile)
```bash
make ci
```
This runs: `format-check lint type-check test-ci security-check`

## Quick fix all formatters
```bash
black orchestrator/ tests/ && python -m ruff check --fix orchestrator/ tests/
```

## Pre-commit (enforced on git commit)
Pre-commit hooks run Black → Ruff → MyPy → Bandit automatically on `git commit`.
If a commit is blocked by pre-commit, fix the reported issue and retry.

**To run pre-commit manually:**
```bash
pre-commit run --all-files
```

## Common Ruff Issues

- `E501` — Line too long (max 100 chars, configured in pyproject.toml)
- `F401` — Unused import (remove it)
- `UP` — Use modern Python syntax (`list[str]` vs `List[str]`)
- Auto-fix most issues: `python -m ruff check --fix orchestrator/`
