# quality-check

Full code quality pipeline for the AI Orchestrator. Use when the user wants to lint, format, type-check, or run security scans.

## Full pipeline (run in this order)

```bash
# 1. Lint + auto-fix
ruff check orchestrator/ --fix

# 2. Format check (read-only)
black orchestrator/ --check
# Auto-format:
black orchestrator/

# 3. Type check (strict — see pyproject.toml overrides for legacy modules)
mypy orchestrator/

# 4. Security scan
bandit -r orchestrator/ -c pyproject.toml

# 5. Dependency vulnerability check
safety check

# 6. All pre-commit hooks at once
pre-commit run --all-files
```

## Quick single-file check

```bash
ruff check orchestrator/<file>.py --fix
black orchestrator/<file>.py
mypy orchestrator/<file>.py
```

## Interpreting ruff output

Ruff is configured with `line-length = 100`. Many legacy violations are suppressed in `pyproject.toml` — focus on violations NOT in the ignore list.

**Suppressed (pre-existing baseline — do not fix these wholesale):**
`F401` (unused imports), `F821` (undefined name), `I001` (import order),
`E402` (module-level import not at top), `UP035` (deprecated typing), and ~30 others.

**New code must NOT introduce:** `E501` violations above 100 chars, `B006` (mutable defaults), `SIM201`/`SIM202`, or any `S`-series (security) rules.

## Interpreting mypy output

- `strict = true` globally
- **Legacy modules in `ignore_errors = true`** (see `[[tool.mypy.overrides]]` in `pyproject.toml`):
  `orchestrator.engine`, `orchestrator.dashboard_mission_control`, `orchestrator.ara_pipelines`, etc.
- **New code MUST pass strict checks** — do NOT add new modules to the override list

## Interpreting bandit output

- `B101` (assert_used) is skipped globally
- High-severity issues to fix: `B501`–`B509` (SSL), `B601`–`B612` (injection), `B301`–`B307` (pickle/eval)
- Low-severity can be annotated with `# nosec B<code>` if intentional

## Coverage check

```bash
pytest tests/ --cov=orchestrator --cov-report=term-missing
# Current fail_under = 12% (ratchet baseline)
# Target = 80% — raise fail_under in pyproject.toml as coverage improves
```

## Pre-commit hook

```bash
pre-commit install           # install hooks (run once after clone)
pre-commit run --all-files   # run all hooks on entire codebase
pre-commit run ruff          # run just ruff hook
```
