# test-smart

Smart pytest invocation reference for the AI Orchestrator. Use this when the user wants to run tests, check CI status locally, or verify a change didn't break anything.

## Marker quick-reference

| Situation | Command |
|-----------|---------|
| CI-safe subset (fastest, no API calls) | `pytest tests/ -m "not slow and not requires_api and not stress and not e2e" --tb=short -q` |
| Unit tests only | `pytest tests/ -m unit -v` |
| Integration tests | `pytest tests/ -m integration -v` |
| Single file | `pytest tests/<file>.py -v` |
| Single function | `pytest tests/<file>.py::<TestClass>::<test_fn> -v` |
| With coverage | `pytest tests/ --cov=orchestrator --cov-report=term-missing` |
| Parallel (faster) | `pytest -n auto tests/ -m "not slow and not requires_api"` |
| NexusScope unit tests | `pytest tests/test_nexusscope.py -m unit -v --no-cov` |
| NexusScope integration | `pytest tests/test_nexusscope.py -m integration -v --no-cov` |
| Before committing | `pytest tests/ -m "not slow and not requires_api and not stress and not e2e" --tb=short -q --no-cov` |

## All available markers

| Marker | Meaning |
|--------|---------|
| `unit` | Pure unit tests — no I/O, no API |
| `integration` | Multi-component tests — may use SQLite, no LLM calls |
| `slow` | Takes >5s — skip for fast feedback |
| `requires_api` | Makes real LLM calls — needs API keys |
| `e2e` | Full end-to-end pipeline |
| `load` | Load tests |
| `stress` | Stress tests |
| `benchmark` | Benchmark tests |
| `mock` | Uses mocked dependencies |
| `asyncio` | Uses asyncio (auto-mode, no extra decorator needed) |
| `edge_case` | Boundary / edge case tests |
| `profiling` | NexusScope profiling tests |

## Ignored test files (always excluded by pyproject.toml)

These files exist but are skipped globally — do not try to run them directly:
`test_api.py`, `test_assembler_debug.py`, `test_basic_imports.py`,
`test_e2e_comprehensive.py`, `test_git_integration.py`, `test_mission_control.py`,
`test_startup.py`, `test_syntax.py`, `test_v65_fix.py`, and ~25 others.

Run `grep "ignore" pyproject.toml` for the full list.

## Coverage baseline

- Current `fail_under = 12%` (temporary ratchet — pre-existing state)
- Target: **80%** — raise `fail_under` after adding new tests
- HTML report: `coverage_html/index.html` after running with `--cov-report=html`

## When to use which marker

| Just edited… | Run… |
|---|---|
| A single module | `pytest tests/test_<module>.py -v` |
| `orchestrator/engine.py` | Full CI-safe subset |
| A new `infrastructure/` module | `pytest tests/ -m unit -v` |
| `engine_core/stages/` | `pytest tests/ -m unit -v` then `integration` |
| `cli.py` | `pytest tests/test_cli*.py -v` |
| Before pushing | CI-safe subset with `--no-cov` for speed |

## TDD cycle commands

```bash
# RED — write test, confirm it fails
pytest tests/test_<module>.py::test_<fn> -v
# Expected: FAILED

# GREEN — implement, confirm it passes
pytest tests/test_<module>.py::test_<fn> -v
# Expected: PASSED

# FULL — no regressions
pytest tests/ -m "not slow and not requires_api and not stress and not e2e" --tb=short -q
```
