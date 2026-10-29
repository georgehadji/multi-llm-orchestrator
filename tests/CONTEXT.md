# tests/ — pytest suite

**Job:** prove changes work. Real tests match `test_*.py`. Many other root files are one-off debug or migration scripts and are listed under `--ignore` in `pyproject.toml` `[tool.pytest.ini_options]`.

## Where a test goes
| Dir | Holds |
|---|---|
| `unit/` | Isolated tests (subdirs `design/`, `quality/`, `security/`) |
| `integration/` | Multi-component tests |
| `contracts/` | Architecture and port contracts. Run with empty API keys |
| `smoke/`, `regression/` | Quick sanity, pinned bug regressions |
| `fixtures/`, `conftest.py` | Shared setup |

Placement is checked by `scripts/check_test_placement.py` and `scripts/check_test_markers.py`.

## Markers
`unit`, `integration`, `slow`, `requires_api`, `e2e`, `load`, `stress`, `benchmark`, `mock`.

## Commands
- CI-equivalent: `pytest -m "not slow and not requires_api and not stress and not e2e" --tb=short -q`
- Contracts: `pytest tests/contracts/ -v --tb=short --no-cov`
- One file: `pytest tests/test_circuit_breaker.py -v`
- Do not run the whole directory unfiltered.

## Rules
- TDD: write the failing test first.
- Never lower the coverage floor, never add a file to `--ignore` to hide a failure.
- Load skill `orchestrator-validation-and-qa` before claiming a change works.

## Human check
New test placed in the right dir with a marker; no new `--ignore` entry.
