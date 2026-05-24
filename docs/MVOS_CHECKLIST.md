# MVOS Checklist — Minimum Viable Operational State

**Version:** 1.0  
**Last Updated:** 2026-04-20  
**Author:** Georgios-Chrysovalantis Chatzivantsidis  

---

## Purpose

This checklist verifies that the Multi-LLM Orchestrator remains in a **Minimum Viable Operational State (MVOS)** after any code change, deployment, or refactor. All items must pass before a release is considered safe.

> **Origin:** Derived from ARCHITECTURAL_AUDIT_V5.md invariant definitions.

---

## Quick Start

```bash
# Automated check (recommended)
python scripts/mvos_audit.py --verbose

# Manual check (use this document)
# Go through each section below and tick the boxes.
```

---

## Invariant 1 — CLI Health Check

**Requirement:** `python -m orchestrator --help` exits 0 and prints usage.

```bash
python -m orchestrator --help
```

- [ ] Exit code is `0`
- [ ] Output contains "Multi-LLM Orchestrator"
- [ ] No `ModuleNotFoundError` or `ImportError` in stderr

**Troubleshooting:**
- If import errors occur, check `.env` file and `pip install -e ".[dev]"`
- Circular imports may have been introduced

---

## Invariant 2 — CLI Run Project

**Requirement:** The `run_project` entry point is reachable and starts without import errors.

```bash
python -m orchestrator \
  --project "MVOS smoke test" \
  --criteria "Exit immediately" \
  --budget 0.01 \
  --time 1 \
  --concurrency 1
```

- [ ] Command starts (does not crash on import)
- [ ] No `Traceback` in stderr
- [ ] May exit non-zero due to budget/API key — that's OK

**Troubleshooting:**
- Budget-exhausted or missing-API-key errors are expected on smoke tests
- Any `Traceback` indicates a regression

---

## Invariant 3 — API Execute Task

**Requirement:** `APIServer` instantiates cleanly and `/execute` endpoint is reachable.

```bash
# Start server in background
python -c "from orchestrator.api_server import APIServer; import asyncio; s=APIServer(port=8765, auth_required=False); asyncio.run(s.start())" &

# Test health
curl -s http://localhost:8765/health | python -m json.tool
```

- [ ] `GET /health` returns HTTP 200
- [ ] Response JSON contains `"status"`
- [ ] `GET /models` returns JSON list or dict
- [ ] `POST /execute` with `{}` returns 400/422 (not 500)

**Troubleshooting:**
- Port conflicts: change `port=` argument
- `aiohttp` not installed: `pip install aiohttp`

---

## Invariant 4 — State Resume

**Requirement:** `StateManager.load_project()` returns the last saved state or `None`. Never raises.

```bash
python -c "
import asyncio, tempfile
from orchestrator.state import StateManager
from orchestrator.models import ProjectState

with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db = f.name

async def test():
    sm = StateManager(db_path=db)
    s = ProjectState(project_description='test', success_criteria='test', budget=None)
    await sm.save_project('test-id', s)
    loaded = await sm.load_project('test-id')
    print('OK' if loaded and loaded.project_description == 'test' else 'FAIL')
    await sm.close()

asyncio.run(test())
"
```

- [ ] Load returns the saved state
- [ ] Load of non-existent project returns `None`
- [ ] Load of corrupted JSON returns `None` (no exception)

**Troubleshooting:**
- If `None` on valid state: schema mismatch after models.py change
- If exception on corrupted state: `_deserialize_state()` is broken

---

## Invariant 5 — Results Persisted Within 5 Seconds

**Requirement:** `save_project` → `load_project` round-trip completes in < 5 seconds.

```bash
python -c "
import asyncio, time, tempfile
from orchestrator.state import StateManager
from orchestrator.models import ProjectState

with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    db = f.name

async def test():
    sm = StateManager(db_path=db)
    s = ProjectState(project_description='speed', success_criteria='speed', budget=None)
    t0 = time.monotonic()
    await sm.save_project('speed', s)
    await sm.load_project('speed')
    print(f'{time.monotonic()-t0:.2f}s')
    await sm.close()

asyncio.run(test())
"
```

- [ ] Round-trip time < 5.0 seconds

**Troubleshooting:**
- Slow disk I/O on Windows with large WAL files: run `PRAGMA wal_checkpoint(TRUNCATE)`

---

## Invariant 6 — Circuit Breaker Trips Within 30 Seconds

**Requirement:** After 3 consecutive LLM API failures, the circuit breaker opens and subsequent calls fail fast (< 30s, ideally < 1s).

```bash
python -c "
import time
from orchestrator.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError

cb = CircuitBreaker(failure_threshold=3, reset_timeout_seconds=1)
for _ in range(3):
    cb.record_failure()

assert cb.is_open()
t0 = time.monotonic()
try:
    with cb.context():
        pass
except CircuitBreakerOpenError:
    pass
print(f'{time.monotonic()-t0:.2f}s')
"
```

- [ ] Circuit is open after 3 failures
- [ ] Context entry raises `CircuitBreakerOpenError`
- [ ] Time from call to exception < 1.0 second

**Troubleshooting:**
- If circuit doesn't open: check `failure_threshold` value
- If slow: `reset_timeout_seconds` may be too high, or context manager has side effects

---

## Extended Checks (Optional but Recommended)

### Test Suite

```bash
pytest tests/ -m "not slow and not integration and not requires_api"
```

- [ ] All unit tests pass (target: 101 tests)
- [ ] Coverage does not drop below previous baseline

### Integration Tests

```bash
pytest tests/integration/ -v
```

- [ ] `test_full_run.py` — end-to-end flow passes
- [ ] `test_circuit_breaker_fail_fast.py` — fail-fast < 5s
- [ ] `test_resume_after_crash.py` — crash recovery works

### Smoke Tests

```bash
pytest tests/smoke/ -v
```

- [ ] CLI `--help`, `--list-projects`, `analyze --help` all exit cleanly
- [ ] API health, rate limiting, CORS, size limits behave correctly

### Lint & Type Check

```bash
ruff check orchestrator/ tests/
black --check orchestrator/ tests/
```

- [ ] ruff reports zero errors on new code
- [ ] black formatting is clean

---

## Sign-off

| Role | Name | Date | Result |
|------|------|------|--------|
| Developer | | | |
| Reviewer | | | |
| Deployer | | | |

**Release is BLOCKED if any invariant fails.**
