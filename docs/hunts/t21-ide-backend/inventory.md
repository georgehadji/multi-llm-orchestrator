# T21 — `ide_backend/`, the blocked region

Wave 21 of the depth pass. This is the one region in the whole programme whose
blocker was **environmental rather than budgetary**: 16 files / 5,152 lines
that had never been importable here, so its coverage claim was the weakest in
the ledger.

## Phase 0 — clearing the blocker

`pip install "fastapi" "uvicorn[standard]" "websockets<13" "httpx"` (the
`dashboard` extra). All 16 modules then import cleanly — **0 import failures**,
verified by exit code, not by stderr.

The install itself produced the first finding.

## Phase 3–4 — triage

| ID | Disposition | Summary |
|---|---|---|
| C1 | **VERIFIED DEFECT — FIXED** | `test_color_regex.py` contained **no `assert`**. It counted failures, printed them, and `return failed == 0`; pytest discards a test's return value, so the test passed no matter what. Proven by sabotaging the pattern to match nothing — still `1 passed`. Made fail-closed, **it failed for real**: 9 of 10 cases passed, 1 did not. |
| C2 | **VERIFIED DEFECT — FIXED** | The same file *copy-pasted* production's regex rather than calling it, so even a green run said nothing about `ide_orchestrator_server.py`. Production's substitution is now extracted as `replace_accent_color()` and the test drives it. |
| C3 | **VERIFIED DEFECT — FIXED** | `orchestrator/ide_backend/test_server.py` is **not a test**. It is a FastAPI server (`app = FastAPI(title="IDE Test")`, `uvicorn.run(..., port=8765)`) — and the file `start-ide.bat` launched. Renamed `standalone_server.py`; the launcher and the archived doc that referenced it updated. |
| C4 | **VERIFIED DEFECT — FIXED** | 695 lines of `test_*.py` lived inside `orchestrator/`, where `testpaths = ["tests"]` means pytest never collects them. 13 tests that nobody had ever run. |
| C5 | **VERIFIED DEFECT — FIXED** | `test_ide_modifications.py`'s two async tests were written against `ide_orchestrator_server.SessionManager` (synchronous, has `broadcast`) but imported `session_manager.SessionManager` (async, no `broadcast`) — the T17 "landed on the wrong copy of a pair" shape, kept invisible by C4. |
| C6 | **VERIFIED — ESCALATED** | Three parallel IDE entry points: `launch.py`→`server.py` (with `api/routes.py`, `websocket/handlers.py`, `session_manager.py`), `ide_orchestrator_server.py` (3,005 lines, self-contained, own `SessionManager`), and `standalone_server.py`. The latter two both bind port **8765**. Which is canonical is a product decision — same disposition as T15's `mcp_server.py` fork. |
| C7 | **VERIFIED — ESCALATED** | The `dashboard` extra pins `websockets>=11.0,<13.0`, while the **required** `google-genai>=1.0,<2.0` needs `websockets>=13.0.0,<17.0`. `pip install -e ".[dashboard]"` reports the conflict. `google.genai` and `google.genai.live` still import under websockets 12.0, so nothing breaks at import; the Live-API websocket path is UNKNOWN and untested here. |
| C8 | FALSE — hypothesis falsified | `grep` showed `app = FastAPI(` at line 2056 *and* 2921 plus two `if __name__ == "__main__"` blocks on different ports, which read as a second app silently shadowing the first and discarding every route registered on it. AST says there is exactly **one** module-level `app`: line 2056 sits inside a triple-quoted string — a FastAPI project *template* the IDE server emits for generated projects. No shadowing, no defect. |
| C9 | FALSE — hypothesis falsified | Production calls `session_manager.update_session(...)` and `get_session(...)` **without `await`** at lines 2189–2212 and passes the result straight into `broadcast` as the payload — the exact shape of C5's `'coroutine' object has no attribute` failure. But Subsystem B's `SessionManager` is fully **synchronous** except `broadcast`, so the calls are correct. |
| C10 | Recorded, not elevated | `test_ide_modifications.py`'s `test_broadcast_order_session_state_first` was **tautological**: it replaced `broadcast` with a recorder, then called `broadcast` three times itself and asserted those three calls arrived in the order it had just made them. No production code ran. Even with C5 fixed it could never fail for a product reason. |

## Phase 5 — fixes

**Extracted one function** from the 3,005-line handler so the behaviour is
reachable from a test:

```python
def replace_accent_color(css_content: str, new_color: str) -> tuple[str, str | None]:
    """Swap the first ``--accent`` declaration for ``new_color``. ..."""
```

The handler keeps its I/O, logging and terminal-line side effects and loses
four lines.

**Consolidated the tests.** `test_ide_modifications.py` was deleted, not
repaired. Its 12 tests broke down as: 9 exercising Python's own `re` module,
1 tautological (C10), and 2 pointed at the wrong class (C5) — its coverage of
production code was **zero**. Everything worth keeping was rewritten into
`tests/unit/test_ide_color_regex.py` as 12 parametrized tests that call
production, plus a file round-trip test so the read-substitute-write path the
deleted file touched stays covered.

**Coverage genuinely lost:** none of the product. The *claimed* coverage of
broadcast ordering (FIX-001a+003) is gone — but it never existed; C10 shows
the test asserted only against its own mock. That property is now honestly
recorded as uncovered rather than falsely reported as tested.

## Phase 8 — gate

`scripts/check_test_placement.py` fails any file under `orchestrator/` that
pytest would collect by name **and** that defines module-level tests.

Deliberately narrow: a name alone is not a violation. `design/slop_test.py`
("Slop Test Engine — 61 deterministic anti-slop gates"),
`test_first_generator.py` (test-first generation), `test_fixer.py` and
`test_validator.py` are production modules whose *domain* is testing and
define no tests — all five such files are correctly ignored, verified
individually. One manual script (`test_instructor_tenacity.py`, whose three
module-level test functions make **live API calls**) is baselined with its
reason rather than renamed, because renaming a root module would move the
root-module freeze baseline.

Verified to catch the defect: restore the deleted files and it reports both
and exits 1.
