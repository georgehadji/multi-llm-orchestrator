# T21 coverage and residual risk

## Counters

| | |
|---|---|
| `hunt_iterations` | 1 |
| `fix_revisions` | 3 (gate rewritten from name-only to AST after it flagged 4 production modules; test file rewritten as parametrized rather than patched; `test_ide_modifications.py` consolidated rather than repaired) |
| `budget_spent` | 1 source file changed (helper extracted, handler −4 lines), 2 files renamed/moved, 1 deleted, 1 test file rewritten (12 tests), 1 gate script, 2 reference updates |

## What was actually examined

**Fully examined:** all 16 modules for importability (0 failures). The three
`test_*.py` files (695 lines) in full. The `SessionManager` API of both
subsystems, method by method, to settle C5 and C9. The three entry points and
the ports they bind. The `replace_accent_color` call site and its handler.

**Examined only at the surface:** `ide_orchestrator_server.py` — 3,005 lines,
of which perhaps 300 were read closely (the SessionManager class, the
websocket event dispatch, `handle_modification_request`, the app/main blocks,
and the region around line 2056 to settle C8). **The remaining ~2,700 lines
were not read.** This is the single largest unexamined file in the programme
and this wave does not change that.

**Not examined at all:** `api/routes.py` (237), `websocket/handlers.py` (215),
`websocket_manager.py` (152), `session_manager.py` (408 — its API was read,
its logic was not), `integration/orchestrator_bridge.py` (138), `server.py`
(147). That is ~1,300 lines of Subsystem A, now importable for the first time
and still unhunted.

## Residual risk

1. **The region's coverage claim is improved, not closed.** T21 cleared the
   *blocker* and swept the test surface. Roughly 4,000 of the 5,152 lines
   still have no individual disposition. Anyone reading "T21 complete" should
   read it as "importable and its tests are now real", not "audited".
2. **C6 is the item most worth a human decision.** Three IDE backends, two
   claiming the same port, with no statement anywhere of which is canonical.
   Until that is settled, any fix risks landing on the dead copy — the exact
   failure mode T17 catalogued 65 times.
3. **C7 is unresolved and may bite silently.** `websockets` 12.0 satisfies the
   `dashboard` extra and violates `google-genai`'s floor. Imports succeed, so
   the failure — if any — would appear only on the Gemini Live API path at
   runtime. UNKNOWN whether that path is used; not tested here.
4. **`replace_accent_color` is now covered; the handler around it is not.**
   The extraction makes the substitution testable, but the read/write/log/
   terminal-line orchestration in `handle_modification_request` still has no
   test, because testing it needs a WebSocket double.
5. **Broadcast ordering is uncovered.** C10's test claimed it and never tested
   it. Deleting the false signal does not create a real one.
6. **The gate's baseline hides a live-API hazard.** `test_instructor_tenacity.py`
   stays as-is; if `testpaths` is ever widened, its three module-level test
   functions will make real API calls.

## Claims NOT made

- Not claimed: that `ide_backend/` is defect-free. Most of it is unread.
- Not claimed: that the three IDE servers work. None was started; no port was
  bound; no request was served. Only importability was verified.
- Not claimed: that `websockets` 12.0 is safe for `google-genai`. Only that
  both packages import under it.
- Not claimed: that deleting `test_ide_modifications.py` preserved every
  assertion. It preserved every assertion that touched product code — of
  which there were none — plus the file round-trip, rewritten.
