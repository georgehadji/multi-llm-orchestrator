# P1 — V4 precision audit, wave 1 (highest-priority region)

Files (from `docs/hunts/v4_waves.tsv`, DEEP tier, priority 8–12): `generators/website_generator.py`,
`testing/first_generator.py`, `engine.py`, `entrypoints/cli_dispatch.py`,
`ide_backend/ide_orchestrator_server.py`, `infrastructure/streaming.py`,
`infrastructure/llm_client.py`, `api_server.py`. 8 files, 13,786 LOC at measurement time.

Config: `MODE=AGENTIC-REPO`, `EXECUTION=AUTO`, `APPLY_FIXES=ON`, `TOGGLE_B_INNOCENCE=ON`,
`TOGGLE_C_TAIL_SWEEP=ON` (DEEP tier), `ELICITATION_K=8`.

INVENTORY.md consulted before elicitation (V4 rule 1C); no P1 candidate re-raises a
disposition already recorded there.

| ID | Severity | Evidence | Reach | Location | Category | Violated Property |
|----|----------|----------|-------|----------|----------|-------------------|
| P1-7 | HIGH | HYPOTHESIS (well-mechanized) | REACHABLE | `api_server.py` :476,568,666 | Concurrency | A shared long-running Orchestrator's `_run_ctx.budget` is overwritten by every concurrent request with no isolation and no lock |
| P1-3 | MEDIUM-HIGH | STATIC + HYPOTHESIS | REACHABLE | `ide_orchestrator_server.py::SessionManager.start_server` | Concurrency (TOCTOU) | Dev-server ports are hardcoded, not session-scoped; the losing side of a collision reports false success |
| P1-6 | MEDIUM | STATIC + EXEC | REACHABLE | `llm_client.py::UnifiedClient` | Logic (shared mutable class state) | `_clients` is a `ClassVar` dict; every instance shares one cache, and any instance's `close()` clears it for all |
| P1-8 | MEDIUM | EXEC | REACHABLE (sandbox path only) | `first_generator.py::_parse_pytest_output` | Logic (wrong comparator) | All-failing pytest summary ("N failed in Xs") parsed as all-passing; the sandbox call site has no cross-check to catch it |
| P1-1 | MEDIUM | STATIC | REACHABLE | `website_generator.py::_get_registry` | Logic (silent failure) | `ImportError` fallback swaps in a 4-section fake registry with zero log line |
| P1-2 | MEDIUM | STATIC | REACHABLE | `cli_dispatch.py` ×4 sites | Dependencies/Config (dead flag) | `--agent-profile` is parsed into a dict applied nowhere; 3 independent competing implementations exist, none wired |
| P1-4 | LOW (DEAD reach) | EXEC | DEAD | `streaming.py::StreamingPipeline._run_pipeline` | Logic (undefined name) | References bare `project_description`, not in scope; `NameError` on every invocation, caught and reported as a generic ERROR event |
| P1-5 | LOW (DEAD reach) | EXEC | DEAD | `streaming.py::StreamingPipeline.__init__` | Concurrency (missing await) | `get_event_bus()` is `async def`, called synchronously; `self.event_bus` is a bare coroutine, every `_emit_to_bus` fails silently |

**Cleared (innocent), not counted above:**
- `engine.py:65-73` — `OPENROUTER_OPTS`/`generate_openrouter_schema`/`get_schema_for_task_type` imported and never used anywhere in the file or repo. Dead import, no violated property; not a V4 finding.
- `engine.py::__aexit__` unguarded `await self._c.shutdown()` — looked inconsistent with 3 sibling guarded calls. Read `container.py::shutdown()` in full: every one of its 7 steps independently self-guards with `try/except Exception: logger.warning`; cannot propagate. FALSE.
- `ide_orchestrator_server.py:2072-2101` — `class Item(BaseModel)` / `@app.get("/items/{{item_id}}")` looked like a live double-brace routing bug. AST-confirmed to sit inside a `JoinedStr` (f-string) inside `generate_fastapi_backend`'s template dict — generated-project boilerplate; `{{item_id}}` is the correct f-string escape. FALSE — same grep-vs-AST trap as an earlier session finding.
- `ide_orchestrator_server.py::FileGenerators` — checked for the already-known `str.format()`-on-JSX-braces defect (fixed elsewhere in this repo). Zero `.format()` calls in this class. Does not reproduce.
- `api_server.py::TokenBucketRateLimiter` — the full check-then-act sequence is inside `async with self._lock`; correctly race-free. `get_retry_after()` reads without the lock, but it is advisory-only (`is_allowed()` is the actual gate) — no violated property of consequence.
- `first_generator.py::_run_pytest_locally` (the OTHER call site of `_parse_pytest_output`) — has its own pre-existing defensive cross-check against `returncode_passed` that fully neutralizes the parse bug for this call site. Confirmed correct, not re-flagged.

**Also noted, not raised as a P1 finding (out of scope for this wave):** a second, independent
`RunContext` class exists at `application/unattended_guard.py:36`, distinct from the
`application/run_context.py` one `engine.py` actually uses. Surfaced while tracing Candidate 7;
not investigated further here — flagged for a future wave rather than chased as a tangent.
