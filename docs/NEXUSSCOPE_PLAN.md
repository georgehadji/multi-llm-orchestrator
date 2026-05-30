# NexusScope — pyinstrument Integration Plan

## Context

The orchestrator has no statistical profiling today. `telemetry.py` tracks per-model EMA latency, and `tracing.py` emits OpenTelemetry-compatible spans, but neither captures *call-stack hotspot data* — the "what is the CPU actually doing?" view that pyinstrument provides. NexusScope wraps pyinstrument behind an internal brand name that fits the project's existing `nexus_*` namespace, adds async-native profiling for the engine's pipeline stages and LLM API calls, and surfaces results via the dashboard and CLI.

This plan also fixes an existing engine.py bug discovered during architecture review: line 478 (`self._pipeline = PipelineRunner(self)`) unconditionally overwrites `container.pipeline` (the real `TaskPipeline`) with a `PipelineRunner` that has **no `run(ctx)` method**, making `_execute_task` crash at runtime.

---

## Pre-condition: Fix the `self._pipeline` double-assignment bug

**File:** `orchestrator/engine.py` — lines 477-478

**Problem:**
```python
# Line 426 — correct
self._pipeline = container.pipeline       # TaskPipeline with 7 stages

# Lines 477-478 — BUG: unconditionally overwrites above
from .pipeline_runner import PipelineRunner
self._pipeline = PipelineRunner(self)     # has no run(ctx) → crash in _execute_task
```

**Fix:** Change line 478 to use a **separate attribute** so both objects are accessible:
```python
from .pipeline_runner import PipelineRunner
self._pipeline_runner = PipelineRunner(self)   # rename: runner owns execute_all()
# self._pipeline is already set correctly at line 426 — do NOT override it
```

Any internal callers of `self._pipeline` that call `execute_all()`/`execute_all_with_retry()` must be updated to use `self._pipeline_runner` instead. Search: `self._pipeline.execute_all`, `self._pipeline.warm_cache_for_level`, `self._pipeline.build_system_prompt`, `self._pipeline.build_project_context` in `engine.py`.

---

## Module: `orchestrator/infrastructure/nexusscope/`

Follows the `nexus_search/` sub-package pattern already in `infrastructure/`.

### Files to create

```
orchestrator/infrastructure/nexusscope/
├── __init__.py        — public API exports
├── config.py          — NexusScopeConfig (env-var driven, from_env())
├── session.py         — ProfileSession dataclass + SessionRingBuffer
├── profiler.py        — NexusScopeProfiler core class + get_profiler() singleton
├── decorators.py      — module-level @profile_sync, @profile_async
├── pipeline_hook.py   — ProfilingTaskPipeline (wraps TaskPipeline stages)
└── llm_adapter.py     — ProfiledLLMClient (wraps UnifiedClient)
```

No middleware file — the dashboard approach (ring buffer + API route) replaces the `?profile=1` middleware pattern, which adds unnecessary overhead for all requests.

---

### `config.py`

```python
@dataclass
class NexusScopeConfig:
    enabled: bool        # env: ORCHESTRATOR_PROFILING=1
    interval: float      # env: NEXUSSCOPE_INTERVAL=0.001
    async_mode: bool     # env: NEXUSSCOPE_ASYNC_MODE=1
    buffer_size: int     # env: NEXUSSCOPE_BUFFER_SIZE=100

    @classmethod
    def from_env(cls) -> "NexusScopeConfig": ...
```

---

### `session.py`

```python
@dataclass
class ProfileSession:
    name: str
    started_at: float
    finished_at: float | None
    _profiler: Any        # pyinstrument.Profiler (None if disabled/unavail)

    @property
    def duration_ms(self) -> float: ...

class SessionRingBuffer:
    def push(self, session: ProfileSession) -> None: ...
    def all(self) -> list[ProfileSession]: ...
    def by_name(self, name: str) -> list[ProfileSession]: ...
    def last(self, n: int = 1) -> list[ProfileSession]: ...
```

---

### `profiler.py` — core API

```python
class NexusScopeProfiler:
    def __init__(self, config: NexusScopeConfig | None = None,
                 event_bus=None) -> None:
        # event_bus: satisfies EventPort protocol (publish coroutine)

    # Sync context manager
    @contextmanager
    def session(self, name: str) -> Generator[ProfileSession, None, None]: ...

    # Async context manager (used for pipeline stages and LLM calls)
    @asynccontextmanager
    async def async_session(self, name: str) -> AsyncGenerator[ProfileSession, None]: ...

    # Decorator factories
    def profile_sync(self, name: str | None = None) -> Callable: ...
    def profile_async(self, name: str | None = None) -> Callable: ...

    # Buffer access
    def get_sessions(self, name: str | None = None,
                     last_n: int | None = None) -> list[ProfileSession]: ...

    # Rendering
    def render_last(self, name: str | None = None,
                    fmt: str = "text") -> str | dict: ...
    # fmt: "text" | "html" | "json" | "speedscope"

def get_profiler() -> NexusScopeProfiler:
    """Module-level singleton. Replace in tests."""
```

**EventBus integration inside `async_session`:**
After the session closes, if `event_bus` was injected:
```python
from orchestrator.unified_events.core import MetricEvent
await self._event_bus.publish(MetricEvent(
    aggregate_id=name,
    metric_name="nexusscope.session",
    value=session.duration_ms,
    metadata={"session_name": name, "has_profile": session._profiler is not None},
))
```
`MetricEvent` already exists in `orchestrator/unified_events/core.py`.

**Tracing integration inside `async_session`:**
Attach pyinstrument summary to the active `tracing.py` span (non-blocking):
```python
from orchestrator.infrastructure.tracing import get_tracer
tracer = get_tracer()
if tracer and tracer._current_span:
    tracer._current_span.set_attribute(
        "nexusscope.duration_ms", session.duration_ms
    )
```
This avoids duplicate span lifecycle — NexusScope piggybacks on existing `traced_task`/`traced_llm_call` spans.

**Lazy pyinstrument import:** Single `_import_pyinstrument()` method tries `import pyinstrument` once; logs a warning and degrades gracefully if absent.

---

### `pipeline_hook.py`

```python
class ProfilingStageWrapper:
    """Wraps a single PipelineStage — adds async_session around process()."""
    async def process(self, ctx: PipelineContext) -> PipelineContext:
        async with self._profiler.async_session(
            f"pipeline.stage.{type(self._stage).__name__}"
        ):
            return await self._stage.process(ctx)

class ProfilingTaskPipeline:
    """Drop-in replacement for TaskPipeline.
    Wraps each stage in ProfilingStageWrapper.
    Replicates TaskPipeline.run() exactly.
    """
    def __init__(self, stages: list[PipelineStage],
                 profiler: NexusScopeProfiler | None = None) -> None: ...

    async def run(self, ctx: PipelineContext) -> PipelineContext: ...
```

---

### `llm_adapter.py`

```python
class ProfiledLLMClient:
    """Satisfies LLMClient Protocol (domain/ports.py:86).
    Wraps UnifiedClient from infrastructure/llm_client.py.
    """
    async def call(self, model, prompt, system="", max_tokens=1500,
                   temperature=0.3, timeout=120, **kwargs):
        async with self._profiler.async_session(
            f"llm.call.{getattr(model, 'value', str(model))}"
        ):
            return await self._inner.call(
                model=model, prompt=prompt, system=system,
                max_tokens=max_tokens, temperature=temperature,
                timeout=timeout, **kwargs,
            )
```

---

## Files to modify

### 1. `orchestrator/engine.py`

- **Lines 477-478**: Rename `self._pipeline = PipelineRunner(self)` → `self._pipeline_runner = PipelineRunner(self)` (keep `self._pipeline = container.pipeline` from line 426 intact)
- **Callers in engine.py**: Update all `self._pipeline.execute_all(...)`, `self._pipeline.warm_cache_for_level(...)`, `self._pipeline.build_system_prompt()`, `self._pipeline.build_project_context()` → `self._pipeline_runner.*`

### 2. `orchestrator/engine_core/container.py`

After `pipeline = TaskPipeline([...])` at line 278, add:
```python
import os
if os.getenv("ORCHESTRATOR_PROFILING", "0") == "1":
    from ..infrastructure.nexusscope.pipeline_hook import ProfilingTaskPipeline
    from ..infrastructure.nexusscope import get_profiler
    pipeline = ProfilingTaskPipeline(pipeline._stages, profiler=get_profiler())
```

After `client = UnifiedClient(cache=cache)` at line 196, add:
```python
if os.getenv("ORCHESTRATOR_PROFILING", "0") == "1":
    from ..infrastructure.nexusscope.llm_adapter import ProfiledLLMClient
    from ..infrastructure.nexusscope import get_profiler
    client = ProfiledLLMClient(inner=client, profiler=get_profiler())
```

`get_profiler()` is the singleton; both the pipeline hook and the LLM adapter share the same `NexusScopeProfiler` instance and ring buffer.

### 3. `orchestrator/config.py`

Add a new class (pure data, no imports):
```python
class NexusScopeDefaults:
    """NexusScope statistical profiler defaults.
    Override via env vars: ORCHESTRATOR_PROFILING, NEXUSSCOPE_INTERVAL,
    NEXUSSCOPE_ASYNC_MODE, NEXUSSCOPE_BUFFER_SIZE.
    """
    ENABLED: bool = False
    INTERVAL: float = 0.001   # 1ms — pyinstrument default
    ASYNC_MODE: bool = True
    BUFFER_SIZE: int = 100
```

### 4. `orchestrator/dashboard_core/core.py`

Inside `create_app()`, add routes using the same inline decorator pattern already used there:

```python
# ── NexusScope profiler routes ────────────────────────────────
try:
    from orchestrator.infrastructure.nexusscope import get_profiler as _get_ns_profiler
    _ns_available = True
except ImportError:
    _ns_available = False

@app.get("/api/nexusscope/sessions")
async def nexusscope_sessions(session: str | None = None, last_n: int = 20):
    if not _ns_available:
        return {"error": "NexusScope not available"}
    profiler = _get_ns_profiler()
    sessions = profiler.get_sessions(name=session, last_n=last_n)
    return [{"name": s.name, "duration_ms": s.duration_ms,
             "has_profile": s._profiler is not None} for s in sessions]

@app.get("/api/nexusscope/report")
async def nexusscope_report(session: str | None = None, fmt: str = "text"):
    if not _ns_available:
        return {"error": "NexusScope not available"}
    from fastapi.responses import HTMLResponse
    profiler = _get_ns_profiler()
    rendered = profiler.render_last(name=session, fmt=fmt)
    if fmt == "html":
        return HTMLResponse(content=rendered)
    return {"report": rendered}
```

### 5. `orchestrator/cli.py`

Follow the `_nexus_subparsers()` pattern. Add a new function and register it:

```python
def _nexusscope_subparsers(subparsers) -> None:
    """Register the 'nexusscope' subcommand."""
    nsp = subparsers.add_parser("nexusscope", help="NexusScope statistical profiler")
    nsp_sub = nsp.add_subparsers(dest="nexusscope_command", metavar="COMMAND")

    # nexusscope sessions — list recent sessions
    sess_p = nsp_sub.add_parser("sessions", help="List recent profiling sessions")
    sess_p.add_argument("--name", "-n", default=None, help="Filter by session name")
    sess_p.add_argument("--last", "-l", type=int, default=20, help="Number to show")
    sess_p.set_defaults(func=cmd_nexusscope_sessions)

    # nexusscope report — render last session
    rep_p = nsp_sub.add_parser("report", help="Render profiling report")
    rep_p.add_argument("--name", "-n", default=None, help="Session name filter")
    rep_p.add_argument("--format", "-f",
                       choices=["text", "html", "json", "speedscope"],
                       default="text", help="Output format")
    rep_p.add_argument("--output", "-o", default=None, help="Write to file")
    rep_p.set_defaults(func=cmd_nexusscope_report)
```

Also add `--profile` flag to the `run` subparser:
```python
run_p.add_argument("--profile", action="store_true", default=False,
                   help="Enable NexusScope profiling for this run")
run_p.add_argument("--profile-output", default=None, metavar="PATH",
                   help="Write profile report to PATH on exit")
run_p.add_argument("--profile-format",
                   choices=["text", "html", "json", "speedscope"],
                   default="text")
```

In `cmd_run()`: if `args.profile`, set `os.environ["ORCHESTRATOR_PROFILING"] = "1"` before creating the Orchestrator.

### 6. `pyproject.toml`

```toml
[project.optional-dependencies]
profiling = [
    "pyinstrument>=4.6,<5.0",
]
```

Add `profiling` pytest marker:
```toml
[tool.pytest.ini_options]
markers = [
    ...
    "profiling: marks tests that exercise NexusScope profiling",
]
```

### 7. `tests/conftest.py`

```python
@pytest.fixture
def nexusscope_config():
    from orchestrator.infrastructure.nexusscope.config import NexusScopeConfig
    return NexusScopeConfig(enabled=True, interval=0.001,
                            async_mode=True, buffer_size=50)

@pytest.fixture
def nexusscope(nexusscope_config):
    """NexusScopeProfiler with profiling enabled. Pyinstrument not required."""
    from orchestrator.infrastructure.nexusscope.profiler import NexusScopeProfiler
    return NexusScopeProfiler(config=nexusscope_config)
```

---

## New test file: `tests/test_nexusscope.py`

All tests `@pytest.mark.unit` except the route tests which are `@pytest.mark.integration`.

**Unit tests (pyinstrument NOT required — mock it):**
```
test_profiling_disabled_is_noop           — disabled config → empty buffer after exit
test_ring_buffer_capacity_eviction        — push capacity+5 → exactly capacity retained
test_ring_buffer_filter_by_name           — by_name("x") returns only "x" sessions
test_sync_context_manager_records_session — duration_ms > 0, session in buffer
test_async_context_manager_records_session
test_profile_sync_decorator_records_session
test_profile_async_decorator_records_session
test_get_profiler_is_singleton            — two calls return same object
test_config_from_env_enabled              — ORCHESTRATOR_PROFILING=1 → enabled=True
test_config_defaults                      — no env → enabled=False, interval=0.001
test_profiled_llm_client_delegates        — inner.call() is called
test_profiled_llm_client_names_session    — session named "llm.call.<model.value>"
test_profiling_pipeline_wraps_stages      — 2 mock stages → 2 sessions in buffer
test_profiling_pipeline_abort_propagates  — stage raises → ctx.should_abort=True
```

**Integration tests (FastAPI TestClient):**
```
test_nexusscope_sessions_route_empty      — GET /api/nexusscope/sessions → []
test_nexusscope_sessions_route_populated  — push session → appears in response
test_nexusscope_report_route_text         — GET /api/nexusscope/report → text body
test_nexusscope_report_html               — ?fmt=html → HTMLResponse
```

---

## Build order

1. `pyproject.toml` — add `profiling` dep group + marker
2. `orchestrator/config.py` — add `NexusScopeDefaults` (zero-risk, no imports)
3. `nexusscope/config.py` — `NexusScopeConfig.from_env()`
4. `nexusscope/session.py` — `ProfileSession`, `SessionRingBuffer`
5. `nexusscope/profiler.py` — `NexusScopeProfiler`, `get_profiler()` singleton
6. `nexusscope/decorators.py` — thin module
7. `nexusscope/__init__.py` — package usable in isolation from this point
8. **Write unit tests** (RED phase) — mock pyinstrument, verify buffer/config/context managers
9. `nexusscope/llm_adapter.py` — `ProfiledLLMClient` + pass tests
10. `nexusscope/pipeline_hook.py` — `ProfilingTaskPipeline` + pass tests
11. **Fix engine.py:477-478** — rename `self._pipeline` → `self._pipeline_runner`; update all callers
12. `engine_core/container.py` — wire both adapters behind env-var guard
13. `cli.py` — add `_nexusscope_subparsers()` + `--profile` on run subparser
14. `dashboard_core/core.py` — add two API routes inside `create_app()`
15. `tests/conftest.py` — add fixtures
16. **Integration tests** (step 8 second phase) — FastAPI TestClient for routes

---

## Verification

```bash
# 1. Install with profiling extras
pip install -e ".[dev,profiling]"

# 2. Run unit tests (pyinstrument not required — mocked)
pytest tests/test_nexusscope.py -m unit -v --no-cov

# 3. Run integration tests
pytest tests/test_nexusscope.py -m integration -v --no-cov

# 4. Smoke test profiling end-to-end (requires OPENROUTER_API_KEY)
ORCHESTRATOR_PROFILING=1 python -m orchestrator run \
  --project "Hello world" --budget 0.10 \
  --profile-output profile.html --profile-format html

# 5. CLI sessions command
ORCHESTRATOR_PROFILING=1 python -m orchestrator nexusscope sessions

# 6. Verify engine bug fix doesn't break existing tests
pytest tests/ -m "not slow and not requires_api and not stress and not e2e" \
  --tb=short -q --no-cov

# 7. Dashboard route check (with server running)
curl http://localhost:8888/api/nexusscope/sessions
```

---
## Implementation Status (2026-05-27)

| Step | Description | Status |
|------|-------------|--------|
| 1 | pyproject.toml marker | ✓ |
| 2 | config.py NexusScopeDefaults | ✓ |
| 3 | nexusscope/config.py | ✓ |
| 4 | nexusscope/session.py | ✓ |
| 5 | nexusscope/profiler.py | ✓ |
| 6 | nexusscope/decorators.py | ✓ |
| 7 | nexusscope/__init__.py | ✓ |
| 8 | Unit tests | Pending |
| 9 | nexusscope/pipeline_hook.py | ✓ |
| 10 | nexusscope/llm_adapter.py | ✓ |
| 11 | **engine.py bug fix** (L478) | ✓ renamed to `self._pipeline_runner` |
| 12 | container.py wiring | ✓ env-var guarded |
| 13 | CLI --profile flag | --profile, --profile-output, --profile-format, nexusscope sessions/report |
| 14 | Dashboard routes | Pending |
| 15 | conftest.py fixtures | Pending |
| 16 | Integration tests | Pending |
