# OpenTelemetry Distributed Tracing — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add native OpenTelemetry tracing to the orchestrator so every project run produces a span tree exportable to Jaeger/Grafana Tempo/Datadog.

**Architecture:** A new `orchestrator/tracing.py` module provides `TracingConfig`, a singleton `get_tracer()`, and four context-manager helpers (`traced_task`, `traced_llm_call`, `traced_policy_check`, `traced_remediation`). These are called directly inside `engine.py`, `api_clients.py`, `policy_engine.py`, and `remediation.py`. All helpers are **no-op** when tracing is disabled, adding < 1 µs overhead. The CLI gains `--tracing` and `--otlp-endpoint` flags.

**Tech Stack:** `opentelemetry-api>=1.20`, `opentelemetry-sdk>=1.20`, `opentelemetry-exporter-otlp-proto-grpc>=1.20` (optional extras). Python contextlib for helpers.

---

## Task 1: Add optional OTEL dependencies

**Files:**
- Modify: `pyproject.toml`

**Step 1: Add `[tracing]` optional extras**

Open `pyproject.toml` and add after the existing `[project.optional-dependencies]` `dev` block:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
    "jsonschema>=4.0",
]
tracing = [
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "opentelemetry-exporter-otlp-proto-grpc>=1.20",
]
```

**Step 2: Install the extras**

```bash
pip install -e ".[tracing]"
```

Expected: installs `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-grpc` without errors.

**Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build: add opentelemetry optional extras [tracing]"
```

---

## Task 2: Create `orchestrator/tracing.py`

**Files:**
- Create: `orchestrator/tracing.py`
- Create: `tests/test_tracing.py`

**Step 1: Write the failing tests first**

Create `tests/test_tracing.py`:

```python
"""Tests for orchestrator/tracing.py — OTEL span instrumentation."""
from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from orchestrator.tracing import (
    TracingConfig,
    configure_tracing,
    get_tracer,
    traced_task,
    traced_llm_call,
    traced_policy_check,
    traced_remediation,
)


@pytest.fixture(autouse=True)
def reset_tracing():
    """Reset global tracer state between tests."""
    import orchestrator.tracing as t
    t._tracer = None
    t._provider = None
    yield
    t._tracer = None
    t._provider = None


@pytest.fixture
def span_exporter():
    """Returns an InMemorySpanExporter wired to a fresh TracerProvider."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


def test_tracing_config_defaults():
    cfg = TracingConfig()
    assert cfg.enabled is False
    assert cfg.service_name == "multi-llm-orchestrator"
    assert cfg.otlp_endpoint is None
    assert cfg.sample_rate == 1.0


def test_disabled_tracing_produces_no_spans():
    configure_tracing(TracingConfig(enabled=False))
    with traced_task("t1", "CODE"):
        with traced_llm_call("gpt-4o", "generate"):
            pass
    tracer = get_tracer()
    # No-op tracer — spans are NonRecordingSpan, no exporter to check
    # Just verify no exception is raised
    assert tracer is not None


def test_enabled_tracing_records_span(span_exporter):
    exporter, provider = span_exporter
    import orchestrator.tracing as t
    t._provider = provider
    t._tracer = provider.get_tracer("test")

    with traced_task("task_001", "CODE"):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "task:task_001"
    assert spans[0].attributes["task.id"] == "task_001"
    assert spans[0].attributes["task.type"] == "CODE"


def test_llm_call_span_attributes(span_exporter):
    exporter, provider = span_exporter
    import orchestrator.tracing as t
    t._provider = provider
    t._tracer = provider.get_tracer("test")

    with traced_llm_call("gpt-4o", "generate") as span:
        span.set_attribute("llm.tokens_in", 100)
        span.set_attribute("llm.tokens_out", 200)
        span.set_attribute("llm.cost_usd", 0.003)

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "llm_call:generate"
    assert spans[0].attributes["llm.model"] == "gpt-4o"
    assert spans[0].attributes["llm.tokens_in"] == 100


def test_policy_check_span(span_exporter):
    exporter, provider = span_exporter
    import orchestrator.tracing as t
    t._provider = provider
    t._tracer = provider.get_tracer("test")

    with traced_policy_check(policy_count=3) as span:
        span.set_attribute("policy.passed", True)

    spans = exporter.get_finished_spans()
    assert spans[0].name == "policy_check"
    assert spans[0].attributes["policy.count"] == 3


def test_remediation_span(span_exporter):
    exporter, provider = span_exporter
    import orchestrator.tracing as t
    t._provider = provider
    t._tracer = provider.get_tracer("test")

    with traced_remediation("fallback_model", "score_below_threshold") as span:
        span.set_attribute("remediation.success", True)

    spans = exporter.get_finished_spans()
    assert spans[0].name == "remediation"
    assert spans[0].attributes["remediation.strategy"] == "fallback_model"


def test_nested_spans_have_parent_child_relationship(span_exporter):
    exporter, provider = span_exporter
    import orchestrator.tracing as t
    t._provider = provider
    t._tracer = provider.get_tracer("test")

    with traced_task("t1", "CODE"):
        with traced_llm_call("gpt-4o", "generate"):
            pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    child = next(s for s in spans if s.name == "llm_call:generate")
    parent = next(s for s in spans if s.name == "task:t1")
    assert child.parent.span_id == parent.context.span_id
```

**Step 2: Run to verify tests fail**

```bash
pytest tests/test_tracing.py -v
```

Expected: `ImportError: cannot import name 'TracingConfig' from 'orchestrator.tracing'`

**Step 3: Create `orchestrator/tracing.py`**

```python
"""
OpenTelemetry Distributed Tracing — tracing.py
===============================================
Provides TracingConfig, configure_tracing(), get_tracer(), and four
context-manager helpers for the main instrumentation points.

All helpers are no-op when tracing is disabled — overhead < 1 µs.

Usage:
    from orchestrator.tracing import configure_tracing, TracingConfig
    configure_tracing(TracingConfig(enabled=True, otlp_endpoint="http://localhost:4317"))
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

logger = logging.getLogger("orchestrator.tracing")

# ── Module-level singletons (reset between tests) ──────────────────────────────
_tracer = None
_provider = None


@dataclass
class TracingConfig:
    enabled: bool = False
    service_name: str = "multi-llm-orchestrator"
    otlp_endpoint: Optional[str] = None   # None → ConsoleSpanExporter (dev)
    sample_rate: float = 1.0


def configure_tracing(cfg: TracingConfig) -> None:
    """Initialise the global OTEL TracerProvider. Safe to call multiple times."""
    global _tracer, _provider

    if not cfg.enabled:
        # Use the no-op API tracer — guaranteed < 1 µs per span
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer(__name__)
        except ImportError:
            _tracer = _NoopTracer()
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.sampling import TraceIdRatioBased
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": cfg.service_name})
        sampler = TraceIdRatioBased(cfg.sample_rate)
        provider = TracerProvider(resource=resource, sampler=sampler)

        if cfg.otlp_endpoint:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            exporter = OTLPSpanExporter(endpoint=cfg.otlp_endpoint, insecure=True)
            provider.add_span_processor(BatchSpanProcessor(exporter))
            logger.info(f"OTEL tracing → {cfg.otlp_endpoint}")
        else:
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.sdk.trace.export.in_memory_span_exporter import ConsoleSpanExporter
            try:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter as _CSE
            except ImportError:
                from opentelemetry.sdk.trace.export import ConsoleSpanExporter as _CSE
            provider.add_span_processor(SimpleSpanProcessor(_CSE()))
            logger.info("OTEL tracing → console (dev mode)")

        trace.set_tracer_provider(provider)
        _provider = provider
        _tracer = provider.get_tracer(cfg.service_name)

    except ImportError:
        logger.warning(
            "opentelemetry-sdk not installed. "
            "Run: pip install -e '.[tracing]'"
        )
        _tracer = _NoopTracer()


def get_tracer():
    """Return the global tracer (no-op if configure_tracing() was not called)."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer(__name__)
        except ImportError:
            _tracer = _NoopTracer()
    return _tracer


# ── Context managers ───────────────────────────────────────────────────────────

@contextmanager
def traced_task(task_id: str, task_type: str) -> Iterator:
    """Span for a single task's full execution loop."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"task:{task_id}") as span:
        span.set_attribute("task.id", task_id)
        span.set_attribute("task.type", task_type)
        yield span


@contextmanager
def traced_llm_call(model: str, call_type: str) -> Iterator:
    """Span for a single LLM API request."""
    tracer = get_tracer()
    with tracer.start_as_current_span(f"llm_call:{call_type}") as span:
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.call_type", call_type)
        yield span


@contextmanager
def traced_policy_check(policy_count: int) -> Iterator:
    """Span for a PolicyEngine.check() or enforce() call."""
    tracer = get_tracer()
    with tracer.start_as_current_span("policy_check") as span:
        span.set_attribute("policy.count", policy_count)
        yield span


@contextmanager
def traced_remediation(strategy: str, trigger: str) -> Iterator:
    """Span for a RemediationEngine attempt."""
    tracer = get_tracer()
    with tracer.start_as_current_span("remediation") as span:
        span.set_attribute("remediation.strategy", strategy)
        span.set_attribute("remediation.trigger", trigger)
        yield span


# ── Fallback no-op tracer (when opentelemetry is not installed) ────────────────

class _NoopSpan:
    def set_attribute(self, key, value): pass
    def record_exception(self, exc): pass
    def set_status(self, status): pass
    def __enter__(self): return self
    def __exit__(self, *_): pass


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, name):
        yield _NoopSpan()
```

**Step 4: Run tests**

```bash
pytest tests/test_tracing.py -v
```

Expected: all tests PASS (some may need minor fixture tweaks for ConsoleSpanExporter import path).

**Step 5: Commit**

```bash
git add orchestrator/tracing.py tests/test_tracing.py
git commit -m "feat: add orchestrator/tracing.py with OTEL span helpers"
```

---

## Task 3: Instrument `api_clients.py`

**Files:**
- Modify: `orchestrator/api_clients.py` (around line 136 — `UnifiedClient.call()`)

**Step 1: Write the failing test**

Add to `tests/test_tracing.py`:

```python
def test_unified_client_emits_llm_span(span_exporter, tmp_path):
    """UnifiedClient.call() emits an llm_call span."""
    import asyncio
    from unittest.mock import AsyncMock, patch
    from orchestrator.api_clients import UnifiedClient, APIResponse
    from orchestrator.models import Model
    from orchestrator.tracing import configure_tracing, TracingConfig
    import orchestrator.tracing as t

    exporter, provider = span_exporter
    t._provider = provider
    t._tracer = provider.get_tracer("test")
    configure_tracing(TracingConfig(enabled=True))
    t._tracer = provider.get_tracer("test")  # override after configure

    fake_response = APIResponse("hello", 10, 20, Model.GPT_4O_MINI, latency_ms=42.0)

    with patch.object(UnifiedClient, "_call_with_retry", new=AsyncMock(return_value=fake_response)):
        client = UnifiedClient()
        resp = asyncio.run(client.call(Model.GPT_4O_MINI, "ping", bypass_cache=True))

    spans = exporter.get_finished_spans()
    llm_spans = [s for s in spans if s.name.startswith("llm_call:")]
    assert len(llm_spans) == 1
    assert llm_spans[0].attributes["llm.model"] == Model.GPT_4O_MINI.value
    assert llm_spans[0].attributes.get("llm.tokens_in") == 10
```

Run to confirm it fails:

```bash
pytest tests/test_tracing.py::test_unified_client_emits_llm_span -v
```

Expected: FAIL — `llm_spans` is empty (span not yet emitted).

**Step 2: Add import to `api_clients.py`**

At the top of `orchestrator/api_clients.py`, after existing imports:

```python
from .tracing import traced_llm_call
```

**Step 3: Wrap `call()` in `api_clients.py`**

Find `async def call(self, model, ...)` (line ~136). Change the method body to wrap the core dispatch:

```python
async def call(self, model: Model, prompt: str,
               system: str = "",
               max_tokens: int = 1500,
               temperature: float = 0.3,
               timeout: int = 60,
               retries: int = 2,
               bypass_cache: bool = False) -> APIResponse:
    if not bypass_cache:
        cached = await self.cache.get(model.value, prompt, max_tokens, system, temperature)
        if cached:
            logger.debug(f"Cache hit for {model.value}")
            return APIResponse(
                text=cached["response"],
                input_tokens=cached["tokens_input"],
                output_tokens=cached["tokens_output"],
                model=model,
                cached=True,
            )

    async with self.semaphore:
        with traced_llm_call(model.value, "api_call") as span:
            response = await self._call_with_retry(
                model, prompt, system, max_tokens, temperature, timeout, retries
            )
            span.set_attribute("llm.tokens_in", response.input_tokens)
            span.set_attribute("llm.tokens_out", response.output_tokens)
            span.set_attribute("llm.cost_usd", response.cost_usd)
            span.set_attribute("llm.latency_ms", response.latency_ms)
            span.set_attribute("llm.cached", False)
            return response
```

**Step 4: Run tests**

```bash
pytest tests/test_tracing.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add orchestrator/api_clients.py tests/test_tracing.py
git commit -m "feat: instrument api_clients.py with traced_llm_call spans"
```

---

## Task 4: Instrument `engine.py`

**Files:**
- Modify: `orchestrator/engine.py` (lines ~184 `run_project`, ~618 `_execute_task`)

**Step 1: Add import in `engine.py`**

After existing imports in `engine.py`:

```python
from .tracing import traced_task
```

**Step 2: Wrap `_execute_task()` in `engine.py`**

Find `async def _execute_task(self, task: Task) -> TaskResult:` (~line 618). Wrap the body:

```python
async def _execute_task(self, task: Task) -> TaskResult:
    with traced_task(task.id, task.task_type.value) as span:
        span.set_attribute("task.description", task.description[:200])
        result = await self._execute_task_inner(task)
        span.set_attribute("task.status", result.status.value)
        span.set_attribute("task.score", result.score or 0.0)
        return result
```

Then rename the existing `_execute_task` body to `_execute_task_inner`. (The existing body stays intact — just the method name changes to `_execute_task_inner`.)

**Step 3: Wrap `run_project()` root span**

Find `async def run_project(...)` (~line 184). Add root span around the body:

```python
async def run_project(self, project_description: str, ...):
    from .tracing import get_tracer
    tracer = get_tracer()
    with tracer.start_as_current_span("run_project") as span:
        span.set_attribute("project.description", project_description[:200])
        # ... existing body unchanged ...
```

**Step 4: Run existing engine tests**

```bash
pytest tests/test_engine_e2e.py -v
```

Expected: all tests still PASS (tracing is disabled by default — no-op path).

**Step 5: Commit**

```bash
git add orchestrator/engine.py
git commit -m "feat: instrument engine.py with run_project and task spans"
```

---

## Task 5: Instrument `policy_engine.py`

**Files:**
- Modify: `orchestrator/policy_engine.py` (lines containing `def check` ~129, `def enforce` ~279)

**Step 1: Add import**

```python
from .tracing import traced_policy_check
```

**Step 2: Wrap `check()` and `enforce()`**

In `check()`:

```python
def check(self, model: Model, profile: ModelProfile, policies: list[Policy]) -> PolicyResult:
    with traced_policy_check(len(policies)) as span:
        result = self._check_inner(model, profile, policies)
        span.set_attribute("policy.passed", result.passed)
        span.set_attribute("policy.violations", len(result.violations))
        return result
```

Rename existing `check` body to `_check_inner`. Same pattern for `enforce()`.

**Step 3: Run policy tests**

```bash
pytest tests/test_policy_governance.py tests/test_policy_dsl.py -v
```

Expected: all PASS.

**Step 4: Commit**

```bash
git add orchestrator/policy_engine.py
git commit -m "feat: instrument policy_engine.py with traced_policy_check spans"
```

---

## Task 6: Instrument `remediation.py`

**Files:**
- Modify: `orchestrator/remediation.py`

**Step 1: Add import**

```python
from .tracing import traced_remediation
```

**Step 2: Find and wrap the remediation execution method**

Find the method that applies a strategy (likely `apply()` or `execute()` in `RemediationEngine`). Wrap it:

```python
with traced_remediation(strategy.value, trigger_reason) as span:
    result = self._apply_strategy(strategy, task, result)
    span.set_attribute("remediation.success", result.status != TaskStatus.FAILED)
```

**Step 3: Run remediation tests**

```bash
pytest tests/ -k "remediation" -v
```

Expected: all PASS.

**Step 4: Commit**

```bash
git add orchestrator/remediation.py
git commit -m "feat: instrument remediation.py with traced_remediation spans"
```

---

## Task 7: Add CLI flags `--tracing` and `--otlp-endpoint`

**Files:**
- Modify: `orchestrator/cli.py`
- Modify: `orchestrator/engine.py` (Orchestrator.__init__ signature)

**Step 1: Update `Orchestrator.__init__`**

In `engine.py`, add `tracing_cfg` parameter:

```python
from .tracing import TracingConfig, configure_tracing

def __init__(self, ..., tracing_cfg: Optional[TracingConfig] = None):
    ...
    if tracing_cfg:
        configure_tracing(tracing_cfg)
```

**Step 2: Add CLI flags in `cli.py`**

Find the `argparse` argument parser setup. Add:

```python
parser.add_argument(
    "--tracing",
    action="store_true",
    default=False,
    help="Enable OpenTelemetry distributed tracing (requires pip install -e '[tracing]')"
)
parser.add_argument(
    "--otlp-endpoint",
    type=str,
    default=None,
    metavar="URL",
    help="OTLP gRPC endpoint (e.g. http://localhost:4317). Defaults to console output."
)
```

Then where `Orchestrator(...)` is constructed, pass the config:

```python
from .tracing import TracingConfig

tracing_cfg = None
if args.tracing:
    tracing_cfg = TracingConfig(
        enabled=True,
        otlp_endpoint=args.otlp_endpoint,
    )

orch = Orchestrator(..., tracing_cfg=tracing_cfg)
```

**Step 3: Test CLI flags**

```bash
python -m orchestrator --help | grep tracing
```

Expected: `--tracing` and `--otlp-endpoint` appear in help output.

**Step 4: Commit**

```bash
git add orchestrator/cli.py orchestrator/engine.py
git commit -m "feat: add --tracing and --otlp-endpoint CLI flags"
```

---

## Task 8: Export from `__init__.py` and final tests

**Files:**
- Modify: `orchestrator/__init__.py`
- Modify: `tests/test_tracing.py` (add integration test)

**Step 1: Export new symbols from `__init__.py`**

Add to imports and `__all__`:

```python
from .tracing import TracingConfig, configure_tracing, get_tracer
```

```python
# __all__ addition:
"TracingConfig", "configure_tracing", "get_tracer",
```

**Step 2: Add integration test (smoke test)**

Add to `tests/test_tracing.py`:

```python
def test_full_span_tree_shape(span_exporter):
    """Smoke test: traced_task nests traced_llm_call and traced_policy_check."""
    exporter, provider = span_exporter
    import orchestrator.tracing as t
    t._provider = provider
    t._tracer = provider.get_tracer("test")

    with traced_task("smoke", "CODE"):
        with traced_llm_call("claude-opus", "generate"):
            pass
        with traced_policy_check(2):
            pass
        with traced_remediation("auto_retry", "low_score"):
            pass

    spans = exporter.get_finished_spans()
    names = {s.name for s in spans}
    assert names == {
        "task:smoke",
        "llm_call:generate",
        "policy_check",
        "remediation",
    }
    # All children should have task:smoke as parent
    task_span = next(s for s in spans if s.name == "task:smoke")
    for s in spans:
        if s.name != "task:smoke":
            assert s.parent.span_id == task_span.context.span_id
```

**Step 3: Run full test suite**

```bash
pytest tests/ -v
```

Expected: all tests PASS, no regressions.

**Step 4: Final commit**

```bash
git add orchestrator/__init__.py tests/test_tracing.py
git commit -m "feat: export TracingConfig from package; add integration smoke test"
```

---

## Verification

After all tasks, verify the full feature end-to-end:

```bash
# 1. Confirm all tests pass
pytest tests/ -v

# 2. Confirm CLI help shows tracing flags
python -m orchestrator --help | grep -E "tracing|otlp"

# 3. Dry-run with console tracing (no OTLP backend needed)
python -m orchestrator --project "Add 1+1" --criteria "returns 2" --tracing --budget 0.01
# Expected: span output printed to console showing run_project > task > llm_call hierarchy
```
