# OpenTelemetry Distributed Tracing — Design

**Date:** 2026-02-23
**Author:** Brainstorming session
**Status:** Approved

---

## Problem

The orchestrator pipeline (decompose → route → generate → critique → revise) has no
distributed tracing. When a project run is slow or expensive it is impossible to pinpoint
which task, which LLM call, or which policy check is responsible. The existing
`TelemetryCollector` and `HookRegistry` record aggregated stats but cannot answer
"where did this specific run spend its time?"

---

## Goal

Add OpenTelemetry tracing so that every project run produces a span tree exportable to any
OTEL-compatible backend (Jaeger, Grafana Tempo, Datadog, Zipkin). Zero overhead when
tracing is disabled (default).

---

## Approach — Native instrumentation (Approach B)

Instrument the three hot paths directly:

| Module | What is instrumented |
|--------|---------------------|
| `engine.py` | `run_project()` root span, `_execute_task()` task span |
| `api_clients.py` | every `call()` — one span per LLM request |
| `policy_engine.py` | `check()` and `enforce()` — one span per policy evaluation |
| `remediation.py` | `run()` — one span per remediation attempt |

---

## Span Hierarchy

```
run_project                    [project_id, status, total_cost_usd]
  └── task:<task_id>           [task_type, model_selected, iteration]
        ├── llm_call:generate  [model, tokens_in, tokens_out, cost_usd, latency_ms]
        ├── llm_call:critique  [model, tokens_in, tokens_out, cost_usd, latency_ms]
        ├── llm_call:revise    [model, tokens_in, tokens_out, cost_usd, latency_ms]
        ├── policy_check       [policies_count, passed, violations_count]
        └── remediation        [strategy, trigger_reason, success]
```

---

## New Module: `orchestrator/tracing.py`

```python
@dataclass
class TracingConfig:
    enabled: bool = False
    service_name: str = "multi-llm-orchestrator"
    otlp_endpoint: str | None = None   # None → ConsoleSpanExporter (dev mode)
    sample_rate: float = 1.0

def configure_tracing(cfg: TracingConfig) -> None:
    """Initialize OTEL TracerProvider. Safe to call multiple times."""

def get_tracer() -> opentelemetry.trace.Tracer:
    """Return the global tracer (no-op tracer when disabled)."""

@contextmanager
def traced_task(task_id: str, task_type: str) -> Iterator[Span]: ...

@contextmanager
def traced_llm_call(model: str, call_type: str) -> Iterator[Span]: ...

@contextmanager
def traced_policy_check(policy_count: int) -> Iterator[Span]: ...

@contextmanager
def traced_remediation(strategy: str, trigger: str) -> Iterator[Span]: ...
```

All context managers are **no-op** when `TracingConfig.enabled = False`, adding
`< 1 µs` overhead per call.

---

## CLI Integration

```bash
# Tracing disabled (default)
python -m orchestrator --project "..." --criteria "..."

# Tracing enabled — prints spans to console (dev)
python -m orchestrator --project "..." --tracing

# Tracing enabled — exports to OTLP endpoint (production)
python -m orchestrator --project "..." --tracing --otlp-endpoint http://localhost:4317
```

`cli.py` changes:
- Add `--tracing` boolean flag
- Add `--otlp-endpoint` optional string flag
- Build `TracingConfig` and pass to `Orchestrator.__init__()`

`Orchestrator.__init__()` change:
- Accept optional `tracing_cfg: TracingConfig | None = None`
- Call `configure_tracing(cfg)` at startup

---

## Dependencies

Added to `pyproject.toml` as optional extras (`[project.optional-dependencies]`):

```toml
[project.optional-dependencies]
tracing = [
    "opentelemetry-api>=1.20",
    "opentelemetry-sdk>=1.20",
    "opentelemetry-exporter-otlp-proto-grpc>=1.20",
]
```

Install: `pip install -e ".[tracing]"`

---

## Testing Strategy

- Unit tests in `tests/test_tracing.py` using `opentelemetry.sdk.trace.export.in_memory_span_exporter.InMemorySpanExporter`
- Assert span names, attributes (model, tokens, cost), and parent-child relationships
- Verify that `enabled=False` produces zero spans (no-op path)
- Integration test: `run_project()` with mock LLMs → verify full span tree shape

---

## Files Changed

| File | Change type |
|------|-------------|
| `orchestrator/tracing.py` | **New** |
| `orchestrator/engine.py` | Modified — add `traced_task`, `traced_llm_call` calls |
| `orchestrator/api_clients.py` | Modified — wrap `call()` in `traced_llm_call` |
| `orchestrator/policy_engine.py` | Modified — wrap `check()`/`enforce()` in `traced_policy_check` |
| `orchestrator/remediation.py` | Modified — wrap `run()` in `traced_remediation` |
| `orchestrator/cli.py` | Modified — add `--tracing`, `--otlp-endpoint` flags |
| `orchestrator/__init__.py` | Modified — export `TracingConfig`, `configure_tracing` |
| `pyproject.toml` | Modified — add `[tracing]` optional extras |
| `tests/test_tracing.py` | **New** |

---

## Non-Goals

- No Prometheus bridge (existing `PrometheusExporter` is unaffected)
- No UI / dashboard (use Jaeger or Grafana Tempo)
- No log correlation (future work)
