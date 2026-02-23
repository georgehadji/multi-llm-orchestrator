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
from dataclasses import dataclass
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
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor, ConsoleSpanExporter
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
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
