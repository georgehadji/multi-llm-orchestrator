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


def test_disabled_tracing_produces_no_exception():
    configure_tracing(TracingConfig(enabled=False))
    with traced_task("t1", "CODE"):
        with traced_llm_call("gpt-4o", "generate"):
            pass
    tracer = get_tracer()
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
