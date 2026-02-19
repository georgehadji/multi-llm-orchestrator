"""
Tests for HookRegistry, EventType, and all MetricsExporter implementations.
Covers: callback registration, fire, error isolation, clear,
        ConsoleExporter, JSONExporter, PrometheusExporter.
"""
from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path

import pytest

from orchestrator.hooks import EventType, HookRegistry
from orchestrator.metrics import (
    ConsoleExporter,
    JSONExporter,
    PrometheusExporter,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def registry():
    return HookRegistry()


SAMPLE_METRICS = {
    "kimi-k2-5": {
        "call_count": 12,
        "failure_count": 0,
        "success_rate": 1.0,
        "avg_latency_ms": 1830.4,
        "latency_p95_ms": 2900.0,
        "quality_score": 0.87,
        "trust_factor": 1.0,
        "avg_cost_usd": 0.000042,
        "validator_fail_count": 0,
        "error_rate": 0.0,
    },
    "gpt-4o": {
        "call_count": 5,
        "failure_count": 1,
        "success_rate": 0.8,
        "avg_latency_ms": 900.0,
        "latency_p95_ms": 1500.0,
        "quality_score": 0.91,
        "trust_factor": 0.97,
        "avg_cost_usd": 0.0025,
        "validator_fail_count": 2,
        "error_rate": 0.2,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# HookRegistry — basic behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_add_and_fire_callback_called(registry):
    called = []
    registry.add(EventType.TASK_STARTED, lambda **kw: called.append(kw))
    registry.fire(EventType.TASK_STARTED, task_id="t_001")
    assert len(called) == 1
    assert called[0]["task_id"] == "t_001"


def test_fire_event_with_string_key(registry):
    """EventType.value strings should also work as event keys."""
    called = []
    registry.add("task_started", lambda **kw: called.append(True))
    registry.fire("task_started")
    assert called == [True]


def test_fire_eventtype_matches_string_key(registry):
    """Registering with EventType and firing with string (or vice-versa) should work."""
    called = []
    registry.add(EventType.TASK_COMPLETED, lambda **kw: called.append("ok"))
    registry.fire("task_completed")
    assert called == ["ok"]


def test_multiple_callbacks_all_called(registry):
    results = []
    registry.add(EventType.TASK_COMPLETED, lambda **kw: results.append(1))
    registry.add(EventType.TASK_COMPLETED, lambda **kw: results.append(2))
    registry.fire(EventType.TASK_COMPLETED)
    assert sorted(results) == [1, 2]


def test_fire_with_no_registered_callbacks_is_safe(registry):
    """Should not raise even with no callbacks."""
    registry.fire(EventType.BUDGET_WARNING, phase="gen", ratio=1.2)


def test_callback_exception_does_not_propagate(registry):
    """A crashing callback must not prevent other callbacks or crash the caller."""
    results = []
    registry.add(EventType.TASK_STARTED, lambda **kw: (_ for _ in ()).throw(RuntimeError("boom")))
    registry.add(EventType.TASK_STARTED, lambda **kw: results.append("safe"))
    registry.fire(EventType.TASK_STARTED)
    assert results == ["safe"]


def test_clear_specific_event(registry):
    called = []
    registry.add(EventType.TASK_STARTED, lambda **kw: called.append(True))
    registry.clear(EventType.TASK_STARTED)
    registry.fire(EventType.TASK_STARTED)
    assert called == []


def test_clear_all_events(registry):
    called = []
    registry.add(EventType.TASK_STARTED, lambda **kw: called.append(1))
    registry.add(EventType.TASK_COMPLETED, lambda **kw: called.append(2))
    registry.clear()
    registry.fire(EventType.TASK_STARTED)
    registry.fire(EventType.TASK_COMPLETED)
    assert called == []


def test_registered_events_returns_populated_events(registry):
    registry.add(EventType.VALIDATION_FAILED, lambda **kw: None)
    events = registry.registered_events()
    assert "validation_failed" in events


def test_registered_events_empty_when_no_hooks(registry):
    assert registry.registered_events() == []


def test_len_counts_all_callbacks(registry):
    registry.add(EventType.TASK_STARTED, lambda **kw: None)
    registry.add(EventType.TASK_STARTED, lambda **kw: None)
    registry.add(EventType.TASK_COMPLETED, lambda **kw: None)
    assert len(registry) == 3


def test_len_zero_when_empty(registry):
    assert len(registry) == 0


def test_all_event_types_exist():
    """Check all expected EventType values are present."""
    values = {e.value for e in EventType}
    assert "task_started" in values
    assert "task_completed" in values
    assert "validation_failed" in values
    assert "budget_warning" in values
    assert "model_selected" in values


# ─────────────────────────────────────────────────────────────────────────────
# ConsoleExporter
# ─────────────────────────────────────────────────────────────────────────────

def test_console_exporter_produces_output(capsys):
    exporter = ConsoleExporter()
    exporter.export(SAMPLE_METRICS)
    out = capsys.readouterr().out
    assert "kimi" in out
    assert "gpt-4o" in out


def test_console_exporter_shows_call_count(capsys):
    exporter = ConsoleExporter()
    exporter.export(SAMPLE_METRICS)
    out = capsys.readouterr().out
    assert "12" in out  # call_count for kimi-k2-5


def test_console_exporter_shows_success_rate(capsys):
    exporter = ConsoleExporter()
    exporter.export(SAMPLE_METRICS)
    out = capsys.readouterr().out
    assert "100.00%" in out


def test_console_exporter_empty_metrics_no_crash(capsys):
    exporter = ConsoleExporter()
    exporter.export({})
    out = capsys.readouterr().out
    assert len(out) > 0  # header still printed


# ─────────────────────────────────────────────────────────────────────────────
# JSONExporter
# ─────────────────────────────────────────────────────────────────────────────

def test_json_exporter_creates_valid_json(tmp_path):
    path = tmp_path / "metrics.json"
    exporter = JSONExporter(path)
    exporter.export(SAMPLE_METRICS)
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert "kimi-k2-5" in data
    assert data["kimi-k2-5"]["call_count"] == 12


def test_json_exporter_indented(tmp_path):
    path = tmp_path / "metrics.json"
    exporter = JSONExporter(path)
    exporter.export(SAMPLE_METRICS)
    content = path.read_text(encoding="utf-8")
    # Indented JSON must contain newlines
    assert "\n" in content


def test_json_exporter_overwrites_on_repeated_call(tmp_path):
    path = tmp_path / "metrics.json"
    exporter = JSONExporter(path)
    exporter.export(SAMPLE_METRICS)
    exporter.export({"model-x": {"call_count": 1}})
    data = json.loads(path.read_text(encoding="utf-8"))
    # Only second export should be in file
    assert "kimi-k2-5" not in data
    assert "model-x" in data


def test_json_exporter_creates_parent_dirs(tmp_path):
    path = tmp_path / "sub" / "dir" / "metrics.json"
    exporter = JSONExporter(path)
    exporter.export({})
    assert path.exists()


def test_json_exporter_accepts_string_path(tmp_path):
    path_str = str(tmp_path / "m.json")
    exporter = JSONExporter(path_str)
    exporter.export(SAMPLE_METRICS)
    assert Path(path_str).exists()


# ─────────────────────────────────────────────────────────────────────────────
# PrometheusExporter
# ─────────────────────────────────────────────────────────────────────────────

def test_prometheus_exporter_writes_to_stdout(capsys):
    exporter = PrometheusExporter()
    exporter.export(SAMPLE_METRICS)
    out = capsys.readouterr().out
    assert "orchestrator_model_calls_total" in out
    assert "orchestrator_success_rate" in out


def test_prometheus_exporter_sanitizes_hyphens(capsys):
    """Hyphens in model names must become underscores in label values."""
    exporter = PrometheusExporter()
    exporter.export({"kimi-k2-5": {"call_count": 1}})
    out = capsys.readouterr().out
    assert 'model="kimi_k2_5"' in out
    assert 'model="kimi-k2-5"' not in out


def test_prometheus_exporter_writes_to_file(tmp_path, capsys):
    path = tmp_path / "metrics.prom"
    exporter = PrometheusExporter(output_file=path)
    exporter.export(SAMPLE_METRICS)
    # Nothing should be written to stdout
    out = capsys.readouterr().out
    assert out == ""
    # File should exist and contain Prometheus format
    content = path.read_text(encoding="utf-8")
    assert "orchestrator_model_calls_total" in content
    assert "# HELP" in content
    assert "# TYPE" in content


def test_prometheus_exporter_help_and_type_lines(capsys):
    exporter = PrometheusExporter()
    exporter.export({"m": {"call_count": 5}})
    out = capsys.readouterr().out
    assert "# HELP orchestrator_model_calls_total" in out
    assert "# TYPE orchestrator_model_calls_total gauge" in out


def test_prometheus_exporter_all_metric_families_present(capsys):
    exporter = PrometheusExporter()
    exporter.export(SAMPLE_METRICS)
    out = capsys.readouterr().out
    expected_metrics = [
        "orchestrator_model_calls_total",
        "orchestrator_success_rate",
        "orchestrator_latency_avg_ms",
        "orchestrator_latency_p95_ms",
        "orchestrator_quality_score",
        "orchestrator_trust_factor",
        "orchestrator_cost_avg_usd",
        "orchestrator_validator_failures_total",
    ]
    for metric in expected_metrics:
        assert metric in out, f"Missing Prometheus metric: {metric}"


def test_prometheus_exporter_empty_metrics(capsys):
    exporter = PrometheusExporter()
    exporter.export({})
    out = capsys.readouterr().out
    # Should still emit # HELP / # TYPE lines
    assert "# HELP" in out
