"""Tests for SQLite-based report generation."""

import json
import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _create_test_db(db_path, entries):
    """Create a test SQLite database with request entries."""
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            request_id TEXT,
            type TEXT,
            status TEXT,
            prompt TEXT,
            selected_model TEXT,
            provider TEXT,
            tier TEXT,
            confidence REAL,
            complexity_score REAL,
            classifier_latency_ms INTEGER,
            total_latency_ms INTEGER,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            total_tokens INTEGER,
            cost REAL,
            daily_spend REAL,
            response_preview TEXT,
            fallback_used TEXT,
            error TEXT,
            tool_count INTEGER,
            has_images INTEGER,
            has_tools INTEGER,
            max_context_tokens INTEGER
        )
    """)
    for e in entries:
        cursor.execute("""
            INSERT INTO requests (timestamp, request_id, type, status, selected_model,
                provider, tier, confidence, classifier_latency_ms, total_latency_ms,
                prompt_tokens, completion_tokens, total_tokens, cost, fallback_used, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            e.get("timestamp"), e.get("request_id"), e.get("type", "chat"),
            e.get("status", "ok"), e.get("selected_model"), e.get("provider"),
            e.get("tier"), e.get("confidence"), e.get("classifier_latency_ms"),
            e.get("total_latency_ms"), e.get("prompt_tokens"), e.get("completion_tokens"),
            e.get("total_tokens"), e.get("cost"), e.get("fallback_used"), e.get("error"),
        ))
    conn.commit()
    conn.close()


SAMPLE_ENTRIES = [
    {
        "timestamp": "2026-03-01T08:00:00+00:00",
        "request_id": "r1",
        "type": "chat",
        "status": "ok",
        "selected_model": "claude-3-haiku",
        "provider": "anthropic",
        "tier": "simple",
        "confidence": 0.95,
        "classifier_latency_ms": 8,
        "total_latency_ms": 450,
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cost": 0.0003,
    },
    {
        "timestamp": "2026-03-01T08:01:00+00:00",
        "request_id": "r2",
        "type": "chat",
        "status": "ok",
        "selected_model": "claude-3-opus",
        "provider": "anthropic",
        "tier": "complex",
        "confidence": 0.88,
        "classifier_latency_ms": 10,
        "total_latency_ms": 3200,
        "prompt_tokens": 500,
        "completion_tokens": 800,
        "total_tokens": 1300,
        "cost": 0.045,
    },
    {
        "timestamp": "2026-03-01T08:02:00+00:00",
        "request_id": "r3",
        "type": "chat",
        "status": "error",
        "selected_model": "claude-3-haiku",
        "provider": "anthropic",
        "tier": "simple",
        "confidence": 0.92,
        "classifier_latency_ms": 7,
        "total_latency_ms": 120,
        "prompt_tokens": 80,
        "completion_tokens": 0,
        "total_tokens": 80,
        "cost": 0.0,
        "error": "rate_limit",
    },
    {
        "timestamp": "2026-03-01T08:03:00+00:00",
        "request_id": "r4",
        "type": "chat",
        "status": "ok",
        "selected_model": "gpt-4o-mini",
        "provider": "openai",
        "tier": "simple",
        "confidence": 0.97,
        "classifier_latency_ms": 6,
        "total_latency_ms": 380,
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "total_tokens": 300,
        "cost": 0.0006,
        "fallback_used": "claude-3-haiku -> gpt-4o-mini",
    },
]


def test_load_sqlite_all():
    from nadirclaw.report import load_log_entries_sqlite

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "requests.db"
        _create_test_db(db_path, SAMPLE_ENTRIES)

        entries = load_log_entries_sqlite(db_path)
        assert len(entries) == 4


def test_load_sqlite_with_model_filter():
    from nadirclaw.report import load_log_entries_sqlite

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "requests.db"
        _create_test_db(db_path, SAMPLE_ENTRIES)

        entries = load_log_entries_sqlite(db_path, model_filter="haiku")
        assert len(entries) == 2
        assert all("haiku" in e["selected_model"] for e in entries)


def test_load_sqlite_with_since():
    from nadirclaw.report import load_log_entries_sqlite, parse_since

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "requests.db"
        _create_test_db(db_path, SAMPLE_ENTRIES)

        since = datetime(2026, 3, 1, 8, 1, 30, tzinfo=timezone.utc)
        entries = load_log_entries_sqlite(db_path, since=since)
        assert len(entries) == 2  # r3 and r4


def test_generate_report_with_cost():
    from nadirclaw.report import generate_report, load_log_entries_sqlite

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "requests.db"
        _create_test_db(db_path, SAMPLE_ENTRIES)

        entries = load_log_entries_sqlite(db_path)
        report = generate_report(entries)

        assert report["total_requests"] == 4
        assert report["error_count"] == 1
        assert report["fallback_count"] == 1
        assert report["total_cost"] == pytest.approx(0.0459, abs=0.001)

        # Cost breakdown by model
        assert "claude-3-opus" in report["model_usage"]
        assert report["model_usage"]["claude-3-opus"]["cost"] == pytest.approx(0.045, abs=0.001)
        assert report["model_usage"]["claude-3-haiku"]["requests"] == 2

        # Latency
        assert "total" in report["latency"]
        assert report["latency"]["total"]["p50"] > 0
        assert report["latency"]["total"]["p95"] > 0


def test_format_report_shows_cost():
    from nadirclaw.report import format_report_text, generate_report, load_log_entries_sqlite

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "requests.db"
        _create_test_db(db_path, SAMPLE_ENTRIES)

        entries = load_log_entries_sqlite(db_path)
        report = generate_report(entries)
        text = format_report_text(report)

        assert "Total cost:" in text
        assert "$0.04" in text
        assert "Cost" in text  # header
        assert "claude-3-opus" in text


def test_json_output():
    from nadirclaw.report import generate_report, load_log_entries_sqlite

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "requests.db"
        _create_test_db(db_path, SAMPLE_ENTRIES)

        entries = load_log_entries_sqlite(db_path)
        report = generate_report(entries)

        # Verify it's JSON-serializable
        output = json.dumps(report, indent=2, default=str)
        parsed = json.loads(output)
        assert parsed["total_requests"] == 4
        assert "total_cost" in parsed
        assert "model_usage" in parsed
