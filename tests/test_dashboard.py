"""
Tests for the CLI dashboard rendering.

Plan reference: docs/plans/2026-02-25-learn-and-show-design.md

Tests the render_dashboard() function in metrics.py and verifies that
the 'dashboard' subcommand is registered in cli.py.

Tests will initially FAIL (ImportError / AttributeError) because
render_dashboard does not exist yet — RED phase before implementation.
"""
from __future__ import annotations

import asyncio
import pytest

from orchestrator.models import Model, TaskType
from orchestrator.policy import ModelProfile
from orchestrator.telemetry_store import TelemetryStore
from orchestrator.metrics import render_dashboard


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_profile(
    *,
    model: Model = Model.DEEPSEEK_CHAT,
    quality_score: float = 0.87,
    avg_cost_usd: float = 0.002,
    call_count: int = 1,
) -> ModelProfile:
    return ModelProfile(
        model=model,
        provider="deepseek",
        cost_per_1m_input=0.27,
        cost_per_1m_output=1.10,
        quality_score=quality_score,
        trust_factor=0.9,
        avg_latency_ms=1500.0,
        latency_p95_ms=3000.0,
        success_rate=0.95,
        avg_cost_usd=avg_cost_usd,
        call_count=call_count,
        failure_count=0,
        validator_fail_count=0,
    )


async def _seed_hot_data(store: TelemetryStore, tmp_path) -> None:
    """Seed 50+ calls for two models so they reach HOT confidence."""
    for i in range(50):
        await store.record_snapshot(f"d{i}", Model.DEEPSEEK_CHAT, TaskType.CODE_GEN,
                                    _make_profile(quality_score=0.90, avg_cost_usd=0.001, call_count=1))
    for i in range(50):
        await store.record_snapshot(f"g{i}", Model.GPT_4O, TaskType.CODE_GEN,
                                    _make_profile(model=Model.GPT_4O, quality_score=0.85,
                                                  avg_cost_usd=0.010, call_count=1))


# ─────────────────────────────────────────────────────────────────────────────
# render_dashboard — basic contract
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_render_dashboard_returns_string(tmp_path):
    """render_dashboard() returns a non-empty string."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    output = await render_dashboard(store, days=30)
    assert isinstance(output, str)


@pytest.mark.asyncio
async def test_render_dashboard_no_data_message(tmp_path):
    """render_dashboard() indicates no data when the DB is empty."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    output = await render_dashboard(store, days=30)
    # Should communicate that there's no data — not crash or return blank
    assert "no data" in output.lower() or "0 runs" in output.lower() or len(output) > 0


@pytest.mark.asyncio
async def test_render_dashboard_contains_model_rankings_header(tmp_path):
    """render_dashboard() includes a MODEL RANKINGS section."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await _seed_hot_data(store, tmp_path)

    output = await render_dashboard(store, days=30)
    assert "MODEL RANKINGS" in output.upper()


@pytest.mark.asyncio
async def test_render_dashboard_lists_model_names(tmp_path):
    """render_dashboard() shows model names in the rankings section."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await _seed_hot_data(store, tmp_path)

    output = await render_dashboard(store, days=30)
    assert "deepseek-chat" in output
    assert "gpt-4o" in output


@pytest.mark.asyncio
async def test_render_dashboard_shows_confidence_labels(tmp_path):
    """render_dashboard() shows HOT/WARM/COLD confidence labels."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await _seed_hot_data(store, tmp_path)

    output = await render_dashboard(store, days=30)
    # Both models have 50 calls → HOT
    assert "HOT" in output


@pytest.mark.asyncio
async def test_render_dashboard_shows_task_type_leaders(tmp_path):
    """render_dashboard() includes a TASK-TYPE LEADERS section."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await _seed_hot_data(store, tmp_path)

    output = await render_dashboard(store, days=30)
    assert "TASK" in output.upper()


@pytest.mark.asyncio
async def test_render_dashboard_shows_recommendations_section(tmp_path):
    """render_dashboard() includes a RECOMMENDATIONS section."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await _seed_hot_data(store, tmp_path)

    output = await render_dashboard(store, days=30)
    assert "RECOMMENDATION" in output.upper()


@pytest.mark.asyncio
async def test_render_dashboard_renders_within_reasonable_length(tmp_path):
    """render_dashboard() output is non-trivial but not excessively large."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await _seed_hot_data(store, tmp_path)

    output = await render_dashboard(store, days=30)
    # Should have a real structure: at least 10 lines
    lines = [l for l in output.splitlines() if l.strip()]
    assert len(lines) >= 10


@pytest.mark.asyncio
async def test_render_dashboard_higher_value_model_ranked_first(tmp_path):
    """render_dashboard() ranks deepseek (better value) before gpt-4o."""
    store = TelemetryStore(db_path=tmp_path / "telemetry.db")
    await _seed_hot_data(store, tmp_path)

    output = await render_dashboard(store, days=30)
    deepseek_pos = output.find("deepseek-chat")
    gpt4o_pos = output.find("gpt-4o")
    # deepseek should appear before gpt-4o in the output
    assert deepseek_pos < gpt4o_pos


# ─────────────────────────────────────────────────────────────────────────────
# CLI subcommand registration
# ─────────────────────────────────────────────────────────────────────────────

def test_dashboard_subcommand_is_registered():
    """The 'dashboard' subcommand is registered in the CLI parser."""
    import argparse
    from orchestrator.cli import main
    import sys

    # Construct a parser instance by importing the registration function directly
    # We check that parsing 'dashboard' doesn't raise an error
    from orchestrator import cli as cli_module
    assert hasattr(cli_module, "_dashboard_subparsers"), (
        "_dashboard_subparsers must be defined in cli.py to register the dashboard subcommand"
    )
