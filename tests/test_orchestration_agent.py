"""
Unit tests for OrchestrationAgent (B4+B5).
"""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from orchestrator.orchestration_agent import AgentDraft, OrchestrationAgent
from orchestrator.specs import JobSpecV2, PolicySpecV2


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_llm_response(goal: str, hard: list[str], rationale: str) -> str:
    """Build a valid JSON response string for the agent."""
    return json.dumps({
        "job": {
            "goal": goal,
            "inputs": {"data_locality": "eu", "contains_pii": False},
            "slas": {"max_cost_usd": 2.0, "min_quality_tier": 0.85},
            "constraints": {"hard": hard, "soft": {}},
            "metrics": [],
        },
        "policy": {
            "allow_deny_rules": [],
            "routing_hints": [],
            "validation_rules": [],
            "escalation_rules": [],
        },
        "rationale": rationale,
    })


# ─────────────────────────────────────────────────────────────────────────────
# draft()
# ─────────────────────────────────────────────────────────────────────────────

def test_draft_returns_agent_draft():
    agent = OrchestrationAgent()
    resp = _make_llm_response("code review", ["eu_only"], "EU data only")
    agent._call_llm = AsyncMock(return_value=resp)

    draft = asyncio.run(agent.draft("code review, EU data only"))

    assert isinstance(draft, AgentDraft)
    assert isinstance(draft.job, JobSpecV2)
    assert isinstance(draft.policy, PolicySpecV2)


def test_draft_parses_goal():
    agent = OrchestrationAgent()
    resp = _make_llm_response("my goal", [], "no constraints")
    agent._call_llm = AsyncMock(return_value=resp)

    draft = asyncio.run(agent.draft("my goal"))
    assert draft.job.goal == "my goal"


def test_draft_parses_hard_constraints():
    agent = OrchestrationAgent()
    resp = _make_llm_response("test", ["eu_only", "no_training"], "EU + no training")
    agent._call_llm = AsyncMock(return_value=resp)

    draft = asyncio.run(agent.draft("test"))
    assert "eu_only" in draft.job.constraints.hard
    assert "no_training" in draft.job.constraints.hard


def test_draft_captures_rationale():
    agent = OrchestrationAgent()
    resp = _make_llm_response("test", [], "Chose eu_only because data is sensitive")
    agent._call_llm = AsyncMock(return_value=resp)

    draft = asyncio.run(agent.draft("test"))
    assert "eu_only" in draft.rationale or "sensitive" in draft.rationale


def test_draft_strips_markdown_fences():
    agent = OrchestrationAgent()
    raw = "```json\n" + _make_llm_response("test", [], "ok") + "\n```"
    agent._call_llm = AsyncMock(return_value=raw)

    draft = asyncio.run(agent.draft("test"))
    assert draft.job.goal == "test"


def test_draft_falls_back_on_bad_json():
    agent = OrchestrationAgent()
    agent._call_llm = AsyncMock(return_value="not json at all")

    draft = asyncio.run(agent.draft("fallback test"))
    assert isinstance(draft, AgentDraft)
    assert draft.job.goal == "fallback test"
    assert "Default" in draft.rationale


# ─────────────────────────────────────────────────────────────────────────────
# refine()
# ─────────────────────────────────────────────────────────────────────────────

def test_refine_returns_updated_draft():
    agent = OrchestrationAgent()
    initial_resp = _make_llm_response("initial goal", [], "initial rationale")
    refined_resp = _make_llm_response("initial goal", ["no_training"], "added no_training")
    agent._call_llm = AsyncMock(side_effect=[initial_resp, refined_resp])

    async def _run():
        draft = await agent.draft("initial goal")
        return await agent.refine(draft, "also block training data")

    refined = asyncio.run(_run())
    assert "no_training" in refined.job.constraints.hard


# ─────────────────────────────────────────────────────────────────────────────
# analyze_run() — B5 telemetry loop
# ─────────────────────────────────────────────────────────────────────────────

def test_analyze_run_returns_suggestions():
    from orchestrator.models import Budget, ProjectState, ProjectStatus

    agent = OrchestrationAgent()
    agent._call_llm = AsyncMock(
        return_value="Suggestion: tighten max_cost_usd to $0.03\nSuggestion: add ruff"
    )

    state = ProjectState(
        project_description="test",
        success_criteria="pass",
        budget=Budget(max_usd=2.0),
        tasks={},
        results={},
        api_health={},
        status=ProjectStatus.SUCCESS,
    )
    job = JobSpecV2(goal="test")
    suggestions = asyncio.run(agent.analyze_run(state, job, PolicySpecV2()))

    assert "Suggestion" in suggestions


# ─────────────────────────────────────────────────────────────────────────────
# Default draft
# ─────────────────────────────────────────────────────────────────────────────

def test_default_draft():
    agent = OrchestrationAgent()
    draft = agent._default_draft("test intent")
    assert draft.nl_intent == "test intent"
    assert isinstance(draft.job, JobSpecV2)
    assert "Default" in draft.rationale
