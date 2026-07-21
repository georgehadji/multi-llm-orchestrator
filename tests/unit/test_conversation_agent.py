"""
Unit tests for orchestrator.application.conversation_agent.

Tests cover:
- ConversationAgent confidence scoring
- ProjectSpec merging
- Ready-signal detection
- spec.to_orchestrator_args() output
- Enhancement acceptance
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.unit

from orchestrator.application.conversation_agent import (
    ConversationAgent,
    ConversationTurn,
    ProjectSpec,
    _READY_SIGNALS,
)

pytestmark = pytest.mark.asyncio


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_agent(llm_response: str | None = None) -> ConversationAgent:
    """Build an agent with a mocked LLM client."""
    client = MagicMock()
    response = MagicMock()
    if llm_response is None:
        llm_response = json.dumps(
            {
                "message": "Tell me more.",
                "suggestions": [],
                "ready": False,
                "spec_partial": {},
            }
        )
    response.text = llm_response
    client.call = AsyncMock(return_value=response)
    return ConversationAgent(client=client)


def _full_spec_response(**overrides) -> str:
    """JSON response that fills all 6 required fields."""
    partial = {
        "project_description": "A SaaS invoicing tool",
        "target_users": "Freelancers and small agencies",
        "core_features": ["Invoice creation", "PDF export", "Payment tracking"],
        "tech_stack": "Next.js + FastAPI + PostgreSQL",
        "platform": "web",
        "auth_requirements": "Email + Google OAuth",
        "data_persistence": "PostgreSQL, invoices and clients tables",
        "success_criteria": "Users can create and send invoices",
    }
    partial.update(overrides)
    return json.dumps(
        {
            "message": "Here is a summary of the spec.",
            "suggestions": ["Stripe integration", "Admin dashboard"],
            "ready": True,
            "spec_partial": partial,
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# ProjectSpec tests
# ─────────────────────────────────────────────────────────────────────────────


def test_spec_to_orchestrator_args_includes_description():
    spec = ProjectSpec(
        project_description="An invoicing app",
        success_criteria="All tests pass",
    )
    args = spec.to_orchestrator_args()
    assert "An invoicing app" in args["project"]
    assert args["criteria"] == "All tests pass"


def test_spec_to_orchestrator_args_includes_features():
    spec = ProjectSpec(
        project_description="App",
        core_features=["Login", "Dashboard", "Export"],
    )
    args = spec.to_orchestrator_args()
    assert "Login" in args["project"]
    assert "Dashboard" in args["project"]


def test_spec_to_orchestrator_args_includes_enhancements():
    spec = ProjectSpec(
        project_description="App",
        enhancements_accepted=["Stripe integration", "Rate limiting"],
    )
    args = spec.to_orchestrator_args()
    assert "Stripe integration" in args["project"]


def test_spec_to_orchestrator_args_default_criteria():
    spec = ProjectSpec(project_description="App", success_criteria="")
    args = spec.to_orchestrator_args()
    assert "production-ready" in args["criteria"]


def test_spec_budget_defaults_to_eight():
    spec = ProjectSpec()
    assert spec.budget_usd == 8.0


# ─────────────────────────────────────────────────────────────────────────────
# ConversationAgent — opening
# ─────────────────────────────────────────────────────────────────────────────


async def test_start_returns_agent_turn():
    agent = _make_agent()
    turn = await agent.start()
    assert isinstance(turn, ConversationTurn)
    assert turn.role == "agent"
    assert len(turn.content) > 10
    assert turn.ready_to_build is False


# ─────────────────────────────────────────────────────────────────────────────
# ConversationAgent — confidence scoring
# ─────────────────────────────────────────────────────────────────────────────


async def test_confidence_zero_on_empty_spec():
    agent = _make_agent()
    assert agent.confidence == 0.0


async def test_confidence_increases_as_fields_filled():
    agent = _make_agent(_full_spec_response())
    await agent.start()
    await agent.process_turn("Here is my project idea")
    # All 6 required fields filled → confidence should be 1.0
    assert agent.confidence == pytest.approx(1.0)


async def test_confidence_partial_when_some_fields_missing():
    agent = _make_agent(
        json.dumps(
            {
                "message": "Got it.",
                "suggestions": [],
                "ready": False,
                "spec_partial": {
                    "project_description": "An app",
                    "target_users": "Developers",
                    # missing features, tech_stack, auth, data_persistence
                },
            }
        )
    )
    await agent.start()
    await agent.process_turn("I want to build an app for developers")
    # 2 out of 6 required fields → ~0.33
    assert 0.1 < agent.confidence < 0.6


# ─────────────────────────────────────────────────────────────────────────────
# ConversationAgent — ready detection
# ─────────────────────────────────────────────────────────────────────────────


async def test_agent_not_ready_initially():
    agent = _make_agent()
    assert agent.ready is False


async def test_agent_ready_when_llm_says_ready_and_confidence_high():
    agent = _make_agent(_full_spec_response())
    await agent.start()
    await agent.process_turn("Here is my full project description")
    assert agent.ready is True


async def test_explicit_go_signal_marks_ready_at_50pct_confidence():
    """User typing 'go' should mark ready even if only half the fields are filled."""
    partial_response = json.dumps(
        {
            "message": "Tell me more.",
            "suggestions": [],
            "ready": False,
            "spec_partial": {
                "project_description": "An invoicing app",
                "target_users": "Freelancers",
                "core_features": ["Invoice creation", "PDF export", "Payments"],
                "tech_stack": "Next.js",
            },
        }
    )
    agent = _make_agent(partial_response)
    await agent.start()
    await agent.process_turn("It's an invoicing tool for freelancers")
    # confidence now ~0.67 (4/6) — enough for explicit signal
    turn = await agent.process_turn("go")
    assert turn.ready_to_build is True
    assert agent.ready is True


async def test_explicit_signal_below_50pct_does_not_mark_ready():
    """'go' with no context should not fire build (confidence too low)."""
    agent = _make_agent()
    await agent.start()
    turn = await agent.process_turn("go")
    # confidence is 0.0 — should not mark ready
    assert turn.ready_to_build is False


def test_ready_signals_set_contains_common_phrases():
    for phrase in ("ok", "go", "build it", "yes", "proceed", "let's go"):
        assert phrase in _READY_SIGNALS


# ─────────────────────────────────────────────────────────────────────────────
# ConversationAgent — spec merging
# ─────────────────────────────────────────────────────────────────────────────


async def test_features_accumulate_across_turns():
    first = json.dumps(
        {
            "message": "Got it.",
            "suggestions": [],
            "ready": False,
            "spec_partial": {"core_features": ["Login", "Dashboard"]},
        }
    )
    second = json.dumps(
        {
            "message": "More features.",
            "suggestions": [],
            "ready": False,
            "spec_partial": {"core_features": ["Export", "Notifications"]},
        }
    )

    client = MagicMock()
    r1, r2 = MagicMock(), MagicMock()
    r1.text = first
    r2.text = second
    client.call = AsyncMock(side_effect=[r1, r2])
    agent = ConversationAgent(client=client)

    await agent.start()
    await agent.process_turn("Turn 1")
    await agent.process_turn("Turn 2")

    assert set(agent.spec.core_features) == {"Login", "Dashboard", "Export", "Notifications"}


async def test_duplicate_features_not_added():
    resp = json.dumps(
        {
            "message": ".",
            "suggestions": [],
            "ready": False,
            "spec_partial": {"core_features": ["Login", "Login", "Dashboard"]},
        }
    )
    agent = _make_agent(resp)
    await agent.start()
    await agent.process_turn("anything")
    assert agent.spec.core_features.count("Login") == 1


# ─────────────────────────────────────────────────────────────────────────────
# ConversationAgent — suggestions
# ─────────────────────────────────────────────────────────────────────────────


async def test_suggestions_returned_in_turn():
    resp = json.dumps(
        {
            "message": "Got it.",
            "suggestions": ["Stripe integration", "Admin dashboard"],
            "ready": False,
            "spec_partial": {},
        }
    )
    agent = _make_agent(resp)
    await agent.start()
    turn = await agent.process_turn("I want a SaaS app")
    assert "Stripe integration" in turn.suggestions


async def test_accept_enhancement_stored():
    agent = _make_agent()
    agent.accept_enhancement("Stripe integration")
    agent.accept_enhancement("Rate limiting")
    assert "Stripe integration" in agent.spec.enhancements_accepted
    assert "Rate limiting" in agent.spec.enhancements_accepted


async def test_accept_enhancement_no_duplicates():
    agent = _make_agent()
    agent.accept_enhancement("Stripe integration")
    agent.accept_enhancement("Stripe integration")
    assert agent.spec.enhancements_accepted.count("Stripe integration") == 1


# ─────────────────────────────────────────────────────────────────────────────
# ConversationAgent — LLM error handling
# ─────────────────────────────────────────────────────────────────────────────


async def test_llm_error_returns_graceful_fallback():
    client = MagicMock()
    client.call = AsyncMock(side_effect=Exception("network timeout"))
    agent = ConversationAgent(client=client)
    await agent.start()
    turn = await agent.process_turn("I want to build something")
    assert turn.role == "agent"
    assert len(turn.content) > 0  # should not crash, should return fallback message


async def test_malformed_json_from_llm_handled_gracefully():
    client = MagicMock()
    response = MagicMock()
    response.text = "Sure! What kind of app did you have in mind?"  # plain text, not JSON
    client.call = AsyncMock(return_value=response)
    agent = ConversationAgent(client=client)
    await agent.start()
    turn = await agent.process_turn("I want to build something")
    assert "app" in turn.content.lower()
    assert turn.ready_to_build is False
