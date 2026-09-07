"""Agent registry — the roster must stay total, and dispatch must resolve.

T23-AGENT1: AgentOrchestrator takes its agents by injection and never builds any,
and `_run_one` degrades silently when a role is missing (it returns a failed
AgentTaskResult rather than raising). With `agents/__init__.py` exporting
implementations for only 3 of the 10 declared roles, the subsystem was a no-op —
most sharply for AgentRole.INVESTIGATOR, which `_decompose_goal` actively
dispatches to while its implementation sat unexported.

Silent degradation is exactly what makes an incomplete roster hard to notice, so
these tests assert the properties that silence would hide.
"""

from __future__ import annotations

import pytest

from orchestrator.agents import (
    AGENT_TYPES,
    AgentBase,
    AgentOrchestrator,
    AgentRole,
    build_default_agents,
)


@pytest.mark.unit
def test_every_declared_role_has_an_implementation() -> None:
    """A role with no implementation dispatches to nothing, silently."""
    missing = set(AgentRole) - set(AGENT_TYPES)
    assert not missing, (
        "AgentRole members with no entry in AGENT_TYPES — dispatch to these "
        f"returns a failed result with no error: {sorted(r.value for r in missing)}"
    )


@pytest.mark.unit
def test_each_agent_binds_the_role_it_is_registered_under() -> None:
    """CodebaseInvestigatorAgent alone used to inherit AgentBase.__init__ without
    binding its role, so it could be constructed under any role at all."""
    for role, agent in build_default_agents().items():
        assert (
            agent.role is role
        ), f"{type(agent).__name__} registered as {role} but bound {agent.role}"


@pytest.mark.unit
def test_default_roster_covers_every_role() -> None:
    roster = build_default_agents()
    assert set(roster) == set(AgentRole)
    assert all(isinstance(a, AgentBase) for a in roster.values())


@pytest.mark.unit
def test_roles_argument_selects_a_subset() -> None:
    roster = build_default_agents(roles=[AgentRole.DEVELOPER, AgentRole.QA])
    assert set(roster) == {AgentRole.DEVELOPER, AgentRole.QA}


@pytest.mark.unit
def test_the_role_the_coordinator_dispatches_to_resolves() -> None:
    """_decompose_goal targets INVESTIGATOR; before the registry, obtaining one
    required reaching past __init__.py into an unexported module."""
    orch = AgentOrchestrator(agents=build_default_agents())
    agent = orch.get_agent(AgentRole.INVESTIGATOR)
    assert agent is not None
    assert type(agent).__name__ == "CodebaseInvestigatorAgent"


@pytest.mark.unit
def test_injection_seam_is_preserved() -> None:
    """The factory must not become the only way to build a roster — tests and
    callers still need to pass partial or mocked ones."""
    orch = AgentOrchestrator(agents={})
    assert orch.get_agent(AgentRole.DEVELOPER) is None
