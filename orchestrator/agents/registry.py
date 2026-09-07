"""
Agent registry — the role → implementation map, and a default-roster factory.
=============================================================================

Why this exists
---------------
``AgentOrchestrator`` takes its agents by injection
(``__init__(self, agents: dict[AgentRole, AgentBase], ...)``) and never constructs
any, and ``_run_one`` degrades silently when a role is missing: it returns a failed
``AgentTaskResult`` rather than raising. That combination made the whole subsystem a
no-op in practice — nothing in the product ever built the dict, and
``agents/__init__.py`` exported implementations for only 3 of the 10 declared roles.

Most consequential: ``_decompose_goal`` dispatches work to
``AgentRole.INVESTIGATOR``, whose implementation was among the unexported seven —
so even a caller who wired the orchestrator correctly through the package's public
API got a silent "no agent for role" failure on every investigation task.

Pattern
-------
Registry + Abstract Factory. The registry is the single place that knows which
class serves which role, which is what the coordinator's role-based dispatch
already implies but had nowhere to look up. Keeping construction here rather than
inside ``AgentOrchestrator`` preserves its injection seam: tests and callers can
still pass a partial or wholly-mocked roster.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import AgentBase, AgentRole
from .developer import ArchitectAgent, DeveloperAgent, TesterAgent
from .devops import DevOpsAgent
from .investigator import CodebaseInvestigatorAgent
from .product_manager import ProductManagerAgent
from .qc import QCAgent
from .researcher import ResearcherAgent
from .reviewer import ReviewerAgent
from .user import UserAgent

if TYPE_CHECKING:
    from collections.abc import Iterable

# Every AgentRole declared in base.py maps to exactly one implementation.
# test_agent_registry.py asserts this stays total: a new role without an
# implementation is a silent no-op at dispatch time, which is the failure mode
# this registry exists to prevent.
AGENT_TYPES: dict[AgentRole, type[AgentBase]] = {
    AgentRole.ARCHITECT: ArchitectAgent,
    AgentRole.DEVELOPER: DeveloperAgent,
    AgentRole.REVIEWER: ReviewerAgent,
    AgentRole.TESTER: TesterAgent,
    AgentRole.DEVOPS: DevOpsAgent,
    AgentRole.RESEARCHER: ResearcherAgent,
    AgentRole.USER: UserAgent,
    AgentRole.PRODUCT_MANAGER: ProductManagerAgent,
    AgentRole.QA: QCAgent,
    AgentRole.INVESTIGATOR: CodebaseInvestigatorAgent,
}


def build_default_agents(
    roles: Iterable[AgentRole] | None = None,
    **agent_kwargs: Any,
) -> dict[AgentRole, AgentBase]:
    """Build a role → agent roster ready to hand to ``AgentOrchestrator``.

    Parameters
    ----------
    roles :
        Which roles to instantiate. Defaults to all of them — the roster the
        coordinator's ``_decompose_goal`` can actually dispatch to.
    **agent_kwargs :
        Forwarded verbatim to every agent constructor (``workspace``, ``client``,
        ``tools``, ``model_preferences``, ``event_bus``). Every implementation
        takes ``**kwargs`` and binds its own role, so one set of collaborators
        serves the whole roster.

    Returns
    -------
    A fresh dict. Callers remain free to add, replace or drop entries before
    constructing the orchestrator — the injection seam is preserved.
    """
    selected = list(roles) if roles is not None else list(AGENT_TYPES)
    return {role: AGENT_TYPES[role](**agent_kwargs) for role in selected}
