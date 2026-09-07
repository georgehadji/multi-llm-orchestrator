"""
Agents package — Multi-agent architecture for cooperative development.
"""

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from .coordinator import AgentOrchestrator
from .developer import ArchitectAgent, DeveloperAgent, TesterAgent
from .devops import DevOpsAgent
from .investigator import CodebaseInvestigatorAgent
from .pool import AgentPool, TaskChannel
from .product_manager import ProductManagerAgent
from .qc import QCAgent
from .registry import AGENT_TYPES, build_default_agents
from .researcher import ResearcherAgent
from .reviewer import ReviewerAgent
from .user import UserAgent

__all__ = [
    # Core
    "AgentBase",
    "AgentRole",
    "AgentTask",
    "AgentTaskResult",
    "AgentOrchestrator",
    # Roster
    "AGENT_TYPES",
    "build_default_agents",
    # Implementations — one per AgentRole
    "ArchitectAgent",
    "DeveloperAgent",
    "TesterAgent",
    "ReviewerAgent",
    "DevOpsAgent",
    "ResearcherAgent",
    "UserAgent",
    "ProductManagerAgent",
    "QCAgent",
    "CodebaseInvestigatorAgent",
    # Meta-controller
    "AgentPool",
    "TaskChannel",
]
