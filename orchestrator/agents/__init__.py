"""
Agents package — Multi-agent architecture for cooperative development.
"""

from .base import AgentBase, AgentRole, AgentTask, AgentTaskResult
from .coordinator import AgentOrchestrator
from .developer import DeveloperAgent, ArchitectAgent, TesterAgent

__all__ = [
    "AgentBase",
    "AgentRole",
    "AgentTask",
    "AgentTaskResult",
    "AgentOrchestrator",
    "DeveloperAgent",
    "ArchitectAgent",
    "TesterAgent",
]
