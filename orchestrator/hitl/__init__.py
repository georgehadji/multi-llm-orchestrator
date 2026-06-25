"""HITL — Human-in-the-loop approval gates."""

from .channel import AutoApproveChannel, CLIDecisionChannel, DecisionChannel, FailClosedChannel
from .gate import Decision, DecisionResult, HumanInTheLoop

__all__ = [
    "Decision",
    "DecisionResult",
    "HumanInTheLoop",
    "DecisionChannel",
    "FailClosedChannel",
    "AutoApproveChannel",
    "CLIDecisionChannel",
]
