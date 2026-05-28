"""
SkillOpt Data Models
====================
Trajectory, SkillPatch, and SkillUpdateResult dataclasses for the
self-improving skill system.  Kept separate from models.py to preserve
the "models.py = pure data, no behaviour" invariant.

These are the in-memory representations; persistence is handled by
orchestrator.application.skill_store.SkillStore.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .models import TaskType


@dataclass
class Trajectory:
    """One task execution observation collected for SkillOpt training.

    Wraps the inputs and outputs of a single pipeline run so the optimizer
    can learn which kinds of outputs correlate with high scores.
    """

    task_id: str
    task_type: "TaskType"
    prompt: str          # task.prompt (truncated if very long)
    output: str          # best output from the generate/critique pipeline
    score: float         # CritiqueReport.score in [0.0, 1.0]
    critique_text: str   # CritiqueState.best_critique (free text)
    model_used: str      # Model.value of the worker model
    cost_usd: float      # total LLM cost for this task
    recorded_at: float   # time.time() at collection point


@dataclass
class SkillPatch:
    """Atomic edit to a skill document proposed by the optimizer model.

    ``token_cost`` is the optimizer's self-reported estimate of how many
    tokens the patch adds/changes.  The sum of token_cost across all
    patches in one epoch must not exceed the edit_budget (default 150).
    """

    op: Literal["append", "insert_after", "replace", "delete"]
    anchor: str    # heading or substring that locates the insertion point;
                   # empty string means "end of document" for append
    content: str   # new text (empty for delete)
    token_cost: int = 0  # estimated tokens; enforced by SkillOptimizer


@dataclass
class SkillUpdateResult:
    """Outcome of one SkillOptimizer epoch.

    ``accepted=True``  means the candidate skill cleared the validation gate
    and has been persisted as the new best skill.

    ``accepted=False`` means the patches were rejected (no improvement on
    held-out val split); the patches were stored in the negative-feedback
    buffer so the optimizer can avoid repeating them.
    """

    task_type: "TaskType"
    epoch: int
    accepted: bool
    score_before: float
    score_after: float
    patches_applied: list[SkillPatch] = field(default_factory=list)
    rejection_reason: str = ""  # non-empty only when accepted=False
