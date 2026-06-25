"""
phase_policy — single source of truth for per-phase reasoning/thinking/temperature.

Pure domain data (no I/O, no behaviour beyond lookups) per the Dependency Rule.
See docs/REASONING_AND_TEMPERATURE.md for the rationale behind every value.

Why this exists
---------------
Temperature literals and ad-hoc "is this a reasoning step?" checks were scattered
across services (52× temperature=0.3, 44× 0.2, …). That drift means a tuning
change has to be hunted across dozens of call sites and inevitably misses some.
This module centralises the policy so each pipeline phase reads ONE table.

Usage
-----
    from orchestrator.domain.phase_policy import Phase, temperature_for, policy_for

    temp = temperature_for(Phase.EVALUATE)            # 0.1
    temp = temperature_for(Phase.GENERATE, TaskType.WRITING)  # 0.8 (creative override)
    pol  = policy_for(Phase.DECOMPOSE)                 # full PhasePolicy
    if pol.use_thinking:
        ...  # request reasoning effort = pol.reasoning_effort.value
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..models import TaskType


class Phase(str, Enum):
    """Pipeline phases that carry distinct reasoning/temperature needs.

    Aligned to the engine control loop: decompose → generate → critique →
    revise → evaluate, plus the simple non-reasoning phases (extract/summarize/
    creative) and the diversity-sampling phase (Verbalized Sampling).
    """

    DECOMPOSE = "decompose"   # plan a project into a task graph
    GENERATE = "generate"     # produce code (default generation)
    CRITIQUE = "critique"     # review output for flaws
    REVISE = "revise"         # apply known fixes
    EVALUATE = "evaluate"     # adversarial scoring
    EXTRACT = "extract"       # structured data extraction
    SUMMARIZE = "summarize"   # faithful summarisation
    CREATIVE = "creative"     # creative writing
    SAMPLING = "sampling"     # diversity / Verbalized Sampling


class ReasoningEffort(str, Enum):
    """OpenRouter ``reasoning.effort`` levels (NONE = thinking disabled)."""

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class PhasePolicy:
    """Immutable per-phase policy.

    Attributes:
        temperature:      Sampling temperature for NON-reasoning models. Reasoning
                          models (o-series, GPT-5 Pro, …) forbid/ignore temperature;
                          the client omits it for them regardless of this value.
        use_thinking:     Whether to request the model's reasoning/thinking path.
        reasoning_effort: Effort level when use_thinking is True (NONE otherwise).
        prefer_reasoning: Whether routing should bias toward a reasoning model.
        exclude_reasoning: Drop the chain-of-thought from the response (keeps
                           structured output clean; you still pay to generate it).
    """

    temperature: float
    use_thinking: bool
    reasoning_effort: ReasoningEffort
    prefer_reasoning: bool
    exclude_reasoning: bool = False


# ── The policy table ──────────────────────────────────────────────────────────
# Values justified in docs/REASONING_AND_TEMPERATURE.md. Touch the doc + this
# table together.

_PHASE_POLICY: dict[Phase, PhasePolicy] = {
    # Verification-heavy phases: reasoning + high effort, near-deterministic temp.
    Phase.DECOMPOSE: PhasePolicy(0.2, True, ReasoningEffort.HIGH, True),
    Phase.CRITIQUE: PhasePolicy(0.1, True, ReasoningEffort.HIGH, True),
    Phase.EVALUATE: PhasePolicy(0.1, True, ReasoningEffort.HIGH, True, exclude_reasoning=True),
    # Code generation: low temp; optional reasoning only for hard logic.
    Phase.GENERATE: PhasePolicy(0.2, False, ReasoningEffort.MEDIUM, False),
    Phase.REVISE: PhasePolicy(0.2, False, ReasoningEffort.LOW, False),
    # Simple phases: no reasoning, deterministic-to-fluent temps.
    Phase.EXTRACT: PhasePolicy(0.0, False, ReasoningEffort.NONE, False),
    Phase.SUMMARIZE: PhasePolicy(0.3, False, ReasoningEffort.NONE, False),
    Phase.CREATIVE: PhasePolicy(0.8, False, ReasoningEffort.NONE, False),
    # Diversity sampling: high temp for spread.
    Phase.SAMPLING: PhasePolicy(0.9, False, ReasoningEffort.NONE, False),
}

# Per-TaskType temperature overrides. When a GENERATE-phase call is actually a
# creative or extraction task, the task's nature wins over the phase default.
# Built by value to stay robust across enum-name changes between versions.
def _task_temp_table() -> dict[TaskType, float]:
    table: dict[TaskType, float] = {}
    # Map by value to stay robust across enum-name changes.
    by_value = {t.value: t for t in TaskType}
    overrides = {
        "creative_writing": 0.8,
        "data_extraction": 0.0,
        "summarization": 0.3,
        "code_generation": 0.2,
        "code_review": 0.1,
        "complex_reasoning": 0.3,
        "evaluation": 0.1,
    }
    for value, temp in overrides.items():
        if value in by_value:
            table[by_value[value]] = temp
    return table


_TASK_TEMPERATURE = _task_temp_table()


# ── Public helpers ────────────────────────────────────────────────────────────

def policy_for(phase: Phase) -> PhasePolicy:
    """Return the immutable policy for *phase* (GENERATE default if unknown)."""
    return _PHASE_POLICY.get(phase, _PHASE_POLICY[Phase.GENERATE])


def temperature_for(phase: Phase, task_type: TaskType | None = None) -> float:
    """Optimal temperature for *phase*, with an optional TaskType override.

    The task-type override exists because the GENERATE phase covers code,
    creative writing, and extraction — which want very different temperatures.
    """
    if task_type is not None and task_type in _TASK_TEMPERATURE:
        return _TASK_TEMPERATURE[task_type]
    return policy_for(phase).temperature


def use_thinking_for(phase: Phase) -> bool:
    """Whether the reasoning/thinking path should be requested for *phase*."""
    return policy_for(phase).use_thinking


def reasoning_effort_for(phase: Phase) -> ReasoningEffort:
    """The reasoning effort level for *phase* (NONE when thinking is off)."""
    return policy_for(phase).reasoning_effort


def prefer_reasoning_for(phase: Phase) -> bool:
    """Whether routing should bias toward a reasoning model for *phase*."""
    return policy_for(phase).prefer_reasoning
