"""
AutonomyConfig — Multi-Mode Orchestrator Selector.
==================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of the Category 1 Enhancement Plan (Wave 2: X4 Multi-Mode Selector).
Defines autonomy levels that control the orchestrator's behavior:
max_iterations, repair attempts, verification depth, critique depth,
model tier, and checkpoint frequency.

Mapping to the 6-mode selector from Create.xyz / Bolt.new:
    Lite      → query only, no code generation
    Standard  → single critique pass, 1 repair attempt
    Auto      → full pipeline, 2 critiques, 2 repairs
    Max       → autonomous loop, 3 critiques, 5 repairs, reasoning tier
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AutonomyLevel(str, Enum):
    """Autonomy level for orchestrator execution."""

    LITE = "lite"  # Query only, no code generation
    STANDARD = "standard"  # Single critique pass, 1 repair
    AUTO = "auto"  # Full pipeline, 2 critiques, 2 repairs
    MAX = "max"  # Autonomous loop, 3 critiques, 5 repairs


@dataclass
class AutonomyConfig:
    """Configuration for autonomy-controlled execution.

    Each level tunes runtime limits, verification strategy,
    critique depth, model tier, and checkpoint frequency.
    """

    level: AutonomyLevel = AutonomyLevel.STANDARD

    # ── Iteration & Repair ──
    max_iterations: int = 3
    repair_attempts: int = 1
    max_runtime_minutes: int = 120

    # ── Verification ──
    verification_mode: str = "unit_tests"  # "none", "syntax", "unit_tests", "behavioral"
    acceptance_threshold: float = 0.85

    # ── Critique ──
    critique_passes: int = 1
    critique_model_tier: str = "balanced"  # "cheap", "balanced", "premium"

    # ── Model Tier ──
    generation_model_tier: str = "balanced"  # "cheap", "balanced", "reasoning"
    decomposition_model_tier: str = "balanced"

    # ── Checkpoint ──
    checkpoint_frequency: int = 1  # Checkpoint every N tasks

    # ── Quality ──
    strict_validation: bool = False
    require_documentation: bool = False

    @classmethod
    def for_level(cls, level: AutonomyLevel) -> AutonomyConfig:
        """Return a pre-configured AutonomyConfig for the given level."""
        configs = {
            AutonomyLevel.LITE: cls(
                level=AutonomyLevel.LITE,
                max_iterations=0,
                repair_attempts=0,
                max_runtime_minutes=5,
                verification_mode="none",
                critique_passes=0,
                generation_model_tier="cheap",
                checkpoint_frequency=0,
            ),
            AutonomyLevel.STANDARD: cls(
                level=AutonomyLevel.STANDARD,
                max_iterations=3,
                repair_attempts=1,
                max_runtime_minutes=120,
                verification_mode="unit_tests",
                critique_passes=1,
                generation_model_tier="balanced",
                decomposition_model_tier="balanced",
                checkpoint_frequency=5,
            ),
            AutonomyLevel.AUTO: cls(
                level=AutonomyLevel.AUTO,
                max_iterations=5,
                repair_attempts=2,
                max_runtime_minutes=240,
                verification_mode="unit_tests",
                critique_passes=2,
                generation_model_tier="balanced",
                checkpoint_frequency=1,
            ),
            AutonomyLevel.MAX: cls(
                level=AutonomyLevel.MAX,
                max_iterations=10,
                repair_attempts=5,
                max_runtime_minutes=480,
                verification_mode="behavioral",
                critique_passes=3,
                generation_model_tier="reasoning",
                decomposition_model_tier="premium",
                checkpoint_frequency=1,
                strict_validation=True,
                require_documentation=True,
            ),
        }
        return configs[level]

    @classmethod
    def from_agent_profile(cls, profile_name: str) -> AutonomyConfig:
        """Convert an agent profile name to an AutonomyConfig.

        Maps: standard→STANDARD, max→MAX, creative→AUTO, conservative→STANDARD
        """
        mapping = {
            "standard": AutonomyLevel.STANDARD,
            "max": AutonomyLevel.MAX,
            "creative": AutonomyLevel.AUTO,
            "conservative": AutonomyLevel.STANDARD,
            "research": AutonomyLevel.AUTO,
        }
        level = mapping.get(profile_name, AutonomyLevel.STANDARD)
        return cls.for_level(level)

    def apply_to_task(self, task: object) -> None:
        """Apply autonomy settings to a task's runtime limits.

        Sets max_iterations and acceptance_threshold on the task object
        if those attributes exist.
        """
        if hasattr(task, "max_iterations"):
            task.max_iterations = self.max_iterations
        if hasattr(task, "acceptance_threshold"):
            task.acceptance_threshold = self.acceptance_threshold

    def model_tier_for(self, purpose: str) -> str:
        """Return the model tier to use for a given purpose."""
        if purpose == "generation":
            return self.generation_model_tier
        if purpose == "critique":
            return self.critique_model_tier
        if purpose == "decomposition":
            return self.decomposition_model_tier
        return "balanced"

    @property
    def is_lite(self) -> bool:
        return self.level == AutonomyLevel.LITE

    @property
    def is_standard(self) -> bool:
        return self.level == AutonomyLevel.STANDARD

    @property
    def is_autonomous(self) -> bool:
        return self.level in (AutonomyLevel.AUTO, AutonomyLevel.MAX)

    @property
    def is_max(self) -> bool:
        return self.level == AutonomyLevel.MAX
