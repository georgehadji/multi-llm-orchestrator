"""
AutonomyConfig — Autonomy level configuration and presets
=========================================================
Defines the three autonomy levels (MANUAL/SUPERVISED/FULL) and their
associated thresholds. Pure sync module — no I/O, no asyncio.

Pattern: Pure Dataclass + Strategy presets
Async: No — pure computation
Layer: L2 Verification
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("orchestrator.autonomy")


class AutonomyLevel(Enum):
    """Three autonomy levels ordered from most restrictive to most permissive."""

    MANUAL = "manual"
    SUPERVISED = "supervised"
    FULL = "full"


@dataclass
class AutonomyConfig:
    """Configuration for autonomy level with associated thresholds."""

    level: AutonomyLevel
    require_approval_above_usd: float
    require_approval_for_external_calls: bool
    max_retries_without_approval: int


AUTONOMY_PRESETS: dict[str, AutonomyConfig] = {
    "manual": AutonomyConfig(
        level=AutonomyLevel.MANUAL,
        require_approval_above_usd=0.0,
        require_approval_for_external_calls=True,
        max_retries_without_approval=0,
    ),
    "supervised": AutonomyConfig(
        level=AutonomyLevel.SUPERVISED,
        require_approval_above_usd=1.0,
        require_approval_for_external_calls=False,
        max_retries_without_approval=3,
    ),
    "full": AutonomyConfig(
        level=AutonomyLevel.FULL,
        require_approval_above_usd=999.0,
        require_approval_for_external_calls=False,
        max_retries_without_approval=10,
    ),
}


def get_autonomy_config(preset: str) -> AutonomyConfig:
    """Return the AutonomyConfig for the named preset.

    Args:
        preset: One of "manual", "supervised", or "full".

    Returns:
        The corresponding AutonomyConfig.

    Raises:
        ValueError: If the preset name is not recognized.
    """
    if preset not in AUTONOMY_PRESETS:
        raise ValueError(f"Unknown autonomy preset: {preset!r}. Valid presets: {list(AUTONOMY_PRESETS)}")
    return AUTONOMY_PRESETS[preset]


def requires_approval(
    config: AutonomyConfig,
    estimated_cost_usd: float,
    is_external_call: bool = False,
) -> bool:
    """Determine whether an action requires user approval under the given config.

    Args:
        config: The active AutonomyConfig.
        estimated_cost_usd: Projected cost of the action in USD.
        is_external_call: Whether the action involves an external API call.

    Returns:
        True if the action must be approved by the user before proceeding.
    """
    if config.level == AutonomyLevel.MANUAL:
        return True
    if config.level == AutonomyLevel.FULL:
        # Full autonomy: only block if cost exceeds the (very high) threshold
        if estimated_cost_usd > config.require_approval_above_usd:
            return True
        return is_external_call and config.require_approval_for_external_calls
    # SUPERVISED: threshold-based + optional external call gate
    if estimated_cost_usd > config.require_approval_above_usd:
        return True
    return is_external_call and config.require_approval_for_external_calls
