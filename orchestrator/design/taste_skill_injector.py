"""taste_skill_injector — builds the skill-prefix string injected into LLM system prompts.

The prefix is structured as:

    DESIGN_VARIANCE=N / MOTION_INTENSITY=N / VISUAL_DENSITY=N

    <default skill content — anti-slop ban list>

    [<variant skill content> when variant != DEFAULT]

The caller wraps this in a ``<skill>…</skill>`` block via the existing
``PipelineContext.skill_prefix`` seam in ``engine_core/stages/generate.py``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from orchestrator.models import DesignVariant

from .taste_skill_loader import TasteSkillLoader, get_default_loader

logger = logging.getLogger(__name__)

# Map DesignVariant enum values → loader keys
_VARIANT_TO_KEY: dict[DesignVariant, str] = {
    DesignVariant.DEFAULT: "default",
    DesignVariant.SOFT: "soft",
    DesignVariant.MINIMALIST: "minimalist",
    DesignVariant.BRUTALIST: "brutalist",
    DesignVariant.REDESIGN: "redesign",
}

_DIAL_CLAMP_MIN = 1
_DIAL_CLAMP_MAX = 10


def _clamp(value: int) -> int:
    return max(_DIAL_CLAMP_MIN, min(_DIAL_CLAMP_MAX, value))


@dataclass(frozen=True)
class DesignDials:
    """Three tunable dials that control aesthetic intensity.

    Each dial is an integer in [1, 10]. Values outside this range are
    silently clamped on construction.
    """

    design_variance: int = 5  # Layout experimentation (1=centered, 10=asymmetric)
    motion_intensity: int = 5  # Animation depth (1=hover-only, 10=scroll/magnetic)
    visual_density: int = 5  # Info per viewport (1=spacious, 10=dense)

    def __post_init__(self) -> None:
        object.__setattr__(self, "design_variance", _clamp(self.design_variance))
        object.__setattr__(self, "motion_intensity", _clamp(self.motion_intensity))
        object.__setattr__(self, "visual_density", _clamp(self.visual_density))


class TasteSkillInjector:
    """Assembles the skill-prefix string for a given variant and dial settings."""

    def __init__(self, loader: TasteSkillLoader | None = None) -> None:
        self._loader = loader or get_default_loader()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_prefix(self, variant: DesignVariant, dials: DesignDials) -> str:
        """Return the full skill-prefix string for *variant* and *dials*.

        Always includes the default anti-slop skill.  When *variant* is not
        DEFAULT, the variant-specific skill is appended after it.
        Returns "" if the default skill file is not bundled.
        """
        default_content = self._loader.load("default")
        if not default_content:
            logger.warning("taste_skill_injector: default skill not bundled; skipping prefix")
            return ""

        parts: list[str] = [
            self._dials_block(dials),
            default_content,
        ]

        if variant != DesignVariant.DEFAULT:
            variant_key = _VARIANT_TO_KEY.get(variant, "default")
            variant_content = self._loader.load(variant_key)
            if variant_content:
                parts.append(f"## Style variant: {variant.value}\n\n{variant_content}")
            else:
                logger.debug(
                    "taste_skill_injector: variant skill %r not bundled; using default only",
                    variant.value,
                )

        return "\n\n".join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _dials_block(self, dials: DesignDials) -> str:
        return (
            f"DESIGN_VARIANCE={dials.design_variance} / "
            f"MOTION_INTENSITY={dials.motion_intensity} / "
            f"VISUAL_DENSITY={dials.visual_density}"
        )
