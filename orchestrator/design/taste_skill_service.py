"""taste_skill_service — resolves design variant and builds the skill prefix for a task.

This is the single entry point engine.py calls for taste-skill integration.
It owns the variant-resolution logic (task field > prompt cue > DEFAULT) and
delegates prefix assembly to TasteSkillInjector.
"""

from __future__ import annotations

import logging

from orchestrator.models import DesignVariant, Task

from .frontend_detect import is_web_frontend_task
from .taste_skill_injector import DesignDials, TasteSkillInjector
from .taste_skill_loader import TasteSkillLoader, get_default_loader

logger = logging.getLogger(__name__)

# Phrases that signal the user wants a redesign audit
_REDESIGN_CUES = frozenset(
    {
        "redesign",
        "improve ui",
        "improve the ui",
        "fix design",
        "update layout",
        "update the layout",
        "fix the design",
        "audit the ui",
        "ui overhaul",
    }
)

# Phrases that signal the user wants an animation review
_ANIMATION_REVIEW_CUES = frozenset(
    {
        "review animations",
        "audit animations",
        "audit the animations",
        "audit motion",
        "check animations",
        "check the animations",
        "fix animations",
        "fix the animations",
        "animation quality",
        "animation review",
    }
)


def _infer_variant_from_prompt(prompt: str) -> DesignVariant | None:
    """Return REDESIGN or ANIMATION_REVIEW if the prompt contains cues, else None."""
    lower = prompt.lower()
    if any(cue in lower for cue in _REDESIGN_CUES):
        return DesignVariant.REDESIGN
    if any(cue in lower for cue in _ANIMATION_REVIEW_CUES):
        return DesignVariant.ANIMATION_REVIEW
    return None


class TasteSkillService:
    """Resolves the appropriate DesignVariant for a task and builds its skill prefix."""

    def __init__(
        self,
        injector: TasteSkillInjector | None = None,
        flags: object | None = None,
        settings: object | None = None,
    ) -> None:
        loader: TasteSkillLoader = get_default_loader()
        self._injector = injector or TasteSkillInjector(loader)
        self._flags = flags
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve_variant(self, task: Task) -> DesignVariant:
        """Determine the design variant for *task*.

        Priority order:
        1. ``task.design_variant`` if explicitly set
        2. Prompt cue detection (redesign, animation review)
        3. Auto-detection of animation-heavy tasks via ``get_animation_weight()``
        4. DEFAULT
        """
        if task.design_variant is not None:
            return task.design_variant

        inferred = _infer_variant_from_prompt(task.prompt)
        if inferred is not None:
            return inferred

        # Auto-detect animation-heavy tasks
        try:
            from .frontend_detect import get_animation_weight as _anim_weight

            _weight = _anim_weight(
                prompt=task.prompt,
                target_path=getattr(task, "target_path", ""),
            )
            if _weight > 0.7:
                logger.debug(
                    "taste_skill: auto-selected ANIMATION_REVIEW for %s (weight=%.2f)",
                    task.id,
                    _weight,
                )
                return DesignVariant.ANIMATION_REVIEW
        except Exception:
            pass

        return DesignVariant.DEFAULT

    def build_prefix(self, task: Task) -> str:
        """Return the skill-prefix string for *task*, or "" when not applicable.

        Returns "" when:
        - ``flags.taste_skill_enabled`` is False
        - the task is not a web-frontend task
        - the default SKILL.md is not bundled
        """
        if self._flags is not None and not getattr(self._flags, "taste_skill_enabled", True):
            return ""

        if not is_web_frontend_task(task.prompt, getattr(task, "target_path", "")):
            return ""

        variant = self.resolve_variant(task)
        dials = self._build_dials()
        prefix = self._injector.get_prefix(variant, dials)

        # ── Auto-inject animation skills based on motion intensity ──
        if dials.motion_intensity >= 4:
            vocab = self._injector.get_skill_content("animation_vocabulary")
            if vocab:
                prefix = f"{prefix}\n\n<animation_vocabulary>\n{vocab}\n</animation_vocabulary>"

        if dials.motion_intensity >= 7:
            apple = self._injector.get_skill_content("apple_design")
            if apple:
                prefix = f"{prefix}\n\n<apple_design>\n{apple}\n</apple_design>"
        # ────────────────────────────────────────────────────────────────

        if prefix:
            logger.debug(
                "taste_skill: injecting variant=%s dials=%s for task %s " "(motion_intensity=%d)",
                variant.value,
                dials,
                task.id,
                dials.motion_intensity,
            )

        return prefix

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_dials(self) -> DesignDials:
        if self._settings is None:
            return DesignDials()
        return DesignDials(
            design_variance=getattr(self._settings, "design_variance", 5),
            motion_intensity=getattr(self._settings, "motion_intensity", 5),
            visual_density=getattr(self._settings, "visual_density", 5),
        )
