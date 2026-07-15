"""taste_skill_loader — loads and caches bundled SKILL.md files from taste-skill.

Skill files live at orchestrator/design/skills/*.SKILL.md.
The loader reads them lazily and caches results in memory so repeated calls
within a process are free. A missing or unreadable file returns "" and never
raises — callers should treat an empty prefix as "feature unavailable".
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

SKILL_DIR = Path(__file__).parent / "skills"

# Maps variant key → filename inside SKILL_DIR
SKILL_FILES: dict[str, str] = {
    "default": "default.SKILL.md",
    "soft": "soft.SKILL.md",
    "minimalist": "minimalist.SKILL.md",
    "brutalist": "brutalist.SKILL.md",
    "redesign": "redesign.SKILL.md",
    "imagegen_web": "imagegen_web.SKILL.md",
    "image_to_code": "image_to_code.SKILL.md",
    # ── Emil Kowalski animation skills (skills-main) ──────────────
    "animation_vocabulary": "animation_vocabulary.SKILL.md",
    "apple_design": "apple_design.SKILL.md",
    "apple_springs": "apple_springs.SKILL.md",
    "review_animations": "review_animations.SKILL.md",
    "animation_standards": "animation_standards.SKILL.md",
    # ───────────────────────────────────────────────────────────────
}

# Trim large skill files to keep prompt overhead manageable.
# The default taste-skill SKILL.md is ~85KB; 6000 chars captures all bans and
# the brief-inference section without overwhelming the context window.
_MAX_SKILL_CHARS = 6_000


class TasteSkillLoader:
    """Loads and in-memory caches vendored SKILL.md content by variant key."""

    def __init__(self, skill_dir: Path = SKILL_DIR, max_chars: int = _MAX_SKILL_CHARS) -> None:
        self._skill_dir = skill_dir
        self._max_chars = max_chars
        self._cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, variant_key: str) -> str:
        """Return the SKILL.md content for *variant_key*, trimmed to max_chars.

        Returns "" if the key is unknown or the file cannot be read.
        Never raises.
        """
        if variant_key in self._cache:
            return self._cache[variant_key]

        content = self._read(variant_key)
        self._cache[variant_key] = content
        return content

    def available(self) -> list[str]:
        """Return variant keys whose .SKILL.md file is present on disk."""
        return [key for key in SKILL_FILES if self.is_bundled(key)]

    def is_bundled(self, variant_key: str) -> bool:
        """Return True if the skill file for *variant_key* exists on disk."""
        filename = SKILL_FILES.get(variant_key)
        if not filename:
            return False
        return (self._skill_dir / filename).is_file()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _read(self, variant_key: str) -> str:
        filename = SKILL_FILES.get(variant_key)
        if not filename:
            logger.debug("taste_skill_loader: unknown variant key %r", variant_key)
            return ""

        path = self._skill_dir / filename
        if not path.is_file():
            logger.warning("taste_skill_loader: skill file missing: %s", path)
            return ""

        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            logger.warning("taste_skill_loader: could not read %s: %s", path, exc)
            return ""

        if len(content) > self._max_chars:
            content = content[: self._max_chars] + "\n\n[...truncated for token efficiency]"

        return content


@lru_cache(maxsize=1)
def get_default_loader() -> TasteSkillLoader:
    """Return a process-level singleton TasteSkillLoader."""
    return TasteSkillLoader()
