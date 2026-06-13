"""
Pre-emit Self-Critique — 6-axis scoring prompt + stamp parser.
=============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Injects a self-critique instruction into the generation prompt and parses
the resulting stamp from the output.

Source: Hallmark design skill (references/slop-test.md § Pre-emit self-critique)
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

logger = logging.getLogger(__name__)


class SelfCritiqueScores(TypedDict):
    P: int
    H: int
    E: int
    S: int
    R: int
    V: int


SELF_CRITIQUE_PROMPT: str = (
    "\n## Pre-emit Self-Critique (mandatory)\n"
    "Before handing back any output, score it 1–5 on each axis.\n"
    "Anything < 3 on any axis triggers a revision pass.\n\n"
    "| Axis | Question |\n"
    "|---|---|\n"
    "| Philosophy | Is there a clear why — a position the page is taking? |\n"
    "| Hierarchy | Can a reader tell in 2 seconds what's primary/secondary/tertiary? |\n"
    "| Execution | Are details (rule weight, accent footprint, text-wrap, contrast) in spec? |\n"
    "| Specificity | Does this look like THIS brief — or a generic page? |\n"
    "| Restraint | Have you removed everything that isn't earning its place? |\n"
    "| Variety | Does this share a structural fingerprint with a previous output? |\n\n"
    "Record in output stamp:\n"
    "/* Hallmark · pre-emit critique: P{score} H{score} E{score} S{score} R{score} V{score} */\n"
    "/* Hallmark · macrostructure: {name} · theme: {name} · nav: {N#} · footer: {Ft#} */"
)

# Regex to parse the pre-emit stamp
_PRE_EMIT_RE = re.compile(
    r"pre-emit\s*critique\s*:\s*P(\d)\s+H(\d)\s+E(\d)\s+S(\d)\s+R(\d)\s+V(\d)",
    re.IGNORECASE,
)

# Regex to parse the macrostructure stamp
_MACRO_STAMP_RE = re.compile(
    r"macrostructure\s*:\s*(\w[\w\s-]*\w)",
    re.IGNORECASE,
)

# Regex to parse the theme stamp
_THEME_STAMP_RE = re.compile(
    r"theme\s*:\s*(\w[\w\s-]*\w)",
    re.IGNORECASE,
)

# Regex to parse nav stamp
_NAV_STAMP_RE = re.compile(
    r"nav\s*:\s*(N\w*)",
    re.IGNORECASE,
)

# Regex to parse footer stamp
_FOOTER_STAMP_RE = re.compile(
    r"footer\s*:\s*(Ft\w*)",
    re.IGNORECASE,
)


def inject_self_critique(prefix: str) -> str:
    """Append the pre-emit self-critique prompt to a skill prefix.

    Args:
        prefix: Existing skill prefix string.

    Returns:
        Prefix with self-critique block appended.
    """
    return prefix + SELF_CRITIQUE_PROMPT


class SelfCritiqueParser:
    """Parses pre-emit self-critique stamps from generated output."""

    @staticmethod
    def parse_scores(output: str) -> SelfCritiqueScores | None:
        """Parse the P-H-E-S-R-V scores from output.

        Returns None if no stamp found.
        """
        match = _PRE_EMIT_RE.search(output)
        if not match:
            return None
        return {
            "P": int(match.group(1)),
            "H": int(match.group(2)),
            "E": int(match.group(3)),
            "S": int(match.group(4)),
            "R": int(match.group(5)),
            "V": int(match.group(6)),
        }

    @staticmethod
    def parse_macrostructure(output: str) -> str | None:
        """Parse macrostructure name from stamp."""
        match = _MACRO_STAMP_RE.search(output)
        return match.group(1).strip() if match else None

    @staticmethod
    def parse_theme(output: str) -> str | None:
        """Parse theme name from stamp."""
        match = _THEME_STAMP_RE.search(output)
        return match.group(1).strip() if match else None

    @staticmethod
    def parse_nav(output: str) -> str | None:
        """Parse nav archetype code from stamp."""
        match = _NAV_STAMP_RE.search(output)
        return match.group(1).strip() if match else None

    @staticmethod
    def parse_footer(output: str) -> str | None:
        """Parse footer archetype code from stamp."""
        match = _FOOTER_STAMP_RE.search(output)
        return match.group(1).strip() if match else None

    @classmethod
    def min_score(cls, output: str) -> float:
        """Return the minimum axis score as a float 0.0–1.0.

        Returns 1.0 if no stamp found (neutral — doesn't penalize).
        """
        scores = cls.parse_scores(output)
        if scores is None:
            return 1.0
        return min(scores.values()) / 5.0
