"""design_validators — anti-slop pattern detection for AI-generated frontend code.

All validators in this module are SOFT/WARN-only and must never be added to
``Task.hard_validators``.  They surface findings as log warnings so developers
can see which generic patterns slipped through, without blocking generation.

Usage (engine.py):
    result = validate_anti_slop(generated_output)
    if not result.passed:
        logger.warning("taste-skill anti_slop: %s", result.details)

Registration:
    The ``anti_slop`` key is registered in ``quality/validators.py::VALIDATORS``
    so callers can reference it by name, but the engine only runs it as a
    soft check — not via ``run_validators`` / ``hard_validators``.
"""

from __future__ import annotations

import re

from orchestrator.quality.validators import ValidationResult

# ---------------------------------------------------------------------------
# Compiled patterns — each targets one of taste-skill's anti-slop rules
# ---------------------------------------------------------------------------

# Banned fonts (taste-skill default skill Section 1)
_BANNED_FONTS = re.compile(
    r"""(?:font-family\s*:\s*[^;]*\b(?:Inter|Roboto|Arial|Helvetica)\b"""
    r"""|['"](Inter|Roboto|Arial|Helvetica)['"]\s*,)""",
    re.IGNORECASE,
)

# AI / generic purple-blue gradient (most common AI design fingerprint)
_AI_GRADIENT = re.compile(
    r"linear-gradient\s*\([^)]*(?:purple|violet|#[89a-fA-F][0-9a-fA-F]{5}|#[6-9][0-9a-fA-F]{5})"
    r"[^)]*(?:blue|#[0-4][0-9a-fA-F]{5})",
    re.IGNORECASE,
)

# Pure #000000 / #000 background (not "#09090b" or similar dark shades)
_PURE_BLACK_BG = re.compile(
    r"background(?:-color)?\s*:\s*#(?:000000|000)\b",
    re.IGNORECASE,
)

# Pure #ffffff / #fff background
_PURE_WHITE_BG = re.compile(
    r"background(?:-color)?\s*:\s*#(?:ffffff|fff)\b",
    re.IGNORECASE,
)

# Uniform 3-column grid (the most generic AI layout)
_THREE_COLS = re.compile(
    r"repeat\s*\(\s*3\s*,\s*1fr\s*\)|grid-cols-3\b",
    re.IGNORECASE,
)

# Generic low-quality box-shadow (dark rgba at 0.1-0.2 opacity, 1-4px)
_GENERIC_SHADOW = re.compile(
    r"box-shadow\s*:\s*0\s+[1-4]px\s+[1-8]px\s+(?:\d+px\s+)?rgba\s*\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\.[12]\d*\s*\)",
    re.IGNORECASE,
)

# Em-dash in copy (banned by taste-skill — use proper punctuation)
_EM_DASH = re.compile(r"—")

# ---------------------------------------------------------------------------
# Public validator
# ---------------------------------------------------------------------------

_CHECKS: list[tuple[re.Pattern[str], str]] = [
    (_BANNED_FONTS, "Banned font detected (Inter/Roboto/Arial/Helvetica)"),
    (_AI_GRADIENT, "AI purple-to-blue gradient detected"),
    (_PURE_BLACK_BG, "Pure #000/#000000 background detected"),
    (_PURE_WHITE_BG, "Pure #fff/#ffffff background detected"),
    (_THREE_COLS, "Uniform 3-column grid detected (most generic AI layout)"),
    (_GENERIC_SHADOW, "Generic low-quality box-shadow detected"),
    (_EM_DASH, "Em-dash (—) in copy detected"),
]


def validate_anti_slop(output: str) -> ValidationResult:
    """Scan *output* for taste-skill anti-slop banned patterns.

    Returns a ValidationResult with ``passed=True`` when no findings,
    ``passed=False`` when one or more patterns matched.
    ``validator_name`` is always ``"anti_slop"``.

    This validator is SOFT/WARN-only — it must never be placed in
    ``Task.hard_validators``.
    """
    findings: list[str] = []

    for pattern, message in _CHECKS:
        if pattern.search(output):
            findings.append(message)

    if findings:
        details = f"Anti-slop findings ({len(findings)} total):\n"
        details += "\n".join(f"  - {f}" for f in findings)
        return ValidationResult(passed=False, details=details, validator_name="anti_slop")

    return ValidationResult(
        passed=True, details="No anti-slop patterns detected", validator_name="anti_slop"
    )
