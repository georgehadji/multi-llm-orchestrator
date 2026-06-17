"""
Atelier Slop Critique — 6-axis anti-slop scoring for generated UIs.
===================================================================
Implements the pre-emit self-critique system from Hallmark's slop-test.md.

Evaluates generated HTML/CSS across 6 axes:
  Structure (30%) — page shape, rhythm, section variety
  Typography (20%) — font pairing, scale, hierarchy
  Color (20%) — palette cohesion, OKLCH usage, gradient discipline
  Imagery (15%) — image selection, illustration style, icon consistency
  Motion (10%) — transition quality, easing, prefers-reduced-motion
  Polish (5%) — interactive states, focus rings, print stylesheet

Each axis is scored 1-5. Total < 3.0 = slop detected.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re


@dataclass
class AxisScore:
    """Score for a single critique axis."""

    name: str
    score: float  # 1.0-5.0
    max_score: float = 5.0
    findings: list[str] = field(default_factory=list)
    passed: bool = True


@dataclass
class AtelierCritiqueResult:
    """Complete slop critique result."""

    axes: list[AxisScore] = field(default_factory=list)
    total_score: float = 0.0
    passed: bool = True
    slop_detected: bool = False
    details: str = ""
    recommendations: list[str] = field(default_factory=list)


# ── Anti-Slop Gate Definitions ─────────────────────────────────────────────

# Structure gates: detect generic layout patterns
_STRUCTURE_GATES = {
    "centered_hero": (
        r'(?:\.hero|#hero).*text-align\s*:\s*center',
        "Centered hero with no visual anchor",
    ),
    "three_even_columns": (
        r'grid-template-columns\s*:\s*(?:repeat\s*\(\s*3\s*,\s*1fr|1fr\s+1fr\s+1fr)',
        "Feature section as 3 even columns",
    ),
    "single_cta_hero": (
        r'(?:<button|<a\s+[^>]*class="[^"]*btn[^"]*")\s*[^>]*>(?:Get\s+Started|Sign\s+Up|Learn\s+More)',
        "Hero CTA uses generic 'Get Started' or 'Sign Up'",
    ),
    "no_section_rhythm": (
        r'(?!.*(?:<section|<div\s+class="[^"]*"))',
        "No semantic sections or div containers — page has no structure",
    ),
    "all_same_height_sections": (
        r'min-height\s*:\s*100vh',
        "All sections are full-viewport — no visual breathing room",
    ),
}

# Typography gates
_TYPOGRAPHY_GATES = {
    "inter_only": (
        r"Inter(?:\s*,?\s*sans-serif)?(?!.*Fraunces|.*Playfair|.*Newsreader|.*Geist\s+Mono|.*Space\s+Grotesk)",
        "Inter is the only font family — no typographic contrast",
    ),
    "font_size_binary": (
        r'(?:font-size\s*:\s*(?:14|16)px|text-(?:sm|base|lg))\s*[;}]',
        "Font sizes are clustered at 14px/16px with nothing between",
    ),
    "heading_weight_monoculture": (
        r'font-weight\s*:\s*700[^;]*;\s*\n\s*(?:h2|h3)',
        "Likely heading weight monoculture — all headings appear to use 700",
    ),
    "no_display_text": (
        r'(?!.*(?:font-size\s*:\s*(?:48|56|64|72)px|text-[4-8]xl))',
        "No display-scale typography (48px+ headings) — page lacks visual anchor",
    ),
}

# Color gates
_COLOR_GATES = {
    "purple_cyan_gradient": (
        r'(?:#[7-9a-f][0-9a-f]{2}(?:f[5-9a-f]|e[0-9a-f])|#[a-f0-9]*[pP][uU])',
        "Purple-to-cyan gradient detected",
    ),
    "no_oklch_colors": (
        r'(?!.*oklch\()',
        "No OKLCH colors used — fallback to hex/rgb",
    ),
    "flat_shadows": (
        r'box-shadow\s*:\s*(?:0\s+\d+px\s+\d+px\s+rgba\(0,\s*0,\s*0)',
        "Shadows use flat black rgba(0,0,0,...) instead of OKLCH-derived",
    ),
}

# Imagery gates
_IMAGERY_GATES = {
    "undersea_cable": (
        r'(?:undersea|cable|circuit|network)\s*(?:illustration|image|graphic)',
        "Undersea-cable illustration trope",
    ),
    "gray_placeholder": (
        r'(?:placeholder|grey\s+box).*landscape',
        "Placeholder grey box with landscape icon",
    ),
    "unsplash_people": (
        r'unsplash.*(?:people|team|office)',
        "Generic Unsplash stock photos of people in offices",
    ),
}

# Motion gates
_MOTION_GATES = {
    "no_reduced_motion": (
        r'(?!.*prefers-reduced-motion)',
        "No prefers-reduced-motion media query",
    ),
    "no_transition_duration": (
        r'(?!.*transition-duration)',
        "No transition-duration on interactive elements",
    ),
    "linear_animations": (
        r'(?:animation|transition).*linear',
        "Animations use 'linear' easing — no character",
    ),
}

# Polish gates
_POLISH_GATES = {
    "no_focus_styles": (
        r'(?!.*:focus-visible)',
        "No :focus-visible styles on interactive elements",
    ),
    "no_print_stylesheet": (
        r'(?!.*@media\s+print)',
        "No @media print stylesheet",
    ),
    "no_loading_states": (
        r'(?!.*aria-busy|aria-live|loading\s*[=:])',
        "No loading states or progress indicators",
    ),
}

# All gates mapped to axes with weights
_AXIS_GATES = {
    "Structure": (_STRUCTURE_GATES, 0.30),
    "Typography": (_TYPOGRAPHY_GATES, 0.20),
    "Color": (_COLOR_GATES, 0.20),
    "Imagery": (_IMAGERY_GATES, 0.15),
    "Motion": (_MOTION_GATES, 0.10),
    "Polish": (_POLISH_GATES, 0.05),
}


def critique_html(html_content: str) -> AtelierCritiqueResult:
    """Evaluate generated HTML/CSS against the Atelier slop gates.

    Returns a scored critique with per-axis breakdown.
    Score ≥ 3.0 = passes. Score < 3.0 = slop detected.
    """
    axes: list[AxisScore] = []
    total_weighted = 0.0
    total_weight = 0.0

    for axis_name, (gates, weight) in _AXIS_GATES.items():
        total_gates = len(gates)
        failures = 0
        findings: list[str] = []

        for gate_id, (pattern, description) in gates.items():
            # Gates with (?!...) are negated — "absence of X is a failure"
            if pattern.startswith("(?!") and pattern.endswith(")"):
                # Check if the pattern is ABSENT (which is the failure case)
                inner = pattern[3:-1]  # strip the (?! and )
                if not re.search(inner, html_content, re.IGNORECASE | re.DOTALL):
                    failures += 1
                    findings.append(f"[{gate_id}] {description}")
            else:
                # Normal gate: matching is a failure
                if re.search(pattern, html_content, re.IGNORECASE | re.DOTALL):
                    failures += 1
                    findings.append(f"[{gate_id}] {description}")

        # Score: 5.0 = 0 failures, 1.0 = all gates fail
        pass_rate = (total_gates - failures) / total_gates if total_gates > 0 else 1.0
        score = 1.0 + (4.0 * pass_rate)

        axes.append(
            AxisScore(
                name=axis_name,
                score=round(score, 2),
                max_score=5.0,
                findings=findings,
                passed=score >= 3.0,
            )
        )
        total_weighted += score * weight
        total_weight += weight

    overall = round(total_weighted / total_weight, 2) if total_weight > 0 else 5.0
    slop = overall < 3.0
    total_findings = [f for a in axes for f in a.findings]

    return AtelierCritiqueResult(
        axes=axes,
        total_score=overall,
        passed=overall >= 3.0,
        slop_detected=slop,
        details=(
            f"Atelier score: {overall:.1f}/5.0 — {'PASS' if not slop else 'SLOP DETECTED'}"
        ),
        recommendations=[
            f"{axis.name}: {f}" for axis in axes for f in axis.findings[:2]
        ][:6],
    )


def critique_file(filepath: Path) -> AtelierCritiqueResult:
    """Evaluate a generated HTML file against the Atelier slop gates."""
    content = filepath.read_text(encoding="utf-8", errors="ignore")
    return critique_html(content)
