"""
Slop Test Engine — 61 deterministic anti-slop gates.
===================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Deterministic validation that catches AI-template patterns in generated
frontend code. Each gate is a regex or structural check with a severity:
- CRITICAL: must fix before shipping (fails hard validator)
- MAJOR: looks AI-generated (triggers revision pass)
- MINOR: small taste issue (logged warning)

Some gates have genre-scoped overrides (e.g. atmospheric allows radial
gradients that editorial forbids).

Source: Hallmark design skill (references/slop-test.md)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class GateSeverity(Enum):
    """Severity classification for slop-test gates."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


@dataclass(frozen=True)
class Gate:
    """A single slop-test gate."""

    number: int
    category: str
    name: str
    severity: GateSeverity
    check: Callable[[str, str | None], bool]
    genre_override: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class GateFinding:
    """A single finding from a gate."""

    gate: Gate
    override_applied: bool = False


@dataclass
class SlopTestResult:
    """Result of running the slop test."""

    findings: list[GateFinding] = field(default_factory=list)

    @property
    def critical_failures(self) -> list[GateFinding]:
        return [
            f
            for f in self.findings
            if f.gate.severity == GateSeverity.CRITICAL and not f.override_applied
        ]

    @property
    def major_failures(self) -> list[GateFinding]:
        return [
            f
            for f in self.findings
            if f.gate.severity == GateSeverity.MAJOR and not f.override_applied
        ]

    @property
    def minor_findings(self) -> list[GateFinding]:
        return [
            f
            for f in self.findings
            if f.gate.severity == GateSeverity.MINOR and not f.override_applied
        ]

    @property
    def passed(self) -> bool:
        return not self.critical_failures and not self.major_failures

    @property
    def summary(self) -> str:
        lines = [f"Slop test: {len(self.findings)} findings"]
        if self.critical_failures:
            lines.append(f"  CRITICAL ({len(self.critical_failures)}):")
            for f in self.critical_failures:
                lines.append(f"    Gate {f.gate.number}: {f.gate.name}")
        if self.major_failures:
            lines.append(f"  MAJOR ({len(self.major_failures)}):")
            for f in self.major_failures:
                lines.append(f"    Gate {f.gate.number}: {f.gate.name}")
        if self.minor_findings:
            lines.append(f"  MINOR ({len(self.minor_findings)}):")
            for f in self.minor_findings:
                lines.append(f"    Gate {f.gate.number}: {f.gate.name}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Compiled regex patterns — reusable across gates
# ═══════════════════════════════════════════════════════════════════════════════

# Banned default fonts
_RE_BANNED_FONT = re.compile(
    r"font-family\s*:\s*[^;]*\b(?:Inter|Roboto|Open Sans|Poppins|Lato|Arial|Helvetica)\b",
    re.IGNORECASE,
)

# Purple-to-blue gradient (the AI aesthetic)
_RE_AI_GRADIENT = re.compile(
    r"linear-gradient\s*\([^)]*(?:purple|violet|#[89a-fA-F][0-9a-fA-F]{5})"
    r"[^)]*(?:blue|#[0-4][0-9a-fA-F]{5})",
    re.IGNORECASE,
)

# Any gradient text (background-clip: text + gradient)
_RE_GRADIENT_TEXT = re.compile(
    r"background-clip\s*:\s*text[^}]*(?:gradient|linear-gradient)",
    re.IGNORECASE | re.DOTALL,
)

# Pure black / pure white backgrounds
_RE_PURE_BLACK = re.compile(r"background(?:-color)?\s*:\s*#(?:000000|000)\b", re.IGNORECASE)
_RE_PURE_WHITE = re.compile(r"background(?:-color)?\s*:\s*#(?:ffffff|fff)\b", re.IGNORECASE)

# 3-equal-column grid with icon tiles (the AI template)
_RE_THREE_COL_GRID = re.compile(
    r"repeat\s*\(\s*3\s*,\s*1fr\s*\)|grid-cols-3\b",
    re.IGNORECASE,
)

# Generic low-quality box-shadow
_RE_GENERIC_SHADOW = re.compile(
    r"box-shadow\s*:\s*0\s+[1-4]px\s+[1-8]px\s+(?:\d+px\s+)?rgba\s*\(\s*0\s*,\s*0\s*,\s*0\s*,\s*0\.[12]\d*\s*\)",
    re.IGNORECASE,
)

# transition-all (performance killer)
_RE_TRANSITION_ALL = re.compile(r"transition\s*:\s*all\b", re.IGNORECASE)

# hover:scale-105 uniform scale
_RE_UNIFORM_SCALE = re.compile(r"scale-105|scale\(\s*1\.05\s*\)", re.IGNORECASE)

# Bouncy / overshoot easing
_RE_BOUNCE_EASING = re.compile(r"cubic-bezier\s*\([^)]*1\.\d+[^)]*\)", re.IGNORECASE)

# Multiple simultaneous hover effects
_RE_MULTI_HOVER = re.compile(
    r"hover\s*:\s*[^;{}]*(?:translate|scale|shadow|colour|rotate)[^;{}]*"
    r"(?:translate|scale|shadow|colour|rotate)",
    re.IGNORECASE,
)

# Animate layout properties
_RE_ANIMATE_LAYOUT = re.compile(
    r"transition[^;{}]*(?:width|height|top|left|margin|padding)",
    re.IGNORECASE,
)

# Focus ring fade-in
_RE_FOCUS_FADE = re.compile(
    r":focus-visible[^;{}]*transition[^;{}]*(?:opacity|visibility)",
    re.IGNORECASE,
)

# Auto-rotating carousel without pause
_RE_CAROUSEL_NO_PAUSE = re.compile(
    r"(?:carousel|slider|banner)[^;{}]*(?:auto-play|autoplay)[^;{}]*(?<!no-)pause",
    re.IGNORECASE,
)

# Placeholder names
_RE_PLACEHOLDER_NAMES = re.compile(
    r"\b(?:Jane Doe|John Smith|Acme|Nexus|Seamless|Unleash)\b",
    re.IGNORECASE,
)

# Missing macrostructure stamp
_RE_MISSING_STAMP = re.compile(r"/\*\s*Hallmark\s*·\s*macrostructure:", re.IGNORECASE)

# Specimen fall-through
_RE_SPECIMEN_FALL = re.compile(
    r"macrostructure\s*:\s*specimen",
    re.IGNORECASE,
)

# Pure grey OKLCH (zero chroma)
_RE_PURE_GREY = re.compile(r"oklch\s*\([^)]*\b0\s*\)", re.IGNORECASE)

# Accent over 5% coverage (rough heuristic: accent background on large elements)
_RE_ACCENT_FLOOD = re.compile(
    r"background\s*:\s*var\s*\(\s*--color-accent\s*\)",
    re.IGNORECASE,
)

# Arbitrary spacing values (not on 4px grid)
_RE_ARBITRARY_SPACING = re.compile(
    r"(?:padding|gap|margin)\s*:\s*(?:1[13579]|2[13579]|3[13579]|[4-9][13579])px",
    re.IGNORECASE,
)

# Prose container too wide or too narrow
_RE_PROSE_WIDTH = re.compile(
    r"max-width\s*:\s*(?:\d+ch|\\[\d+ch\\])",
    re.IGNORECASE,
)

# ── Emil Kowalski animation gates ──────────────────────────────────

# Missing press feedback (button without scale on :active).
# Two separate patterns instead of one lookahead-based pattern: the greedy
# quantifier before a negative lookahead can consume past a real
# :active{scale} block when a button selector appears more than once
# (e.g. .btn:focus-visible before .btn:active), producing false positives.
_RE_HAS_BUTTON = re.compile(r"(?:<button|\.btn\b)", re.IGNORECASE)
_RE_HAS_PRESS_FEEDBACK = re.compile(r":active[\s\S]{0,300}scale", re.IGNORECASE)


def _missing_press_feedback(output: str) -> bool:
    return bool(_RE_HAS_BUTTON.search(output)) and not _RE_HAS_PRESS_FEEDBACK.search(output)


# Duration > 300ms on UI element
_RE_LONG_UI_DURATION = re.compile(
    r"transition[^;{}]*(?:[3-9]\d{2}|[1-9]\d{3,})ms",
    re.IGNORECASE,
)

# Linear easing on enter/exit (should be ease-out)
_RE_LINEAR_ENTER = re.compile(
    r"(?:enter|appear|show|open|mount)[\s\S]{0,200}linear\b",
    re.IGNORECASE,
)

# Missing stagger on group entrance (>3 children, same animation-delay)
_RE_NO_STAGGER = re.compile(
    r"(?:\.item\s*\{[^}]*animation[^}]*\}){3,}",
    re.IGNORECASE,
)
# ────────────────────────────────────────────────────────────────────

# Missing focus-visible or active styling
_RE_MISSING_FOCUS = re.compile(
    r":focus-visible|focus-visible",
    re.IGNORECASE,
)
_RE_MISSING_ACTIVE = re.compile(
    r":active|\.active",
    re.IGNORECASE,
)

# Missing prefers-reduced-motion
_RE_MISSING_REDUCED_MOTION = re.compile(
    r"prefers-reduced-motion",
    re.IGNORECASE,
)

# Lazy-loaded LCP element
_RE_LAZY_LCP = re.compile(
    r'(?:hero|banner|lcp)[^;{}]*loading\s*=\s*["\']lazy["\']',
    re.IGNORECASE,
)

# Mixed icon libraries
_RE_MIXED_ICONS = re.compile(
    r"(?:lucide|heroicons|phosphor|material|tabler)[^;{}]*(?:lucide|heroicons|phosphor|material|tabler)",
    re.IGNORECASE,
)

# Emoji as feature icons
_RE_EMOJI_ICONS = re.compile(
    r"[✨🚀⚡🔥🎯✅📊📈💡🔧🛠️]",
)

# Lottie default
_RE_LOTTIE_DEFAULT = re.compile(
    r"lottie|lottie-react|lottie-web",
    re.IGNORECASE,
)

# Missing aria-label on SVG
_RE_SVG_NO_ARIA = re.compile(
    r"<svg(?![^>]*(?:aria-label|aria-hidden))",
    re.IGNORECASE,
)

# Horizontal scroll (missing overflow-x: clip)
_RE_NO_OVERFLOW_CLIP = re.compile(
    r"overflow-x\s*:\s*clip",
    re.IGNORECASE,
)

# Decorative highlighter at wrong position
_RE_HIGHLIGHTER_BASELINE = re.compile(
    r"linear-gradient\s*\([^)]*transparent\s+\d+%\s*,[^)]*accent",
    re.IGNORECASE,
)

# Flex row without align-items: center
_RE_FLEX_NO_ALIGN = re.compile(
    r"display\s*:\s*flex[^;{}]*\n(?![^;{}]*align-items\s*:\s*center)",
    re.IGNORECASE,
)

# More than 3 font families
_RE_TOO_MANY_FONTS = re.compile(
    r"font-family\s*:\s*[^;]*,\s*[^;]*,\s*[^;]*,\s*[^;]*",
    re.IGNORECASE,
)

# Italic headers
_RE_ITALIC_HEADERS = re.compile(
    r"(?:h[1-6]|\.__title|\bhero__title|\bsection__title)[^;{}]*font-style\s*:\s*italic",
    re.IGNORECASE,
)

# Input border-width shift
_RE_INPUT_BORDER_SHIFT = re.compile(
    r"(?:input|textarea|select)[^;{}]*border-width\s*:\s*(?!1px)",
    re.IGNORECASE,
)

# Focus ring from border instead of outline
_RE_FOCUS_FROM_BORDER = re.compile(
    r":focus-visible[^;{}]*border\s*:\s*(?!.*outline)",
    re.IGNORECASE,
)

# Input height ≠ button height
_RE_INPUT_BUTTON_HEIGHT = re.compile(
    r"(?:height\s*:\s*(\d+)px)[^;{}]*(?:height\s*:\s*(?!\1)\d+px)",
    re.IGNORECASE,
)

# Disabled signalled by opacity alone
_RE_DISABLED_OPACITY_ONLY = re.compile(
    r"(?:disabled|\.disabled)[^;{}]*opacity\s*:\s*[^;{}]*\n(?![^;{}]*(?:cursor|pointer-events))",
    re.IGNORECASE,
)

# Low contrast (approximate: same lightness in OKLCH)
_RE_LOW_CONTRAST = re.compile(
    r"color\s*:\s*var\s*\(\s*--color-ink\s*\)[^;{}]*background\s*:\s*var\s*\(\s*--color-ink\s*\)",
    re.IGNORECASE,
)

# Missing --color-accent-ink
_RE_MISSING_ACCENT_INK = re.compile(
    r"--color-accent(?!-ink)[^;{}]*\n(?![^;{}]*--color-accent-ink)",
    re.IGNORECASE,
)

# Dark section without light text
_RE_DARK_NO_LIGHT_TEXT = re.compile(
    r"background(?:-color)?\s*:\s*var\s*\(\s*--color-ink[^;{}]*\n"
    r"(?![^;{}]*color\s*:\s*var\s*\(\s*--color-paper)",
    re.IGNORECASE,
)

# AI default nav
_RE_AI_NAV = re.compile(
    r"<nav[^>]*>[^<]*(?:wordmark|logo)[^<]*</nav>",
    re.IGNORECASE | re.DOTALL,
)

# AI default footer
_RE_AI_FOOTER = re.compile(
    r"<footer[^>]*>[^<]*(?:Product|Company|Resources|Legal)[^<]*</footer>",
    re.IGNORECASE | re.DOTALL,
)

# Centred hero everything
_RE_CENTRED_HERO = re.compile(
    r"(?:hero|banner)[^;{}]*(?:text-align\s*:\s*center[^;{}]*){3,}",
    re.IGNORECASE,
)

# Decorative without purpose
_RE_DECORATIVE_NO_PURPOSE = re.compile(
    r"(?:cursor|scanline|gradient blob|abstract shape|ornament|badge|sticker)",
    re.IGNORECASE,
)

# Invented metrics
_RE_INVENTED_METRICS = re.compile(
    r"\b(?:10× faster|50,000\+ teams|99\.9% uptime|\+\d+% conversion|"
    r"saves? \d+ hours|trusted by \d+)\b",
    re.IGNORECASE,
)

# Re-drawn chrome
_RE_REDRAWN_CHROME = re.compile(
    r"(?:browser bar|URL pill|traffic-light dots|phone frame|"
    r"mock title bar|IDE chrome|file tabs|activity bar)",
    re.IGNORECASE,
)

# Mid-render token improvisation
_RE_INLINE_COLOR = re.compile(
    r"(?:color|background)\s*:\s*(?:#\w+|oklch\(|rgb\(|hsl\()",
    re.IGNORECASE,
)

# Two-line clickable text
_RE_TWO_LINE_CLICKABLE = re.compile(
    r"(?:button|cta|nav|link)[^;{}]*\n[^;{}]*\n[^;{}]*(?:button|cta|nav|link)",
    re.IGNORECASE,
)

# Image grid without minmax(0, 1fr)
_RE_GRID_NO_MINMAX = re.compile(
    r"grid-template-columns[^;{}]*\b1fr\b(?!.*minmax)",
    re.IGNORECASE,
)

# Display headers without overflow-wrap
_RE_DISPLAY_NO_WRAP = re.compile(
    r"(?:text-[4-9]xl|clamp\s*\()[^;{}]*\n(?![^;{}]*overflow-wrap)",
    re.IGNORECASE,
)

# Eyebrow beside heading (tag-left, header-right pattern)
_RE_EYEBROW_BESIDE = re.compile(
    r"(?:eyebrow|label|kicker)[^;{}]*grid-template-columns[^;{}]*\d+fr[^;{}]*\d+fr",
    re.IGNORECASE,
)

# All-caps display with tight line-height
_RE_CAPS_TIGHT_LH = re.compile(
    r"text-transform\s*:\s*uppercase[^;{}]*line-height\s*:\s*(?:0\.\d+|1\.0)",
    re.IGNORECASE,
)

# Sticky bleed
_RE_STICKY_BLEED = re.compile(
    r"position\s*:\s*sticky[^;{}]*top\s*:\s*0[^;{}]*\n"
    r"[^;{}]*position\s*:\s*sticky[^;{}]*top\s*:\s*0",
    re.IGNORECASE,
)

# Studied DNA discarded
_RE_DISCARDED_DNA = re.compile(
    r"theme\s*:\s*(?!studied-DNA)",
    re.IGNORECASE,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Gate definitions
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_GATES: list[Gate] = [
    # ── Visual gates ──────────────────────────────────────────────────────────
    Gate(
        1,
        "visual",
        "Banned display font",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_BANNED_FONT.search(o)),
        {},
        "Inter, Roboto, Open Sans, Poppins, Lato used as display font",
    ),
    Gate(
        2,
        "visual",
        "Purple-to-blue gradient",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_AI_GRADIENT.search(o)),
        {"atmospheric": "ALLOW radial gradients on background only"},
        "Purple-to-blue or cyan-to-magenta gradient anywhere",
    ),
    Gate(
        3,
        "visual",
        "3-equal-column icon tiles",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_THREE_COL_GRID.search(o)) and "icon" in o.lower(),
        {},
        "Three equal columns with icon-above-heading tiles",
    ),
    Gate(
        4,
        "visual",
        "Gradient text",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_GRADIENT_TEXT.search(o)),
        {},
        "background-clip: text with gradient fill",
    ),
    Gate(
        5,
        "visual",
        "Pure black background",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_PURE_BLACK.search(o)),
        {},
        "Pure #000 or #000000 used as background",
    ),
    Gate(
        6,
        "visual",
        "Pure white background",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_PURE_WHITE.search(o)),
        {"modern-minimal": "ALLOW pure #fff paper (Stripe/ElevenLabs school)"},
        "Pure #fff or #ffffff used as background",
    ),
    Gate(
        7,
        "visual",
        "Centred everything hero",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_CENTRED_HERO.search(o)),
        {
            "atmospheric": "ALLOW centred hero when canvas itself is the design",
            "playful": "ALLOW centred hero",
        },
        "Hero with everything centred — eyebrow, title, lede, CTA all on same axis",
    ),
    # ── Structural gates ──────────────────────────────────────────────────────
    Gate(
        8,
        "structural",
        "Macrostructure fingerprint repetition",
        GateSeverity.MAJOR,
        lambda o, g: False,  # Requires design_log context; checked externally
        {},
        "Same macrostructure as previous output in project",
    ),
    Gate(
        9,
        "structural",
        "Sections separated only by equal whitespace",
        GateSeverity.MINOR,
        lambda o, g: o.count("<section") > 3 and o.count("<hr") < 2,
        {},
        "No rule, ornament, or colour shift between sections",
    ),
    # ── Microinteraction gates ────────────────────────────────────────────────
    Gate(
        10,
        "microinteractions",
        "transition-all used",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_TRANSITION_ALL.search(o)),
        {},
        "transition: all or transition-all anywhere",
    ),
    Gate(
        11,
        "microinteractions",
        "Uniform hover scale",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_UNIFORM_SCALE.search(o)),
        {},
        "hover:scale-105 or uniform scale applied across unrelated elements",
    ),
    Gate(
        12,
        "microinteractions",
        "Bouncy overshoot easing",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_BOUNCE_EASING.search(o)),
        {"playful": "ALLOW spring overshoot for playful themes"},
        "Bouncy / overshoot cubic-bezier on UI state changes",
    ),
    Gate(
        13,
        "microinteractions",
        "Multiple simultaneous hover effects",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_MULTI_HOVER.search(o)),
        {},
        "More than one hover effect at same time (translate + scale + shadow)",
    ),
    Gate(
        14,
        "microinteractions",
        "Animating layout properties",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_ANIMATE_LAYOUT.search(o)),
        {},
        "Animating width, height, top, left, margin, or padding",
    ),
    Gate(
        15,
        "microinteractions",
        "Focus ring fades in",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_FOCUS_FADE.search(o)),
        {},
        "Focus ring transition uses opacity fade-in",
    ),
    Gate(
        16,
        "microinteractions",
        "Celebratory success toast",
        GateSeverity.MINOR,
        lambda o, g: "toast" in o.lower() and "success" in o.lower(),
        {},
        "Celebratory success toast for visible action effects",
    ),
    Gate(
        17,
        "microinteractions",
        "Tooltip hover-delay equals focus-delay",
        GateSeverity.MINOR,
        lambda o, g: False,  # Requires parsing CSS values; heuristic too complex
        {},
        "Tooltip hover-delay and focus-delay should differ (hover 800-1000ms, focus 0ms)",
    ),
    Gate(
        18,
        "microinteractions",
        "Auto-rotating content without pause",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_CAROUSEL_NO_PAUSE.search(o)),
        {},
        "Carousel or banner auto-rotates without pause-on-hover",
    ),
    # ── Variety gates ─────────────────────────────────────────────────────────
    Gate(
        19,
        "variety",
        "Placeholder names",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_PLACEHOLDER_NAMES.search(o)),
        {},
        "Jane Doe / John Smith / Acme / Nexus placeholder names",
    ),
    Gate(
        20,
        "variety",
        "Missing macrostructure stamp",
        GateSeverity.MAJOR,
        lambda o, g: not bool(_RE_MISSING_STAMP.search(o)),
        {},
        "CSS comment stamp /* Hallmark · macrostructure: ... */ is missing",
    ),
    Gate(
        21,
        "variety",
        "Specimen fall-through",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_SPECIMEN_FALL.search(o)),
        {},
        "Specimen macrostructure used when brief did not explicitly request editorial/foundry",
    ),
    # ── Implementation gates ──────────────────────────────────────────────────
    Gate(
        22,
        "implementation",
        "Pure grey OKLCH neutrals",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_PURE_GREY.search(o)),
        {"modern-minimal": "ALLOW zero-chroma neutrals"},
        "Neutral surface with oklch(... 0 ...) — zero chroma reads flat",
    ),
    Gate(
        23,
        "implementation",
        "Accent flood",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_ACCENT_FLOOD.search(o)),
        {"atmospheric": "ALLOW accent-tinted blooms up to ~20% canvas"},
        "Accent colour covers more than ~5% of any viewport",
    ),
    Gate(
        24,
        "implementation",
        "Arbitrary spacing values",
        GateSeverity.MINOR,
        lambda o, g: bool(_RE_ARBITRARY_SPACING.search(o)),
        {},
        "Padding / gap / margin not on 4px spacing scale",
    ),
    Gate(
        25,
        "implementation",
        "Prose container width out of range",
        GateSeverity.MAJOR,
        lambda o, g: False,  # Requires numeric parsing
        {},
        "max-width outside 45-75ch range",
    ),
    Gate(
        26,
        "implementation",
        "Missing interactive states",
        GateSeverity.CRITICAL,
        lambda o, g: not (bool(_RE_MISSING_FOCUS.search(o)) and bool(_RE_MISSING_ACTIVE.search(o))),
        {},
        "Interactive element lacks :focus-visible or :active styling",
    ),
    Gate(
        27,
        "implementation",
        "Missing prefers-reduced-motion",
        GateSeverity.MAJOR,
        lambda o, g: "@media" in o and not bool(_RE_MISSING_REDUCED_MOTION.search(o)),
        {},
        "Animation present but no @media (prefers-reduced-motion) fallback",
    ),
    # ── Hero enrichment gates ─────────────────────────────────────────────────
    Gate(
        28,
        "hero",
        "Lazy-loaded LCP element",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_LAZY_LCP.search(o)),
        {},
        'loading="lazy" on hero image or video (LCP killer)',
    ),
    Gate(
        29,
        "hero",
        "Abstract background flood",
        GateSeverity.MAJOR,
        lambda o, g: "mesh-gradient" in o.lower() or "aurora" in o.lower(),
        {"atmospheric": "ALLOW up to two warm-toned radial blooms covering ~20-30%"},
        "Animated mesh-gradient or aurora blob covering whole page",
    ),
    Gate(
        30,
        "hero",
        "Mixed icon libraries",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_MIXED_ICONS.search(o)),
        {},
        "Two or more icon libraries mixed on same page",
    ),
    Gate(
        31,
        "hero",
        "Emoji as feature icons",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_EMOJI_ICONS.search(o)),
        {},
        "Emoji glyph used as feature-card / value-prop icon",
    ),
    Gate(
        32,
        "hero",
        "Lottie as default illustration",
        GateSeverity.MINOR,
        lambda o, g: bool(_RE_LOTTIE_DEFAULT.search(o)),
        {},
        "Lottie library used when hand-built SVG or pure-CSS would work",
    ),
    # ── Diversification gates ─────────────────────────────────────────────────
    Gate(
        33,
        "diversification",
        "Missing SVG aria-label",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_SVG_NO_ARIA.search(o)),
        {},
        "Custom SVG or decorative figure lacks aria-label or aria-hidden",
    ),
    Gate(
        34,
        "diversification",
        "Same archetype knobs",
        GateSeverity.MINOR,
        lambda o, g: False,  # Requires design_log context
        {},
        "Same variation knob values as previous output of same archetype",
    ),
    # ── Layout-safety gates ───────────────────────────────────────────────────
    Gate(
        35,
        "layout",
        "Horizontal scroll",
        GateSeverity.CRITICAL,
        lambda o, g: "overflow-x: clip" not in o.lower() and "overflow-x: hidden" not in o.lower(),
        {},
        "Missing overflow-x: clip on html and body",
    ),
    Gate(
        36,
        "layout",
        "Highlighter band at baseline",
        GateSeverity.MAJOR,
        lambda o, g: False,  # Requires visual confirmation; too complex for regex
        {},
        "Highlighter band sits at baseline instead of behind x-height",
    ),
    Gate(
        37,
        "layout",
        "Flex row without align-items: center",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_FLEX_NO_ALIGN.search(o)),
        {},
        "Interactive bar mixing button + text without align-items: center",
    ),
    # ── Typography gates ──────────────────────────────────────────────────────
    Gate(
        38,
        "typography",
        "More than 3 font families",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_TOO_MANY_FONTS.search(o)),
        {},
        "More than 3 distinct font-family families on the page",
    ),
    Gate(
        39,
        "typography",
        "Italic headers",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_ITALIC_HEADERS.search(o)),
        {},
        "Heading or display type set to italic — strongest AI tell",
    ),
    # ── Input-state gates ─────────────────────────────────────────────────────
    Gate(
        40,
        "input",
        "Input border-width shifts",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_INPUT_BORDER_SHIFT.search(o)),
        {},
        "Border-width changes between input states",
    ),
    Gate(
        41,
        "input",
        "Focus ring from border",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_FOCUS_FROM_BORDER.search(o)),
        {},
        "Focus ring built from border instead of outline",
    ),
    Gate(
        42,
        "input",
        "Input height ≠ button height",
        GateSeverity.MAJOR,
        lambda o, g: False,  # Requires numeric comparison
        {},
        "Input and adjacent button have different heights",
    ),
    Gate(
        43,
        "input",
        "Helper text collapses when empty",
        GateSeverity.MINOR,
        lambda o, g: "min-height" not in o.lower() and "helper" in o.lower(),
        {},
        "Helper-text slot lacks min-height reserve",
    ),
    Gate(
        44,
        "input",
        "Disabled signalled by opacity alone",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_DISABLED_OPACITY_ONLY.search(o)),
        {},
        "Disabled state uses only opacity — missing cursor and disabled attribute",
    ),
    # ── Contrast gates ────────────────────────────────────────────────────────
    Gate(
        45,
        "contrast",
        "Low contrast text",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_LOW_CONTRAST.search(o)),
        {},
        "Text colour matches background colour (ink-on-ink)",
    ),
    Gate(
        46,
        "contrast",
        "Missing accent-ink token",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_MISSING_ACCENT_INK.search(o)),
        {},
        "--color-accent used without --color-accent-ink definition",
    ),
    Gate(
        47,
        "contrast",
        "Dark section without light text",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_DARK_NO_LIGHT_TEXT.search(o)),
        {},
        "Dark background section without corresponding light text colour",
    ),
    # ── Nav / footer / hero gates ─────────────────────────────────────────────
    Gate(
        48,
        "chrome",
        "AI default nav",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_AI_NAV.search(o)),
        {},
        "Wordmark-left + 4-5 inline links + button-right nav fingerprint",
    ),
    Gate(
        49,
        "chrome",
        "AI default footer",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_AI_FOOTER.search(o)),
        {},
        "4 columns (Product/Company/Resources/Legal) + social row footer fingerprint",
    ),
    Gate(
        50,
        "chrome",
        "Hero padding too top-heavy",
        GateSeverity.MAJOR,
        lambda o, g: False,  # Requires numeric parsing
        {},
        "padding-block-end < 1.3× padding-block-start",
    ),
    Gate(
        51,
        "chrome",
        "Decorative without purpose",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_DECORATIVE_NO_PURPOSE.search(o)),
        {},
        "Decorative element with no semantic anchor in content",
    ),
    # ── Honest copy gates ─────────────────────────────────────────────────────
    Gate(
        52,
        "copy",
        "Invented metrics",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_INVENTED_METRICS.search(o)),
        {},
        "Fabricated quantitative claims (10× faster, 50,000+ teams, etc.)",
    ),
    # ── Re-drawn chrome gates ─────────────────────────────────────────────────
    Gate(
        53,
        "chrome",
        "Re-drawn UI chrome",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_REDRAWN_CHROME.search(o)),
        {},
        "Fake browser bar, phone frame, IDE chrome, or terminal frame",
    ),
    # ── Token discipline gates ────────────────────────────────────────────────
    Gate(
        54,
        "tokens",
        "Mid-render token improvisation",
        GateSeverity.CRITICAL,
        lambda o, g: bool(_RE_INLINE_COLOR.search(o)),
        {},
        "Colour or font value outside defined token block",
    ),
    # ── Responsive gates ──────────────────────────────────────────────────────
    Gate(
        55,
        "responsive",
        "Two-line clickable text",
        GateSeverity.CRITICAL,
        lambda o, g: False,  # Requires rendering at multiple widths
        {},
        "Button label, nav link, or CTA wraps to two lines",
    ),
    Gate(
        56,
        "responsive",
        "Image grid without minmax(0, 1fr)",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_GRID_NO_MINMAX.search(o)),
        {},
        "Grid track containing image uses bare 1fr instead of minmax(0, 1fr)",
    ),
    Gate(
        57,
        "responsive",
        "Display headers without long-word wrap",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_DISPLAY_NO_WRAP.search(o)),
        {},
        "Display-size text lacks overflow-wrap: anywhere",
    ),
    # ── Emil Kowalski animation gates ────────────────────────────────────────
    Gate(
        58,
        "microinteractions",
        "Missing press feedback",
        GateSeverity.MAJOR,
        lambda o, g: _missing_press_feedback(o),
        {},
        "Button or pressable element has no :active scale feedback",
    ),
    Gate(
        59,
        "microinteractions",
        "UI duration exceeds 300ms",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_LONG_UI_DURATION.search(o)),
        {},
        "UI animation duration > 300ms (budget: 100-250ms for UI elements)",
    ),
    Gate(
        60,
        "microinteractions",
        "Linear easing on enter/exit",
        GateSeverity.MAJOR,
        lambda o, g: bool(_RE_LINEAR_ENTER.search(o)),
        {},
        "Enter/exit animation uses linear instead of ease-out",
    ),
    Gate(
        61,
        "microinteractions",
        "Missing stagger on group entrance",
        GateSeverity.MINOR,
        lambda o, g: bool(_RE_NO_STAGGER.search(o)),
        {},
        "Multiple items animate in without stagger (30-80ms delay between items)",
    ),
    # ─────────────────────────────────────────────────────────────────────────
]


class SlopTestEngine:
    """Deterministic anti-slop validation engine.

    Usage:
        engine = SlopTestEngine()
        result = engine.run(generated_css, genre="editorial")
        if not result.passed:
            print(result.summary)
    """

    def __init__(self, gates: list[Gate] | None = None) -> None:
        self._gates = gates or _DEFAULT_GATES

    def run(self, output: str, genre: str | None = None) -> SlopTestResult:
        """Run all gates against *output*.

        Args:
            output: The generated code/CSS to validate.
            genre: Optional genre for scoped overrides.

        Returns:
            SlopTestResult with all findings.
        """
        findings: list[GateFinding] = []
        for gate in self._gates:
            # Apply genre override
            override = gate.genre_override.get(genre) if genre else None
            if override == "ALLOW":
                continue

            try:
                if gate.check(output, genre):
                    findings.append(GateFinding(gate=gate, override_applied=override is not None))
            except Exception as exc:
                logger.debug("slop_test: gate %d check failed: %s", gate.number, exc)

        return SlopTestResult(findings=findings)
