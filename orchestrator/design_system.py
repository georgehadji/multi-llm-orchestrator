"""
DesignSystem - .design-system.yml brand config injection.
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 4, Phase 6 (Replit-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ColorTokens:
    primary: str = "#818cf8"
    secondary: str = "#a78bfa"
    accent: str = "#4f9eff"
    background: str = "#09090b"
    surface: str = "#111113"
    surface_alt: str = "#1a1a1a"
    text_primary: str = "#fafafa"
    text_secondary: str = "#a1a1aa"
    text: str = "#fafafa"
    border: str = "#27272a"
    success: str = "#34d399"
    warning: str = "#fbbf24"
    error: str = "#f87171"

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class TypographyTokens:
    font_sans: str = "'Inter', sans-serif"
    font_mono: str = "'JetBrains Mono', monospace"
    base_size: int = 16
    scale: float = 1.25
    line_height: float = 1.6
    heading_weight: int = 700

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


class ComponentSource(str, Enum):
    """Upstream library a curated component reference comes from.

    Used by ``design/component_registry.py`` to attribute each ComponentSpec and
    to prefer sources that actually target the framework being generated.
    """

    SHADCN = "shadcn"
    ACETERNITY = "aceternity"
    MAGIC_UI = "magic_ui"
    TWENTYFIRST = "twentyfirst"
    REACTBITS = "reactbits"
    ANIMATE_UI = "animate_ui"
    BITS_UI = "bits_ui"
    SVELTEBITS = "sveltebits"
    CUSTOM = "custom"


# Which framework each source targets.  ``CUSTOM`` is framework-agnostic: it
# describes a section generically, so it is valid for any target.
#
# NOTE ON VERIFICATION: the four libraries added for the component-generation
# feature (reactbits, animate_ui, bits_ui, sveltebits) could NOT be verified
# against their own documentation — this environment's egress proxy blocks
# those domains and their MCP servers are not connected here. The assignments
# below follow each project's name and widely-known framework; correct them
# here if any is wrong, and everything downstream follows.
COMPONENT_SOURCE_FRAMEWORKS: dict[ComponentSource, frozenset[str]] = {
    ComponentSource.SHADCN: frozenset({"react"}),
    ComponentSource.ACETERNITY: frozenset({"react"}),
    ComponentSource.MAGIC_UI: frozenset({"react"}),
    ComponentSource.TWENTYFIRST: frozenset({"react"}),
    ComponentSource.REACTBITS: frozenset({"react"}),
    ComponentSource.ANIMATE_UI: frozenset({"react"}),
    ComponentSource.BITS_UI: frozenset({"svelte"}),
    ComponentSource.SVELTEBITS: frozenset({"svelte"}),
    ComponentSource.CUSTOM: frozenset({"react", "svelte", "html"}),
}


def normalize_framework(framework: str) -> str:
    """Map a CLI/spec framework value onto a component-ecosystem name.

    ``next.js`` and ``react`` share React's component ecosystem; ``sveltekit``
    and ``svelte`` share Svelte's. Anything else (notably ``html``) has no
    component ecosystem and returns ``"html"``.
    """
    value = (framework or "").strip().lower().replace("_", ".")
    if value in ("react", "next.js", "nextjs", "next"):
        return "react"
    if value in ("svelte", "sveltekit", "svelte.kit"):
        return "svelte"
    return "html"


def sources_for_framework(framework: str) -> set[ComponentSource]:
    """Return the sources whose components are usable in *framework*."""
    target = normalize_framework(framework)
    return {
        source for source, frameworks in COMPONENT_SOURCE_FRAMEWORKS.items() if target in frameworks
    }


@dataclass
class ComponentSpec:
    """A curated reference to one upstream component.

    This is a *reference*, not code: ``prompt_reference`` describes the
    component so the generator can steer an LLM toward it, which is why the
    registry can cite libraries without vendoring or building them.
    """

    name: str
    source: ComponentSource
    category: str
    variant: str = "default"
    layout_type: str = "flex"
    primary_color: str | None = None
    animation_style: str = "modern"
    has_aria_labels: bool = True
    keyboard_navigable: bool = True
    responsive: bool = True
    dark_mode_support: bool = True
    prompt_reference: str = ""

    def supports_framework(self, framework: str) -> bool:
        """True when this component's source targets *framework*."""
        target = normalize_framework(framework)
        return target in COMPONENT_SOURCE_FRAMEWORKS.get(
            self.source, frozenset({"react", "svelte", "html"})
        )

    def compatibility_score(self, design_system: DesignSystem) -> float:
        """Score 0.0-1.0 for how well this component fits *design_system*.

        Accessibility and responsiveness are weighted highest because they are
        non-negotiable in generated output; tone and dark-mode agreement break
        ties between otherwise-equivalent candidates.
        """
        score = 0.0
        if self.has_aria_labels:
            score += 0.3
        if self.keyboard_navigable:
            score += 0.2
        if self.responsive:
            score += 0.2
        # `tone` is a plain string on DesignSystem; tolerate an enum too.
        tone = getattr(design_system.tone, "value", design_system.tone)
        if self.animation_style and str(tone) in str(self.animation_style):
            score += 0.2
        if self.dark_mode_support == bool(getattr(design_system, "dark_mode", True)):
            score += 0.1
        return round(min(score, 1.0), 3)


@dataclass
class DesignSystem:
    name: str = ""
    version: str = "1.0.0"
    colors: ColorTokens = field(default_factory=ColorTokens)
    typography: TypographyTokens = field(default_factory=TypographyTokens)
    border_radius: int = 8
    spacing_unit: int = 4
    dark_mode: bool = True

    # Additional fields used by WebsiteGenerator — defaults to None for compatibility
    tone: str = "modern"
    font_heading: str = "Inter"
    font_body: str = "Inter"
    accessibility: Any = None

    def __post_init__(self):
        # Ensure nested objects exist as property-like access
        if self.accessibility is None:
            from types import SimpleNamespace

            self.accessibility = SimpleNamespace(min_contrast_ratio=4.5)
        # Make spacing, shadow, animation, border_radius accessible
        if not hasattr(self, "spacing"):
            from types import SimpleNamespace

            self.spacing = SimpleNamespace(unit=f"{self.spacing_unit}px")
            self.shadow = SimpleNamespace(
                sm="0 1px 2px rgba(0,0,0,.05)",
                md="0 4px 6px rgba(0,0,0,.07)",
                lg="0 10px 15px rgba(0,0,0,.1)",
            )
            self.animation = SimpleNamespace(duration="200ms", easing="ease-in-out")
            self.border_radius = SimpleNamespace(sm="4px", md="8px", lg="12px", full="9999px")

    def to_dict(self):
        from types import SimpleNamespace

        def _safe_dict(obj):
            if isinstance(obj, SimpleNamespace):
                return {k: _safe_dict(v) for k, v in obj.__dict__.items()}
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)

        return {
            "name": self.name,
            "version": self.version,
            "colors": self.colors.to_dict(),
            "typography": self.typography.to_dict(),
            "border_radius": _safe_dict(self.border_radius),
            "spacing": _safe_dict(self.spacing),
            "shadow": _safe_dict(self.shadow),
            "animation": _safe_dict(self.animation),
            "dark_mode": self.dark_mode,
            "tone": self.tone,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(
            name=d.get("name", ""),
            version=d.get("version", "1.0.0"),
            colors=ColorTokens(**d.get("colors", {})),
            typography=TypographyTokens(**d.get("typography", {})),
            border_radius=d.get("border_radius", 8),
            spacing_unit=d.get("spacing_unit", 4),
            dark_mode=d.get("dark_mode", True),
        )

    def to_css(self):
        return f"""/* Generated from {self.name} v{self.version} */
:root {{
  --color-primary: {self.colors.primary};
  --color-secondary: {self.colors.secondary};
  --color-bg: {self.colors.background};
  --color-surface: {self.colors.surface};
  --color-text: {self.colors.text};
  --color-text-secondary: {self.colors.text_secondary};
  --color-success: {self.colors.success};
  --color-warning: {self.colors.warning};
  --color-error: {self.colors.error};
  --font-sans: {self.typography.font_sans};
  --font-mono: {self.typography.font_mono};
  --border-radius: {self.border_radius}px;
  --spacing-unit: {self.spacing_unit}px;
}}"""

    def to_tailwind(self):
        return f"""// Generated from {self.name} v{self.version}
module.exports = {{
  theme: {{
    extend: {{
      colors: {{
        primary: "{self.colors.primary}",
        secondary: "{self.colors.secondary}",
        surface: "{self.colors.surface}",
      }},
      fontFamily: {{
        sans: [{self.typography.font_sans}],
        mono: [{self.typography.font_mono}],
      }}
    }}
  }}
}};"""

    def to_prompt_injection(self):
        nl = chr(10)
        return f'DESIGN SYSTEM ({self.name} v{self.version}):{nl}- Primary: {self.colors.primary}{nl}- Font: {self.typography.font_sans}{nl}- Dark: {"yes" if self.dark_mode else "no"}{nl}Use these tokens in all generated UI code.'

    def to_prompt_context(self):
        """Alias for to_prompt_injection — used by WebsiteGenerator."""
        return self.to_prompt_injection()


class DesignSystemManager:
    def __init__(self, project_dir="."):
        self._dir = Path(project_dir)
        self.design_system = DesignSystem()
        self._load()

    def _load(self):
        for name in [".design-system.yml", ".design-system.yaml", ".design-system.json"]:
            fp = self._dir / name
            if fp.exists():
                try:
                    text = fp.read_text(encoding="utf-8")
                    if name.endswith(".json"):
                        self.design_system = DesignSystem.from_dict(json.loads(text))
                    else:
                        import yaml

                        self.design_system = DesignSystem.from_dict(yaml.safe_load(text) or {})
                    return
                except Exception:
                    pass

    def save(self, path=".design-system.yml"):
        import yaml

        (self._dir / path).write_text(
            yaml.dump(self.design_system.to_dict(), default_flow_style=False), encoding="utf-8"
        )

    def get_prompt(self):
        return self.design_system.to_prompt_injection()

    def to_prompt_context(self) -> str:
        """Alias for to_prompt_injection — used by WebsiteGenerator."""
        return self.to_prompt_injection()

    def export_frontend(self, output_dir="."):
        out = Path(output_dir)
        (out / "design-tokens.css").write_text(self.design_system.to_css(), encoding="utf-8")
        (out / "tailwind.design.js").write_text(self.design_system.to_tailwind(), encoding="utf-8")


# ── Quality types used by website_validator ─────────────────────────────────


class QualityCheck:
    """Individual quality check result — used by website_validator."""

    def __init__(
        self,
        name: str,
        passed: bool,
        score: float,
        details: str,
        recommendations: list[str] | None = None,
        applicable: bool = True,
    ):
        self.name = name
        self.passed = passed
        self.score = score
        self.details = details
        self.recommendations = recommendations if recommendations is not None else []
        # False when the check has nothing to judge (e.g. rate limiting on a
        # static brochure site with no endpoints). Such a check must be excluded
        # from the aggregate rather than contributing a free 1.0 — otherwise
        # every score is inflated by the checks that did not apply.
        self.applicable = applicable


class QualityReport:
    """Aggregated quality report — accepts arbitrary kwargs for compatibility.

    Separate from DesignSystem — this is the output of WebsiteQualityValidator.validate(),
    not a design token definition.
    """

    def __init__(self, **kwargs):
        self.checks: list = kwargs.pop("checks", [])
        # `score`/`passed` are DERIVED from `checks` unless a caller supplies them
        # explicitly. Deriving here (rather than in each validator) keeps a single
        # aggregation rule: both shipped validator copies built this report without
        # passing either field, so the aggregate silently defaulted to 0.0/False no
        # matter how the individual checks scored.
        self.score: float = (
            kwargs.pop("score") if "score" in kwargs else self._derive_score(self.checks)
        )
        self.lighthouse_score: float = kwargs.pop("lighthouse_score", 0)
        self.wcag_level: str = kwargs.pop("wcag_level", "A")
        self.responsive_breakpoints_tested: int = kwargs.pop("responsive_breakpoints_tested", 0)
        self.issues: list = kwargs.pop("issues", [])
        self.warnings: list = kwargs.pop("warnings", [])
        # Conservative: every check must pass. A mean score hides a single fatal
        # dimension (e.g. responsive=0.5 => mobile is broken) behind good siblings.
        self.passed: bool = (
            kwargs.pop("passed") if "passed" in kwargs else self._derive_passed(self.checks)
        )
        self.__dict__.update(kwargs)

    @staticmethod
    def _applicable(checks: list) -> list:
        """Checks that had something to judge (see QualityCheck.applicable)."""
        return [c for c in checks if bool(getattr(c, "applicable", True))]

    # Weight given to the worst check when aggregating. 0.5 = "half average,
    # half worst". See _derive_score.
    WORST_WEIGHT = 0.5

    @classmethod
    def _derive_score(cls, checks: list) -> float:
        """Conservative aggregate over APPLICABLE checks; 0.0 when nothing to judge.

        Two departures from a plain mean, each fixing a measured defect:

        1. Inapplicable checks are EXCLUDED rather than counted as 1.0. On a
           static brochure site three of eight checks report "not applicable";
           counting them pinned ~0.375 of every score at maximum no matter what
           the generator produced.

        2. The aggregate is ``0.5 * mean + 0.5 * min`` — a site is only as
           shippable as its weakest dimension. Under a plain mean a single fatal
           check was almost invisible: measured against a clean fixture, a
           placeholder <title> cost 0.021 and six megabytes of images cost
           0.029. The gate detected every defect and cared about none of them.
           This is the same conservative-aggregation doctrine the evaluator uses
           for self-consistency: disagreement resolves downward, not to the mean.
        """
        scores = [
            float(c.score) for c in cls._applicable(checks) if getattr(c, "score", None) is not None
        ]
        if not scores:
            return 0.0
        mean = sum(scores) / len(scores)
        return round((1.0 - cls.WORST_WEIGHT) * mean + cls.WORST_WEIGHT * min(scores), 4)

    @classmethod
    def _derive_passed(cls, checks: list) -> bool:
        """True only when every applicable check passed; an empty report never passes."""
        applicable = cls._applicable(checks)
        return bool(applicable) and all(bool(getattr(c, "passed", False)) for c in applicable)

    def failed_checks(self) -> list:
        """Applicable checks that did not pass — what a gate reports to the caller."""
        return [c for c in self._applicable(self.checks) if not bool(getattr(c, "passed", False))]
