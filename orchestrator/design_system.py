"""
DesignSystem - .design-system.yml brand config injection.
===========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 4, Phase 6 (Replit-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass, field
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
            self.shadow = SimpleNamespace(sm="0 1px 2px rgba(0,0,0,.05)", md="0 4px 6px rgba(0,0,0,.07)", lg="0 10px 15px rgba(0,0,0,.1)")
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
