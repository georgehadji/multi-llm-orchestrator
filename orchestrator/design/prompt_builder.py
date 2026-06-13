"""
Hallmark Prompt Builder — Constructs structured constraint blocks for LLM prompts.
================================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Builds the macrostructure, theme, and self-critique blocks that are injected
into the skill prefix before the existing taste-skill content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .catalogs import Archetype, Macrostructure, Theme


class HallmarkPromptBuilder:
    """Builds structured constraint blocks for Hallmark design generation."""

    def build_macrostructure_block(self, macro: Macrostructure) -> str:
        """Build a macrostructure constraint block."""
        sections = ""
        if macro.saas_sections:
            sections = (
                "\n## SaaS Section Sequence\n"
                + "\n".join(f"- {s}" for s in macro.saas_sections)
            )
        return (
            f"## Selected Macrostructure: {macro.name}\n"
            f"- Heading placement: {macro.heading}\n"
            f"- Body composition: {macro.body}\n"
            f"- Divider language: {macro.divider}\n"
            f"- Button voice: {macro.button}\n"
            f"- Image treatment: {macro.image}\n"
            f"- Reveal pattern: {macro.reveal}\n"
            f"- Nav archetype: {macro.nav_default}\n"
            f"- Footer archetype: {macro.footer_default}\n"
            f"{sections}"
        )

    def build_theme_block(self, theme: Theme) -> str:
        """Build a theme constraint block."""
        moves = "\n".join(f"- {m}" for m in theme.signature_moves)
        anti = "\n".join(f"- {a}" for a in theme.anti_patterns)
        voices = "\n".join(f"- {v}" for v in theme.voice_fixtures)
        return (
            f"## Selected Theme: {theme.name} ({theme.genre.value})\n"
            f"- Paper: {theme.paper_oklch}\n"
            f"- Ink: {theme.ink_oklch}\n"
            f"- Accent: {theme.accent_oklch}\n"
            f"- Accent ink: {theme.accent_ink_oklch}\n"
            f"- Display font: {theme.font_display}\n"
            f"- Body font: {theme.font_body}\n"
            f"- Mono font: {theme.font_mono}\n"
            f"- Motion stance: {theme.motion_stance}\n"
            f"\n## Signature moves (exhibit at least 5):\n{moves}\n"
            f"\n## Voice fixtures:\n{voices}\n"
            f"\n## Anti-patterns (never ship):\n{anti}"
        )

    def build_nav_block(self, nav: Archetype) -> str:
        """Build a nav archetype constraint block."""
        return (
            f"## Selected Nav: {nav.name} ({nav.code})\n"
            f"{nav.description}\n"
            f"- Variation knobs: {', '.join(nav.knobs.keys())}"
        )

    def build_footer_block(self, footer: Archetype) -> str:
        """Build a footer archetype constraint block."""
        return (
            f"## Selected Footer: {footer.name} ({footer.code})\n"
            f"{footer.description}\n"
            f"- Variation knobs: {', '.join(footer.knobs.keys())}"
        )

    def build_component_block(self) -> str:
        """Build component-scope constraint block (8-state checklist)."""
        return (
            "## Component Scope\n"
            "This is a single component, not a full page.\n"
            "- Skip macrostructure, nav, footer, hero enrichment\n"
            "- Every interactive component MUST ship code for ALL 8 states:\n"
            "  1. default\n"
            "  2. hover\n"
            "  3. :focus-visible\n"
            "  4. :active\n"
            "  5. disabled\n"
            "  6. loading\n"
            "  7. error\n"
            "  8. success\n"
            "- Include an 8-state demo wrapper (.preview.html / .preview.tsx)\n"
            "- Use project tokens; no mid-render improvisation"
        )

    def build_self_critique_block(self) -> str:
        """Build the pre-emit self-critique instruction block."""
        return (
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

    def build_full_prefix(
        self,
        macrostructure: Macrostructure | None,
        theme: Theme | None,
        nav: Archetype | None,
        footer: Archetype | None,
        scope: str = "page",
    ) -> str:
        """Build the complete Hallmark constraint prefix."""
        parts: list[str] = []

        if scope == "component":
            parts.append(self.build_component_block())
        else:
            if macrostructure is not None:
                parts.append(self.build_macrostructure_block(macrostructure))
            if theme is not None:
                parts.append(self.build_theme_block(theme))
            if nav is not None:
                parts.append(self.build_nav_block(nav))
            if footer is not None:
                parts.append(self.build_footer_block(footer))

        parts.append(self.build_self_critique_block())
        return "\n\n".join(parts)
