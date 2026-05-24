"""
UXDesignEnhancer — High-end UI/UX standards for generated frontends
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Ensures the orchestrator produces high-quality user interfaces that follow
modern UX standards: responsive design, accessibility (WCAG), consistent
spacing/typography, dark mode support, and component-based architecture.

Integrates with DeveloperAgent when generating frontend code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.ux.design_enhancer")


@dataclass
class UXStandard:
    """A UX standard that generated code should follow."""
    name: str
    description: str
    category: str  # "layout", "accessibility", "typography", "color", "interaction"

    def to_prompt(self) -> str:
        return f"- **{self.name}**: {self.description}"


# Modern UX standards for generated interfaces
UX_STANDARDS: list[UXStandard] = [
    # Layout & Responsiveness
    UXStandard("Responsive Design", "Use CSS Grid and Flexbox. Support mobile (320px), tablet (768px), desktop (1024px+). Never use fixed widths.", "layout"),
    UXStandard("Mobile-First", "Default to mobile layout, use min-width media queries for larger screens.", "layout"),
    UXStandard("Consistent Spacing", "Use a 4px or 8px spacing scale. Consistent padding, margins, and gaps throughout.", "layout"),
    UXStandard("Visual Hierarchy", "Clear heading hierarchy (h1→h6). Most important content is visually prominent.", "layout"),

    # Accessibility (WCAG 2.1 AA)
    UXStandard("Color Contrast", "Text must meet 4.5:1 contrast ratio (3:1 for large text). Use tools to verify.", "accessibility"),
    UXStandard("Keyboard Navigation", "All interactive elements reachable via Tab. Visible focus indicators. No keyboard traps.", "accessibility"),
    UXStandard("ARIA Labels", "Interactive elements have aria-label or aria-labelledby. Landmarks: nav, main, complementary.", "accessibility"),
    UXStandard("Screen Reader Support", "Images have alt text. Forms have associated labels. Use semantic HTML elements.", "accessibility"),
    UXStandard("Touch Targets", "Interactive elements minimum 44×44px touch target size.", "accessibility"),

    # Typography
    UXStandard("Font Pairing", "Use a maximum of 2 fonts: one for headings, one for body. System fonts or variable fonts preferred.", "typography"),
    UXStandard("Readable Line Length", "Body text between 60-80 characters per line. Use max-width on text containers.", "typography"),
    UXStandard("Line Height", "Body text: 1.5-1.6 line height. Headings: 1.2-1.3.", "typography"),
    UXStandard("Font Size Scale", "Use a consistent type scale. Default: 16px body, scale ratio of 1.25 or 1.333.", "typography"),

    # Color & Theme
    UXStandard("Design System", "Define CSS custom properties for colors, spacing, and typography. Use them consistently.", "color"),
    UXStandard("Dark Mode", "Support prefers-color-scheme media query. Don't use pure black (#000) — use dark gray instead.", "color"),
    UXStandard("Color Palette", "Define 3-6 colors maximum: primary, secondary, accent, background, text, error. Don't use raw colors inline.", "color"),

    # Interaction & Animation
    UXStandard("Micro-interactions", "Subtle hover/focus states on buttons, links, and inputs. 150-300ms transitions.", "interaction"),
    UXStandard("Loading States", "Show loading spinners or skeleton screens for async operations. Disable buttons during submission.", "interaction"),
    UXStandard("Error Handling", "Inline form validation errors. Toast notifications for actions. Clear error messages.", "interaction"),
    UXStandard("Empty States", "When no data exists, show a helpful illustration or message, not a blank page.", "interaction"),
]


def ux_system_prompt() -> str:
    """Get the UX standards as an LLM system prompt injection."""
    lines = [
        "## UI/UX Standards — Must Follow",
        "",
        "When generating HTML, CSS, or frontend code, you MUST follow these standards:",
        "",
    ]

    for cat in ["layout", "accessibility", "typography", "color", "interaction"]:
        cat_name = cat.replace("_", " ").title()
        items = [s for s in UX_STANDARDS if s.category == cat]
        if items:
            lines.append(f"### {cat_name}")
            lines.append("")
            for s in items:
                lines.append(s.to_prompt())
            lines.append("")

    lines.append("### Tech Stack Preference")
    lines.append("- Use Tailwind CSS for styling when the project doesn't specify a framework.")
    lines.append("- If the user mentions a framework (React, Vue, etc.), use its standard tooling.")
    lines.append("- Use semantic HTML5 elements throughout.")
    lines.append("")

    return "\n".join(lines)


class UXDesignReviewer:
    """Reviews generated HTML/CSS/JS against UX standards.

    Can be called by DeveloperAgent after generating frontend code
    to ensure quality before delivery.
    """

    def __init__(self) -> None:
        self.standards = UX_STANDARDS

    def review(self, html: str, css: str = "", js: str = "") -> list[dict[str, Any]]:
        """Review generated code against UX standards.

        Args:
            html: Generated HTML code.
            css: Generated CSS code.
            js: Generated JavaScript code.

        Returns:
            List of findings — each with standard_name, passed, suggestion.
        """
        findings: list[dict[str, Any]] = []
        combined = (html + " " + css + " " + js).lower()

        for standard in self.standards:
            passed = True
            suggestion = ""

            if standard.name == "Responsive Design":
                if "max-width" not in combined and "grid" not in combined and "flex" not in combined:
                    passed = False
                    suggestion = "Add responsive layout using CSS Grid or Flexbox with max-width containers."

            elif standard.name == "Keyboard Navigation":
                if "tabindex" not in combined and "focus" not in combined and ":focus" not in combined:
                    passed = False
                    suggestion = "Add keyboard navigation support: tabindex attributes and visible :focus styles."

            elif standard.name == "Color Contrast":
                if "color:" in combined and "contrast" not in combined:
                    passed = False
                    suggestion = "Ensure text/background color combinations meet 4.5:1 contrast ratio."

            elif standard.name == "Dark Mode":
                if "prefers-color-scheme" not in combined:
                    passed = True  # This is optional but recommended
                    suggestion = "Consider adding dark mode support via prefers-color-scheme."

            elif standard.name == "Design System":
                if "--" not in combined and "var(" not in combined:
                    passed = False
                    suggestion = "Define CSS custom properties for colors, spacing, and typography."

            elif standard.name == "Responsive Design":
                passed = any(kw in combined for kw in ["@media", "grid", "flex"])
                if not passed:
                    suggestion = "Add media queries or use CSS Grid/Flexbox for responsive layout."

            findings.append({
                "standard": standard.name,
                "category": standard.category,
                "passed": passed,
                "suggestion": suggestion,
            })

        return findings

    def score(self, html: str, css: str = "", js: str = "") -> float:
        """Calculate a UX quality score from 0.0 to 10.0."""
        findings = self.review(html, css, js)
        if not findings:
            return 10.0
        passed = sum(1 for f in findings if f["passed"])
        return round((passed / len(findings)) * 10.0, 1)

    def critique(self, html: str, css: str = "", js: str = "") -> str:
        """Generate a critique string for failed standards."""
        findings = self.review(html, css, js)
        failed = [f for f in findings if not f["passed"]]
        if not failed:
            return "All UX standards pass."
        lines = ["## UX Review Findings", ""]
        for f in failed:
            lines.append(f"- **{f['standard']}** ({f['category']}): {f['suggestion']}")
        return "\n".join(lines)
