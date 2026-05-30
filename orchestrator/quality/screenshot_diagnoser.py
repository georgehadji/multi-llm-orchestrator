"""
ScreenshotDiagnoser — Diagnose issues from screenshots.
=========================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 1 (Wave 3: N3 Screenshot-Based Debugging).
Accepts screenshot images (or description text), diagnoses UI/functional
issues, and produces fix suggestions. Light — full vision pipeline
requires V3 (Browser-Use Agent) infrastructure.
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
# FIXED: from ..infrastructure.llm_client import UnifiedClient
    from ..infrastructure.llm_client import UnifiedClient

logger = logging.getLogger(__name__)

_DIAGNOSE_PROMPT = """You are a UI/UX debugging expert. Analyze this application screenshot
(or description) and identify issues.

Context:
{context}

Screenshot description (or OCR text if available):
{description}

Identify:
1. Visual bugs (misaligned elements, clipping, color issues)
2. Layout problems (responsive breakpoints, spacing)
3. Functional issues (broken buttons, missing elements)
4. Accessibility problems (contrast, labels, keyboard nav)

Return exactly this JSON:
{{
    "issues": [
        {{
            "severity": "critical|warning|info",
            "category": "visual|layout|functional|accessibility",
            "element": "the specific UI element affected",
            "description": "what is wrong",
            "fix_suggestion": "how to fix it"
        }}
    ],
    "overall_assessment": "A 1-2 sentence summary of the UI state"
}}"""


@dataclass
class ScreenshotIssue:
    """A single issue found in a screenshot."""

    severity: str  # critical, warning, info
    category: str  # visual, layout, functional, accessibility
    element: str
    description: str
    fix_suggestion: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "category": self.category,
            "element": self.element,
            "description": self.description,
            "fix_suggestion": self.fix_suggestion,
        }


@dataclass
class ScreenshotDiagnosis:
    """Complete diagnosis from a screenshot."""

    issues: list[ScreenshotIssue]
    overall_assessment: str

    @property
    def critical_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "critical")

    @property
    def issues_by_category(self) -> dict[str, list[ScreenshotIssue]]:
        result: dict[str, list[ScreenshotIssue]] = {}
        for issue in self.issues:
            result.setdefault(issue.category, []).append(issue)
        return result


class ScreenshotDiagnoser:
    """Diagnose UI/functional issues from screenshots.

    Accepts image files (base64-encoded for the LLM) or text descriptions.
    Requires a vision-capable model for image analysis; falls back to
    text-only analysis if no vision model is available.
    """

    def __init__(self, client: UnifiedClient | None = None):
        self._client = client

    async def diagnose(
        self,
        screenshot_path: Path | None = None,
        description: str = "",
        context: str = "",
        use_vision: bool = True,
    ) -> ScreenshotDiagnosis:
        """Diagnose issues from a screenshot or description.

        Args:
            screenshot_path: Path to screenshot image file (optional)
            description: Text description of the screenshot (used if no image)
            context: Additional context (e.g., task description)
            use_vision: Whether to use vision model for image analysis

        Returns:
            ScreenshotDiagnosis with identified issues
        """
        if not self._client:
            return ScreenshotDiagnosis(
                issues=[],
                overall_assessment="No LLM client available for screenshot diagnosis",
            )

        # Build the description from image or text
        if screenshot_path and use_vision and screenshot_path.exists():
            img_description = self._encode_image(screenshot_path)
            prompt_text = description or "Analyze this screenshot for UI issues"
        else:
            img_description = ""
            prompt_text = description or "No screenshot description provided"

        import json

        prompt = _DIAGNOSE_PROMPT.format(
            context=context or "No additional context",
            description=prompt_text,
        )

        try:
            response = await self._client.call(
                model=None,
                prompt=prompt,
                system="You are a UI/UX debugging expert. Return only valid JSON.",
                max_tokens=1000,
                temperature=0.2,
                timeout=60,
                # If image is available, pass it as a multimodal input
                # (implementation depends on the specific client's multimodal API)
            )
            parsed = self._parse_diagnosis(response.text)
            issues = [
                ScreenshotIssue(
                    severity=i.get("severity", "info"),
                    category=i.get("category", "visual"),
                    element=i.get("element", "unknown"),
                    description=i.get("description", ""),
                    fix_suggestion=i.get("fix_suggestion", ""),
                )
                for i in parsed.get("issues", [])
            ]
            return ScreenshotDiagnosis(
                issues=issues,
                overall_assessment=parsed.get("overall_assessment", "No assessment provided"),
            )
        except Exception as exc:
            logger.warning(f"Screenshot diagnosis failed: {exc}")
            return ScreenshotDiagnosis(
                issues=[],
                overall_assessment=f"Diagnosis failed: {exc}",
            )

    @staticmethod
    def _encode_image(path: Path) -> str:
        """Base64 encode an image for API transport."""
        import base64

        if not path.exists():
            return ""
        ext = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
        }
        mime = mime_map.get(ext, "image/png")
        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        return f"data:{mime};base64,{data}"

    @staticmethod
    def _parse_diagnosis(text: str) -> dict:
        """Parse the diagnosis response from the LLM."""
        import json
        import re

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"issues": [], "overall_assessment": "Unable to parse diagnosis"}

    def to_markdown(self, diagnosis: ScreenshotDiagnosis) -> str:
        """Format a diagnosis as a markdown report."""
        lines = [
            "## Screenshot Diagnosis",
            "",
            f"**Assessment**: {diagnosis.overall_assessment}",
            "",
            f"**Issues found**: {len(diagnosis.issues)} " f"({diagnosis.critical_count} critical)",
            "",
        ]
        for issue in diagnosis.issues:
            emoji = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(issue.severity, "⚪")
            lines.append(f"- {emoji} **[{issue.category}]** {issue.element}")
            lines.append(f"  *{issue.description}*")
            lines.append(f"  Fix: {issue.fix_suggestion}")
            lines.append("")
        return "\n".join(lines)