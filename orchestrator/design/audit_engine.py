"""
Audit Engine — Structured design audit of existing code.
=======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Reads existing frontend files and scores them against the anti-pattern list.
Returns a ranked punch list. Does not edit.

Usage:
    engine = AuditEngine()
    findings = await engine.audit([Path("app/page.tsx"), Path("app/globals.css")])
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .slop_test import SlopTestEngine

logger = logging.getLogger(__name__)


@dataclass
class AuditFinding:
    """A single audit finding."""

    tell: str
    where: str
    severity: Literal["critical", "major", "minor"]
    fix: str
    gate_number: int = 0


class AuditEngine:
    """Structured design audit engine."""

    def __init__(self) -> None:
        self._slop = SlopTestEngine()

    async def audit(self, files: list[Path]) -> list[AuditFinding]:
        """Audit a list of files and return ranked findings.

        Args:
            files: List of file paths to audit.

        Returns:
            List of AuditFinding, sorted by severity (critical first).
        """
        findings: list[AuditFinding] = []

        for file_path in files:
            try:
                content = file_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                logger.warning("audit: could not read %s: %s", file_path, exc)
                continue

            # Run slop-test gates
            slop_result = self._slop.run(content)
            for finding in slop_result.findings:
                gate = finding.gate
                severity = (
                    "critical"
                    if gate.severity.value == "critical"
                    else "major" if gate.severity.value == "major" else "minor"
                )
                findings.append(
                    AuditFinding(
                        tell=gate.name,
                        where=f"{file_path}:{self._approximate_line(content, gate.number)}",
                        severity=severity,
                        fix=f"See Hallmark slop-test gate {gate.number}: {gate.description}",
                        gate_number=gate.number,
                    )
                )

            # Check structural fingerprint
            findings.extend(self._check_fingerprint(content, file_path))

            # Check design.md compliance
            findings.extend(self._check_design_md(content, file_path))

        # Sort: critical → major → minor
        severity_order = {"critical": 0, "major": 1, "minor": 2}
        findings.sort(key=lambda f: severity_order.get(f.severity, 3))
        return findings

    def _approximate_line(self, content: str, gate_number: int) -> str:
        """Approximate line number for a gate finding."""
        # Simple heuristic: gate number * 10 as rough line estimate
        return f"~{gate_number * 10}"

    def _check_fingerprint(self, content: str, file_path: Path) -> list[AuditFinding]:
        """Check if the page matches a known AI template fingerprint."""
        findings: list[AuditFinding] = []

        # Check for generic AI template: hero → 3 features → CTA → footer
        hero_count = content.lower().count("<header") + content.lower().count('class="hero')
        feature_count = content.lower().count("feature") + content.lower().count("<section")
        cta_count = content.lower().count("cta") + content.lower().count('class="cta')

        if hero_count > 0 and feature_count >= 3 and cta_count > 0:
            # Check if it has structural variety
            has_asymmetry = "grid-template-columns" in content and "1fr 2fr" in content
            has_varied_sections = (
                len(set(re.findall(r'<section[^>]*class="([^"]*)"', content))) >= 4
            )

            if not has_asymmetry and not has_varied_sections:
                findings.append(
                    AuditFinding(
                        tell="AI template fingerprint",
                        where=str(file_path),
                        severity="critical",
                        fix="Break the template: vary column widths, move headings off-centre, use asymmetric spans",
                    )
                )

        return findings

    def _check_design_md(self, content: str, file_path: Path) -> list[AuditFinding]:
        """Check compliance with design.md if one exists."""
        findings: list[AuditFinding] = []

        # Check for design.md in project root
        design_md = file_path.parent / "design.md"
        if not design_md.exists():
            design_md = file_path.parent / "DESIGN.md"

        if design_md.exists():
            try:
                design_content = design_md.read_text(encoding="utf-8")
                # Check theme drift
                if "theme:" in design_content:
                    design_theme = design_content.split("theme:")[1].split("\n")[0].strip()
                    if design_theme.lower() not in content.lower():
                        findings.append(
                            AuditFinding(
                                tell="Theme drift",
                                where=str(file_path),
                                severity="critical",
                                fix=f"Apply design.md theme '{design_theme}' tokens to this file",
                            )
                        )
            except (OSError, UnicodeDecodeError):
                pass

        return findings

    def format_report(self, findings: list[AuditFinding]) -> str:
        """Format findings into a human-readable audit report."""
        critical = [f for f in findings if f.severity == "critical"]
        major = [f for f in findings if f.severity == "major"]
        minor = [f for f in findings if f.severity == "minor"]

        lines = [
            "# Hallmark Design Audit",
            "",
            f"**{len(critical)} critical · {len(major)} major · {len(minor)} minor**",
            "",
        ]

        if critical:
            lines.append("## Critical (ships as slop)")
            for f in critical:
                lines.append(f"- **{f.tell}** at `{f.where}`")
                lines.append(f"  - Fix: {f.fix}")
            lines.append("")

        if major:
            lines.append("## Major (looks AI-generated)")
            for f in major:
                lines.append(f"- **{f.tell}** at `{f.where}`")
                lines.append(f"  - Fix: {f.fix}")
            lines.append("")

        if minor:
            lines.append("## Minor (small taste issue)")
            for f in minor:
                lines.append(f"- **{f.tell}** at `{f.where}`")
                lines.append(f"  - Fix: {f.fix}")
            lines.append("")

        lines.append(f"---\n**Total: {len(findings)} findings**")
        return "\n".join(lines)
