"""
RegressionDetector — Compare quality reports to detect backsliding
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Pillar 4: Prevents merging code that makes quality worse.
"""

from __future__ import annotations

from ..quality.quality_report import QualityReport


class RegressionDetector:
    """Compares quality reports to detect regressions."""

    def compare(self, before: QualityReport, after: QualityReport) -> bool:
        """Returns True if quality regressed."""
        if after.overall_score < before.overall_score:
            return True
        if after.lint_errors > before.lint_errors:
            return True
        if after.type_errors > before.type_errors:
            return True
        if after.tests_failed > before.tests_failed:
            return True
        if after.security_issues > before.security_issues:
            return True
        return False

    def report(self, before: QualityReport, after: QualityReport) -> str:
        """Generate a human-readable diff of quality changes."""
        lines = ["Quality Change Report:", ""]
        delta_score = after.overall_score - before.overall_score
        icon = "+" if delta_score >= 0 else ""
        lines.append(f"  Score: {icon}{delta_score:.1f}")
        lines.append(f"  Lint errors: {after.lint_errors - before.lint_errors:+d}")
        lines.append(f"  Type errors: {after.type_errors - before.type_errors:+d}")
        lines.append(f"  Tests failed: {after.tests_failed - before.tests_failed:+d}")
        lines.append(f"  Security issues: {after.security_issues - before.security_issues:+d}")
        return "\n".join(lines)

    def is_regression(self, before: QualityReport, after: QualityReport) -> bool:
        """Alias for compare()."""
        return self.compare(before, after)
