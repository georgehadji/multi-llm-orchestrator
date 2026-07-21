"""
Tests for orchestrator/security_review.py — Security scanning.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit

from orchestrator.safety.security_review import (
    SecurityReviewer,
    SecurityReport,
    SecurityFinding,
    SecurityRule,
    Severity,
    Category,
    DEFAULT_SECURITY_RULES,
)


class TestSecurityReviewer:
    """Tests for SecurityReviewer."""

    def test_rules_loaded(self):
        """Default rules must be non-empty."""
        assert len(DEFAULT_SECURITY_RULES) >= 10

    # ── Quick Scan Detection ──
    @pytest.mark.parametrize(
        "code,expected_rule",
        [
            ("api_key = 'sk-abc123def456'", "SEC-001"),
            ("password = '12345678'", "SEC-001"),
            ("hashlib.md5(b'data')", "SEC-005"),
            ("DEBUG = True", "SEC-007"),
            ("pickle.loads(data)", "SEC-004"),
        ],
    )
    def test_detection_parametrized(self, code, expected_rule):
        """Quick scan must detect known anti-patterns."""
        sr = SecurityReviewer()
        report = sr.quick_scan(code)
        matching = [f for f in report.findings if f.rule_id == expected_rule]
        assert (
            len(matching) > 0
        ), f"Expected {expected_rule}, got {[f.rule_id for f in report.findings]}"

    # ── Clean Code Passes ──
    @pytest.mark.parametrize(
        "code",
        [
            "def add(a, b): return a + b",
            "import os; x = os.environ.get('VAR')",
            "class User: pass",
        ],
    )
    def test_clean_code_passes(self, code):
        """Clean code must produce zero findings."""
        sr = SecurityReviewer()
        report = sr.quick_scan(code)
        assert report.total_findings == 0
        assert report.passed

    # ── Report Properties ──
    def test_report_passed_no_critical(self):
        """Report must pass when no critical/high findings."""
        report = SecurityReport(
            total_findings=1,
            low_count=1,
            findings=[
                SecurityFinding("SEC-013", "CORS", Severity.LOW, Category.CONFIG, "Wildcard CORS")
            ],
        )
        assert report.passed

    def test_report_failed_with_critical(self):
        """Report must fail when critical findings exist."""
        report = SecurityReport(
            total_findings=1,
            critical_count=1,
            findings=[
                SecurityFinding(
                    "SEC-001", "API Key", Severity.CRITICAL, Category.SECRETS, "Hardcoded key"
                )
            ],
        )
        assert not report.passed

    def test_markdown_generation(self):
        """Markdown report must include severity and CWE."""
        finding = SecurityFinding(
            "SEC-001",
            "Hardcoded Key",
            Severity.CRITICAL,
            Category.SECRETS,
            "Found key",
            "config.py:5",
            "CWE-798",
            "Use env var",
        )
        report = SecurityReport(findings=[finding], total_findings=1, critical_count=1)
        md = report.to_markdown()
        assert "CRITICAL" in md
        assert "CWE-798" in md
        assert "config.py:5" in md

    # ── Edge Cases ──
    def test_empty_code(self):
        """Empty code must produce no findings."""
        sr = SecurityReviewer()
        report = sr.quick_scan("")
        assert report.total_findings == 0

    def test_custom_rules(self):
        """Must accept custom rules."""
        sr = SecurityReviewer(
            rules=[
                SecurityRule(
                    "CUSTOM-1",
                    "Custom check",
                    "Custom description",
                    Severity.HIGH,
                    Category.CONFIG,
                    "CWE-999",
                ),
            ]
        )
        report = sr.quick_scan("def foo(): pass")
        assert report.total_findings == 0
