"""
SecurityReview — Structured AI security auditing.
====================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of Category 6, Phase D2 (Dyad-inspired): Severity-level security review
with CWE IDs, SECURITY_RULES.md integration, and structured findings.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    CRITICAL = "critical"  # Data breach, auth bypass
    HIGH = "high"  # Sensitive data exposure, injection
    MEDIUM = "medium"  # Misconfiguration, missing hardening
    LOW = "low"  # Best practice violations
    INFO = "info"  # Informational


class Category(str, Enum):
    AUTH = "authentication"
    AUTHZ = "authorization"
    INJECTION = "injection"
    DATA_EXPOSURE = "data_exposure"
    CRYPTO = "cryptography"
    CONFIG = "configuration"
    DEPENDENCY = "dependency"
    INPUT_VALIDATION = "input_validation"
    LOGGING = "logging"
    SECRETS = "secrets_management"


@dataclass
class SecurityRule:
    """A single security rule from SECURITY_RULES.md."""

    rule_id: str
    title: str
    description: str
    severity: Severity = Severity.MEDIUM
    category: Category = Category.CONFIG
    cwe_id: str = ""
    remediation: str = ""

    def to_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "severity": self.severity.value,
            "category": self.category.value,
            "cwe_id": self.cwe_id,
        }


@dataclass
class SecurityFinding:
    """A single finding from a security review."""

    rule_id: str
    title: str
    severity: Severity
    category: Category
    description: str
    location: str = ""  # File:line
    cwe_id: str = ""
    remediation: str = ""
    fixed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "title": self.title,
            "severity": self.severity.value,
            "category": self.category.value,
            "description": self.description,
            "location": self.location,
            "cwe_id": self.cwe_id,
            "remediation": self.remediation,
            "fixed": self.fixed,
        }


@dataclass
class SecurityReport:
    """Complete security review report."""

    findings: list[SecurityFinding] = field(default_factory=list)
    rules_checked: int = 0
    total_findings: int = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0

    @property
    def passed(self) -> bool:
        return self.critical_count == 0 and self.high_count == 0

    @property
    def summary(self) -> str:
        parts = [
            f"Security Review: {self.total_findings} findings "
            f"({self.critical_count}C/{self.high_count}H/{self.medium_count}M/{self.low_count}L)",
        ]
        if self.passed:
            parts.append("Status: PASSED")
        else:
            parts.append("Status: FAILED — critical or high severity issues found")
        return "\n".join(parts)

    def to_markdown(self) -> str:
        lines = ["# Security Review Report", "", self.summary, ""]
        for sev in (Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO):
            sev_findings = [f for f in self.findings if f.severity == sev]
            if not sev_findings:
                continue
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🔵", "info": "⚪"}
            lines.append(f"## {emoji.get(sev.value, '')} {sev.value.upper()} ({len(sev_findings)})")
            lines.append("")
            for f in sev_findings:
                lines.append(f"### {f.title}")
                lines.append(f"- **Severity**: {f.severity.value}")
                lines.append(f"- **Category**: {f.category.value}")
                if f.cwe_id:
                    lines.append(
                        f"- **CWE**: [{f.cwe_id}](https://cwe.mitre.org/data/definitions/{f.cwe_id.replace('CWE-','')}.html)"
                    )
                if f.location:
                    lines.append(f"- **Location**: `{f.location}`")
                lines.append(f"- **Description**: {f.description}")
                if f.remediation:
                    lines.append(f"- **Fix**: {f.remediation}")
                lines.append("")
        return "\n".join(lines)


# Default security rules (CWE-mapped)
DEFAULT_SECURITY_RULES: list[SecurityRule] = [
    SecurityRule(
        "SEC-001",
        "Hardcoded API Keys",
        "API keys or tokens found in source code",
        Severity.CRITICAL,
        Category.SECRETS,
        "CWE-798",
    ),
    SecurityRule(
        "SEC-002",
        "SQL Injection",
        "Unparameterized SQL queries",
        Severity.CRITICAL,
        Category.INJECTION,
        "CWE-89",
    ),
    SecurityRule(
        "SEC-003",
        "Missing Input Validation",
        "User input used without sanitization",
        Severity.HIGH,
        Category.INPUT_VALIDATION,
        "CWE-20",
    ),
    SecurityRule(
        "SEC-004",
        "Insecure Deserialization",
        "Unsafe pickle/yaml.load usage",
        Severity.HIGH,
        Category.INJECTION,
        "CWE-502",
    ),
    SecurityRule(
        "SEC-005",
        "Weak Cryptography",
        "MD5/SHA1 for passwords, weak RNG",
        Severity.HIGH,
        Category.CRYPTO,
        "CWE-327",
    ),
    SecurityRule(
        "SEC-006",
        "Missing HTTPS Enforcement",
        "No TLS/SSL configuration",
        Severity.HIGH,
        Category.CONFIG,
        "CWE-319",
    ),
    SecurityRule(
        "SEC-007",
        "Debug Mode Enabled",
        "DEBUG=True in production config",
        Severity.MEDIUM,
        Category.CONFIG,
        "CWE-489",
    ),
    SecurityRule(
        "SEC-008",
        "Excessive Logging",
        "PII/sensitive data in log statements",
        Severity.MEDIUM,
        Category.LOGGING,
        "CWE-532",
    ),
    SecurityRule(
        "SEC-009",
        "Outdated Dependency",
        "Dependency with known CVE",
        Severity.MEDIUM,
        Category.DEPENDENCY,
        "CWE-1104",
    ),
    SecurityRule(
        "SEC-010",
        "Missing Rate Limiting",
        "No request rate limiting",
        Severity.MEDIUM,
        Category.CONFIG,
        "CWE-770",
    ),
    SecurityRule(
        "SEC-011",
        "Path Traversal",
        "Unsanitized file paths from user input",
        Severity.HIGH,
        Category.INPUT_VALIDATION,
        "CWE-22",
    ),
    SecurityRule(
        "SEC-012",
        "Missing Auth on Endpoint",
        "API endpoint without authentication",
        Severity.CRITICAL,
        Category.AUTH,
        "CWE-306",
    ),
    SecurityRule(
        "SEC-013",
        "CORS Misconfiguration",
        "Wildcard CORS origin",
        Severity.LOW,
        Category.CONFIG,
        "CWE-942",
    ),
    SecurityRule(
        "SEC-014",
        "Sensitive Data in Error Messages",
        "Stack traces exposed to users",
        Severity.MEDIUM,
        Category.DATA_EXPOSURE,
        "CWE-209",
    ),
    SecurityRule(
        "SEC-015",
        "Missing Content Security Policy",
        "No CSP header configured",
        Severity.LOW,
        Category.CONFIG,
        "CWE-1021",
    ),
]


class SecurityReviewer:
    """Runs structured security reviews against generated code.

    Supports pattern-based scanning and LLM-based review for deeper analysis.
    """

    def __init__(self, rules: list[SecurityRule] | None = None):
        self._rules = rules or DEFAULT_SECURITY_RULES

    def quick_scan(self, code: str, source_file: str = "") -> SecurityReport:
        """Fast pattern-based security scan (no LLM).

        Scans for common security anti-patterns in source code.
        """
        findings: list[SecurityFinding] = []
        patterns = {
            "SEC-001": [
                (
                    r"(?:api[_-]?key|secret|password|token)\s*=\s*[\"'][^\"']{8,}[\"']",
                    Severity.CRITICAL,
                ),
                (r"(?:sk-|AIza)[a-zA-Z0-9_-]{20,}", Severity.CRITICAL),
            ],
            "SEC-002": [
                (
                    r"(?:execute|cursor\.execute)\s*\(\s*(?:f[\"']|[\"'].*?%.*?[\"'])",
                    Severity.CRITICAL,
                ),
            ],
            "SEC-003": [
                (r"eval\s*\(\s*(?:request|input|user)", Severity.HIGH),
            ],
            "SEC-004": [
                (r"pickle\.loads?\s*\(", Severity.HIGH),
                (r"yaml\.load\s*\([^s]", Severity.HIGH),
            ],
            "SEC-005": [
                (r"hashlib\.md5\s*\(", Severity.HIGH),
                (r"random\.(?:random|randint)\s*\(", Severity.HIGH),
            ],
            "SEC-007": [
                (r"DEBUG\s*=\s*True", Severity.MEDIUM),
            ],
            "SEC-011": [
                (r"os\.path\.join\s*\(.*?(?:request|input|user|argv)", Severity.HIGH),
            ],
            "SEC-013": [
                (r"Access-Control-Allow-Origin.*?\*", Severity.LOW),
            ],
        }

        import re

        for rule in self._rules:
            pats = patterns.get(rule.rule_id, [])
            for pattern, sev in pats:
                matches = re.finditer(pattern, code, re.IGNORECASE)
                for match in matches:
                    line_no = code[: match.start()].count("\n") + 1
                    location = f"{source_file}:{line_no}" if source_file else f"line {line_no}"
                    findings.append(
                        SecurityFinding(
                            rule_id=rule.rule_id,
                            title=rule.title,
                            severity=sev,
                            category=rule.category,
                            description=f"Matched pattern: {match.group()[:80]}",
                            location=location,
                            cwe_id=rule.cwe_id,
                            remediation=rule.remediation,
                        )
                    )

        return self._build_report(findings)

    async def llm_review(self, code: str, source_file: str = "", client=None) -> SecurityReport:
        """LLM-based deep security review.

        Args:
            code: Source code to review
            source_file: File path for locations
            client: LLM client for analysis

        Returns:
            SecurityReport with LLM-identified findings
        """
        if not client:
            return self.quick_scan(code, source_file)

        rules_text = "\n".join(
            f"- {r.rule_id}: {r.title} ({r.cwe_id})"
            for r in self._rules[:8]  # Top 8 rules to keep prompt small
        )

        prompt = f"""Review this code for security vulnerabilities. Check against these rules:

{rules_text}

Code to review (from {source_file or 'unknown'}):
```python
{code[:4000]}
```

For each vulnerability found, return exactly this JSON format:
{{"findings": [
    {{"rule_id": "SEC-XXX", "title": "short title", "severity": "critical|high|medium|low",
     "category": "auth|injection|config|etc", "description": "what is wrong",
     "location": "filename:line", "cwe_id": "CWE-XXX", "remediation": "how to fix"}}
]}}

If no vulnerabilities found, return {{"findings": []}}."""

        try:
            response = await client.call(
                model=None,
                prompt=prompt,
                system="You are a senior security auditor. Return only valid JSON.",
                max_tokens=1500,
                temperature=0.1,
                timeout=60,
            )
            parsed = self._parse_response(response.text)
            findings = [
                SecurityFinding(
                    rule_id=f.get("rule_id", "SEC-UNKNOWN"),
                    title=f.get("title", "Untitled"),
                    severity=Severity(f.get("severity", "medium")),
                    category=Category(f.get("category", "config")),
                    description=f.get("description", ""),
                    location=f.get("location", source_file),
                    cwe_id=f.get("cwe_id", ""),
                    remediation=f.get("remediation", ""),
                )
                for f in parsed.get("findings", [])
            ]
        except Exception as e:
            logger.warning(f"LLM security review failed: {e}")
            findings = []

        return self._build_report(findings)

    def _parse_response(self, text: str) -> dict:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"findings": []}

    def _build_report(self, findings: list[SecurityFinding]) -> SecurityReport:
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for f in findings:
            counts[f.severity.value] = counts.get(f.severity.value, 0) + 1
        return SecurityReport(
            findings=findings,
            rules_checked=len(self._rules),
            total_findings=len(findings),
            critical_count=counts.get("critical", 0),
            high_count=counts.get("high", 0),
            medium_count=counts.get("medium", 0),
            low_count=counts.get("low", 0),
        )
