"""
Security Validator — Functional Security Scanning
==================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Functional security validation with immutable findings.
Extends existing analyzer.py security scoring.

Paradigm: Functional Programming (pure functions, immutable data)

Usage:
    from orchestrator.security_validator import validate_security, SecurityFinding

    findings = validate_security(code, file_path)
    score = calculate_security_score(findings)
    is_ready = is_production_ready(findings)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SecuritySeverity(str, Enum):
    """Security violation severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class SecurityCategory(str, Enum):
    """OWASP security categories."""

    SECRETS = "secrets"
    INJECTION = "injection"
    XSS = "xss"
    CSRF = "csrf"
    AUTH = "authentication"
    SESSION = "session"
    HEADERS = "headers"
    CRYPTO = "cryptography"
    DEPENDENCIES = "dependencies"
    CONFIG = "configuration"


@dataclass(frozen=True)  # Immutable
class SecurityFinding:
    """
    Immutable security finding.

    Attributes:
        id: Unique finding identifier
        severity: SecuritySeverity (CRITICAL/HIGH/MEDIUM/LOW/INFO)
        category: OWASP security category
        location: File path with line number
        description: Human-readable description
        recommendation: Remediation guidance
        cwe_id: Common Weakness Enumeration ID
        code_snippet: Optional code snippet showing issue
    """

    id: str
    severity: SecuritySeverity
    category: SecurityCategory
    location: str
    description: str
    recommendation: str
    cwe_id: str
    code_snippet: str = ""

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "severity": self.severity.value,
            "category": self.category.value,
            "location": self.location,
            "description": self.description,
            "recommendation": self.recommendation,
            "cwe_id": self.cwe_id,
            "code_snippet": self.code_snippet,
        }


@dataclass(frozen=True)
class SecurityReport:
    """
    Immutable security audit report.

    Attributes:
        file_path: Audited file path
        findings: List of security findings
        score: Security score (0-100)
        is_production_ready: Boolean production readiness
        timestamp: Audit timestamp
    """

    file_path: str
    findings: tuple[SecurityFinding, ...]  # Immutable tuple
    score: float
    is_production_ready: bool
    timestamp: str

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "file_path": self.file_path,
            "findings": [f.to_dict() for f in self.findings],
            "score": self.score,
            "is_production_ready": self.is_production_ready,
            "timestamp": self.timestamp,
        }


# ═══════════════════════════════════════════════════════
# SECURITY CHECK FUNCTIONS (Pure Functions)
# ═══════════════════════════════════════════════════════


def check_hardcoded_secrets(code: str, file_path: str) -> list[SecurityFinding]:
    """
    Check for hardcoded secrets (pure function).

    Detects: API keys, passwords, tokens, private keys

    Args:
        code: Source code to check
        file_path: File path for location reporting

    Returns:
        List of security findings
    """
    findings = []

    patterns = {
        "API_KEY": (
            r"(?i)(api[_-]?key|apikey|api_secret)\s*[=:]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]",
            "CWE-798",
        ),
        "PASSWORD": (r"(?i)(password|passwd|pwd|pass)\s*[=:]\s*['\"][^'\"]{4,}['\"]", "CWE-257"),
        "SECRET": (
            r"(?i)(secret|secret_key|private_key)\s*[=:]\s*['\"][a-zA-Z0-9_\-]{16,}['\"]",
            "CWE-798",
        ),
        "TOKEN": (
            r"(?i)(auth_token|access_token|bearer)\s*[=:]\s*['\"][a-zA-Z0-9_\-\.]{20,}['\"]",
            "CWE-798",
        ),
        "PRIVATE_KEY": (r"-----BEGIN (RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----", "CWE-321"),
    }

    line_number = 1
    for line in code.split("\n"):
        # Skip comments
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            line_number += 1
            continue

        for finding_type, (pattern, cwe_id) in patterns.items():
            if re.search(pattern, line):
                findings.append(
                    SecurityFinding(
                        id=f"HARDCODED_{finding_type}",
                        severity=SecuritySeverity.CRITICAL,
                        category=SecurityCategory.SECRETS,
                        location=f"{file_path}:{line_number}",
                        description=f"Hardcoded {finding_type.replace('_', ' ').title()} detected",
                        recommendation="Use environment variables (os.getenv()) or secrets manager",
                        cwe_id=cwe_id,
                        code_snippet=line.strip()[:100],
                    )
                )
        line_number += 1

    return findings


def check_sql_injection(code: str, file_path: str) -> list[SecurityFinding]:
    """
    Check for SQL injection vulnerabilities (pure function).

    Detects: String concatenation in SQL queries

    Args:
        code: Source code to check
        file_path: File path for location reporting

    Returns:
        List of security findings
    """
    findings = []

    # Patterns indicating SQL injection risk
    patterns = [
        # Python string formatting in SQL
        (r"execute\s*\(\s*['\"].*%s.*['\"]\s*%", "Python % formatting in SQL"),
        (r"cursor\.execute\s*\([^,]+,\s*\[\]\)", "Empty parameters in execute()"),
        # String concatenation
        (r"(execute|query|raw|cursor\.execute)\s*\(\s*[^)]*\+", "String concatenation in SQL"),
        (r"(execute|query|raw)\s*\(\s*f['\"][^)]*\{", "f-string in SQL query"),
        (r"(execute|query|raw)\s*\(\s*[^)]*\.format\s*\(", ".format() in SQL query"),
        # Direct SQL in code (potential risk)
        (r"SELECT \* FROM \w+ WHERE \w+ = ['\"]", "Hardcoded SQL with literal values"),
    ]

    for pattern, description in patterns:
        if re.search(pattern, code, re.IGNORECASE):
            findings.append(
                SecurityFinding(
                    id="SQL_INJECTION_RISK",
                    severity=SecuritySeverity.CRITICAL,
                    category=SecurityCategory.INJECTION,
                    location=file_path,
                    description=f"Potential SQL injection: {description}",
                    recommendation="Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = %s', (user_id,))",
                    cwe_id="CWE-89",
                )
            )

    return findings


def check_xss(code: str, file_path: str) -> list[SecurityFinding]:
    """
    Check for XSS vulnerabilities (pure function).

    Detects: Unescaped user input in HTML

    Args:
        code: Source code to check
        file_path: File path for location reporting

    Returns:
        List of security findings
    """
    findings = []

    # Patterns for unescaped HTML output
    patterns = [
        # React
        (r"dangerouslySetInnerHTML", "React dangerouslySetInnerHTML"),
        # Vue
        (r"v-html\s*=", "Vue v-html directive"),
        # Vanilla JS
        (r"\.innerHTML\s*=", "Direct innerHTML assignment"),
        (r"\.outerHTML\s*=", "Direct outerHTML assignment"),
        # Angular
        (r"\[innerHTML\]\s*=", "Angular [innerHTML] binding"),
        (r"\{\{.*\|.*async\s*\}\}", "Angular async pipe without sanitization"),
        # Django
        (r"\{\% autoescape off \%\}", "Django autoescape disabled"),
        (r"\|safe", "Django |safe filter"),
    ]

    for pattern, description in patterns:
        if re.search(pattern, code, re.IGNORECASE):
            findings.append(
                SecurityFinding(
                    id="XSS_RISK",
                    severity=SecuritySeverity.HIGH,
                    category=SecurityCategory.XSS,
                    location=file_path,
                    description=f"Potential XSS vulnerability: {description}",
                    recommendation="Use escaped output or sanitize input with DOMPurify/bleach",
                    cwe_id="CWE-79",
                )
            )

    return findings


def check_security_headers(code: str, file_path: str) -> list[SecurityFinding]:
    """
    Check for missing security headers (pure function).

    Detects: Missing CSP, HSTS, X-Frame-Options, etc.
    Applies to any file with a common backend/web-adjacent extension (.py,
    .js, .ts, .go, .java, .rb, .php) of at least 50 non-whitespace
    characters -- this is a coarse extension-based heuristic, not a true
    backend/server-code classifier, so it will also flag non-HTTP files
    (data models, CLI scripts, utilities) that happen to share one of
    those extensions.

    Args:
        code: Source code to check
        file_path: File path for location reporting

    Returns:
        List of security findings
    """
    findings = []

    # Only check backend/server code files
    if not any(
        ext in file_path.lower() for ext in [".py", ".js", ".ts", ".go", ".java", ".rb", ".php"]
    ):
        return findings

    # Skip empty or very short files
    if len(code.strip()) < 50:
        return findings

    required_headers = {
        "Content-Security-Policy": (
            r"Content-Security-Policy",
            "Missing CSP header - allows XSS attacks",
        ),
        "Strict-Transport-Security": (
            r"Strict-Transport-Security",
            "Missing HSTS header - vulnerable to downgrade attacks",
        ),
        "X-Frame-Options": (
            r"X-Frame-Options",
            "Missing X-Frame-Options - vulnerable to clickjacking",
        ),
        "X-Content-Type-Options": (
            r"X-Content-Type-Options",
            "Missing X-Content-Type-Options - MIME sniffing risk",
        ),
    }

    for header, (pattern, description) in required_headers.items():
        if not re.search(pattern, code, re.IGNORECASE):
            findings.append(
                SecurityFinding(
                    id=f"MISSING_{header.replace('-', '_').upper()}",
                    severity=SecuritySeverity.MEDIUM,
                    category=SecurityCategory.HEADERS,
                    location=file_path,
                    description=description,
                    recommendation=f"Add header: {header}: <appropriate-value>",
                    cwe_id="CWE-693",
                )
            )

    return findings


def check_insecure_crypto(code: str, file_path: str) -> list[SecurityFinding]:
    """
    Check for insecure cryptographic algorithms (pure function).

    Detects: MD5, SHA1, DES, RC4

    Args:
        code: Source code to check
        file_path: File path for location reporting

    Returns:
        List of security findings
    """
    findings = []

    insecure_algorithms = {
        "MD5": (r"\bmd5\b", "MD5 is cryptographically broken"),
        "SHA1": (r"\bsha1\b", "SHA1 is deprecated for security use"),
        "DES": (r"\bDES\b", "DES has insufficient key length"),
        "RC4": (r"\bRC4\b", "RC4 has multiple vulnerabilities"),
        "MD4": (r"\bmd4\b", "MD4 is cryptographically broken"),
    }

    for algo, (pattern, description) in insecure_algorithms.items():
        if re.search(pattern, code, re.IGNORECASE):
            # Skip if it's in a comment or for non-security use
            if not re.search(r"#.*" + pattern, code, re.IGNORECASE):
                findings.append(
                    SecurityFinding(
                        id=f"INSECURE_CRYPTO_{algo}",
                        severity=SecuritySeverity.HIGH,
                        category=SecurityCategory.CRYPTO,
                        location=file_path,
                        description=f"Insecure algorithm: {description}",
                        recommendation=f"Use SHA-256, SHA-3, or Argon2 instead of {algo}",
                        cwe_id="CWE-327",
                    )
                )

    return findings


# ═══════════════════════════════════════════════════════
# MAIN VALIDATION FUNCTION (Pure Function Pipeline)
# ═══════════════════════════════════════════════════════


def validate_security(code: str, file_path: str) -> SecurityReport:
    """
    Validate code for security issues (main pipeline).

    Runs all security checks and returns comprehensive report.

    Args:
        code: Source code to validate
        file_path: File path for location reporting

    Returns:
        Immutable SecurityReport
    """
    from datetime import datetime, timezone

    # Run all checks (function pipeline)
    all_findings: list[SecurityFinding] = []
    all_findings.extend(check_hardcoded_secrets(code, file_path))
    all_findings.extend(check_sql_injection(code, file_path))
    all_findings.extend(check_xss(code, file_path))
    all_findings.extend(check_security_headers(code, file_path))
    all_findings.extend(check_insecure_crypto(code, file_path))

    # Calculate score
    score = calculate_security_score(all_findings)

    # Determine production readiness
    is_ready = is_production_ready(all_findings)

    return SecurityReport(
        file_path=file_path,
        findings=tuple(all_findings),  # Immutable
        score=score,
        is_production_ready=is_ready,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def calculate_security_score(findings: list[SecurityFinding]) -> float:
    """
    Calculate security score from findings (pure function).

    Scoring:
    - Start: 100
    - CRITICAL: -25 points each
    - HIGH: -15 points each
    - MEDIUM: -8 points each
    - LOW: -3 points each
    - INFO: -1 point each

    Args:
        findings: List of security findings

    Returns:
        Score 0-100
    """
    if not findings:
        return 100.0

    severity_weights = {
        SecuritySeverity.CRITICAL: 25,
        SecuritySeverity.HIGH: 15,
        SecuritySeverity.MEDIUM: 8,
        SecuritySeverity.LOW: 3,
        SecuritySeverity.INFO: 1,
    }

    total_deductions = sum(severity_weights.get(finding.severity, 0) for finding in findings)

    return max(0.0, min(100.0, 100.0 - total_deductions))


def is_production_ready(findings: list[SecurityFinding]) -> bool:
    """
    Check if code is production-ready (pure function).

    Production-ready = No CRITICAL or HIGH severity findings

    Args:
        findings: List of security findings

    Returns:
        True if production-ready, False otherwise
    """
    critical_or_high = {
        SecuritySeverity.CRITICAL,
        SecuritySeverity.HIGH,
    }

    return not any(finding.severity in critical_or_high for finding in findings)


# ═══════════════════════════════════════════════════════
# SECURITY TEMPLATES (Constants)
# ═══════════════════════════════════════════════════════

SECURITY_HEADERS_TEMPLATE = """
# Security Headers Configuration
# Add to your web server or middleware

Content-Security-Policy: default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'
Strict-Transport-Security: max-age=31536000; includeSubDomains; preload
X-Frame-Options: DENY
X-Content-Type-Options: nosniff
X-XSS-Protection: 1; mode=block
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy: geolocation=(), microphone=(), camera=()
"""

JWT_CONFIG_TEMPLATE = """
# JWT Configuration (add to .env)
# NEVER commit actual values to Git

JWT_SECRET=your-super-secret-key-min-32-chars
JWT_EXPIRY=15m
JWT_REFRESH_EXPIRY=7d
JWT_ALGORITHM=HS256
JWT_ISSUER=your-app-name
"""

RATE_LIMIT_TEMPLATE = """
# Rate Limiting Configuration
# Add to your API middleware

RATE_LIMIT_WINDOW=60  # seconds
RATE_LIMIT_MAX_REQUESTS=100  # per window per IP
RATE_LIMIT_AUTH=5  # auth endpoints (stricter)
"""
