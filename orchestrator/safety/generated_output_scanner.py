"""
Generated-Output Security Scanner
==================================
Defense-in-depth gate that scans code produced by the orchestrator's generators
(apps, SaaS, micro-SaaS, websites) for hardcoded secrets and common insecure
patterns BEFORE the output is delivered to the user.

This is intentionally dependency-free (pure regex + stdlib) so it can run in the
delivery pipeline without pulling in bandit/semgrep. It is a fast first line of
defense, not a replacement for full SAST.

Usage:
    from orchestrator.safety.generated_output_scanner import scan_output_dir

    report = scan_output_dir(Path("outputs/my_app"))
    if report.has_blocking_findings:
        ...  # surface to the user / fail delivery
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

# Files/dirs we never scan (build artifacts, deps, vcs, lockfiles).
_SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
        "target",
        ".pytest_cache",
        ".mypy_cache",
    }
)

# Only scan text source files; skip binaries and large data files.
_SCAN_SUFFIXES = frozenset(
    {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".vue",
        ".svelte",
        ".go",
        ".rs",
        ".java",
        ".rb",
        ".php",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".env",
        ".sh",
        ".html",
        ".sql",
    }
)

# .env.example / template files are *meant* to contain empty/placeholder keys.
_EXAMPLE_NAME_RE = re.compile(r"(?i)(\.example$|\.sample$|\.template$|\.dist$)")

_MAX_FILE_BYTES = 1_000_000  # skip files larger than 1MB

Severity = str  # "CRITICAL" | "HIGH" | "MEDIUM"


@dataclass(frozen=True)
class Finding:
    """A single security finding in a generated file."""

    path: str
    line: int
    severity: Severity
    rule: str
    detail: str


@dataclass
class ScanReport:
    """Aggregate result of scanning a generated-output directory."""

    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    files_skipped: int = 0

    @property
    def has_blocking_findings(self) -> bool:
        """True if any CRITICAL or HIGH finding exists (delivery should warn/fail)."""
        return any(f.severity in ("CRITICAL", "HIGH") for f in self.findings)

    def to_dict(self) -> dict:
        return {
            "files_scanned": self.files_scanned,
            "files_skipped": self.files_skipped,
            "total_findings": len(self.findings),
            "blocking": self.has_blocking_findings,
            "by_severity": {
                sev: sum(1 for f in self.findings if f.severity == sev)
                for sev in ("CRITICAL", "HIGH", "MEDIUM")
            },
            "findings": [
                {
                    "path": f.path,
                    "line": f.line,
                    "severity": f.severity,
                    "rule": f.rule,
                    "detail": f.detail,
                }
                for f in self.findings
            ],
        }


# ── Detection rules ──
# Each rule: (name, severity, compiled regex, human detail).
# Patterns aim for high precision (assigned non-placeholder values) to avoid
# drowning the user in false positives.

_PLACEHOLDER_RE = re.compile(
    r"(?i)(your[_-]?|xxx|placeholder|example|changeme|dummy|<.*>|\$\{|\{\{|todo|replace)"
)


def _is_placeholder(value: str) -> bool:
    v = value.strip().strip("\"'")
    if len(v) < 3:
        return True
    return bool(_PLACEHOLDER_RE.search(v))


_RULES: list[tuple[str, Severity, re.Pattern, str]] = [
    (
        "aws-access-key",
        "CRITICAL",
        re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
        "Hardcoded AWS access key ID",
    ),
    (
        "private-key-block",
        "CRITICAL",
        re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----"),
        "Embedded private key material",
    ),
    (
        "openai-key",
        "CRITICAL",
        re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
        "Hardcoded OpenAI-style API key",
    ),
    (
        "anthropic-key",
        "CRITICAL",
        re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b"),
        "Hardcoded Anthropic API key",
    ),
    (
        "github-token",
        "CRITICAL",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
        "Hardcoded GitHub token",
    ),
    (
        "slack-token",
        "HIGH",
        re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
        "Hardcoded Slack token",
    ),
    (
        "google-api-key",
        "HIGH",
        re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
        "Hardcoded Google API key",
    ),
]

# Assignment-style secret: `password = "literal"`, `SECRET: 'literal'`, etc.
_ASSIGN_SECRET_RE = re.compile(
    r"(?i)\b(password|passwd|secret|api[_-]?key|access[_-]?token|"
    r"private[_-]?key|client[_-]?secret|jwt[_-]?secret|db[_-]?password)\b"
    r"\s*[:=]\s*(\"[^\"]+\"|'[^']+')"
)

# Insecure-pattern rules (not secrets, but unsafe code shipped to users).
_INSECURE_PATTERNS: list[tuple[str, Severity, re.Pattern, str]] = [
    (
        "cors-wildcard-credentials",
        "HIGH",
        re.compile(r"allow_origins\s*=\s*\[\s*[\"']\*[\"']\s*\]"),
        "CORS allow_origins=['*'] (wildcard) — restrict to an explicit allowlist",
    ),
    (
        "flask-debug-true",
        "HIGH",
        re.compile(r"\.run\([^)]*\bdebug\s*=\s*True"),
        "Flask/web server started with debug=True — disable in production",
    ),
    (
        "eval-call",
        "HIGH",
        re.compile(r"(?<![A-Za-z_.])eval\s*\("),
        "Use of eval() — avoid executing dynamic strings",
    ),
    (
        "bind-all-interfaces",
        "MEDIUM",
        re.compile(r"[\"']0\.0\.0\.0[\"']"),
        "Server binds 0.0.0.0 (all interfaces) — prefer 127.0.0.1 unless behind a proxy",
    ),
    (
        "verify-ssl-false",
        "HIGH",
        re.compile(r"\bverify\s*=\s*False"),
        "TLS verification disabled (verify=False) — do not ship this",
    ),
]


# Dotenv-style files (".env", ".env.local", ".env.production", ...) have no
# usable Path.suffix — a bare ".env" yields suffix == "" and ".env.local"
# yields suffix == ".local" (Python only returns the *last* dot-segment, and
# never treats a leading dot as introducing one), so neither ever matches the
# literal ".env" entry in _SCAN_SUFFIXES. Match by name for this convention.
_DOTENV_NAME_RE = re.compile(r"(?i)^\.env(\..+)?$")


def _iter_scannable_files(root: Path):
    """Yield text source files under root, skipping vendored/build/binary paths."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() not in _SCAN_SUFFIXES and not _DOTENV_NAME_RE.match(path.name):
            continue
        try:
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _scan_text(rel_path: str, is_example: bool, text: str) -> list[Finding]:
    findings: list[Finding] = []
    lines = text.splitlines()
    for lineno, line in enumerate(lines, start=1):
        # High-precision token rules
        for name, sev, pattern, detail in _RULES:
            if pattern.search(line):
                findings.append(Finding(rel_path, lineno, sev, name, detail))
        # Assignment-style secrets — a real (non-placeholder-looking) value
        # is still a leak even inside a *.example/.env.example template, so
        # only downgrade severity there instead of skipping the check
        # outright (which previously made _is_placeholder unreachable for
        # every example file, real value or not).
        # Also probe a 2-line window: a key and its literal value can be
        # split across lines (continuation, or a dict key on one line with
        # the value on the next) and are invisible to a pure per-line regex.
        # Strip a trailing line-continuation backslash before joining — left
        # in place, it sits between "=" and the quote and breaks the
        # \s*[:=]\s* adjacency the regex requires.
        window = line if lineno == len(lines) else line.rstrip("\\") + " " + lines[lineno]
        m = _ASSIGN_SECRET_RE.search(line) or _ASSIGN_SECRET_RE.search(window)
        if m and not _is_placeholder(m.group(2)):
            findings.append(
                Finding(
                    rel_path,
                    lineno,
                    "MEDIUM" if is_example else "HIGH",
                    "hardcoded-secret-assignment",
                    f"Hardcoded {m.group(1).lower()} literal — read from environment instead",
                )
            )
        # Insecure code patterns (apply even to examples)
        for name, sev, pattern, detail in _INSECURE_PATTERNS:
            if pattern.search(line):
                findings.append(Finding(rel_path, lineno, sev, name, detail))
    return findings


def scan_output_dir(output_dir: Path) -> ScanReport:
    """Scan a generated-output directory for secrets and insecure patterns.

    Returns a ScanReport. Never raises on individual file errors — a scan that
    cannot read a file simply skips it (the gate must not crash delivery).
    """
    output_dir = Path(output_dir)
    report = ScanReport()
    if not output_dir.exists():
        return report

    for path in _iter_scannable_files(output_dir):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            # Intentionally still skip-and-continue (see docstring) — but a
            # skipped file must be countable, not indistinguishable from
            # "scanned, found nothing" (hunt T9, same shape as hunt T8's C5).
            logger.warning("Security scan could not read %s: %s", path, exc)
            report.files_skipped += 1
            continue
        report.files_scanned += 1
        rel = str(path.relative_to(output_dir))
        is_example = bool(_EXAMPLE_NAME_RE.search(path.name)) or path.name == ".env.example"
        report.findings.extend(_scan_text(rel, is_example, text))

    return report
