"""
Shell-Command Safety Guard — 4-Tier Risk Classifier
=====================================================
Author: Orchestrator core

Gating any shell dispatch before a feature executes generated commands
(GAP-13). Pure classifier only — no execution logic lives here.

Tiers:
    SAFE        — known-safe, read-only, or informational commands
    SUSPICIOUS  — some risk indicators (pipes, network) but may be legitimate
    DANGEROUS   — clearly destructive without explicit ``allow_dangerous=True``
    BLOCKED     — always blocked (fork bombs, ``rm -rf /``, etc.)

Design:
    - Regex/allowlist rules
    - ``RiskLevel`` enum + ``rationale: str``
    - Secure-by-default: ``deny on unknown``
    - Pattern-matching only, no execution
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar


class RiskLevel(str, Enum):
    """Risk classification for a shell command."""

    SAFE = "safe"
    SUSPICIOUS = "suspicious"
    DANGEROUS = "dangerous"
    BLOCKED = "blocked"


@dataclass(frozen=True)
class RiskAssessment:
    """Result of a command risk assessment."""

    level: RiskLevel
    rationale: str
    command: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Rule definitions
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CommandRule:
    """A single rule in the classification table."""

    pattern: re.Pattern
    level: RiskLevel
    rationale: str


_COMMAND_RULES: list[CommandRule] = [
    # ── BLOCKED ──────────────────────────────────────────────────────────────
    # Fork bombs and shell bombs
    CommandRule(
        pattern=re.compile(r"^\s*[:;]\s*\(|\)\{|:\|:\||\bbash\b.*-c\b.*:\s*\(|\)\s*\{"),
        level=RiskLevel.BLOCKED,
        rationale="Fork bomb or shell bomb pattern detected",
    ),
    # rm -rf / or rm -rf ~
    CommandRule(
        pattern=re.compile(r"\brm\s+(-rf|-[rR][fF]|--recursive\s+--force)\s+[/~](\s|$|;)"),
        level=RiskLevel.BLOCKED,
        rationale="Recursive forced deletion of root/home directory",
    ),
    # :(){ :|:& };:  (classic bash fork bomb)
    CommandRule(
        pattern=re.compile(r":\s*\(\s*\)\s*\{|\|\s*:\s*&\s*\}"),
        level=RiskLevel.BLOCKED,
        rationale="Classic bash fork bomb",
    ),
    # ── DANGEROUS ────────────────────────────────────────────────────────────
    # Destructive disk operations
    CommandRule(
        pattern=re.compile(r"\b(mkfs|fdisk|dd|format|diskpart|gparted)\b"),
        level=RiskLevel.DANGEROUS,
        rationale="Destructive disk/filesystem operation",
    ),
    # chmod -R 777 or similar
    CommandRule(
        pattern=re.compile(r"\bchmod\s+(-R|--recursive)?\s*777\b"),
        level=RiskLevel.DANGEROUS,
        rationale="Overly permissive recursive chmod",
    ),
    # wget/curl piped to shell
    CommandRule(
        pattern=re.compile(r"\b(curl|wget)\s+.*\|\s*(bash|sh|zsh|powershell)"),
        level=RiskLevel.DANGEROUS,
        rationale="Pipe-from-network to shell — remote code execution risk",
    ),
    # sudo with broad permissions
    CommandRule(
        pattern=re.compile(r"\bsudo\s+(rm|chmod|chown|dd|mkfs)\b"),
        level=RiskLevel.DANGEROUS,
        rationale="Privileged destructive command",
    ),
    # ── SUSPICIOUS ───────────────────────────────────────────────────────────
    # Network operations (may be legitimate in CI)
    CommandRule(
        pattern=re.compile(r"\b(curl|wget|nc|netcat|telnet|ssh)\b"),
        level=RiskLevel.SUSPICIOUS,
        rationale="Network operation",
    ),
    # Pipe to shell (not from network)
    CommandRule(
        pattern=re.compile(r"\|\s*(bash|sh|zsh|powershell)\b"),
        level=RiskLevel.SUSPICIOUS,
        rationale="Output piped to shell interpreter",
    ),
    # Writing to sensitive locations
    CommandRule(
        pattern=re.compile(r">\s*/etc/|>\s*/usr/|>\s*/var/"),
        level=RiskLevel.SUSPICIOUS,
        rationale="Writing to system directory",
    ),
    # Kill signals
    CommandRule(
        pattern=re.compile(r"\bkill\s+-9\b"),
        level=RiskLevel.SUSPICIOUS,
        rationale="Force-kill signal",
    ),
    # ── SAFE (explicit allowlist) ─────────────────────────────────────────────
    # Python commands
    CommandRule(
        pattern=re.compile(r"^\s*(python|python3|pytest|pip|uv)\b"),
        level=RiskLevel.SAFE,
        rationale="Python toolchain command",
    ),
    # Git read-only operations
    CommandRule(
        pattern=re.compile(r"^\s*git\s+(status|log|diff|show|branch|remote|ls-files)\b"),
        level=RiskLevel.SAFE,
        rationale="Git read-only operation",
    ),
    # List/read-only filesystem
    CommandRule(
        pattern=re.compile(r"^\s*(ls|find|cat|head|tail|wc|file|du|df)\b"),
        level=RiskLevel.SAFE,
        rationale="Read-only filesystem command",
    ),
    # Directory operations
    CommandRule(
        pattern=re.compile(r"^\s*(pwd|cd|mkdir|which|type|where)\b"),
        level=RiskLevel.SAFE,
        rationale="Directory/utility operation",
    ),
    # Lint/format tools
    CommandRule(
        pattern=re.compile(r"^\s*(ruff|black|mypy|flake8|pylint|eslint|prettier|bandit|safety)\b"),
        level=RiskLevel.SAFE,
        rationale="Lint/format/security tool",
    ),
    # grep and text processing
    CommandRule(
        pattern=re.compile(r"^\s*(grep|rg|ag|ack|sed|awk|sort|uniq|cut|tr|jq)\b"),
        level=RiskLevel.SAFE,
        rationale="Text processing tool",
    ),
    # Make
    CommandRule(
        pattern=re.compile(r"^\s*(make|cargo|npm|npx|yarn|dotnet|go)\b"),
        level=RiskLevel.SAFE,
        rationale="Build tool",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────


def classify_command(command: str) -> RiskAssessment:
    """Classify a shell command into a 4-tier risk level.

    Parameters
    ----------
    command: The shell command string to classify.

    Returns
    -------
    ``RiskAssessment`` with the matched level and rationale.

    Notes
    -----
    - Rules are checked in order: BLOCKED → DANGEROUS → SUSPICIOUS → SAFE.
    - The **first** matching rule wins (blocked/dangerous rules take priority).
    - Unknown commands (no rule matches) default to SUSPICIOUS
      (secure-by-default: deny on unknown).
    """
    stripped = command.strip()
    if not stripped:
        return RiskAssessment(
            level=RiskLevel.SAFE,
            rationale="Empty command — no risk",
            command=command,
        )

    # Check in priority order
    priority_order = [RiskLevel.BLOCKED, RiskLevel.DANGEROUS, RiskLevel.SUSPICIOUS, RiskLevel.SAFE]

    for level in priority_order:
        for rule in _COMMAND_RULES:
            if rule.level != level:
                continue
            if rule.pattern.search(stripped):
                return RiskAssessment(
                    level=level,
                    rationale=rule.rationale,
                    command=command,
                )

    # Default: deny on unknown
    return RiskAssessment(
        level=RiskLevel.SUSPICIOUS,
        rationale=f"Unknown command — defaulting to SUSPICIOUS (secure-by-default)",
        command=command,
    )


def requires_explicit_approval(assessment: RiskAssessment, allow_dangerous: bool = False) -> bool:
    """Check if a command needs extra approval before execution.

    Parameters
    ----------
    assessment: The risk assessment result.
    allow_dangerous: If True, DANGEROUS commands are permitted
        (SUSPICIOUS and BLOCKED still gate).

    Returns
    -------
    True if the command should be blocked or requires explicit approval.
    """
    if assessment.level == RiskLevel.BLOCKED:
        return True
    if assessment.level == RiskLevel.DANGEROUS and not allow_dangerous:
        return True
    return False
