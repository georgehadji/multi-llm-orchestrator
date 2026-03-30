"""
Preflight Validation — Response Quality Control
=================================================

Implements Mnemo Cortex preflight validation:
- Check responses before sending to user
- Four modes: PASS, ENRICH, WARN, BLOCK

Usage:
    from orchestrator.preflight import PreflightValidator, PreflightResult, PreflightMode

    validator = PreflightValidator()

    # Check a response before sending
    result = validator.validate(
        response="Here's the code you requested...",
        context={"task": "code_generation", "user_request": "Write a function"},
        mode=PreflightMode.ENRICH
    )

    if result.action == PreflightAction.BLOCK:
        print(f"Blocked: {result.reason}")
    elif result.action == PreflightAction.ENRICH:
        print(f"Enriched with: {result.enrichment}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .log_config import get_logger

logger = get_logger(__name__)


class PreflightMode(Enum):
    """Preflight validation modes."""
    PASS = "pass"       # Allow all responses
    ENRICH = "enrich"   # Add missing context
    WARN = "warn"       # Flag potential issues
    BLOCK = "block"     # Prevent problematic responses
    AUTO = "auto"       # Use default behavior per check


class PreflightAction(Enum):
    """Actions to take based on validation."""
    PASS = "pass"       # Response is good, proceed
    ENRICH = "enrich"  # Add context/information
    WARN = "warn"      # Allow but flag for review
    BLOCK = "block"    # Don't allow this response


class CheckType(Enum):
    """Types of preflight checks."""
    SAFETY = "safety"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    TONE = "tone"
    FORMAT = "format"
    PRIVACY = "privacy"
    SECURITY = "security"


@dataclass
class CheckResult:
    """Result of a single check."""
    check_type: CheckType
    passed: bool
    severity: int  # 0-10
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class PreflightResult:
    """Result of preflight validation."""
    action: PreflightAction
    passed: bool
    checks: list[CheckResult] = field(default_factory=list)
    reason: str | None = None
    enrichment: str | None = None
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_blocking_issues(self) -> bool:
        return any(c.severity >= 8 and not c.passed for c in self.checks)

    @property
    def has_warnings(self) -> bool:
        return any(c.severity >= 5 and c.severity < 8 and not c.passed for c in self.checks)

    @property
    def summary(self) -> str:
        if self.action == PreflightAction.BLOCK:
            return f"BLOCKED: {self.reason or 'Blocking issues detected'}"
        elif self.action == PreflightAction.WARN:
            return f"WARN: {len(self.warnings)} warnings"
        elif self.action == PreflightAction.ENRICH:
            return f"ENRICHED: {self.enrichment[:50] if self.enrichment else 'No enrichment'}"
        else:
            return "PASS: All checks passed"


class PreflightValidator:
    """
    Validates responses before sending to user.

    Implements four modes from Mnemo Cortex:
    - PASS: Allow all responses
    - ENRICH: Add missing context
    - WARN: Flag potential issues
    - BLOCK: Prevent problematic responses
    """

    # Dangerous patterns that should always be blocked
    DANGEROUS_PATTERNS = [
        (r'eval\s*\(', "Use of eval() detected"),
        (r'exec\s*\(', "Use of exec() detected"),
        (r'__import__\s*\(', "Dynamic import detected"),
        (r'subprocess\s*\.\s*call\s*\(\s*\[.*shell\s*=\s*True', "Shell=True in subprocess"),
        (r'os\.system\s*\(', "os.system() call detected"),
    ]

    # Patterns that indicate incomplete responses
    INCOMPLETE_PATTERNS = [
        (r'\[TODO\]', "TODO placeholder found"),
        (r'\[FIXME\]', "FIXME placeholder found"),
        (r'\{\{.*\}\}', "Template placeholder found"),
        (r'<[^>]*>', "HTML/XML tags detected (possible incomplete)"),
    ]

    # Privacy-sensitive patterns
    PRIVACY_PATTERNS = [
        (r'api[_-]?key["\']?\s*[:=]\s*["\'][^"\']{10,}', "API key detected"),
        (r'secret["\']?\s*[:=]\s*["\'][^"\']{10,}', "Secret detected"),
        (r'password["\']?\s*[:=]\s*["\'][^"\']{6,}', "Password detected"),
        (r'Bearer\s+[A-Za-z0-9\-_]+\.[A-Za-z0-9\-_]+', "Bearer token detected"),
    ]

    def __init__(
        self,
        mode: PreflightMode = PreflightMode.AUTO,
        strict_safety: bool = True,
        strict_privacy: bool = True,
        enable_completeness: bool = True,
        enable_tone: bool = False,
    ):
        self.mode = mode
        self.strict_safety = strict_safety
        self.strict_privacy = strict_privacy
        self.enable_completeness = enable_completeness
        self.enable_tone = enable_tone

        # Custom rules
        self._custom_checks: list[callable] = []

    def add_custom_check(self, check_fn: callable) -> None:
        """Add a custom validation function."""
        self._custom_checks.append(check_fn)

    def _check_safety(self, response: str, context: dict[str, Any]) -> CheckResult:
        """Check for dangerous patterns."""
        for pattern, message in self.DANGEROUS_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                return CheckResult(
                    check_type=CheckType.SAFETY,
                    passed=False,
                    severity=10,
                    message=message,
                    details={"pattern": pattern},
                )

        return CheckResult(
            check_type=CheckType.SAFETY,
            passed=True,
            severity=0,
            message="No safety issues detected",
        )

    def _check_privacy(self, response: str, context: dict[str, Any]) -> CheckResult:
        """Check for privacy-sensitive data."""
        for pattern, message in self.PRIVACY_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                return CheckResult(
                    check_type=CheckType.PRIVACY,
                    passed=False,
                    severity=9,
                    message=message,
                    details={"pattern": pattern},
                )

        return CheckResult(
            check_type=CheckType.PRIVACY,
            passed=True,
            severity=0,
            message="No privacy issues detected",
        )

    def _check_completeness(self, response: str, context: dict[str, Any]) -> CheckResult:
        """Check for incomplete responses."""
        # Check for placeholders
        for pattern, message in self.INCOMPLETE_PATTERNS:
            if re.search(pattern, response, re.IGNORECASE):
                return CheckResult(
                    check_type=CheckType.COMPLETENESS,
                    passed=False,
                    severity=7,
                    message=message,
                    details={"pattern": pattern},
                )

        # Check minimum length
        if len(response.strip()) < 20:
            return CheckResult(
                check_type=CheckType.COMPLETENESS,
                passed=False,
                severity=5,
                message="Response too short",
                details={"length": len(response)},
            )

        # Check if response seems cut off
        if not response.rstrip().endswith(('.', '!', '?', ')', ']', '}', '"', "'")):
            # Might be truncated
            if len(response) > 100:
                return CheckResult(
                    check_type=CheckType.COMPLETENESS,
                    passed=True,
                    severity=3,
                    message="Response may be truncated",
                    details={"ends_with": response[-5:]},
                )

        return CheckResult(
            check_type=CheckType.COMPLETENESS,
            passed=True,
            severity=0,
            message="Response appears complete",
        )

    def _check_accuracy(self, response: str, context: dict[str, Any]) -> CheckResult:
        """Check for factual accuracy issues."""
        # Check for contradictory statements
        contradictions = [
            (r'\bhowever\b.*\btherefore\b', "Contradictory logic"),
            (r'\bbut\b.*\bbut\b', "Conflicting statements"),
        ]

        for pattern, message in contradictions:
            if re.search(pattern, response, re.IGNORECASE):
                return CheckResult(
                    check_type=CheckType.ACCURACY,
                    passed=False,
                    severity=6,
                    message=message,
                    details={"pattern": pattern},
                )

        # Check if response addresses the request
        user_request = context.get("user_request", "")
        if user_request:
            # Simple keyword check - response should contain some request keywords
            request_words = set(re.findall(r'\b\w{4,}\b', user_request.lower()))
            response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))

            # Check overlap
            overlap = request_words & response_words
            if len(overlap) < min(2, len(request_words) * 0.3):
                return CheckResult(
                    check_type=CheckType.ACCURACY,
                    passed=False,
                    severity=4,
                    message="Response may not address the request",
                    details={"overlap": list(overlap)[:5]},
                )

        return CheckResult(
            check_type=CheckType.ACCURACY,
            passed=True,
            severity=0,
            message="No accuracy issues detected",
        )

    def _check_format(self, response: str, context: dict[str, Any]) -> CheckResult:
        """Check response format."""
        task_type = context.get("task_type", "")

        # Code generation should have code blocks
        if task_type == "code_gen" or task_type == "code_generation":
            if '```' not in response and 'def ' not in response and 'class ' not in response:
                return CheckResult(
                    check_type=CheckType.FORMAT,
                    passed=False,
                    severity=5,
                    message="Code generation task but no code found",
                    details={"task_type": task_type},
                )

        return CheckResult(
            check_type=CheckType.FORMAT,
            passed=True,
            severity=0,
            message="Format appears correct",
        )

    def _determine_action(
        self,
        checks: list[CheckResult],
        mode: PreflightMode,
    ) -> tuple[PreflightAction, str | None, str | None]:
        """Determine action based on check results and mode."""
        blocking = [c for c in checks if c.severity >= 8 and not c.passed]
        warnings = [c for c in checks if 5 <= c.severity < 8 and not c.passed]
        enrichments = [c for c in checks if c.severity >= 3 and c.severity < 8 and not c.passed]

        if mode == PreflightMode.PASS:
            return PreflightAction.PASS, None, None

        if blocking and mode in (PreflightMode.BLOCK, PreflightMode.AUTO):
            return (
                PreflightAction.BLOCK,
                f"Blocking issues: {', '.join(c.message for c in blocking)}",
                None,
            )

        if warnings and mode in (PreflightMode.WARN, PreflightMode.BLOCK):
            return (
                PreflightAction.WARN,
                f"Warnings: {', '.join(c.message for c in warnings)}",
                None,
            )

        if enrichments and mode in (PreflightMode.ENRICH, PreflightMode.AUTO):
            enrichment_text = ". ".join(c.message for c in enrichments)
            return (
                PreflightAction.ENRICH,
                None,
                f"Consider adding: {enrichment_text}",
            )

        return PreflightAction.PASS, None, None

    def validate(
        self,
        response: str,
        context: dict[str, Any] | None = None,
        mode: PreflightMode | None = None,
    ) -> PreflightResult:
        """
        Validate a response before sending.

        Args:
            response: The response to validate
            context: Additional context (task_type, user_request, etc.)
            mode: Override default validation mode

        Returns:
            PreflightResult with action and details
        """
        context = context or {}
        mode = mode or self.mode

        checks: list[CheckResult] = []

        # Run safety check
        checks.append(self._check_safety(response, context))

        # Run privacy check
        if self.strict_privacy:
            checks.append(self._check_privacy(response, context))

        # Run completeness check
        if self.enable_completeness:
            checks.append(self._check_completeness(response, context))

        # Run accuracy check
        checks.append(self._check_accuracy(response, context))

        # Run format check
        checks.append(self._check_format(response, context))

        # Run custom checks
        for custom_check in self._custom_checks:
            try:
                result = custom_check(response, context)
                if result:
                    checks.append(result)
            except Exception as e:
                logger.warning(f"Custom check failed: {e}")

        # Determine action
        action, reason, enrichment = self._determine_action(checks, mode)

        # Collect warnings
        warnings = [
            c.message for c in checks
            if c.severity >= 5 and c.severity < 8 and not c.passed
        ]

        return PreflightResult(
            action=action,
            passed=action == PreflightAction.PASS,
            checks=checks,
            reason=reason,
            enrichment=enrichment,
            warnings=warnings,
            metadata={
                "mode": mode.value,
                "context_keys": list(context.keys()),
            },
        )

    def validate_and_modify(
        self,
        response: str,
        context: dict[str, Any] | None = None,
        mode: PreflightMode | None = None,
    ) -> tuple[str, PreflightResult]:
        """
        Validate and potentially modify a response.

        Returns the (potentially modified) response and validation result.
        """
        result = self.validate(response, context, mode)

        modified_response = response

        if result.action == PreflightAction.BLOCK:
            # Replace with safe message
            modified_response = "[Response blocked for safety reasons]"
        elif result.action == PreflightAction.ENRICH and result.enrichment:
            # Append enrichment note
            modified_response = response + f"\n\n[Note: {result.enrichment}]"

        return modified_response, result


# Global validator instance
_default_validator: PreflightValidator | None = None


def get_validator() -> PreflightValidator:
    """Get the default validator instance."""
    global _default_validator
    if _default_validator is None:
        _default_validator = PreflightValidator()
    return _default_validator


def preflight_check(
    response: str,
    context: dict[str, Any] | None = None,
    mode: PreflightMode = PreflightMode.AUTO,
) -> PreflightResult:
    """Convenience function for preflight validation."""
    return get_validator().validate(response, context, mode)
