"""
Integration Circuit Breaker with Explicit Failure Mode
======================================================

Prevents silent failures in CI/CD integrations by:
1. Failing fast when integrations are misconfigured
2. Providing clear error messages with remediation steps
3. Offering --fail-without-integration flag for strict CI

Implements minimax regret: worst case is explicit failure with
clear next steps, not silent success with missing artifacts.
"""

from __future__ import annotations

import asyncio
import enum
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

T = TypeVar("T")


class IntegrationState(enum.Enum):
    """Circuit breaker states for integrations."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, rejecting requests
    HALF_OPEN = "half_open"  # Testing if recovered


class IntegrationType(enum.Enum):
    """Types of integrations that can be guarded."""

    GITHUB = "github"
    GITLAB = "gitlab"
    SLACK = "slack"
    JIRA = "jira"
    LINEAR = "linear"


@dataclass
class IntegrationHealth:
    """Health status of an integration."""

    integration_type: IntegrationType
    state: IntegrationState
    consecutive_failures: int
    last_failure: datetime | None
    last_success: datetime | None
    error_message: str | None
    config_present: bool

    @property
    def is_healthy(self) -> bool:
        return self.state == IntegrationState.CLOSED

    @property
    def downtime_seconds(self) -> float:
        if self.last_failure is None:
            return 0.0
        return (datetime.utcnow() - self.last_failure).total_seconds()


class IntegrationFailure(Exception):
    """Raised when integration fails and strict mode is enabled."""

    def __init__(
        self,
        integration: IntegrationType,
        operation: str,
        reason: str,
        remediation: str,
    ):
        self.integration = integration
        self.operation = operation
        self.reason = reason
        self.remediation = remediation
        super().__init__(
            f"{integration.value} integration failed during {operation}: {reason}\n"
            f"Remediation: {remediation}"
        )


class IntegrationCircuitBreaker:
    """
    Circuit breaker for external integrations.

    Prevents silent failures by tracking health and failing explicitly
    when integrations are required but unavailable.

    Usage:
        @cb.guard(IntegrationType.GITHUB, operation="create_check_run")
        async def create_check_run(...) -> CheckRun:
            ...

    If GitHub is down/misconfigured and strict mode is on, this raises
    IntegrationFailure immediately with clear remediation steps.
    """

    # Thresholds
    FAILURE_THRESHOLD = 3
    RECOVERY_TIMEOUT_SECONDS = 60
    HALF_OPEN_MAX_CALLS = 3

    def __init__(
        self,
        strict_mode: bool = False,
        fail_silently: bool = False,
    ):
        """
        Args:
            strict_mode: If True, fail immediately on first integration error
            fail_silently: If True, log errors but don't raise (legacy behavior)
        """
        self.strict_mode = strict_mode
        self.fail_silently = fail_silently
        self._states: dict[IntegrationType, IntegrationState] = dict.fromkeys(
            IntegrationType, IntegrationState.CLOSED
        )
        self._failures: dict[IntegrationType, int] = dict.fromkeys(IntegrationType, 0)
        self._last_failure: dict[IntegrationType, datetime | None] = dict.fromkeys(IntegrationType)
        self._last_success: dict[IntegrationType, datetime | None] = dict.fromkeys(IntegrationType)
        self._half_open_calls: dict[IntegrationType, int] = dict.fromkeys(IntegrationType, 0)
        self._error_messages: dict[IntegrationType, str | None] = dict.fromkeys(IntegrationType)

    def _get_config_status(self, integration: IntegrationType) -> bool:
        """Check if configuration is present for an integration."""
        import os

        config_vars = {
            IntegrationType.GITHUB: ["GITHUB_TOKEN", "GIT_TOKEN"],
            IntegrationType.GITLAB: ["GITLAB_TOKEN", "GIT_TOKEN"],
            IntegrationType.SLACK: ["ORCHESTRATOR_SLACK_WEBHOOK_URL", "SLACK_WEBHOOK_URL"],
            IntegrationType.JIRA: ["ISSUE_TRACKER_API_TOKEN", "JIRA_API_TOKEN"],
            IntegrationType.LINEAR: ["ISSUE_TRACKER_API_TOKEN", "LINEAR_API_TOKEN"],
        }

        vars_to_check = config_vars.get(integration, [])
        return any(os.environ.get(v) for v in vars_to_check)

    def _record_success(self, integration: IntegrationType) -> None:
        """Record a successful call."""
        self._failures[integration] = 0
        self._last_success[integration] = datetime.utcnow()

        if self._states[integration] == IntegrationState.HALF_OPEN:
            self._half_open_calls[integration] += 1
            if self._half_open_calls[integration] >= self.HALF_OPEN_MAX_CALLS:
                logger.info(f"{integration.value} integration recovered, closing circuit")
                self._states[integration] = IntegrationState.CLOSED
                self._half_open_calls[integration] = 0

    def _record_failure(
        self,
        integration: IntegrationType,
        error: Exception,
    ) -> None:
        """Record a failed call."""
        self._failures[integration] += 1
        self._last_failure[integration] = datetime.utcnow()
        self._error_messages[integration] = str(error)

        if self._states[integration] == IntegrationState.HALF_OPEN:
            # Failed during recovery test - back to open
            logger.warning(
                f"{integration.value} integration failed during recovery test, "
                f"re-opening circuit"
            )
            self._states[integration] = IntegrationState.OPEN
            return

        if self._states[integration] == IntegrationState.CLOSED:
            if self._failures[integration] >= self.FAILURE_THRESHOLD:
                logger.error(
                    f"{integration.value} integration circuit breaker OPEN after "
                    f"{self.FAILURE_THRESHOLD} failures"
                )
                self._states[integration] = IntegrationState.OPEN

    def _can_execute(self, integration: IntegrationType) -> bool:
        """Check if a call can be executed based on circuit state."""
        state = self._states[integration]

        if state == IntegrationState.CLOSED:
            return True

        if state == IntegrationState.OPEN:
            # Check if recovery timeout has passed
            last_fail = self._last_failure[integration]
            if last_fail:
                elapsed = (datetime.utcnow() - last_fail).total_seconds()
                if elapsed > self.RECOVERY_TIMEOUT_SECONDS:
                    logger.info(f"{integration.value} recovery timeout passed, entering half-open")
                    self._states[integration] = IntegrationState.HALF_OPEN
                    self._half_open_calls[integration] = 0
                    return True
            return False

        if state == IntegrationState.HALF_OPEN:
            return self._half_open_calls[integration] < self.HALF_OPEN_MAX_CALLS

        return True

    def guard(
        self,
        integration: IntegrationType,
        operation: str,
        remediation: str | None = None,
    ) -> Callable:
        """
        Decorator to guard an integration call.

        Args:
            integration: The type of integration
            operation: Description of the operation (for error messages)
            remediation: Steps to fix the issue

        Example:
            @cb.guard(IntegrationType.GITHUB, "create_check_run",
                     remediation="Check GITHUB_TOKEN is valid and has 'checks:write' scope")
            async def create_check_run(...):
                ...
        """

        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                # Check if circuit allows execution
                if not self._can_execute(integration):
                    msg = self._error_messages.get(integration) or "Circuit breaker is OPEN"
                    error = IntegrationFailure(
                        integration=integration,
                        operation=operation,
                        reason=f"Circuit breaker OPEN: {msg}",
                        remediation=remediation
                        or "Wait for circuit to recover or check integration config",
                    )

                    if self.strict_mode:
                        raise error
                    elif self.fail_silently:
                        logger.warning(f"Suppressing integration failure: {error}")
                        return None  # type: ignore
                    else:
                        # Default: log warning but don't fail the whole run
                        logger.warning(
                            f"Integration {integration.value} unavailable, "
                            f"continuing without {operation}. Error: {msg}"
                        )
                        return None  # type: ignore

                try:
                    result = await func(*args, **kwargs)
                    self._record_success(integration)
                    return result

                except Exception as e:
                    self._record_failure(integration, e)

                    error = IntegrationFailure(
                        integration=integration,
                        operation=operation,
                        reason=str(e),
                        remediation=remediation
                        or "Check integration configuration and network connectivity",
                    )

                    if self.strict_mode:
                        raise error from e
                    elif self.fail_silently:
                        logger.warning(f"Suppressing integration failure: {error}")
                        return None  # type: ignore
                    else:
                        logger.warning(
                            f"Integration {integration.value} failed for {operation}, "
                            f"continuing. Error: {e}"
                        )
                        return None  # type: ignore

            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                # Run async wrapper
                return asyncio.run(async_wrapper(*args, **kwargs))

            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

        return decorator

    def get_health(self, integration: IntegrationType) -> IntegrationHealth:
        """Get health status for an integration."""
        return IntegrationHealth(
            integration_type=integration,
            state=self._states[integration],
            consecutive_failures=self._failures[integration],
            last_failure=self._last_failure[integration],
            last_success=self._last_success[integration],
            error_message=self._error_messages[integration],
            config_present=self._get_config_status(integration),
        )

    def get_all_health(self) -> dict[IntegrationType, IntegrationHealth]:
        """Get health status for all integrations."""
        return {t: self.get_health(t) for t in IntegrationType}

    def manual_trip(self, integration: IntegrationType, reason: str) -> None:
        """Manually trip the circuit breaker (for testing or emergency)."""
        self._states[integration] = IntegrationState.OPEN
        self._last_failure[integration] = datetime.utcnow()
        self._error_messages[integration] = f"Manually tripped: {reason}"
        logger.warning(f"Manually tripped {integration.value} circuit breaker: {reason}")

    def manual_reset(self, integration: IntegrationType) -> None:
        """Manually reset the circuit breaker."""
        self._states[integration] = IntegrationState.CLOSED
        self._failures[integration] = 0
        self._error_messages[integration] = None
        logger.info(f"Manually reset {integration.value} circuit breaker")


# CLI Integration for strict mode
# Usage: python -m orchestrator --strict-integrations


def get_circuit_breaker(strict: bool = False) -> IntegrationCircuitBreaker:
    """Get or create a circuit breaker instance."""
    import os

    strict_mode = strict or os.environ.get("ORCHESTRATOR_STRICT_INTEGRATIONS") == "true"
    fail_silently = os.environ.get("ORCHESTRATOR_INTEGRATIONS_SILENT") == "true"

    return IntegrationCircuitBreaker(
        strict_mode=strict_mode,
        fail_silently=fail_silently,
    )


class IntegrationHealthReporter:
    """
    Reports integration health in CI-friendly formats.

    Outputs warnings/errors in GitHub Actions annotation format
    for visibility in PR checks.
    """

    @staticmethod
    def github_annotation(health: IntegrationHealth) -> str:
        """Format health status as GitHub Actions annotation."""
        if not health.config_present:
            return f"::warning title=Integration Config Missing::{health.integration_type.value} integration not configured"

        if health.state == IntegrationState.OPEN:
            return (
                f"::error title=Integration Unhealthy::{health.integration_type.value} "
                f"integration circuit breaker OPEN. Last error: {health.error_message}"
            )

        if health.state == IntegrationState.HALF_OPEN:
            return (
                f"::warning title=Integration Recovering::{health.integration_type.value} "
                f"integration in recovery mode"
            )

        return f"::notice title=Integration Healthy::{health.integration_type.value} integration operational"

    @staticmethod
    def print_health_report(cb: IntegrationCircuitBreaker) -> None:
        """Print a formatted health report."""
        import os

        healths = cb.get_all_health()
        in_github = os.environ.get("GITHUB_ACTIONS") == "true"

        print("\n" + "=" * 60)
        print("INTEGRATION HEALTH REPORT")
        print("=" * 60)

        for integration, health in healths.items():
            if not health.config_present:
                status = "⚪ NOT CONFIGURED"
            elif health.is_healthy:
                status = "🟢 HEALTHY"
            elif health.state == IntegrationState.HALF_OPEN:
                status = "🟡 RECOVERING"
            else:
                status = "🔴 UNHEALTHY"

            print(f"\n{integration.value.upper()}: {status}")

            if in_github:
                annotation = IntegrationHealthReporter.github_annotation(health)
                print(annotation)

            if health.config_present and not health.is_healthy:
                print(f"  Last error: {health.error_message}")
                print(f"  Downtime: {health.downtime_seconds:.0f}s")

        print("\n" + "=" * 60)
