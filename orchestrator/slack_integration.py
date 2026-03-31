"""
Slack Integration Module for Multi-LLM Orchestrator
===================================================

Provides alerting, run summaries, and slash command handling for Slack integration.

Environment Variables:
    ORCHESTRATOR_SLACK_WEBHOOK_URL: Webhook URL for sending messages
    ORCHESTRATOR_SLACK_SIGNING_SECRET: Secret for verifying slash commands
    ORCHESTRATOR_HOST: Host for dashboard URLs (default: localhost:8888)

Usage:
    from orchestrator.slack_integration import SlackNotifier, RunSummaryFormatter

    notifier = SlackNotifier()
    await notifier.notify_budget_alert(payload)
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Data Models
# ═══════════════════════════════════════════════════════════════════════════════


class AlertSeverity(Enum):
    """Severity levels for alerts."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class BudgetStats:
    """Budget statistics for a project/run."""

    total_budget: float
    spent: float
    remaining: float
    percentage_used: float
    currency: str = "USD"


@dataclass
class BudgetAlertPayload:
    """Payload for budget threshold alerts."""

    project_id: str
    run_id: str
    threshold_crossed: float  # 0.5, 0.8, 1.0, etc.
    stats: BudgetStats
    dashboard_url: str
    escalate_url: str  # Callback URL to increase budget
    severity: AlertSeverity = AlertSeverity.WARNING
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class FailedCheck:
    """A single failed quality check."""

    name: str
    expected: str
    actual: str
    severity: str = "error"


@dataclass
class QualityGateFailurePayload:
    """Payload for quality gate failure alerts."""

    project_id: str
    run_id: str
    quality_score: float
    failed_checks: list[FailedCheck]
    dashboard_url: str
    rerun_url: str  # Callback URL to retrigger run
    report_url: str | None = None
    severity: AlertSeverity = AlertSeverity.ERROR
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class CircuitBreakerPayload:
    """Payload for model circuit breaker alerts."""

    model_name: str
    error_count: int
    last_error_message: str
    suggested_fallback: str
    failure_threshold: int
    dashboard_url: str
    severity: AlertSeverity = AlertSeverity.WARNING
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class IssueItem:
    """A single issue in a run summary."""

    category: str  # SECURITY, TEST, PERFORMANCE, etc.
    description: str
    severity: str = "medium"
    file_path: str | None = None
    line_number: int | None = None


@dataclass
class CostBreakdown:
    """Cost breakdown by model."""

    model_name: str
    cost_usd: float
    tokens_used: int | None = None


@dataclass
class RunSummaryPayload:
    """Payload for end-of-run summaries."""

    project_id: str
    run_id: str
    status: str  # "PASSED" or "FAILED"
    total_cost_usd: float
    cost_breakdown: list[CostBreakdown]
    quality_score: float
    quality_gate_passed: bool
    top_issues: list[IssueItem]
    duration_seconds: float | None = None
    dashboard_url: str = ""
    logs_url: str = ""
    artifacts_url: str = ""
    audit_log_url: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SlashCommandRequest:
    """Incoming Slack slash command request."""

    command: str
    text: str  # Everything after the command
    user_id: str
    user_name: str
    team_id: str
    channel_id: str
    channel_name: str
    response_url: str
    trigger_id: str


@dataclass
class SlashCommandResponse:
    """Response to a slash command."""

    text: str
    response_type: str = "ephemeral"  # "ephemeral" or "in_channel"
    blocks: list[dict] | None = None


# ═══════════════════════════════════════════════════════════════════════════════
# Low-Level Slack Client
# ═══════════════════════════════════════════════════════════════════════════════


class SlackClient:
    """
    Low-level client for sending messages to Slack via webhooks.

    Usage:
        client = SlackClient(webhook_url="https://hooks.slack.com/...")
        await client.send_message({"text": "Hello", "blocks": [...]})
    """

    def __init__(
        self,
        webhook_url: str | None = None,
        timeout_seconds: float = 30.0,
    ):
        self.webhook_url = webhook_url or os.environ.get("ORCHESTRATOR_SLACK_WEBHOOK_URL")
        self.timeout = timeout_seconds
        self._client: httpx.AsyncClient | None = None

    @property
    def is_configured(self) -> bool:
        """Check if the client has a valid webhook URL."""
        return bool(self.webhook_url)

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def send_message(
        self,
        payload: dict,
        webhook_url: str | None = None,
    ) -> bool:
        """
        Send a message payload to Slack.

        Args:
            payload: The JSON payload to send (follows Slack Block Kit format)
            webhook_url: Optional override webhook URL

        Returns:
            True if message was sent successfully
        """
        url = webhook_url or self.webhook_url

        if not url:
            logger.warning("Slack webhook URL not configured, skipping message")
            return False

        try:
            client = await self._get_client()
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200 and response.text == "ok":
                logger.debug("Slack message sent successfully")
                return True
            else:
                logger.error(f"Slack API error: {response.status_code} - {response.text}")
                return False

        except httpx.TimeoutException:
            logger.error("Timeout sending Slack message")
            return False
        except Exception as e:
            logger.error(f"Error sending Slack message: {e}")
            return False

    async def send_text(self, text: str, webhook_url: str | None = None) -> bool:
        """Send a simple text message."""
        return await self.send_message({"text": text}, webhook_url)

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# ═══════════════════════════════════════════════════════════════════════════════
# Slack Notifier (High-Level Alerting)
# ═══════════════════════════════════════════════════════════════════════════════


class SlackNotifier:
    """
    High-level notifier for orchestrator alerts.

    Handles formatting and sending of:
    - Budget threshold alerts
    - Quality gate failures
    - Model circuit breaker trips
    """

    def __init__(self, client: SlackClient | None = None):
        self.client = client or SlackClient()

    def _severity_to_emoji(self, severity: AlertSeverity) -> str:
        """Get emoji for severity level."""
        return {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.ERROR: "❌",
            AlertSeverity.CRITICAL: "🚨",
        }.get(severity, "📢")

    async def notify_budget_alert(self, payload: BudgetAlertPayload) -> bool:
        """Send a budget threshold alert to Slack."""
        emoji = self._severity_to_emoji(payload.severity)
        percentage = int(payload.threshold_crossed * 100)

        message = {
            "text": f"{emoji} Orchestrator Budget Alert: {payload.project_id}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Orchestrator Budget Alert",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Project:*\n`{payload.project_id}`"},
                        {"type": "mrkdwn", "text": f"*Run:*\n`{payload.run_id}`"},
                        {
                            "type": "mrkdwn",
                            "text": f"*Threshold Crossed:*\n{percentage}%",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Severity:*\n{payload.severity.value.upper()}",
                        },
                    ],
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Budget Total:*\n${payload.stats.total_budget:.2f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Budget Spent:*\n${payload.stats.spent:.2f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Remaining:*\n${payload.stats.remaining:.2f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Used:*\n{payload.stats.percentage_used:.1f}%",
                        },
                    ],
                },
                {"type": "divider"},
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "🔗 View Run",
                                "emoji": True,
                            },
                            "url": payload.dashboard_url,
                            "action_id": "view_run",
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "💰 Escalate Budget +10%",
                                "emoji": True,
                            },
                            "url": payload.escalate_url,
                            "style": "primary",
                            "action_id": "escalate_budget",
                        },
                    ],
                },
            ],
        }

        success = await self.client.send_message(message)
        if success:
            logger.info(f"Budget alert sent for project {payload.project_id}")
        return success

    async def notify_quality_gate_failure(self, payload: QualityGateFailurePayload) -> bool:
        """Send a quality gate failure alert to Slack."""
        emoji = self._severity_to_emoji(payload.severity)

        checks_text = ""
        for check in payload.failed_checks[:5]:
            checks_text += f"• *{check.name}*: Expected `{check.expected}`, got `{check.actual}`\n"

        if len(payload.failed_checks) > 5:
            checks_text += f"_... and {len(payload.failed_checks) - 5} more_"

        message = {
            "text": f"{emoji} Quality Gate Failed: {payload.project_id}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Quality Gate Failed",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Project:*\n`{payload.project_id}`"},
                        {"type": "mrkdwn", "text": f"*Run:*\n`{payload.run_id}`"},
                        {
                            "type": "mrkdwn",
                            "text": f"*Quality Score:*\n{payload.quality_score:.2f}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Failed Checks:*\n{len(payload.failed_checks)}",
                        },
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Failed Checks:*\n{checks_text}",
                    },
                },
            ],
        }

        if payload.report_url:
            message["blocks"].append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"<🔗 {payload.report_url}|View Full Report>",
                    },
                }
            )

        message["blocks"].extend(
            [
                {"type": "divider"},
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "🔄 Rerun with Same Settings",
                                "emoji": True,
                            },
                            "url": payload.rerun_url,
                            "style": "primary",
                            "action_id": "rerun_project",
                        },
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "📊 Open in Dashboard",
                                "emoji": True,
                            },
                            "url": payload.dashboard_url,
                            "action_id": "open_dashboard",
                        },
                    ],
                },
            ]
        )

        success = await self.client.send_message(message)
        if success:
            logger.info(f"Quality gate failure alert sent for project {payload.project_id}")
        return success

    async def notify_model_circuit_breaker(self, payload: CircuitBreakerPayload) -> bool:
        """Send a model circuit breaker alert to Slack."""
        emoji = self._severity_to_emoji(payload.severity)

        error_msg = payload.last_error_message
        if len(error_msg) > 200:
            error_msg = error_msg[:197] + "..."

        message = {
            "text": f"{emoji} Model Circuit Breaker: {payload.model_name}",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"{emoji} Model Circuit Breaker Tripped",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Model:*\n`{payload.model_name}`",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Error Count:*\n{payload.error_count}/{payload.failure_threshold}",
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Suggested Fallback:*\n`{payload.suggested_fallback}`",
                        },
                    ],
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Last Error:*\n```\n{error_msg}\n```",
                    },
                },
                {"type": "divider"},
                {
                    "type": "actions",
                    "elements": [
                        {
                            "type": "button",
                            "text": {
                                "type": "plain_text",
                                "text": "📊 View Model Metrics",
                                "emoji": True,
                            },
                            "url": payload.dashboard_url,
                            "action_id": "view_metrics",
                        },
                    ],
                },
            ],
        }

        success = await self.client.send_message(message)
        if success:
            logger.info(f"Circuit breaker alert sent for model {payload.model_name}")
        return success


# ═══════════════════════════════════════════════════════════════════════════════
# Run Summary Formatter
# ═══════════════════════════════════════════════════════════════════════════════


class RunSummaryFormatter:
    """Formats end-of-run summaries for Slack."""

    @staticmethod
    def format_cost_breakdown(costs: list[CostBreakdown]) -> str:
        """Format cost breakdown as a readable string."""
        if not costs:
            return "N/A"

        lines = []
        for cost in costs:
            if cost.tokens_used:
                lines.append(
                    f"• {cost.model_name}: ${cost.cost_usd:.3f} ({cost.tokens_used:,} tokens)"
                )
            else:
                lines.append(f"• {cost.model_name}: ${cost.cost_usd:.3f}")
        return "\n".join(lines)

    @staticmethod
    def format_issues(issues: list[IssueItem]) -> str:
        """Format top issues for display."""
        if not issues:
            return "✅ No issues found"

        lines = []
        for issue in issues[:3]:
            emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(issue.severity, "⚪")
            lines.append(f"{emoji} *[{issue.category}]* {issue.description}")

        if len(issues) > 3:
            lines.append(f"_... and {len(issues) - 3} more issues_")

        return "\n".join(lines)

    @classmethod
    def build_slack_message(cls, payload: RunSummaryPayload) -> dict:
        """Build a complete Slack message from a run summary payload."""
        status_emoji = "✅" if payload.status == "PASSED" else "❌"
        quality_emoji = (
            "🟢" if payload.quality_score >= 0.8 else "🟡" if payload.quality_score >= 0.6 else "🔴"
        )

        duration_str = ""
        if payload.duration_seconds:
            minutes = int(payload.duration_seconds // 60)
            seconds = int(payload.duration_seconds % 60)
            duration_str = f" ({minutes}m {seconds}s)"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"Orchestrator Run Summary: {payload.project_id}",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Status:*\n{status_emoji} {payload.status}{duration_str}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Run ID:*\n`{payload.run_id}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Total Cost:*\n💰 ${payload.total_cost_usd:.3f}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Quality Score:*\n{quality_emoji} {payload.quality_score:.2f}",
                    },
                ],
            },
        ]

        # Add cost breakdown
        if payload.cost_breakdown:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Cost Breakdown:*\n{cls.format_cost_breakdown(payload.cost_breakdown)}",
                    },
                }
            )

        # Add issues
        blocks.append(
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Top Issues:*\n{cls.format_issues(payload.top_issues)}",
                },
            }
        )

        # Add action buttons
        actions = {"type": "actions", "elements": []}

        if payload.dashboard_url:
            actions["elements"].append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📊 Open Dashboard", "emoji": True},
                    "url": payload.dashboard_url,
                    "action_id": "open_dashboard",
                }
            )

        if payload.artifacts_url:
            actions["elements"].append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📦 Download Artifacts", "emoji": True},
                    "url": payload.artifacts_url,
                    "action_id": "download_artifacts",
                }
            )

        if payload.audit_log_url:
            actions["elements"].append(
                {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "📋 View Audit Log", "emoji": True},
                    "url": payload.audit_log_url,
                    "action_id": "view_audit_log",
                }
            )

        if actions["elements"]:
            blocks.extend([{"type": "divider"}, actions])

        return {
            "text": f"Orchestrator Run Summary: {payload.project_id} - {payload.status}",
            "blocks": blocks,
        }


async def send_run_summary_to_slack(
    payload: RunSummaryPayload,
    client: SlackClient | None = None,
) -> bool:
    """
    Send a run summary to Slack.

    Convenience function that formats and sends a run summary.
    """
    client = client or SlackClient()
    message = RunSummaryFormatter.build_slack_message(payload)
    return await client.send_message(message)


# ═══════════════════════════════════════════════════════════════════════════════
# Slash Command Handling
# ═══════════════════════════════════════════════════════════════════════════════


class RateLimiter:
    """
    Simple in-memory rate limiter for Slack commands.

    Tracks requests per workspace/channel and enimits limits.
    Can be replaced with Redis for distributed deployments.
    """

    def __init__(
        self,
        max_requests: int = 10,
        window_seconds: int = 60,
    ):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        # key -> list of timestamps
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _clean_old_requests(self, timestamps: list[float]) -> list[float]:
        """Remove requests outside the time window."""
        now = time.time()
        cutoff = now - self.window_seconds
        return [ts for ts in timestamps if ts > cutoff]

    def is_allowed(self, key: str) -> tuple[bool, int]:
        """
        Check if a request is allowed.

        Returns:
            (allowed, remaining_requests)
        """
        timestamps = self._requests.get(key, [])
        timestamps = self._clean_old_requests(timestamps)

        if len(timestamps) >= self.max_requests:
            return False, 0

        return True, self.max_requests - len(timestamps) - 1

    def record_request(self, key: str):
        """Record a request for the given key."""
        now = time.time()
        self._requests[key].append(now)

    def get_retry_after(self, key: str) -> int:
        """Get seconds until the next request is allowed."""
        timestamps = self._requests.get(key, [])
        if not timestamps:
            return 0

        oldest = min(timestamps)
        retry_after = int(oldest + self.window_seconds - time.time())
        return max(0, retry_after)


@dataclass
class PresetTemplate:
    """A predefined template for project runs."""

    name: str
    description: str
    policy_overrides: dict[str, Any]
    default_budget: float = 5.0
    allowed_overrides: list[str] = field(default_factory=list)


class TemplateRegistry:
    """Registry of available templates for slash commands."""

    def __init__(self):
        self._templates: dict[str, PresetTemplate] = {}
        self._register_defaults()

    def _register_defaults(self):
        """Register default templates."""
        self.register(
            PresetTemplate(
                name="secure-api-starter",
                description="FastAPI + JWT auth with strict security policies",
                policy_overrides={
                    "allowed_models": ["gpt-4o", "claude-3-5-sonnet"],
                    "security_checks": ["bandit", "safety", "secrets"],
                    "required_tests": True,
                    "min_coverage": 80,
                },
                default_budget=8.0,
                allowed_overrides=["budget", "description"],
            )
        )

        self.register(
            PresetTemplate(
                name="internal-dashboard",
                description="Next.js dashboard with basic auth (relaxed settings)",
                policy_overrides={
                    "allowed_models": ["gpt-4o-mini", "deepseek-chat"],
                    "security_checks": ["basic"],
                    "required_tests": False,
                    "min_coverage": 50,
                },
                default_budget=3.0,
                allowed_overrides=["budget", "description", "models"],
            )
        )

        self.register(
            PresetTemplate(
                name="python-cli",
                description="Python CLI tool with standard quality gates",
                policy_overrides={
                    "allowed_models": ["deepseek-chat", "gpt-4o-mini"],
                    "security_checks": ["bandit"],
                    "required_tests": True,
                    "min_coverage": 70,
                },
                default_budget=2.0,
                allowed_overrides=["budget", "description"],
            )
        )

    def register(self, template: PresetTemplate):
        """Register a new template."""
        self._templates[template.name] = template

    def get(self, name: str) -> PresetTemplate | None:
        """Get a template by name."""
        return self._templates.get(name)

    def list_templates(self) -> list[PresetTemplate]:
        """List all available templates."""
        return list(self._templates.values())

    def parse_overrides(self, text: str, template: PresetTemplate) -> dict[str, Any]:
        """
        Parse override arguments from command text.

        Format: key=value key2=value2
        """
        overrides = {}
        parts = text.split()

        for part in parts:
            if "=" in part:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()

                if key not in template.allowed_overrides:
                    continue

                # Try to convert to appropriate type
                if value.lower() in ("true", "false"):
                    overrides[key] = value.lower() == "true"
                elif value.replace(".", "").isdigit():
                    overrides[key] = float(value) if "." in value else int(value)
                else:
                    overrides[key] = value

        return overrides


class TemplateRunner(ABC):
    """
    Abstract interface for running templates.

    Implement this to integrate with your orchestrator's project creation.
    """

    @abstractmethod
    async def run_template(
        self,
        template_name: str,
        user_id: str,
        overrides: dict[str, Any],
    ) -> str:
        """
        Run a template and return the run ID.

        Args:
            template_name: Name of the template to run
            user_id: ID of the user triggering the run
            overrides: Override parameters from the command

        Returns:
            The run ID of the created project
        """
        pass

    @abstractmethod
    def get_dashboard_url(self, run_id: str) -> str:
        """Get the dashboard URL for a run."""
        pass


class SlashCommandHandler:
    """
    Handles Slack slash commands for the orchestrator.

    Supports:
    - /orchestrator run <template> [overrides...]
    - /orchestrator list
    - /orchestrator help
    """

    def __init__(
        self,
        template_runner: TemplateRunner | None = None,
        signing_secret: str | None = None,
        rate_limiter: RateLimiter | None = None,
    ):
        self.templates = TemplateRegistry()
        self.runner = template_runner
        self.signing_secret = signing_secret or os.environ.get("ORCHESTRATOR_SLACK_SIGNING_SECRET")
        self.rate_limiter = rate_limiter or RateLimiter(max_requests=10, window_seconds=60)
        self.host = os.environ.get("ORCHESTRATOR_HOST", "localhost:8888")

    def verify_signature(
        self,
        timestamp: str,
        signature: str,
        body: str,
    ) -> bool:
        """
        Verify Slack request signature.

        Args:
            timestamp: X-Slack-Request-Timestamp header
            signature: X-Slack-Signature header
            body: Raw request body

        Returns:
            True if signature is valid
        """
        if not self.signing_secret:
            logger.warning("Slack signing secret not configured, skipping verification")
            return True

        # Check timestamp to prevent replay attacks
        now = time.time()
        if abs(now - int(timestamp)) > 300:  # 5 minutes
            return False

        # Build signature base string
        sig_basestring = f"v0:{timestamp}:{body}"

        # Calculate signature
        my_signature = (
            "v0="
            + hmac.new(
                self.signing_secret.encode(),
                sig_basestring.encode(),
                hashlib.sha256,
            ).hexdigest()
        )

        return hmac.compare_digest(my_signature, signature)

    def parse_request(self, form_data: dict[str, str]) -> SlashCommandRequest:
        """Parse form data into a SlashCommandRequest."""
        return SlashCommandRequest(
            command=form_data.get("command", ""),
            text=form_data.get("text", ""),
            user_id=form_data.get("user_id", ""),
            user_name=form_data.get("user_name", ""),
            team_id=form_data.get("team_id", ""),
            channel_id=form_data.get("channel_id", ""),
            channel_name=form_data.get("channel_name", ""),
            response_url=form_data.get("response_url", ""),
            trigger_id=form_data.get("trigger_id", ""),
        )

    def _get_rate_limit_key(self, request: SlashCommandRequest) -> str:
        """Get rate limit key for a request."""
        return f"{request.team_id}:{request.channel_id}"

    async def handle(self, request: SlashCommandRequest) -> SlashCommandResponse:
        """
        Handle a slash command request.

        Returns:
            Response to send back to Slack
        """
        # Check rate limit
        rate_key = self._get_rate_limit_key(request)
        allowed, remaining = self.rate_limiter.is_allowed(rate_key)

        if not allowed:
            retry_after = self.rate_limiter.get_retry_after(rate_key)
            return SlashCommandResponse(
                text=f"⏱️ Rate limit exceeded. Please try again in {retry_after} seconds.",
                response_type="ephemeral",
            )

        self.rate_limiter.record_request(rate_key)

        # Parse command
        parts = request.text.strip().split(maxsplit=1)
        subcommand = parts[0].lower() if parts else "help"
        args = parts[1] if len(parts) > 1 else ""

        if subcommand == "run":
            return await self._handle_run(args, request)
        elif subcommand == "list":
            return self._handle_list()
        elif subcommand == "help":
            return self._handle_help()
        else:
            return SlashCommandResponse(
                text=f"❓ Unknown command: `{subcommand}`. Try `/orchestrator help`",
                response_type="ephemeral",
            )

    async def _handle_run(self, args: str, request: SlashCommandRequest) -> SlashCommandResponse:
        """Handle the 'run' subcommand."""
        if not self.runner:
            return SlashCommandResponse(
                text="❌ Template runner not configured.",
                response_type="ephemeral",
            )

        parts = args.split(maxsplit=1)
        template_name = parts[0] if parts else ""
        override_text = parts[1] if len(parts) > 1 else ""

        if not template_name:
            return SlashCommandResponse(
                text="❌ Please specify a template. Usage: `/orchestrator run <template>`",
                response_type="ephemeral",
            )

        template = self.templates.get(template_name)
        if not template:
            available = ", ".join(t.name for t in self.templates.list_templates())
            return SlashCommandResponse(
                text=f"❌ Unknown template: `{template_name}`.\nAvailable: {available}",
                response_type="ephemeral",
            )

        # Parse overrides
        overrides = self.templates.parse_overrides(override_text, template)

        try:
            # Start the run
            run_id = await self.runner.run_template(
                template_name=template_name,
                user_id=request.user_id,
                overrides=overrides,
            )

            dashboard_url = self.runner.get_dashboard_url(run_id)
            budget = overrides.get("budget", template.default_budget)

            return SlashCommandResponse(
                text="",
                response_type="in_channel",
                blocks=[
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"🚀 *Starting Orchestrator Run*\n• Template: `{template_name}`\n• Budget: ${budget}\n• Run ID: `{run_id}`",
                        },
                    },
                    {
                        "type": "actions",
                        "elements": [
                            {
                                "type": "button",
                                "text": {
                                    "type": "plain_text",
                                    "text": "📊 View Dashboard",
                                    "emoji": True,
                                },
                                "url": dashboard_url,
                                "style": "primary",
                            },
                        ],
                    },
                ],
            )

        except Exception as e:
            logger.error(f"Failed to start template run: {e}")
            return SlashCommandResponse(
                text=f"❌ Failed to start run: {str(e)}",
                response_type="ephemeral",
            )

    def _handle_list(self) -> SlashCommandResponse:
        """Handle the 'list' subcommand."""
        templates = self.templates.list_templates()

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📋 Available Templates",
                    "emoji": True,
                },
            },
        ]

        for template in templates:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*`{template.name}`*\n{template.description}\nDefault budget: ${template.default_budget}",
                    },
                }
            )

        return SlashCommandResponse(
            text="Available templates",
            response_type="ephemeral",
            blocks=blocks,
        )

    def _handle_help(self) -> SlashCommandResponse:
        """Handle the 'help' subcommand."""
        return SlashCommandResponse(
            text="",
            response_type="ephemeral",
            blocks=[
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🤖 Orchestrator Bot Help",
                        "emoji": True,
                    },
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            "*Commands:*\n"
                            '• `/orchestrator run <template> [budget=X] [description="..."]` - Start a new run\n'
                            "• `/orchestrator list` - List available templates\n"
                            "• `/orchestrator help` - Show this help message\n\n"
                            "*Examples:*\n"
                            "• `/orchestrator run secure-api-starter`\n"
                            '• `/orchestrator run internal-dashboard budget=10 description="Q4 dashboard"`'
                        ),
                    },
                },
            ],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# HTTP Endpoint Handler (FastAPI/Starlette)
# ═══════════════════════════════════════════════════════════════════════════════


class SlackEndpointHandler:
    """
    HTTP endpoint handler for Slack slash commands.

    Works with FastAPI, Starlette, or any ASGI framework.

    FastAPI Example:
        from fastapi import FastAPI, Request, Response
        from orchestrator.slack_integration import SlackEndpointHandler

        app = FastAPI()
        handler = SlackEndpointHandler()

        @app.post("/slack/slash/orchestrator")
        async def slack_slash(request: Request):
            return await handler.handle_request(request)

    Starlette Example:
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        async def slack_endpoint(request: Request):
            return await handler.handle_request(request)

        routes = [
            Route("/slack/slash/orchestrator", slack_endpoint, methods=["POST"]),
        ]
    """

    def __init__(
        self,
        command_handler: SlashCommandHandler | None = None,
        template_runner: TemplateRunner | None = None,
    ):
        self.command_handler = command_handler or SlashCommandHandler(
            template_runner=template_runner
        )

    async def handle_request(self, request: Any) -> Any:
        """
        Handle an incoming HTTP request.

        Works with FastAPI/Starlette Request objects.
        Returns a response compatible with the framework.
        """
        # Import here to avoid dependency if not using FastAPI
        try:
            from fastapi import Request as FastAPIRequest
            from fastapi.responses import JSONResponse as FastAPIJSONResponse

            ResponseType = FastAPIJSONResponse
        except ImportError:
            try:
                from starlette.requests import Request as StarletteRequest
                from starlette.responses import JSONResponse as StarletteJSONResponse

                ResponseType = StarletteJSONResponse
            except ImportError:
                raise ImportError(
                    "Neither FastAPI nor Starlette is installed. "
                    "Install one to use the Slack endpoint handler."
                )

        # Verify signature
        timestamp = request.headers.get("X-Slack-Request-Timestamp", "")
        signature = request.headers.get("X-Slack-Signature", "")
        body = await request.body()
        body_text = body.decode("utf-8")

        if not self.command_handler.verify_signature(timestamp, signature, body_text):
            return ResponseType(
                content={"error": "Invalid signature"},
                status_code=401,
            )

        # Parse form data
        form_data = await request.form()
        slash_request = self.command_handler.parse_request(dict(form_data))

        # Handle command
        response = await self.command_handler.handle(slash_request)

        return ResponseType(
            content={
                "text": response.text,
                "response_type": response.response_type,
                "blocks": response.blocks,
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Orchestrator Lifecycle Hooks
# ═══════════════════════════════════════════════════════════════════════════════


class SlackIntegrationHooks:
    """
    Integration hooks for the orchestrator lifecycle.

    Connects orchestrator events to Slack notifications.

    Usage:
        hooks = SlackIntegrationHooks()

        # In your orchestrator:
        await hooks.on_budget_threshold_crossed(project, run, 0.8)
        await hooks.on_quality_gate_evaluated(project, run, passed=False)
        await hooks.on_model_circuit_breaker_tripped(model_state)
        await hooks.on_run_completed(project, run, metrics)
    """

    def __init__(
        self,
        notifier: SlackNotifier | None = None,
        host: str | None = None,
    ):
        self.notifier = notifier or SlackNotifier()
        self.host = host or os.environ.get("ORCHESTRATOR_HOST", "localhost:8888")

    def _get_dashboard_url(self, project_id: str, run_id: str) -> str:
        """Generate dashboard URL."""
        return f"https://{self.host}/runs/{run_id}"

    async def on_budget_threshold_crossed(
        self,
        project_id: str,
        run_id: str,
        threshold: float,
        budget_stats: BudgetStats,
    ) -> bool:
        """
        Called when budget crosses a threshold (0.5, 0.8, 1.0).

        Args:
            project_id: Project identifier
            run_id: Run identifier
            threshold: Threshold crossed (e.g., 0.8 for 80%)
            budget_stats: Current budget statistics

        Returns:
            True if notification was sent
        """
        if not self.notifier.client.is_configured:
            return False

        dashboard_url = self._get_dashboard_url(project_id, run_id)
        escalate_url = f"https://{self.host}/api/projects/{project_id}/escalate?percent=10"

        severity = AlertSeverity.WARNING
        if threshold >= 1.0:
            severity = AlertSeverity.CRITICAL
        elif threshold >= 0.8:
            severity = AlertSeverity.ERROR

        payload = BudgetAlertPayload(
            project_id=project_id,
            run_id=run_id,
            threshold_crossed=threshold,
            stats=budget_stats,
            dashboard_url=dashboard_url,
            escalate_url=escalate_url,
            severity=severity,
        )

        return await self.notifier.notify_budget_alert(payload)

    async def on_quality_gate_evaluated(
        self,
        project_id: str,
        run_id: str,
        passed: bool,
        quality_score: float,
        failed_checks: list[FailedCheck],
    ) -> bool:
        """
        Called when quality gate is evaluated.

        Only sends notification if the gate failed.
        """
        if passed or not self.notifier.client.is_configured:
            return False

        dashboard_url = self._get_dashboard_url(project_id, run_id)
        rerun_url = f"https://{self.host}/api/runs/{run_id}/rerun"

        payload = QualityGateFailurePayload(
            project_id=project_id,
            run_id=run_id,
            quality_score=quality_score,
            failed_checks=failed_checks,
            dashboard_url=dashboard_url,
            rerun_url=rerun_url,
        )

        return await self.notifier.notify_quality_gate_failure(payload)

    async def on_model_circuit_breaker_tripped(
        self,
        model_name: str,
        error_count: int,
        last_error: str,
        suggested_fallback: str,
        failure_threshold: int,
    ) -> bool:
        """Called when a model circuit breaker trips."""
        if not self.notifier.client.is_configured:
            return False

        dashboard_url = f"https://{self.host}/models/{model_name}"

        payload = CircuitBreakerPayload(
            model_name=model_name,
            error_count=error_count,
            last_error_message=last_error,
            suggested_fallback=suggested_fallback,
            failure_threshold=failure_threshold,
            dashboard_url=dashboard_url,
        )

        return await self.notifier.notify_model_circuit_breaker(payload)

    async def on_run_completed(
        self,
        project_id: str,
        run_id: str,
        status: str,
        total_cost: float,
        cost_breakdown: list[CostBreakdown],
        quality_score: float,
        quality_gate_passed: bool,
        top_issues: list[IssueItem],
        duration_seconds: float | None = None,
    ) -> bool:
        """Called when a run completes (success or failure)."""
        if not self.notifier.client.is_configured:
            return False

        payload = RunSummaryPayload(
            project_id=project_id,
            run_id=run_id,
            status=status,
            total_cost_usd=total_cost,
            cost_breakdown=cost_breakdown,
            quality_score=quality_score,
            quality_gate_passed=quality_gate_passed,
            top_issues=top_issues,
            duration_seconds=duration_seconds,
            dashboard_url=self._get_dashboard_url(project_id, run_id),
            logs_url=f"https://{self.host}/runs/{run_id}/logs",
            artifacts_url=f"https://{self.host}/runs/{run_id}/artifacts",
            audit_log_url=f"https://{self.host}/runs/{run_id}/audit",
        )

        return await send_run_summary_to_slack(payload, self.notifier.client)


# ═══════════════════════════════════════════════════════════════════════════════
# Example Usage and Configuration
# ═══════════════════════════════════════════════════════════════════════════════

"""
## Environment Variables

Required:
    ORCHESTRATOR_SLACK_WEBHOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz

Optional:
    ORCHESTRATOR_SLACK_SIGNING_SECRET=your_signing_secret
    ORCHESTRATOR_HOST=dashboard.example.com

## FastAPI Integration Example

```python
from fastapi import FastAPI, Request
from orchestrator.slack_integration import (
    SlackEndpointHandler,
    SlashCommandHandler,
    TemplateRunner,
    SlackIntegrationHooks,
)

app = FastAPI()

# Implement your template runner
class MyTemplateRunner(TemplateRunner):
    async def run_template(self, template_name, user_id, overrides):
        # Create and start your orchestrator project
        # Return the run_id
        run_id = await create_project_from_template(template_name, overrides)
        return run_id

    def get_dashboard_url(self, run_id):
        return f"https://dashboard.example.com/runs/{run_id}"

# Set up the handler
template_runner = MyTemplateRunner()
slash_handler = SlashCommandHandler(template_runner=template_runner)
endpoint_handler = SlackEndpointHandler(command_handler=slash_handler)

# HTTP endpoint
@app.post("/slack/slash/orchestrator")
async def slack_slash(request: Request):
    return await endpoint_handler.handle_request(request)

# Lifecycle hooks
hooks = SlackIntegrationHooks()

# In your orchestrator:
@app.on_event("startup")
async def setup_hooks():
    orchestrator.register_hook("budget_threshold", hooks.on_budget_threshold_crossed)
    orchestrator.register_hook("quality_gate", hooks.on_quality_gate_evaluated)
    orchestrator.register_hook("circuit_breaker", hooks.on_model_circuit_breaker_tripped)
    orchestrator.register_hook("run_completed", hooks.on_run_completed)
```

## Manual Usage Example

```python
import asyncio
from orchestrator.slack_integration import (
    SlackNotifier,
    BudgetAlertPayload,
    BudgetStats,
    AlertSeverity,
)

async def main():
    notifier = SlackNotifier()

    payload = BudgetAlertPayload(
        project_id="proj-123",
        run_id="run-456",
        threshold_crossed=0.8,
        stats=BudgetStats(
            total_budget=10.0,
            spent=8.5,
            remaining=1.5,
            percentage_used=85.0,
        ),
        dashboard_url="https://dash.example.com/runs/run-456",
        escalate_url="https://dash.example.com/api/proj-123/escalate",
        severity=AlertSeverity.WARNING,
    )

    success = await notifier.notify_budget_alert(payload)
    print(f"Notification sent: {success}")

asyncio.run(main())
```
"""
