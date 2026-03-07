"""
Orchestrator Integrations Plugin
================================
Save this as orchestrator_plugins/integrations/__init__.py

Third-party integrations for the orchestrator.
These were moved from slack_integration.py and similar files.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

try:
    from orchestrator.unified_events_core import (
        DomainEvent, EventType, get_event_bus, CapabilityUsedEvent
    )
    HAS_CORE = True
except ImportError:
    HAS_CORE = False


logger = logging.getLogger("orchestrator_plugins.integrations")


# ═══════════════════════════════════════════════════════════════════════════════
# Slack Integration
# ═══════════════════════════════════════════════════════════════════════════════

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class SlackMessage:
    text: str
    blocks: Optional[List[Dict]] = None
    channel: Optional[str] = None


class SlackClient:
    """Slack API client."""
    
    def __init__(self, webhook_url: str, token: Optional[str] = None):
        self.webhook_url = webhook_url
        self.token = token
        self._client: Optional[Any] = None
    
    async def _get_client(self):
        if self._client is None:
            if not HAS_HTTPX:
                raise ImportError("httpx required for Slack integration")
            self._client = httpx.AsyncClient()
        return self._client
    
    async def send_message(self, message: SlackMessage) -> bool:
        """Send message to Slack."""
        try:
            client = await self._get_client()
            
            payload = {
                "text": message.text,
            }
            if message.blocks:
                payload["blocks"] = message.blocks
            
            response = await client.post(
                self.webhook_url,
                json=payload,
                timeout=30.0,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Slack message: {e}")
            return False
    
    async def close(self):
        if self._client:
            await self._client.aclose()
            self._client = None


class SlackIntegration:
    """
    Slack integration for orchestrator notifications.
    
    Features:
    - Budget alerts
    - Quality gate failures
    - Circuit breaker triggers
    - Run summaries
    """
    
    def __init__(self, webhook_url: str, default_channel: str = "#orchestrator"):
        self.client = SlackClient(webhook_url) if webhook_url else None
        self.default_channel = default_channel
        self.enabled = webhook_url is not None and HAS_HTTPX
    
    async def send_budget_alert(self, project_id: str, spent: float, 
                                budget: float, severity: AlertSeverity = AlertSeverity.WARNING) -> bool:
        """Send budget threshold alert."""
        if not self.enabled:
            return False
        
        pct = (spent / budget * 100) if budget > 0 else 0
        color = {
            AlertSeverity.INFO: "#3b82f6",
            AlertSeverity.WARNING: "#f59e0b",
            AlertSeverity.ERROR: "#ef4444",
            AlertSeverity.CRITICAL: "#dc2626",
        }.get(severity, "#3b82f6")
        
        message = SlackMessage(
            text=f"⚠️ Budget Alert: {project_id}",
            blocks=[
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"⚠️ Budget {severity.value.upper()}: {project_id}",
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Project:*\n{project_id}"},
                        {"type": "mrkdwn", "text": f"*Status:*\n{severity.value.upper()}"},
                        {"type": "mrkdwn", "text": f"*Spent:*\n${spent:.2f}"},
                        {"type": "mrkdwn", "text": f"*Budget:*\n${budget:.2f} ({pct:.1f}%)"},
                    ]
                },
                {
                    "type": "context",
                    "elements": [
                        {"type": "mrkdwn", "text": f"⏰ {datetime.utcnow().isoformat()}"}
                    ]
                }
            ]
        )
        return await self.client.send_message(message)
    
    async def send_run_summary(self, project_id: str, tasks_completed: int,
                               tasks_failed: int, total_cost: float, 
                               duration_seconds: float, models_used: List[str]) -> bool:
        """Send end-of-run summary."""
        if not self.enabled:
            return False
        
        status = "✅ Success" if tasks_failed == 0 else f"⚠️ {tasks_failed} Failed"
        
        message = SlackMessage(
            text=f"Orchestrator Run Complete: {project_id}",
            blocks=[
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"🚀 Run Complete: {project_id}",
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Status:*\n{status}"},
                        {"type": "mrkdwn", "text": f"*Cost:*\n${total_cost:.2f}"},
                        {"type": "mrkdwn", "text": f"*Tasks:*\n{tasks_completed} completed"},
                        {"type": "mrkdwn", "text": f"*Duration:*\n{duration_seconds/60:.1f} min"},
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Models Used:*\n{', '.join(models_used)}"
                    }
                },
            ]
        )
        return await self.client.send_message(message)
    
    async def send_quality_gate_failure(self, project_id: str, task_id: str,
                                        score: float, threshold: float,
                                        failed_checks: List[str]) -> bool:
        """Send quality gate failure notification."""
        if not self.enabled:
            return False
        
        message = SlackMessage(
            text=f"❌ Quality Gate Failed: {project_id}",
            blocks=[
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"❌ Quality Gate Failed",
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Project:*\n{project_id}"},
                        {"type": "mrkdwn", "text": f"*Task:*\n{task_id}"},
                        {"type": "mrkdwn", "text": f"*Score:*\n{score:.2f} / {threshold:.2f}"},
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Failed Checks:*\n• {'\n• '.join(failed_checks)}"
                    }
                },
            ]
        )
        return await self.client.send_message(message)
    
    def register_event_handlers(self):
        """Register for orchestrator events."""
        if not HAS_CORE:
            return
        
        async def handle_event(event: DomainEvent):
            if event.event_type == EventType.BUDGET_WARNING:
                await self.send_budget_alert(
                    event.aggregate_id,
                    event.metadata.get("spent", 0),
                    event.metadata.get("cap", 0),
                    AlertSeverity.WARNING,
                )
            elif event.event_type == EventType.PROJECT_COMPLETED:
                await self.send_run_summary(
                    event.metadata.get("project_id", event.aggregate_id),
                    event.metadata.get("tasks_completed", 0),
                    event.metadata.get("tasks_failed", 0),
                    event.metadata.get("total_cost", 0),
                    0,  # duration not in event
                    [],  # models not in event
                )
        
        # Subscribe to events
        asyncio.create_task(self._subscribe(handle_event))
    
    async def _subscribe(self, handler):
        bus = await get_event_bus()
        bus.subscribe(handler)


# ═══════════════════════════════════════════════════════════════════════════════
# Discord Integration
# ═══════════════════════════════════════════════════════════════════════════════

class DiscordIntegration:
    """Discord webhook integration."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = webhook_url is not None and HAS_HTTPX
        self._client: Optional[Any] = None
    
    async def _get_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient()
        return self._client
    
    async def send_embed(self, title: str, description: str, 
                         fields: List[Dict], color: int = 0x3b82f6) -> bool:
        """Send Discord embed message."""
        if not self.enabled:
            return False
        
        try:
            client = await self._get_client()
            
            payload = {
                "embeds": [{
                    "title": title,
                    "description": description,
                    "fields": fields,
                    "color": color,
                    "timestamp": datetime.utcnow().isoformat(),
                }]
            }
            
            response = await client.post(self.webhook_url, json=payload, timeout=30.0)
            return response.status_code == 204
        except Exception as e:
            logger.error(f"Failed to send Discord message: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Teams Integration
# ═══════════════════════════════════════════════════════════════════════════════

class TeamsIntegration:
    """Microsoft Teams webhook integration."""
    
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
        self.enabled = webhook_url is not None and HAS_HTTPX
        self._client: Optional[Any] = None
    
    async def send_adaptive_card(self, title: str, text: str, 
                                  facts: List[Dict]) -> bool:
        """Send Teams adaptive card."""
        if not self.enabled:
            return False
        
        try:
            client = await self._get_client()
            
            payload = {
                "@type": "MessageCard",
                "@context": "https://schema.org/extensions",
                "themeColor": "0078D4",
                "summary": title,
                "sections": [{
                    "activityTitle": title,
                    "activitySubtitle": datetime.utcnow().isoformat(),
                    "facts": facts,
                    "markdown": True,
                }]
            }
            
            response = await client.post(self.webhook_url, json=payload, timeout=30.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to send Teams message: {e}")
            return False


# ═══════════════════════════════════════════════════════════════════════════════
# Factory & Setup
# ═══════════════════════════════════════════════════════════════════════════════

def create_slack_integration(webhook_url: Optional[str] = None) -> SlackIntegration:
    """Create Slack integration from environment or config."""
    import os
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    return SlackIntegration(url)


def create_discord_integration(webhook_url: Optional[str] = None) -> DiscordIntegration:
    """Create Discord integration from environment or config."""
    import os
    url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")
    return DiscordIntegration(url)


def create_teams_integration(webhook_url: Optional[str] = None) -> TeamsIntegration:
    """Create Teams integration from environment or config."""
    import os
    url = webhook_url or os.getenv("TEAMS_WEBHOOK_URL")
    return TeamsIntegration(url)


# Auto-setup if environment variables are set
if HAS_CORE:
    slack = create_slack_integration()
    if slack.enabled:
        slack.register_event_handlers()
        logger.info("Slack integration auto-configured")


__all__ = [
    "SlackIntegration",
    "DiscordIntegration",
    "TeamsIntegration",
    "AlertSeverity",
    "create_slack_integration",
    "create_discord_integration",
    "create_teams_integration",
]
