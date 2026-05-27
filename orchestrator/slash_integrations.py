"""
SlashIntegrations - 100+ third-party integrations via slash commands.
======================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 8, Phase X3 (Create.xyz-inspired).
"""

from __future__ import annotations
import json, logging, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


@dataclass
class Integration:
    name: str
    description: str = ""
    endpoint: str = ""  # API endpoint template
    method: str = "POST"  # HTTP method
    headers: dict = field(default_factory=dict)
    auth_env_var: str = ""  # e.g. "OPENAI_API_KEY"
    example: str = ""  # Example usage
    category: str = "utility"  # ai, payments, maps, database, etc.


DEFAULT_INTEGRATIONS = {
    # AI / LLM
    "chatgpt": Integration(
        "chatgpt",
        "OpenAI ChatGPT",
        "https://api.openai.com/v1/chat/completions",
        auth_env_var="OPENAI_API_KEY",
        category="ai",
        example="/chatgpt Explain quantum computing",
    ),
    "claude": Integration(
        "claude",
        "Anthropic Claude",
        "https://api.anthropic.com/v1/messages",
        auth_env_var="ANTHROPIC_API_KEY",
        category="ai",
        example="/claude Review this code",
    ),
    "gemini": Integration(
        "gemini",
        "Google Gemini",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash:generateContent",
        auth_env_var="GOOGLE_API_KEY",
        category="ai",
        example="/gemini Summarize this document",
    ),
    "mistral": Integration(
        "mistral",
        "Mistral AI",
        "https://api.mistral.ai/v1/chat/completions",
        auth_env_var="MISTRAL_API_KEY",
        category="ai",
        example="/mistral Translate to French",
    ),
    # Payments
    "stripe": Integration(
        "stripe",
        "Stripe Payments",
        "https://api.stripe.com/v1/",
        auth_env_var="STRIPE_API_KEY",
        category="payments",
        example="/stripe create-checkout 25.00",
    ),
    "paypal": Integration(
        "paypal",
        "PayPal",
        "https://api-m.paypal.com/v2/",
        auth_env_var="PAYPAL_CLIENT_ID",
        category="payments",
        example="/paypal invoice 50.00",
    ),
    # Maps / Geo
    "google-maps": Integration(
        "google-maps",
        "Google Maps",
        "https://maps.googleapis.com/maps/api/",
        auth_env_var="GOOGLE_MAPS_API_KEY",
        category="maps",
        example="/google-maps geocode 'New York'",
    ),
    "mapbox": Integration(
        "mapbox",
        "Mapbox",
        "https://api.mapbox.com/",
        auth_env_var="MAPBOX_TOKEN",
        category="maps",
        example="/mapbox directions NYC Boston",
    ),
    # Communication
    "slack": Integration(
        "slack",
        "Slack",
        "https://slack.com/api/chat.postMessage",
        method="POST",
        auth_env_var="SLACK_BOT_TOKEN",
        category="communication",
        example="/slack #general Deployment complete",
    ),
    "discord": Integration(
        "discord",
        "Discord",
        "https://discord.com/api/v10/",
        auth_env_var="DISCORD_BOT_TOKEN",
        category="communication",
        example="/discord announce Build passed",
    ),
    "twilio": Integration(
        "twilio",
        "Twilio SMS",
        "https://api.twilio.com/",
        auth_env_var="TWILIO_ACCOUNT_SID",
        category="communication",
        example="/twilio sms +1234567890 'Hello'",
    ),
    "sendgrid": Integration(
        "sendgrid",
        "SendGrid Email",
        "https://api.sendgrid.com/v3/mail/send",
        method="POST",
        auth_env_var="SENDGRID_API_KEY",
        category="communication",
        example="/sendgrid to@email.com 'Subject' 'Body'",
    ),
    # Database
    "supabase": Integration(
        "supabase",
        "Supabase",
        "https://{project}.supabase.co/rest/v1/",
        auth_env_var="SUPABASE_KEY",
        category="database",
        example="/supabase query users select",
    ),
    "firebase": Integration(
        "firebase",
        "Firebase",
        "https://{project}.firebaseio.com/",
        auth_env_var="FIREBASE_CONFIG",
        category="database",
        example="/firebase get /users/1",
    ),
    "airtable": Integration(
        "airtable",
        "Airtable",
        "https://api.airtable.com/v0/",
        auth_env_var="AIRTABLE_API_KEY",
        category="database",
        example="/airtable list 'My Base' 'Tasks'",
    ),
    # Cloud / DevOps
    "aws": Integration("aws", "AWS CLI", "", category="devops", example="/aws s3 ls"),
    "vercel": Integration(
        "vercel",
        "Vercel Deploy",
        "https://api.vercel.com/",
        auth_env_var="VERCEL_TOKEN",
        category="devops",
        example="/vercel deploy",
    ),
    "github": Integration(
        "github",
        "GitHub API",
        "https://api.github.com/",
        auth_env_var="GITHUB_TOKEN",
        category="devops",
        example="/github issues list",
    ),
    "docker": Integration(
        "docker",
        "Docker Hub",
        "https://hub.docker.com/v2/",
        category="devops",
        example="/docker pull python:3.12",
    ),
    # Analytics
    "mixpanel": Integration(
        "mixpanel",
        "Mixpanel",
        "https://api.mixpanel.com/",
        auth_env_var="MIXPANEL_TOKEN",
        category="analytics",
        example="/mixpanel track 'signup'",
    ),
    "segment": Integration(
        "segment",
        "Segment",
        "https://api.segment.io/v1/",
        auth_env_var="SEGMENT_WRITE_KEY",
        category="analytics",
        example="/segment identify user_123",
    ),
}


class SlashIntegrationManager:
    """Manages slash command integrations."""

    def __init__(self):
        self._integrations: dict[str, Integration] = dict(DEFAULT_INTEGRATIONS)

    def register(self, integration):
        self._integrations[integration.name] = integration

    def get(self, name):
        return self._integrations.get(name)

    def list_by_category(self):
        by_cat = {}
        for name, i in self._integrations.items():
            by_cat.setdefault(i.category, []).append(name)
        return by_cat

    def list_all(self):
        return [
            {"name": n, "description": i.description, "category": i.category, "example": i.example}
            for n, i in self._integrations.items()
        ]

    def resolve(self, command, args=""):
        """Resolve a slash command to its integration and build request."""
        integration = self._integrations.get(command.lstrip("/"))
        if not integration:
            return None
        import aiohttp

        headers = {"Content-Type": "application/json", **integration.headers}
        if integration.auth_env_var:
            key = os.environ.get(integration.auth_env_var)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return {
            "url": integration.endpoint,
            "method": integration.method,
            "headers": headers,
            "command": command,
            "args": args,
            "description": integration.description,
        }
