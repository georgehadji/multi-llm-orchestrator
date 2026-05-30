"""
ConversationAgent — Interactive spec-gathering dialogue.

Drives a back-and-forth conversation with the user to build a complete,
production-grade ProjectSpec before handing off to the orchestrator pipeline.

Design goals:
- Stops asking when confident (≥ 0.85 on required fields), or when the user
  explicitly signals readiness ("ok", "go", "proceed", "build it", etc.)
- Proactively suggests enhancements for SaaS / micro-SaaS / website quality
  without waiting to be asked
- Each turn returns a ConversationTurn with the agent message + metadata so
  both the CLI and the dashboard can render it identically
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ConversationTurn:
    role: str  # "user" | "agent"
    content: str
    suggestions: list[str] = field(default_factory=list)  # proactive suggestions shown this turn
    confidence: float = 0.0  # 0.0 – 1.0 spec completeness
    ready_to_build: bool = False  # agent believes spec is complete


@dataclass
class ProjectSpec:
    """Structured output produced by the conversation; fed directly into the orchestrator."""

    project_description: str = ""
    success_criteria: str = ""
    target_users: str = ""
    core_features: list[str] = field(default_factory=list)
    tech_stack: str = ""
    platform: str = ""  # web / desktop / api / mobile / cli
    auth_requirements: str = ""
    data_persistence: str = ""
    integrations: list[str] = field(default_factory=list)
    performance_requirements: str = ""
    budget_usd: float = 8.0
    extra_context: str = ""
    enhancements_accepted: list[str] = field(default_factory=list)

    def to_orchestrator_args(self) -> dict[str, Any]:
        """Convert to kwargs accepted by Orchestrator / CLI."""
        features = "\n".join(f"- {f}" for f in self.core_features)
        enhancements = "\n".join(f"- {e}" for e in self.enhancements_accepted)

        description = self.project_description
        if self.target_users:
            description += f"\n\nTarget users: {self.target_users}"
        if self.tech_stack:
            description += f"\nTech stack: {self.tech_stack}"
        if self.platform:
            description += f"\nPlatform: {self.platform}"
        if self.auth_requirements:
            description += f"\nAuth: {self.auth_requirements}"
        if self.data_persistence:
            description += f"\nData persistence: {self.data_persistence}"
        if self.integrations:
            description += f"\nIntegrations: {', '.join(self.integrations)}"
        if self.performance_requirements:
            description += f"\nPerformance: {self.performance_requirements}"
        if features:
            description += f"\n\nCore features:\n{features}"
        if enhancements:
            description += f"\n\nEnhancements to include:\n{enhancements}"
        if self.extra_context:
            description += f"\n\nExtra context: {self.extra_context}"

        return {
            "project": description,
            "criteria": self.success_criteria
            or "All core features implemented, tests pass, production-ready code",
            "budget": self.budget_usd,
        }


# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are the intake agent for an AI software orchestrator that generates
production-grade code. Your job is to interview the user about what they
want to build, then produce a complete, unambiguous specification.

## Your personality
- Conversational, direct, no corporate fluff
- Ask one or two focused questions per turn — never a wall of questions
- When you spot a gap that would hurt quality, ask about it immediately
- Proactively suggest enhancements that a senior engineer would add to any
  production SaaS / micro-SaaS / website — even if the user didn't ask

## Required fields (you MUST establish all of these before signalling readiness)
1. What the product does (core purpose, 1–2 sentences)
2. Who uses it (target users)
3. Core features (at least 3 concrete features)
4. Tech stack / platform (frontend framework, backend language, DB, hosting)
5. Authentication / user accounts (yes/no + how: email, OAuth, magic link…)
6. Data persistence needs (what data is stored, rough schema)

## Confidence scoring (internal — do NOT print the score)
Track how many required fields are established. When all 6 are clear,
set ready=true in your JSON response.

## Enhancement categories to proactively suggest (pick the 2–3 most relevant
   each turn based on what's been established — spread them across turns,
   don't dump all at once)
- Stripe subscription / one-time payments
- Role-based access control (RBAC)
- Email notifications (transactional via Resend / SendGrid)
- Admin dashboard (user management, metrics)
- REST API with API key authentication
- Webhook system for integrations
- Rate limiting and request quotas
- Redis caching layer
- Background job queue (ARQ / Celery)
- File uploads (S3 / Cloudflare R2)
- Full-text search (Meilisearch / Postgres FTS)
- Real-time features (WebSockets / SSE)
- Audit log (who did what, when)
- Feature flags (gradual rollout)
- Multi-tenancy (organisation / workspace model)
- Error monitoring (Sentry integration)
- Analytics and usage tracking
- Internationalization / multi-language
- Dark mode / theme support
- Progressive Web App (PWA) manifest + offline support
- SEO optimization (meta tags, sitemap, structured data)
- Accessibility (WCAG 2.1 AA)
- CI/CD pipeline (GitHub Actions)
- Docker / docker-compose setup
- Environment-based config with .env.example

## Response format — ALWAYS respond with valid JSON only, no markdown fences
{
  "message": "<your conversational reply to the user>",
  "suggestions": ["<suggestion 1>", "<suggestion 2>"],  // proactive suggestions this turn (0–3)
  "ready": false,   // true only when all 6 required fields are established
  "spec_partial": { // update only the fields you now know; omit fields still unknown
    "project_description": "...",
    "target_users": "...",
    "core_features": ["...", "..."],
    "tech_stack": "...",
    "platform": "...",
    "auth_requirements": "...",
    "data_persistence": "...",
    "integrations": ["..."],
    "performance_requirements": "...",
    "success_criteria": "..."
  }
}

## Tone for the message field
- Acknowledge what the user said briefly, then move forward
- When making suggestions, frame them as "Should I also add X? Most production
  [SaaS / sites] need it." — one sentence each
- When ready, summarise the spec in bullet form and ask for confirmation
"""

_READY_SIGNALS = frozenset(
    [
        "ok",
        "okay",
        "go",
        "go ahead",
        "proceed",
        "build",
        "build it",
        "yes",
        "yep",
        "yeah",
        "sure",
        "fine",
        "start",
        "do it",
        "let's go",
        "let's do it",
        "looks good",
        "that's good",
        "good",
        "great",
        "perfect",
        "sounds good",
        "confirmed",
        "confirm",
        "yes please",
        "go for it",
    ]
)


# ─────────────────────────────────────────────────────────────────────────────
# ConversationAgent
# ─────────────────────────────────────────────────────────────────────────────


class ConversationAgent:
    """
    LLM-driven intake agent. Stateful — one instance per conversation session.

    Usage::

        agent = ConversationAgent(client)
        opening = await agent.start()
        # show opening.content to user
        while not agent.ready:
            user_text = input("> ")
            turn = await agent.process_turn(user_text)
            # show turn.content + turn.suggestions
        spec = agent.spec
    """

    def __init__(self, client: Any) -> None:
        self._client = client
        self._history: list[dict[str, str]] = []
        self._spec = ProjectSpec()
        self._ready = False
        self._confidence = 0.0

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def spec(self) -> ProjectSpec:
        return self._spec

    @property
    def confidence(self) -> float:
        return self._confidence

    async def start(self) -> ConversationTurn:
        """Open the conversation with the first agent message."""
        opening_msg = (
            "Hey! Tell me what you want to build. "
            "Don't worry about having all the details — just describe the idea "
            "and I'll ask the right questions to fill in the gaps."
        )
        self._history.append({"role": "assistant", "content": opening_msg})
        return ConversationTurn(
            role="agent",
            content=opening_msg,
            confidence=0.0,
            ready_to_build=False,
        )

    async def process_turn(self, user_message: str) -> ConversationTurn:
        """Process one user turn; return the agent's reply with metadata."""
        # Check for explicit ready signal first
        if user_message.strip().lower() in _READY_SIGNALS and self._confidence >= 0.5:
            self._ready = True
            return ConversationTurn(
                role="agent",
                content="Great — I have enough to get started. Launching the build now...",
                confidence=self._confidence,
                ready_to_build=True,
            )

        # Add user message to history
        self._history.append({"role": "user", "content": user_message})

        # Call LLM
        raw = await self._call_llm()

        # Parse response
        parsed = self._parse_response(raw)
        agent_message = parsed.get("message", raw)
        suggestions = parsed.get("suggestions", [])
        ready = parsed.get("ready", False)
        spec_partial = parsed.get("spec_partial", {})

        # Merge partial spec
        self._merge_spec(spec_partial)
        self._confidence = self._compute_confidence()

        if ready and self._confidence >= 0.7:
            self._ready = True

        # Add agent message to history
        self._history.append({"role": "assistant", "content": agent_message})

        return ConversationTurn(
            role="agent",
            content=agent_message,
            suggestions=suggestions,
            confidence=self._confidence,
            ready_to_build=self._ready,
        )

    def accept_enhancement(self, enhancement: str) -> None:
        """Record a user-accepted enhancement suggestion."""
        if enhancement not in self._spec.enhancements_accepted:
            self._spec.enhancements_accepted.append(enhancement)

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    async def _call_llm(self) -> str:
        """Call the LLM with the full conversation history."""
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + self._history
        try:
            response = await self._client.call(
                model="gemini-2.5-flash",  # fast + cheap for conversation
                messages=messages,
                temperature=0.7,
                max_tokens=800,
            )
            return response.text if hasattr(response, "text") else str(response)
        except Exception as exc:
            logger.warning("LLM call failed in ConversationAgent: %s", exc)
            return json.dumps(
                {
                    "message": "Sorry, I had a connection issue. Could you repeat that?",
                    "suggestions": [],
                    "ready": False,
                    "spec_partial": {},
                }
            )

    def _parse_response(self, raw: str) -> dict:  # type: ignore[type-arg]
        """Parse the LLM JSON response; fall back gracefully on malformed output."""
        text = raw.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(line for line in lines if not line.startswith("```"))
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except json.JSONDecodeError:
            # LLM returned plain text — treat it as the message
            return {
                "message": raw.strip(),
                "suggestions": [],
                "ready": False,
                "spec_partial": {},
            }

    def _merge_spec(self, partial: dict) -> None:  # type: ignore[type-arg]
        """Merge a partial spec dict into self._spec."""
        for key, value in partial.items():
            if not value:
                continue
            if key == "core_features" and isinstance(value, list):
                for f in value:
                    if f and f not in self._spec.core_features:
                        self._spec.core_features.append(f)
            elif key == "integrations" and isinstance(value, list):
                for i in value:
                    if i and i not in self._spec.integrations:
                        self._spec.integrations.append(i)
            elif hasattr(self._spec, key):
                setattr(self._spec, key, value)

    def _compute_confidence(self) -> float:
        """Score 0.0–1.0 based on how many required fields are established."""
        checks = [
            bool(self._spec.project_description),
            bool(self._spec.target_users),
            len(self._spec.core_features) >= 3,
            bool(self._spec.tech_stack) or bool(self._spec.platform),
            bool(self._spec.auth_requirements),
            bool(self._spec.data_persistence),
        ]
        return sum(checks) / len(checks)
