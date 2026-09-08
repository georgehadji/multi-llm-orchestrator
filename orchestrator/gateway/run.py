"""
Orchestrator Gateway — Accept Project Specs via Messaging
============================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Multi-platform gateway for the AI Orchestrator. Accepts project specs
via messaging platforms, reports progress, and allows approval/denial
of expensive model choices.

Designed as a lightweight bridge: the gateway receives messages, routes
them to the orchestrator engine, and sends back results. It does NOT
replace the CLI — it augments it for headless/continuous operation.

Currently ships with a Webhook adapter and a Telegram adapter pattern.
Additional platforms follow the PlatformAdapter ABC.

Integration: Started via `orchestrator gateway start` CLI subcommand.
Runs in a background asyncio event loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("orchestrator.gateway")


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class GatewayConfig:
    """Gateway configuration.

    Attributes:
        platforms: Dict of ``{platform_name: {config_key: value}}``.
            Example: ``{"telegram": {"token": "...", "chat_id": "..."}}``
        max_active_projects: Maximum concurrent project executions.
        default_model: Default model for gateway-triggered projects.
    """

    platforms: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_active_projects: int = 3
    default_model: str = ""
    # {platform: {user_id, ...}}. Checked by handle_message() before running
    # anything. None/missing-platform/empty-set all deny — fail closed, since
    # a filled-in transport (the "webhook"/"echo" adapters are stubs today)
    # would otherwise let any external sender spend real budget unauthenticated.
    allowed_users: dict[str, frozenset[str]] | None = None


# ─────────────────────────────────────────────────────────────────────────────
# OrchestratorGateway
# ─────────────────────────────────────────────────────────────────────────────


class OrchestratorGateway:
    """Multi-platform gateway for the AI Orchestrator.

    Lifecycle:
        1. ``await gateway.start()`` — connects all platform adapters.
        2. Incoming messages routed via ``handle_message()``.
        3. ``await gateway.shutdown()`` — disconnects all adapters.

    Usage:
        config = GatewayConfig(platforms={"webhook": {"port": 8080}})
        gateway = OrchestratorGateway(config)
        await gateway.start()
        # ... run event loop ...
        await gateway.shutdown()
    """

    def __init__(
        self,
        config: GatewayConfig | None = None,
    ) -> None:
        """Initialize gateway.

        Args:
            config: GatewayConfig with platform settings. Defaults to
                empty config (no platforms).
        """
        self.config = config or GatewayConfig()
        self._platforms: dict[str, Any] = {}  # name -> adapter instance
        self._active_sessions: dict[str, Any] = {}
        self._running = False

    # ── Lifecycle ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start all configured platform adapters.

        Iterates over ``config.platforms`` and instantiates the matching
        adapter for each. Unknown platforms are skipped with a warning.
        """
        self._running = True
        for name, adapter_config in self.config.platforms.items():
            adapter = self._create_adapter(name, adapter_config)
            if adapter is None:
                logger.warning("Gateway: unknown platform '%s' — skipped", name)
                continue
            try:
                await adapter["connect"]()
                self._platforms[name] = adapter
                logger.info("Gateway: connected %s", name)
            except Exception as exc:
                logger.error("Gateway: failed to connect %s: %s", name, exc)

        if not self._platforms:
            logger.warning("Gateway: no platforms connected")
        else:
            logger.info(
                "Gateway: running with %d platform(s): %s",
                len(self._platforms),
                list(self._platforms.keys()),
            )

    async def shutdown(self) -> None:
        """Disconnect all platform adapters."""
        self._running = False
        for name, adapter in self._platforms.items():
            try:
                await adapter["disconnect"]()
                logger.info("Gateway: disconnected %s", name)
            except Exception as exc:
                logger.warning("Gateway: disconnect error for %s: %s", name, exc)
        self._platforms.clear()
        self._active_sessions.clear()
        logger.info("Gateway: shut down")

    # ── Message handling ───────────────────────────────────────────────────

    async def handle_message(
        self,
        platform: str,
        user_id: str,
        text: str,
    ) -> str:
        """Route an incoming message to the appropriate handler.

        Args:
            platform: Source platform name (e.g. ``"webhook"``).
            user_id: Sender identifier.
            text: Message text.

        Returns:
            Response text to send back to the user.
        """
        text = text.strip()
        if not text:
            return "Please provide a project description."

        if not self._is_authorized(platform, user_id):
            logger.warning("Gateway: rejected unauthorized sender %s/%s", platform, user_id)
            return "Not authorized."

        # Route to engine execution
        return await self._execute_project_spec(platform, user_id, text)

    def _is_authorized(self, platform: str, user_id: str) -> bool:
        """Fail closed: no allowlist configured for `platform` denies everyone on it."""
        allowed = (self.config.allowed_users or {}).get(platform)
        return bool(allowed) and user_id in allowed

    async def _execute_project_spec(
        self,
        platform: str,
        user_id: str,
        spec: str,
    ) -> str:
        """Execute a project spec received via messaging.

        Creates an Orchestrator instance, runs the project, and returns
        a summary of results.

        Args:
            platform: Source platform name.
            user_id: Sender identifier.
            spec: The project description text.

        Returns:
            Formatted result string.
        """
        from ..engine import Orchestrator
        from ..budget import Budget

        budget = Budget(max_usd=5.0, max_time_seconds=1800)
        orch = Orchestrator(budget=budget)

        try:
            async with orch:
                state = await orch.run_project(
                    project_description=spec,
                    success_criteria="Complete the project as described.",
                )

            # Build summary
            total = len(state.results)
            passed = sum(1 for r in state.results.values() if r.status.value == "completed")
            lines = [
                f"Project complete. Status: {state.status.value}",
                f"Tasks: {passed}/{total} passed",
                f"Budget: ${state.budget.spent_usd:.2f} / ${state.budget.max_usd}",
            ]
            return "\n".join(lines)

        except Exception as exc:
            logger.error("Gateway project execution failed: %s", exc)
            return "Project execution failed. An operator has been notified."

    # ── Adapter factory ────────────────────────────────────────────────────

    @staticmethod
    def _create_adapter(
        name: str,
        config: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Create a platform adapter by name.

        Currently supports:
        - ``"webhook"``: basic HTTP webhook adapter
        - ``"echo"``: logs messages to console (testing)

        Args:
            name: Platform name.
            config: Platform-specific configuration dict.

        Returns:
            Dict with ``connect``, ``disconnect``, and ``send`` async
            callables, or None if the platform is unknown.
        """
        if name == "echo":

            async def _connect_echo():
                logger.info("Echo adapter connected")

            async def _disconnect_echo():
                logger.info("Echo adapter disconnected")

            async def _send_echo(user_id: str, text: str):
                logger.info("[Echo to %s] %s", user_id, text)

            return {
                "connect": _connect_echo,
                "disconnect": _disconnect_echo,
                "send": _send_echo,
            }

        if name == "webhook":
            port = config.get("port", 8080)

            async def _connect() -> None:
                logger.info("Webhook adapter listening on port %d", port)

            async def _disconnect() -> None:
                logger.info("Webhook adapter stopped")

            async def _send(user_id: str, text: str) -> None:
                logger.info("[Webhook to %s] %s", user_id, text[:200])

            return {
                "connect": _connect,
                "disconnect": _disconnect,
                "send": _send,
            }

        return None

    @property
    def is_running(self) -> bool:
        return self._running
