"""
Slash Command Registry — Single Source of Truth for CLI Commands
================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Inspired by Hermes Agent's COMMAND_REGISTRY pattern: all CLI commands,
their aliases, categories, and metadata are defined in one place.
Every downstream consumer (CLI help, autocomplete, future gateway)
derives from this registry.

This does NOT replace argparse — it augments it with a declarative
source of truth for help text, autocomplete, and command resolution.

Usage:
    from orchestrator.command_registry import (
        COMMAND_REGISTRY, CommandDef, resolve_command,
        commands_by_category,
    )

    cmd = resolve_command("/resume")
    print(cmd.description)  # "Resume a previous project"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────

if TYPE_CHECKING:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

CommandHandler = Callable[..., Awaitable[str]]


@dataclass
class CommandDef:
    """Declarative command definition.

    Attributes:
        name: Canonical command name (without leading slash).
        description: Human-readable one-line description.
        category: Grouping category for help display.
            Standard: "Project", "Configuration", "Tools & Skills", "Info", "Exit"
        aliases: Alternative names that resolve to this command.
        args_hint: Argument placeholder shown in help (e.g. "<prompt>", "[name]").
        cli_only: If True, only available in interactive CLI (not gateway).
        handler: Optional async callable. When set, dispatch can use this
            directly instead of routing through argparse.
    """

    name: str
    description: str
    category: str = "Info"
    aliases: tuple[str, ...] = ()
    args_hint: str = ""
    cli_only: bool = False
    handler: CommandHandler | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Registry — single source of truth
# ─────────────────────────────────────────────────────────────────────────────

COMMAND_REGISTRY: list[CommandDef] = [
    # ── Project commands ──────────────────────────────────────────────────
    CommandDef(
        "new",
        "Start a new project",
        category="Project",
        aliases=("project", "start"),
        args_hint="<description>",
    ),
    CommandDef(
        "resume",
        "Resume a previous project",
        category="Project",
        args_hint="<project_id>",
    ),
    CommandDef(
        "analyze",
        "Analyze a codebase and produce an improvement report",
        category="Project",
        args_hint="<path>",
    ),
    CommandDef(
        "build",
        "Build an app from a description",
        category="Project",
        args_hint="<description>",
    ),
    CommandDef(
        "file",
        "Load project spec from a YAML file",
        category="Project",
        args_hint="<file>",
    ),
    CommandDef(
        "dashboard",
        "Show persistent cross-run model rankings",
        category="Project",
        args_hint="[--days N]",
    ),
    CommandDef(
        "visualize",
        "Print the task dependency graph",
        category="Project",
        aliases=("dag",),
        args_hint="[mermaid|ascii]",
    ),
    # ── Configuration commands ────────────────────────────────────────────
    CommandDef(
        "model",
        "Show or change the active model",
        category="Configuration",
        args_hint="[model_name]",
    ),
    CommandDef(
        "budget",
        "Show remaining budget and time",
        category="Configuration",
        aliases=("funds",),
    ),
    CommandDef(
        "agent",
        "Convert natural language intent to typed specs",
        category="Configuration",
        args_hint="<intent>",
    ),
    CommandDef(
        "cache-stats",
        "Show cache statistics and manage cache",
        category="Configuration",
        aliases=("cache",),
    ),
    CommandDef(
        "nexus",
        "Nexus Search — web search for the AI Orchestrator",
        category="Configuration",
        args_hint="<query>",
    ),
    CommandDef(
        "meta",
        "Meta-optimization management (A/B testing, HITL, rollout)",
        category="Configuration",
    ),
    # ── Tools & Skills ────────────────────────────────────────────────────
    CommandDef(
        "slash",
        "Execute slash commands (/analyst, /architect, /implement, etc.)",
        category="Tools & Skills",
        args_hint="<command> [args]",
    ),
    CommandDef(
        "nash",
        "Nash stability management",
        category="Tools & Skills",
        args_hint="<verb> [options]",
    ),
    # ── Info commands ─────────────────────────────────────────────────────
    CommandDef(
        "help",
        "Show available commands and their descriptions",
        category="Info",
        aliases=("h", "?"),
    ),
    CommandDef(
        "list-projects",
        "List all saved projects",
        category="Info",
        aliases=("projects", "ls"),
    ),
    CommandDef(
        "dry-run",
        "Show execution plan without running any tasks",
        category="Info",
        args_hint="<description>",
    ),
    # ── Exit commands ─────────────────────────────────────────────────────
    CommandDef(
        "quit",
        "Exit the orchestrator",
        category="Exit",
        aliases=("exit", "q"),
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def resolve_command(name: str) -> CommandDef | None:
    """Resolve a command name or alias to its CommandDef.

    Handles leading slashes, lowercasing, and alias lookup.

    Args:
        name: Command name or alias, optionally with leading ``/``
              (e.g. ``"/resume"``, ``"funds"``, ``"/h"``).

    Returns:
        The matching CommandDef or None if not found.
    """
    name = name.lstrip("/").lower().strip()
    for cmd in COMMAND_REGISTRY:
        if cmd.name == name or name in cmd.aliases:
            return cmd
    return None


def commands_by_category() -> dict[str, list[CommandDef]]:
    """Group all commands by their category for help display.

    Returns:
        Dict mapping category name → list of CommandDef in that category.
        Categories preserve registry order within each group.
    """
    result: dict[str, list[CommandDef]] = {}
    for cmd in COMMAND_REGISTRY:
        result.setdefault(cmd.category, []).append(cmd)
    return result


def command_names() -> list[str]:
    """Return all canonical command names for autocomplete."""
    return [cmd.name for cmd in COMMAND_REGISTRY]


def command_aliases() -> list[str]:
    """Return all aliases for autocomplete."""
    result: list[str] = []
    for cmd in COMMAND_REGISTRY:
        result.extend(cmd.aliases)
    return result


def all_slash_commands() -> list[str]:
    """Return all canonical names and aliases with leading slashes."""
    return [f"/{cmd.name}" for cmd in COMMAND_REGISTRY] + [
        f"/{alias}" for cmd in COMMAND_REGISTRY for alias in cmd.aliases
    ]
