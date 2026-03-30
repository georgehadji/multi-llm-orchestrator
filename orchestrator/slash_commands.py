"""
Slash Commands - Interactive agent commands inspired by bmalph
===============================================================
Provides quick access to orchestrator agents via slash commands:
- /analyst - Product analysis and brief creation
- /architect - Architecture planning and rules generation
- /implement - Start implementation phase
- /decompose - Task decomposition
- /status - Show project status
- /help - List all available commands
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .models import Model, TaskType

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .api_clients import UnifiedClient

logger = logging.getLogger("orchestrator.slash")


@dataclass
class SlashCommand:
    """Definition of a slash command."""
    name: str
    description: str
    handler: Callable[[str, SlashCommandContext], Awaitable[str]]
    aliases: list[str] = None

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


@dataclass
class SlashCommandContext:
    """Context passed to slash command handlers."""
    client: UnifiedClient
    output_dir: Path
    project_id: str
    conversation_history: list[dict]

    def __init__(self, client: UnifiedClient, output_dir: Path, project_id: str):
        self.client = client
        self.output_dir = output_dir
        self.project_id = project_id
        self.conversation_history = []


class SlashCommandRegistry:
    """Registry and dispatcher for slash commands."""

    def __init__(self):
        self._commands: dict[str, SlashCommand] = {}
        self._setup_default_commands()

    def _setup_default_commands(self):
        """Register default slash commands."""
        self.register(SlashCommand(
            name="analyst",
            description="Product analysis agent - create briefs, research market/domain",
            handler=self._cmd_analyst,
            aliases=["analysis", "research"]
        ))

        self.register(SlashCommand(
            name="architect",
            description="Architecture planning - generate rules and technical design",
            handler=self._cmd_architect,
            aliases=["architecture", "design"]
        ))

        self.register(SlashCommand(
            name="implement",
            description="Start implementation phase with generated tasks",
            handler=self._cmd_implement,
            aliases=["build", "code"]
        ))

        self.register(SlashCommand(
            name="decompose",
            description="Decompose project into atomic tasks",
            handler=self._cmd_decompose,
            aliases=["breakdown", "tasks"]
        ))

        self.register(SlashCommand(
            name="status",
            description="Show current project status and progress",
            handler=self._cmd_status,
            aliases=["info", "progress"]
        ))

        self.register(SlashCommand(
            name="help",
            description="List all available slash commands",
            handler=self._cmd_help,
            aliases=["?", "commands"]
        ))

        self.register(SlashCommand(
            name="review",
            description="Cross-model code review",
            handler=self._cmd_review,
            aliases=["critique", "audit"]
        ))

        self.register(SlashCommand(
            name="models",
            description="Show available models and their health status",
            handler=self._cmd_models,
            aliases=["providers", "llms"]
        ))

    def register(self, cmd: SlashCommand):
        """Register a slash command."""
        self._commands[cmd.name] = cmd
        for alias in cmd.aliases:
            self._commands[alias] = cmd

    def get(self, name: str) -> SlashCommand | None:
        """Get a command by name or alias."""
        return self._commands.get(name.lower().lstrip('/'))

    def list_commands(self) -> list[SlashCommand]:
        """List all unique commands (excluding aliases)."""
        seen = set()
        result = []
        for cmd in self._commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                result.append(cmd)
        return sorted(result, key=lambda c: c.name)

    async def execute(self, command_line: str, ctx: SlashCommandContext) -> str:
        """Execute a slash command from user input."""
        parts = command_line.strip().split(maxsplit=1)
        if not parts:
            return "No command provided. Type /help for available commands."

        cmd_name = parts[0].lower().lstrip('/')
        args = parts[1] if len(parts) > 1 else ""

        cmd = self.get(cmd_name)
        if not cmd:
            return f"Unknown command: /{cmd_name}. Type /help for available commands."

        try:
            result = await cmd.handler(args, ctx)
            ctx.conversation_history.append({
                "role": "user",
                "content": command_line,
            })
            ctx.conversation_history.append({
                "role": "assistant",
                "content": result,
            })
            return result
        except Exception as e:
            logger.error(f"Command /{cmd.name} failed: {e}")
            return f"Error executing /{cmd.name}: {str(e)}"

    # ─────────────────────────────────────────
    # Command Handlers
    # ─────────────────────────────────────────

    async def _cmd_analyst(self, args: str, ctx: SlashCommandContext) -> str:
        """Product analysis agent."""
        prompt = args or "Analyze this project and create a product brief"

        system = """You are a Product Analyst. Your job is to:
1. Understand user needs and market context
2. Research competitors and domain requirements
3. Create clear product briefs with goals and constraints
4. Identify key features and success metrics

Provide structured output with clear sections."""

        response = await ctx.client.call(
            Model.GPT_4O, prompt, system=system, max_tokens=2000
        )

        # Save to progressive output
        brief_dir = ctx.output_dir / "001-analysis"
        brief_dir.mkdir(parents=True, exist_ok=True)
        brief_file = brief_dir / "product-brief.md"
        brief_file.write_text(response.text, encoding="utf-8")

        return f"**Product Analysis Complete**\n\n{response.text[:500]}...\n\nSaved to: {brief_file}"

    async def _cmd_architect(self, args: str, ctx: SlashCommandContext) -> str:
        """Architecture planning agent."""
        prompt = args or "Generate architecture rules and technical design"

        system = """You are a Software Architect. Your job is to:
1. Design system architecture based on requirements
2. Choose appropriate tech stack and patterns
3. Define component structure and interfaces
4. Plan for scalability, security, and maintainability

Use BMAD-style architecture documentation format."""

        response = await ctx.client.call(
            Model.GPT_4O, prompt, system=system, max_tokens=2000
        )

        # Save to progressive output
        arch_dir = ctx.output_dir / "002-architecture"
        arch_dir.mkdir(parents=True, exist_ok=True)
        arch_file = arch_dir / "architecture.md"
        arch_file.write_text(response.text, encoding="utf-8")

        return f"**Architecture Design Complete**\n\n{response.text[:500]}...\n\nSaved to: {arch_file}"

    async def _cmd_implement(self, args: str, ctx: SlashCommandContext) -> str:
        """Start implementation phase."""
        return """**Implementation Phase Started**

The orchestrator will now:
1. Decompose project into tasks
2. Execute tasks with optimal model routing
3. Cross-review outputs
4. Validate with deterministic checks

Use `python -m orchestrator --file <project.yaml>` to run full workflow.

For interactive mode, use the dashboard."""

    async def _cmd_decompose(self, args: str, ctx: SlashCommandContext) -> str:
        """Task decomposition."""
        prompt = args or "Decompose this project into atomic, executable tasks"

        system = """You are a Project Decomposer. Break down projects into:
- Atomic tasks (one clear deliverable each)
- With dependencies forming a DAG
- Task types: code_generation, code_review, data_extraction, etc.

Return JSON array with id, type, prompt, dependencies for each task."""

        response = await ctx.client.call(
            Model.GPT_4O_MINI, prompt, system=system, max_tokens=1500
        )

        # Save to progressive output
        decomp_dir = ctx.output_dir / "003-decomposition"
        decomp_dir.mkdir(parents=True, exist_ok=True)
        decomp_file = decomp_dir / "tasks.json"
        decomp_file.write_text(response.text, encoding="utf-8")

        return f"**Task Decomposition Complete**\n\n{response.text[:800]}...\n\nSaved to: {decomp_file}"

    async def _cmd_status(self, args: str, ctx: SlashCommandContext) -> str:
        """Show project status."""
        status_file = ctx.output_dir / "status.json"
        if status_file.exists():
            import json
            status = json.loads(status_file.read_text())
            return f"""**Project Status: {status.get('status', 'Unknown')}**

- Budget: ${status.get('budget_used', 0):.4f} / ${status.get('budget_total', 0):.2f}
- Tasks: {status.get('completed_tasks', 0)} / {status.get('total_tasks', 0)} completed
- Time: {status.get('elapsed_time', 0):.1f}s elapsed
"""
        return "No status file found. Project may not have started yet."

    async def _cmd_help(self, args: str, ctx: SlashCommandContext) -> str:
        """List all commands."""
        cmds = self.list_commands()
        lines = ["**Available Slash Commands**\n"]
        for cmd in cmds:
            aliases = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
            lines.append(f"**/{cmd.name}**{aliases}\n  {cmd.description}\n")
        return "\n".join(lines)

    async def _cmd_review(self, args: str, ctx: SlashCommandContext) -> str:
        """Cross-model code review."""
        if not args:
            return "Usage: /review <code or file path>"

        # If args is a file path, read it
        code = args
        code_path = Path(args)
        if code_path.exists():
            code = code_path.read_text(encoding="utf-8")

        # Primary review with GPT-4o
        review1 = await ctx.client.call(
            Model.GPT_4O,
            f"Review this code for correctness, completeness, and quality:\n\n{code}",
            system="You are a code reviewer. Be thorough and specific.",
            max_tokens=1500
        )

        # Secondary review with Gemini
        review2 = await ctx.client.call(
            Model.GEMINI_FLASH,
            f"Review this code focusing on different aspects than:\n{review1.text}\n\nCode:\n{code}",
            system="You are a code reviewer. Find issues the first reviewer missed.",
            max_tokens=1500
        )

        return f"""**Cross-Model Code Review**

**GPT-4o Review:**
{review1.text[:600]}...

**Gemini Flash Review:**
{review2.text[:600]}...

Combined cost: ${review1.cost_usd + review2.cost_usd:.4f}"""

    async def _cmd_models(self, args: str, ctx: SlashCommandContext) -> str:
        """Show available models."""
        from .models import COST_TABLE, ROUTING_TABLE

        lines = ["**Available Models**\n"]

        # Group by provider
        providers = {}
        for model in Model:
            from .models import get_provider
            provider = get_provider(model)
            if provider not in providers:
                providers[provider] = []
            providers[provider].append(model)

        for provider, models in sorted(providers.items()):
            lines.append(f"\n**{provider.upper()}**")
            for model in models[:3]:  # Show first 3 per provider
                cost = COST_TABLE.get(model, {"input": 0, "output": 0})
                lines.append(f"  • {model.value}: ${cost['input']:.3f}/${cost['output']:.3f} per 1M tokens")
            if len(models) > 3:
                lines.append(f"  ... and {len(models) - 3} more")

        lines.append("\n**Routing Priority (CODE_GEN):**")
        for i, model in enumerate(ROUTING_TABLE.get(TaskType.CODE_GEN, [])[:5], 1):
            cost = COST_TABLE.get(model, {"input": 0, "output": 0})
            lines.append(f"  {i}. {model.value} (${cost['output']:.3f})")

        return "\n".join(lines)


# Global registry instance
_slash_registry: SlashCommandRegistry | None = None


def get_slash_registry() -> SlashCommandRegistry:
    """Get or create the global slash command registry."""
    global _slash_registry
    if _slash_registry is None:
        _slash_registry = SlashCommandRegistry()
    return _slash_registry
