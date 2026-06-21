"""cmd_chat command handler — extracted from cli.py."""

from __future__ import annotations

def execute(args) -> int:
    """Handle the 'chat' subcommand — launch the interactive session."""
    from orchestrator.application.chat_cli import run_chat

    asyncio.run(
        run_chat(
            budget=getattr(args, "budget", 8.0),
            dry_run=getattr(args, "dry_run", False),
            output_dir=getattr(args, "output_dir", ""),
        )
    )
    return 0
