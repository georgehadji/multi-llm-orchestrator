"""cmd_chat command handler — extracted from cli.py."""

from __future__ import annotations

def execute(args) -> int:
    """Handle the 'chat' subcommand — launch the interactive session."""
    import asyncio

    from orchestrator.application.chat_cli import run_chat

    asyncio.run(
        run_chat(
            budget=getattr(args, "budget", 8.0),
            dry_run=getattr(args, "dry_run", False),
            output_dir=getattr(args, "output_dir", ""),
        )
    )
    return 0


def register(subparsers) -> None:
    """Register the 'chat' subcommand."""
    p = subparsers.add_parser(
        "chat",
        help="Interactive mode — describe what you want to build in conversation",
    )
    p.add_argument(
        "--budget",
        "-b",
        type=float,
        default=8.0,
        help="Max LLM budget in USD for the build (default: 8.0)",
    )
    p.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="",
        help="Write generated files to this directory",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the final spec but do not start the build",
    )
    p.set_defaults(func=execute)
