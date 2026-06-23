"""cmd_slash command handler — extracted from cli.py."""

from __future__ import annotations

def execute(args) -> None:
    """Handle the 'slash' subcommand."""
    import asyncio
    from datetime import datetime
    from pathlib import Path

    from ..api_clients import UnifiedClient
    from ..cache import DiskCache
    from ..slash_commands import SlashCommandContext, get_slash_registry

    registry = get_slash_registry()
    cache = DiskCache()
    client = UnifiedClient(cache=cache)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ctx = SlashCommandContext(
        client=client,
        output_dir=output_dir,
        project_id=f"slash_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    if args.interactive or not args.command:
        # Interactive REPL mode
        print("╔══════════════════════════════════════════════════════════╗")
        print("║     Multi-LLM Orchestrator - Slash Command Mode          ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print("\nType /help for available commands, or /quit to exit\n")
        print(asyncio.run(registry.execute("/help", ctx)))

        while True:
            try:
                user_input = input("orchestrator> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                    print("Goodbye!")
                    break
                if not user_input.startswith("/"):
                    user_input = "/" + user_input

                result = asyncio.run(registry.execute(user_input, ctx))
                print(f"\n{result}\n")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    else:
        # Single command mode
        cmd_line = f"/{args.command} {args.args}"
        result = asyncio.run(registry.execute(cmd_line, ctx))
        print(result)


def register(subparsers) -> None:
    """Register the 'slash' subcommand for interactive agent commands."""
    sp = subparsers.add_parser(
        "slash",
        help="Interactive slash commands (/analyst, /architect, /implement, etc.)",
    )
    sp.add_argument(
        "command",
        nargs="?",
        default="",
        help="Slash command to execute (e.g., 'analyst', 'architect', 'help')",
    )
    sp.add_argument(
        "--args",
        "-a",
        default="",
        help="Arguments for the slash command",
    )
    sp.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="./slash_outputs",
        help="Directory for progressive output",
    )
    sp.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enter interactive REPL mode",
    )
    sp.set_defaults(func=execute)
