"""codebase command module."""
from __future__ import annotations



def register(subparsers) -> None:
    # Register the modify subcommand for codebase-aware operations.
    mp = subparsers.add_parser(
        "modify",
        help="Modify an existing codebase using AI reasoning",
    )
    mp.add_argument("--repo", required=True, help="Path to codebase root")
    mp.add_argument("--objective", required=True, help="What to do")
    mp.add_argument("--budget", type=float, default=10.0, help="Max LLM budget USD")
    mp.add_argument("--dry-run", action="store_true", help="Plan only")
    mp.set_defaults(func=_handle_modify_command)
