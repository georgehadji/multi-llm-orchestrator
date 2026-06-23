"""cmd_dashboard command handler — extracted from cli.py."""

from __future__ import annotations


def execute(args) -> None:
    """Handle the 'dashboard' subcommand: render persistent cross-run learning."""
    import asyncio

    from ..metrics import render_dashboard
    from ..telemetry_store import TelemetryStore

    store = TelemetryStore()
    output = asyncio.run(render_dashboard(store, days=args.days))
    print(output)


def register(subparsers) -> None:
    """Register the 'dashboard' subcommand."""
    dp = subparsers.add_parser(
        "dashboard",
        help="Show persistent cross-run model rankings, task leaders, and recommendations",
    )
    dp.add_argument(
        "--days",
        type=int,
        default=30,
        metavar="N",
        help="Lookback window in days (default: 30)",
    )
    dp.set_defaults(func=execute)
