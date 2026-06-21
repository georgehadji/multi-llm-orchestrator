"""cmd_dashboard command handler — extracted from cli.py."""

from __future__ import annotations

def execute(args) -> None:
    """Handle the 'dashboard' subcommand: render persistent cross-run learning."""
    from .metrics import render_dashboard
    from .telemetry_store import TelemetryStore

    store = TelemetryStore()
    output = asyncio.run(render_dashboard(store, days=args.days))
    print(output)
