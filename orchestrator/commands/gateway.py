"""cmd_gateway command handler — extracted from cli.py."""

from __future__ import annotations


def execute(args) -> None:
    """Handle the 'gateway' subcommand: start/stop the messaging gateway."""
    import asyncio
    from .gateway.run import OrchestratorGateway, GatewayConfig

    async def _run():
        config = GatewayConfig()
        if args.platforms:
            for pair in args.platforms:
                if ":" in pair:
                    name, val = pair.split(":", 1)
                    try:
                        port = int(val.strip())
                    except ValueError as exc:
                        raise ValueError(
                            f"Invalid port for platform {name.strip()!r}: {val!r}"
                        ) from exc
                    if not 1 <= port <= 65535:
                        raise ValueError(
                            f"Port for platform {name.strip()!r} must be 1-65535, got {port}"
                        )
                    config.platforms[name.strip()] = {"port": port}
                else:
                    config.platforms[pair.strip()] = {}

        if args.command == "start":
            gw = OrchestratorGateway(config)
            try:
                await gw.start()
                print("Gateway running. Press Ctrl+C to stop.")
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down gateway...")
                await gw.shutdown()
        elif args.command == "status":
            gw = OrchestratorGateway(config)
            await gw.start()
            print(f"Gateway running: {gw.is_running}")
            print(f"Platforms: {list(config.platforms.keys())}")
            await gw.shutdown()

    asyncio.run(_run())


def register(subparsers) -> None:
    """Register the 'gateway' subcommand."""
    gp = subparsers.add_parser("gateway", help="Multi-platform messaging gateway")
    gp.add_argument("command", choices=["start", "status"], help="Gateway command")
    gp.add_argument(
        "--platforms",
        "-p",
        nargs="*",
        default=[],
        help="Platforms to enable (e.g. echo webhook:8080)",
    )
    gp.set_defaults(func=execute)
