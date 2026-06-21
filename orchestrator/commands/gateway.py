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
                    config.platforms[name.strip()] = {"port": int(val.strip())}
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
