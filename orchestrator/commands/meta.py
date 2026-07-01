"""cmd_meta command handler — extracted from cli.py."""

from __future__ import annotations


def execute(args) -> None:
    """Handle meta-optimization subcommands."""
    import asyncio

    # Initialize a minimal orchestrator to access meta_v2
    from ..engine import Orchestrator
    from ..meta_integration import get_meta_status

    async def run():
        orch = Orchestrator()
        await orch.__aenter__()

        try:
            if orch.meta_v2:
                status = get_meta_status(orch.meta_v2)

                if args.meta_cmd == "status":
                    print("\n=== META-OPTIMIZATION STATUS ===")
                    print(f"Enabled: {status.get('enabled', False)}")
                    print(f"Optimizations run: {status.get('optimization_count', 0)}")

                    if "archive_stats" in status:
                        print("\nArchive:")
                        print(
                            f"  Total projects: {status['archive_stats'].get('total_projects', 0)}"
                        )
                        print(
                            f"  Total executions: {status['archive_stats'].get('total_executions', 0)}"
                        )

                    if "ab_testing" in status:
                        print("\nA/B Testing:")
                        print(
                            f"  Total experiments: {status['ab_testing'].get('total_experiments', 0)}"
                        )
                        print(
                            f"  Active experiments: {status['ab_testing'].get('active_experiments', 0)}"
                        )

                    if "hitl" in status:
                        print("\nHITL:")
                        print(f"  Pending requests: {status['hitl'].get('pending_count', 0)}")
                        print(f"  Auto-approved: {status['hitl'].get('auto_approved_count', 0)}")

                    if "rollout" in status:
                        print("\nGradual Rollout:")
                        print(f"  Active rollouts: {status['rollout'].get('active_rollouts', 0)}")
                        print(
                            f"  Completed rollouts: {status['rollout'].get('completed_rollouts', 0)}"
                        )

                elif args.meta_cmd == "optimize":
                    print("\nRunning meta-optimization...")
                    outcomes = await orch.meta_v2.maybe_optimize()

                    if outcomes:
                        print(f"\nGenerated {len(outcomes)} proposals:")
                        for outcome in outcomes:
                            print(
                                f"  [{outcome.decision.value}] {outcome.proposal.description[:60]}..."
                            )
                    else:
                        print(
                            "No optimization proposals generated (insufficient data or no improvements found)"
                        )

                elif args.meta_cmd == "transfer":
                    from ..transfer_learning import get_transfer_engine

                    transfer_engine = get_transfer_engine()

                    if transfer_engine:
                        stats = transfer_engine.get_stats()
                        print("\n=== TRANSFER LEARNING STATUS ===")
                        print(f"Indexed projects: {stats.get('indexed_projects', 0)}")
                        print(f"Active patterns: {stats.get('active_patterns', 0)}")
                        print(f"Total patterns: {stats.get('total_patterns', 0)}")
                        print(
                            f"Average pattern confidence: {stats.get('avg_pattern_confidence', 0):.2f}"
                        )
                    else:
                        print("Transfer learning not initialized")
            else:
                print("Meta-optimization not available")

        finally:
            await orch.__aexit__(None, None, None)

    asyncio.run(run())


def register(subparsers):
    """Setup meta-optimization subcommand parser."""
    meta_parser = subparsers.add_parser(
        "meta",
        help="Meta-optimization management (A/B testing, HITL, rollout, transfer learning)",
    )

    meta_subparsers = meta_parser.add_subparsers(dest="meta_cmd", help="Meta-optimization command")

    # Status command
    status_parser = meta_subparsers.add_parser("status", help="Show meta-optimization status")
    status_parser.set_defaults(func=execute)

    # Optimize command
    optimize_parser = meta_subparsers.add_parser("optimize", help="Run meta-optimization cycle")
    optimize_parser.set_defaults(func=execute)

    # Transfer command
    transfer_parser = meta_subparsers.add_parser("transfer", help="Show transfer learning status")
    transfer_parser.set_defaults(func=execute)
