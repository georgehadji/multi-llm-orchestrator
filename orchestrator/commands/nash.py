"""nash command module."""
from __future__ import annotations



def register(subparsers):
    """Add Nash stability subcommands."""
    nash_parser = subparsers.add_parser("nash", help="Nash stability management")
    nash_subparsers = nash_parser.add_subparsers(dest="nash_command", metavar="COMMAND")

    # nash status
    status_parser = nash_subparsers.add_parser("status", help="Show Nash stability status")
    status_parser.add_argument("--format", choices=["table", "json"], default="table")
    status_parser.add_argument("--watch", action="store_true", help="Watch mode")
    status_parser.set_defaults(func=status)

    # nash backup
    backup_parser = nash_subparsers.add_parser(
        "backup", help="Backup/restore accumulated knowledge"
    )
    backup_parser.add_argument("--list", action="store_true", help="List backups")
    backup_parser.add_argument("--restore", type=str, help="Restore from backup file")
    backup_parser.add_argument("--value", action="store_true", help="Show estimated value")
    backup_parser.set_defaults(func=backup)

    # nash tuning
    tuning_parser = nash_subparsers.add_parser("tuning", help="Auto-tuning control")
    tuning_parser.add_argument("--status", action="store_true", help="Show tuning status")
    tuning_parser.add_argument("--tune", type=str, help="Parameter to tune")
    tuning_parser.add_argument("--value", type=float, help="New value for parameter")
    tuning_parser.set_defaults(func=tuning)

    # nash compare
    compare_parser = nash_subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("model_a", help="First model to compare")
    compare_parser.add_argument("model_b", help="Second model to compare")
    compare_parser.add_argument("--task-type", default="CODE_GEN", help="Task type")
    compare_parser.set_defaults(func=compare)

def backup(args):
    """Handle nash backup command."""
    import asyncio

    from orchestrator.nash_backup import get_backup_manager

    async def run():
        mgr = get_backup_manager()

        if args.list:
            backups = mgr.list_backups()
            if not backups:
                print("No backups found.")
                return
            print(f"\n{'Backup ID':<30} {'Date':<20} {'Size':<10} {'Value':<10}")
            print("-" * 70)
            for b in backups:
                date_str = b.created_at.strftime("%Y-%m-%d %H:%M")
                size_str = f"{b.total_size_bytes / 1024:.1f} KB"
                value_str = f"${b.estimated_value_usd:.2f}"
                print(f"{b.backup_id:<30} {date_str:<20} {size_str:<10} {value_str:<10}")

        elif args.restore:
            result = await mgr.restore_backup(args.restore)
            if result.success:
                print(f"✓ Restored: {result.backup_id}")
            else:
                print("✗ Restore failed")
                for error in result.errors:
                    print(f"  - {error}")

        elif args.value:
            estimate = mgr.estimate_switching_cost()
            print(f"\nEstimated Value: ${estimate['total_value_usd']:.2f}")
            print(f"Total Records: {estimate['total_records']}")

        else:
            manifest = await mgr.create_backup()
            print(f"✓ Backup created: {manifest.backup_id}")
            print(f"  Components: {len(manifest.components)}")
            print(f"  Size: {manifest.total_size_bytes / 1024:.1f} KB")
            print(f"  Value: ${manifest.estimated_value_usd:.2f}")

    asyncio.run(run())


def status(args):
    """Handle nash status command."""
    import asyncio

    from orchestrator.nash_stable_orchestrator import get_nash_stable_orchestrator

    async def show():
        orch = get_nash_stable_orchestrator()
        report = orch.get_nash_stability_report()

        if args.format == "json":
            import json

            print(json.dumps(report, indent=2))
        else:
            _print_nash_status(report)

    if args.watch:
        import time

        try:
            while True:
                import os

                os.system("cls" if os.name == "nt" else "clear")  # nosec B605 — terminal clear only
                asyncio.run(show())
                print("\n[Press Ctrl+C to exit]")
                # Sync poll loop — asyncio.sleep() cannot be used outside an async
                # function. time.sleep() between asyncio.run() calls is correct here.
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nExiting...")
    else:
        asyncio.run(show())


def _print_nash_status(report):
    """Print Nash status in table format."""
    print("\n" + "=" * 60)
    print("NASH STABILITY REPORT".center(60))
    print("=" * 60)

    score = report.get("nash_stability_score", 0)
    score_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
    print(f"\n  Stability Score: {score:.2f} [{score_bar}]")
    print(f"  Status: {report.get('interpretation', 'Unknown')}")

    switching = report.get("switching_cost_analysis", {})
    print(f"\n  Switching Cost: ${switching.get('total_switching_cost_usd', 0):.2f}")

    assets = report.get("accumulated_assets", {})
    print("\n  Accumulated Assets:")
    print(f"    • Knowledge Graph: {assets.get('knowledge_graph_relationships', 0)} relationships")
    print(f"    • Learned Patterns: {assets.get('learned_patterns', 0)}")
    print(f"    • Template Variants: {assets.get('optimized_templates', 0)}")
    print(f"    • Calibrated Predictions: {assets.get('calibrated_predictions', 0)}")

    print("\n" + "=" * 60)


def tuning(args):
    """Handle nash tuning command."""
    from orchestrator.nash_auto_tuning import get_auto_tuner

    tuner = get_auto_tuner()

    if args.status or (not args.tune):
        report = tuner.get_tuning_report()
        print("\nAuto-Tuning Status:")
        print("=" * 50)
        for name, info in report.get("parameters", {}).items():
            print(f"\n{name}:")
            print(f"  Current: {info['current_value']:.4f}")
            print(f"  Strategy: {info['strategy']}")
            print(f"  Samples: {info['samples']}")

    elif args.tune and args.value is not None:
        param = tuner._parameters.get(args.tune)
        if param:
            old = param.current_value
            param.current_value = max(param.min_value, min(param.max_value, args.value))
            tuner._save_state()
            print(f"✓ Tuned {args.tune}: {old:.4f} → {param.current_value:.4f}")
        else:
            print(f"Unknown parameter: {args.tune}")


def compare(args):
    """Handle nash compare command."""
    import asyncio

    from orchestrator.models import Model, TaskType
    from orchestrator.pareto_frontier import get_cost_quality_frontier

    async def run():
        frontier = get_cost_quality_frontier()
        try:
            model_a = Model(args.model_a)
            model_b = Model(args.model_b)
            task_type = TaskType(args.task_type)

            comparison = frontier.compare_models(model_a, model_b, task_type)

            print("\n" + "=" * 70)
            print(f"MODEL COMPARISON: {args.model_a} vs {args.model_b}".center(70))
            print("=" * 70)

            data_a = comparison.get("model_a", {})
            data_b = comparison.get("model_b", {})

            print(f"\n  {'Metric':<15} {args.model_a:<12} {args.model_b:<12}")
            print("  " + "-" * 40)
            print(
                f"  {'Quality':<15} {data_a.get('quality', 0):<12.3f} {data_b.get('quality', 0):<12.3f}"
            )
            print(f"  {'Cost':<15} ${data_a.get('cost', 0):<11.4f} ${data_b.get('cost', 0):<11.4f}")
            print(
                f"  {'Efficiency':<15} {data_a.get('efficiency', 0):<12.1f} {data_b.get('efficiency', 0):<12.1f}"
            )

            print(f"\n  {comparison.get('recommendation', '')}")
            print("=" * 70 + "\n")

        except ValueError as e:
            print(f"Error: {e}")

    asyncio.run(run())
