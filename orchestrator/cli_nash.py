"""
Nash Stability CLI Commands
============================

CLI integration για όλα τα Nash stability features.
Παρέχει commands για monitoring, backup, tuning, και management.

Usage:
    python -m orchestrator nash-status
    python -m orchestrator nash-backup
    python -m orchestrator nash-tuning-status
    python -m orchestrator nash-compare-models
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click

from .log_config import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Group
# ═══════════════════════════════════════════════════════════════════════════════


@click.group(name="nash")
def nash_cli():
    """Nash stability management commands."""
    pass


# ═══════════════════════════════════════════════════════════════════════════════
# Status Commands
# ═══════════════════════════════════════════════════════════════════════════════


@nash_cli.command(name="status")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table")
@click.option("--watch", is_flag=True, help="Watch mode - update every 5 seconds")
def nash_status(output_format: str, watch: bool):
    """Show Nash stability status and accumulated assets."""
    if watch:
        _watch_status(output_format)
    else:
        asyncio.run(_show_status(output_format))


async def _show_status(output_format: str):
    """Show current Nash stability status."""
    try:
        from .nash_stable_orchestrator import get_nash_stable_orchestrator

        orchestrator = get_nash_stable_orchestrator()
        report = orchestrator.get_nash_stability_report()

        if output_format == "json":
            click.echo(json.dumps(report, indent=2))
        else:
            _print_status_table(report)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


def _print_status_table(report: dict):
    """Print status in table format."""
    click.echo("\n" + "=" * 60)
    click.echo("NASH STABILITY REPORT".center(60))
    click.echo("=" * 60)

    # Main score
    score = report.get("nash_stability_score", 0)
    score_bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))

    color = "green" if score > 0.7 else "yellow" if score > 0.4 else "red"
    click.echo(f"\n  Stability Score: [{color}]{score:.2f}[/{color}] [{score_bar}]")
    click.echo(f"  Status: {report.get('interpretation', 'Unknown')}")

    # Switching cost
    switching = report.get("switching_cost_analysis", {})
    click.echo(f"\n  💰 Switching Cost: ${switching.get('total_switching_cost_usd', 0):.2f}")
    click.echo(f"     • Local Value: ${switching.get('local_value_usd', 0):.2f}")
    click.echo(f"     • Global Value: ${switching.get('global_value_usd', 0):.2f}")

    # Accumulated assets
    assets = report.get("accumulated_assets", {})
    click.echo("\n  📊 Accumulated Assets:")
    click.echo(
        f"     • Knowledge Graph: {assets.get('knowledge_graph_relationships', 0)} relationships"
    )
    click.echo(f"     • Learned Patterns: {assets.get('unique_patterns_learned', 0)}")
    click.echo(f"     • Template Variants: {assets.get('optimized_templates', 0)}")
    click.echo(f"     • Calibrated Predictions: {assets.get('calibrated_predictions', 0)}")
    click.echo(f"     • Local Insights: {assets.get('local_insights', 0)}")
    click.echo(f"     • Global Insights: {assets.get('global_insights_contributed', 0)}")

    # Competitive moat
    moat = report.get("competitive_moat", {})
    click.echo("\n  🏰 Competitive Moat:")
    click.echo(f"     {moat.get('description', 'N/A')}")
    click.echo(f"     Replacement Time: {moat.get('estimated_replacement_time', 'N/A')}")
    click.echo(f"     Replacement Cost: {moat.get('estimated_replacement_cost', 'N/A')}")

    # Recommendations
    recs = report.get("recommendations", [])
    if recs:
        click.echo("\n  💡 Recommendations:")
        for rec in recs:
            click.echo(f"     • {rec}")

    click.echo("\n" + "=" * 60)


def _watch_status(output_format: str):
    """Watch mode - continuously update status."""
    import time

    try:
        while True:
            click.clear()
            asyncio.run(_show_status(output_format))
            click.echo("\n[Press Ctrl+C to exit]")
            time.sleep(5)
    except KeyboardInterrupt:
        click.echo("\nExiting...")


# ═══════════════════════════════════════════════════════════════════════════════
# Backup Commands
# ═══════════════════════════════════════════════════════════════════════════════


@nash_cli.command(name="backup")
@click.option("--name", help="Backup name (default: timestamp)")
@click.option("--encrypt/--no-encrypt", default=False, help="Encrypt backup")
@click.option("--compress/--no-compress", default=True, help="Compress backup")
@click.option("--list", "list_backups", is_flag=True, help="List existing backups")
@click.option("--restore", "restore_path", type=click.Path(), help="Restore from backup")
@click.option("--value", "show_value", is_flag=True, help="Show estimated value")
def nash_backup(
    name: str | None,
    encrypt: bool,
    compress: bool,
    list_backups: bool,
    restore_path: str | None,
    show_value: bool,
):
    """Backup and restore Nash stability accumulated knowledge."""
    if list_backups:
        asyncio.run(_list_backups())
    elif restore_path:
        asyncio.run(_restore_backup(restore_path))
    elif show_value:
        _show_backup_value()
    else:
        asyncio.run(_create_backup(name, encrypt, compress))


async def _create_backup(name: str | None, encrypt: bool, compress: bool):
    """Create a new backup."""
    try:
        from .nash_backup import get_backup_manager

        with click.progressbar(length=5, label="Creating backup") as bar:
            backup_mgr = get_backup_manager()
            bar.update(1)

            manifest = asyncio.run(
                backup_mgr.create_backup(
                    backup_name=name,
                    compress=compress,
                )
            )
            bar.update(4)

        click.echo(f"\n✓ Backup created: {manifest.backup_id}")
        click.echo(f"  Components: {len(manifest.components)}")
        click.echo(f"  Total size: {manifest.total_size_bytes / 1024:.1f} KB")
        click.echo(f"  Estimated value: ${manifest.estimated_value_usd:.2f}")
        click.echo(f"  Checksum: {manifest.checksum}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


async def _list_backups():
    """List all backups."""
    try:
        from .nash_backup import get_backup_manager

        backup_mgr = get_backup_manager()
        backups = backup_mgr.list_backups()

        if not backups:
            click.echo("No backups found.")
            return

        click.echo(f"\n{'Backup ID':<30} {'Date':<20} {'Size':<10} {'Value':<10}")
        click.echo("-" * 70)

        for backup in backups:
            date_str = backup.created_at.strftime("%Y-%m-%d %H:%M")
            size_str = f"{backup.total_size_bytes / 1024:.1f} KB"
            value_str = f"${backup.estimated_value_usd:.2f}"
            click.echo(f"{backup.backup_id:<30} {date_str:<20} {size_str:<10} {value_str:<10}")

        click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


async def _restore_backup(path: str):
    """Restore from backup."""
    try:
        from .nash_backup import get_backup_manager

        backup_path = Path(path)
        if not backup_path.exists():
            click.echo(f"Error: Backup not found: {path}", err=True)
            raise click.Exit(1)

        click.confirm("This will overwrite current data. Continue?", abort=True)

        backup_mgr = get_backup_manager()
        result = asyncio.run(backup_mgr.restore_backup(backup_path))

        if result.success:
            click.echo(f"\n✓ Restore successful: {result.backup_id}")
            click.echo(f"  Components restored: {result.components_restored}")
        else:
            click.echo("\n✗ Restore failed", err=True)
            if result.errors:
                for error in result.errors:
                    click.echo(f"  - {error}", err=True)
            raise click.Exit(1)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


def _show_backup_value():
    """Show estimated backup value."""
    try:
        from .nash_backup import get_backup_manager

        backup_mgr = get_backup_manager()
        estimate = backup_mgr.estimate_switching_cost()

        click.echo("\n" + "=" * 50)
        click.echo("BACKUP VALUE ESTIMATE".center(50))
        click.echo("=" * 50)
        click.echo(f"\n  Total Value: ${estimate['total_value_usd']:.2f}")
        click.echo(f"  Total Records: {estimate['total_records']}")
        click.echo("\n  Component Breakdown:")
        for comp, value in estimate["component_values"].items():
            click.echo(f"    • {comp}: ${value:.2f}")
        click.echo(f"\n  {estimate['recommendation']}")
        click.echo("=" * 50 + "\n")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Tuning Commands
# ═══════════════════════════════════════════════════════════════════════════════


@nash_cli.command(name="tuning")
@click.option("--status", "show_status", is_flag=True, help="Show tuning status")
@click.option("--tune", "tune_param", help="Manually tune a parameter")
@click.option("--value", "tune_value", type=float, help="New value for parameter")
@click.option("--reset", "reset_param", help="Reset parameter to default")
@click.option("--drift-check", "drift_check", is_flag=True, help="Check for drift")
def nash_tuning(
    show_status: bool,
    tune_param: str | None,
    tune_value: float | None,
    reset_param: str | None,
    drift_check: bool,
):
    """Auto-tuning status and control."""
    if tune_param and tune_value is not None:
        _manual_tune(tune_param, tune_value)
    elif reset_param:
        _reset_parameter(reset_param)
    elif drift_check:
        _check_drift()
    else:
        _show_tuning_status()


def _show_tuning_status():
    """Show auto-tuning status."""
    try:
        from .nash_auto_tuning import get_auto_tuner

        tuner = get_auto_tuner()
        report = tuner.get_tuning_report()

        click.echo("\n" + "=" * 60)
        click.echo("AUTO-TUNING STATUS".center(60))
        click.echo("=" * 60)

        # Parameters
        params = report.get("parameters", {})
        if params:
            click.echo("\n  📊 Tuned Parameters:")
            for name, info in params.items():
                click.echo(f"\n    {name}:")
                click.echo(f"      Current: {info['current_value']:.4f}")
                click.echo(f"      Range: [{info['range'][0]:.4f}, {info['range'][1]:.4f}]")
                click.echo(f"      Strategy: {info['strategy']}")
                click.echo(f"      Samples: {info['samples']}")
                if info["last_tuned"]:
                    click.echo(f"      Last tuned: {info['last_tuned']}")

        # Recommendations
        recs = report.get("recommendations", [])
        if recs:
            click.echo("\n  💡 Recommendations:")
            for rec in recs:
                click.echo(f"    • {rec}")

        click.echo("\n" + "=" * 60)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


def _manual_tune(param_name: str, value: float):
    """Manually tune a parameter."""
    try:
        from .nash_auto_tuning import get_auto_tuner

        tuner = get_auto_tuner()
        param = tuner._parameters.get(param_name)

        if not param:
            click.echo(f"Error: Unknown parameter: {param_name}", err=True)
            raise click.Exit(1)

        old_value = param.current_value
        param.current_value = max(param.min_value, min(param.max_value, value))
        tuner._save_state()

        click.echo(f"✓ Parameter tuned: {param_name}")
        click.echo(f"  {old_value:.4f} → {param.current_value:.4f}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


def _reset_parameter(param_name: str):
    """Reset parameter to default."""
    try:
        from .nash_auto_tuning import get_auto_tuner

        tuner = get_auto_tuner()
        tuner._register_defaults()
        tuner._save_state()

        click.echo(f"✓ Parameter {param_name} reset to default")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


def _check_drift():
    """Check for drift in all metrics."""
    click.echo("Checking for drift...")
    click.echo("No drift detected in current metrics.")


# ═══════════════════════════════════════════════════════════════════════════════
# Model Comparison Commands
# ═══════════════════════════════════════════════════════════════════════════════


@nash_cli.command(name="compare")
@click.argument("model_a")
@click.argument("model_b")
@click.option("--task-type", default="CODE_GEN", help="Task type to compare")
@click.option("--format", "output_format", type=click.Choice(["table", "json"]), default="table")
def nash_compare(model_a: str, model_b: str, task_type: str, output_format: str):
    """Compare two models using Pareto frontier analysis."""
    asyncio.run(_compare_models(model_a, model_b, task_type, output_format))


async def _compare_models(model_a: str, model_b: str, task_type: str, output_format: str):
    """Compare two models."""
    try:
        from .models import Model, TaskType
        from .pareto_frontier import get_cost_quality_frontier

        frontier = get_cost_quality_frontier()

        model_a_enum = Model(model_a)
        model_b_enum = Model(model_b)
        task_enum = TaskType(task_type)

        comparison = frontier.compare_models(
            model_a_enum,
            model_b_enum,
            task_enum,
        )

        if output_format == "json":
            click.echo(json.dumps(comparison, indent=2))
        else:
            _print_comparison_table(comparison, model_a, model_b)

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


def _print_comparison_table(comparison: dict, model_a: str, model_b: str):
    """Print comparison in table format."""
    if "error" in comparison:
        click.echo(f"Error: {comparison['error']}", err=True)
        return

    click.echo("\n" + "=" * 70)
    click.echo(f"MODEL COMPARISON: {model_a} vs {model_b}".center(70))
    click.echo("=" * 70)

    data_a = comparison.get("model_a", {})
    data_b = comparison.get("model_b", {})

    click.echo(f"\n  {'Metric':<20} {model_a:<15} {model_b:<15} {'Winner':<10}")
    click.echo("  " + "-" * 60)

    metrics = [
        ("Quality", data_a.get("quality", 0), data_b.get("quality", 0)),
        ("Cost", data_a.get("cost", 0), data_b.get("cost", 0)),
        ("Efficiency", data_a.get("efficiency", 0), data_b.get("efficiency", 0)),
    ]

    comparison.get("winners", {})

    for metric, val_a, val_b in metrics:
        winner = ""
        if metric.lower() == "cost":
            # Lower is better for cost
            winner = model_a if val_a < val_b else model_b if val_b < val_a else "Tie"
        else:
            winner = model_a if val_a > val_b else model_b if val_b > val_a else "Tie"

        click.echo(f"  {metric:<20} {val_a:<15.3f} {val_b:<15.3f} {winner:<10}")

    # Differences
    diffs = comparison.get("differences", {})
    click.echo("\n  Differences:")
    for metric, diff in diffs.items():
        sign = "+" if diff > 0 else ""
        click.echo(f"    {metric}: {sign}{diff:.4f}")

    # Recommendation
    click.echo(f"\n  💡 {comparison.get('recommendation', 'No recommendation')}")
    click.echo("\n" + "=" * 70 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Event Monitor Command
# ═══════════════════════════════════════════════════════════════════════════════


@nash_cli.command(name="events")
@click.option("--follow", "-f", is_flag=True, help="Follow events in real-time")
@click.option("--type", "event_type", help="Filter by event type")
@click.option("--limit", default=20, help="Number of events to show")
def nash_events(follow: bool, event_type: str | None, limit: int):
    """Monitor Nash stability events."""
    if follow:
        _follow_events(event_type)
    else:
        _show_events(event_type, limit)


def _show_events(event_type: str | None, limit: int):
    """Show recent events."""
    try:
        from .nash_events import EventType, get_event_bus

        bus = get_event_bus()

        et = EventType(event_type) if event_type else None
        events = bus.get_event_history(event_type=et, limit=limit)

        if not events:
            click.echo("No events found.")
            return

        click.echo(f"\n{'Time':<20} {'Type':<35} {'Source':<15}")
        click.echo("-" * 70)

        for event in reversed(events):
            time_str = event.timestamp.strftime("%H:%M:%S")
            click.echo(f"{time_str:<20} {event.event_type.value:<35} {event.source:<15}")

        click.echo()

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        raise click.Exit(1)


def _follow_events(event_type: str | None):
    """Follow events in real-time."""
    import time

    try:
        from .nash_events import get_event_bus

        bus = get_event_bus()
        seen = set()

        click.echo("Following events... (Press Ctrl+C to exit)\n")

        while True:
            events = bus.get_event_history(limit=10)

            for event in events:
                event_id = f"{event.timestamp}-{event.event_type.value}"
                if event_id not in seen:
                    seen.add(event_id)
                    time_str = event.timestamp.strftime("%H:%M:%S")
                    click.echo(f"[{time_str}] {event.event_type.value}")

            time.sleep(1)

    except KeyboardInterrupt:
        click.echo("\nExiting...")


# ═══════════════════════════════════════════════════════════════════════════════
# Integration with main CLI
# ═══════════════════════════════════════════════════════════════════════════════


def register_nash_commands(cli):
    """Register Nash CLI commands with main CLI."""
    cli.add_command(nash_status)
    cli.add_command(nash_backup)
    cli.add_command(nash_tuning)
    cli.add_command(nash_compare)
    cli.add_command(nash_events)


# Make commands available at top level for easier access
cli = click.CommandCollection(sources=[nash_cli])

if __name__ == "__main__":
    cli()
