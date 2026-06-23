#!/usr/bin/env python3

"""
CLI Entry Point — thin dispatch to application.cli_dispatch.run()
===================================================================
Phase 5 Strangler Fig: argument parsing, async handlers, and helpers
extracted to ``application/cli_dispatch.py`` and ``application/cli_helpers.py``.

This file retains only the Nash/NexusScope stubs (needed by dynamic command
registration), the Click-based codebase-analysis entry point, and the
``main()`` dispatch wrapper that delegates to ``cli_dispatch.run()``.
"""

from __future__ import annotations

import logging

from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger("orchestrator.cli")

# ---------------------------------------------------------------------------
# Nash subcommand stubs (registered via commands/nash.py:register())
# ---------------------------------------------------------------------------


def _cmd_nash_status(args):
    """Delegates to commands.nash.status."""
    from .commands.nash import status

    status(args)


def _print_nash_status(report):
    """Print Nash stability report in table format."""
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
    print(
        f"    \u2022 Knowledge Graph: {assets.get('knowledge_graph_relationships', 0)} relationships"
    )
    print(f"    \u2022 Learned Patterns: {assets.get('learned_patterns', 0)}")
    print(f"    \u2022 Template Variants: {assets.get('optimized_templates', 0)}")
    print(f"    \u2022 Calibrated Predictions: {assets.get('calibrated_predictions', 0)}")
    print("\n" + "=" * 60)


def _cmd_nash_backup(args):
    """Delegates to commands.nash.backup."""
    from .commands.nash import backup

    backup(args)


def _cmd_nash_tuning(args):
    """Delegates to commands.nash.tuning."""
    from .commands.nash import tuning

    tuning(args)


def _cmd_nash_compare(args):
    """Delegates to commands.nash.compare."""
    from .commands.nash import compare

    compare(args)


# Keep cmd_cache_stats for potential external imports
def cmd_cache_stats(args) -> None:
    """Cache stats — delegates to commands.cache_stats."""
    from .commands.cache_stats import execute

    execute(args)


# ---------------------------------------------------------------------------
# Codebase modification helpers
# ---------------------------------------------------------------------------


def _handle_modify_command(args):
    """Delegates to commands.codebase.execute."""
    from .commands.codebase import execute

    execute(args)


# ---------------------------------------------------------------------------
# Click-based CLI for codebase analysis feature (optional)
# ---------------------------------------------------------------------------

try:
    import click

    @click.command()
    @click.option(
        "--analyze-codebase", type=click.Path(exists=True), help="Analyze an existing codebase"
    )
    def cli(analyze_codebase):
        """AI Orchestrator: Codebase Analysis"""
        if analyze_codebase:
            import asyncio

            from .codebase_understanding import CodebaseUnderstanding
            from .improvement_suggester import ImprovementSuggester

            async def run_analysis():
                understanding = CodebaseUnderstanding(analyze_codebase)
                report = await understanding.analyze()

                suggester = ImprovementSuggester(report)
                suggestions = suggester.suggest()

                print(f"\nCodebase Analysis: {analyze_codebase}")
                print(f"  Files: {len(report.files)}")
                print(f"  Patterns: {len(report.patterns)}")
                print(f"  Improvements proposed: {len(suggestions)}")
                print()
                for s in suggestions:
                    print(f"  [{s.category}] {s.title}")
                    print(f"   Impact: {s.impact}")
                    print()

            asyncio.run(run_analysis())
        else:
            click.echo("Specify --analyze-codebase")

except ImportError:
    # Click not available, define a no-op function
    def cli(*args, **kwargs):
        print("Click not installed")


# ---------------------------------------------------------------------------
# NexusScope stubs (registered via commands/nexusscope.py:register())
# ---------------------------------------------------------------------------


def _cmd_nexusscope_sessions(args):
    """Delegates to commands.nexusscope.sessions."""
    from .commands.nexusscope import sessions

    sessions(args)


def _cmd_nexusscope_report(args):
    """Delegates to commands.nexusscope.report."""
    from .commands.nexusscope import report

    report(args)


# ---------------------------------------------------------------------------
# Chat — Interactive spec-gathering mode
# ---------------------------------------------------------------------------


def cmd_chat(args) -> None:
    """Delegates to commands.chat."""
    from .commands.chat import execute

    execute(args)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------


def main():
    """Entry point: delegates to application.cli_dispatch.run()."""
    from .application.cli_dispatch import run

    run()


if __name__ == "__main__":
    main()
