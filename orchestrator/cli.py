#!/usr/bin/env python3
"""
CLI Entry Point — run orchestrator from terminal
=================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Usage:
    python -m orchestrator --project "Build a FastAPI auth service" \
                           --criteria "All tests pass, docs complete" \
                           --budget 8.0 --time 5400

    python -m orchestrator --resume <project_id>
    python -m orchestrator --list-projects

FIX #10 cascade: CLI now uses asyncio.run for StateManager async calls.
FEAT:   Output is always written to a folder. If --output-dir is omitted,
        a default path of ./outputs/<project_id> is used automatically.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(override=True)  # override=True: .env values win over empty system env vars

from .models import Budget
from .engine import Orchestrator
from .state import StateManager
from .project_file import load_project_file
from .output_writer import write_output_dir


def cmd_build(args) -> None:
    """
    Handle the 'build' subcommand: build a complete app from a description.

    Uses the AppBuilder pipeline to generate, assemble, and verify an app.
    """
    import tempfile
    from pathlib import Path
    from orchestrator.app_builder import AppBuilder

    output_dir = args.output_dir or tempfile.mkdtemp(prefix="app-builder-")

    builder = AppBuilder()
    result = asyncio.run(builder.build(
        description=args.description,
        criteria=args.criteria,
        output_dir=Path(output_dir),
        app_type_override=args.app_type or None,
        docker=args.docker,
    ))

    if result.success:
        print(f"Build successful: {result.output_dir}")
    else:
        errors = ", ".join(result.errors) if result.errors else "unknown error"
        print(f"Build failed: {errors}")


def _build_subparsers(subparsers) -> None:
    """Register the 'build' subcommand on the given subparsers action."""
    build_parser = subparsers.add_parser(
        "build",
        help="Build a complete app from a description",
    )
    build_parser.add_argument(
        "--description", "-d",
        required=True,
        help="App description",
    )
    build_parser.add_argument(
        "--criteria", "-c",
        default="The app must work correctly",
        help="Success criteria (default: 'The app must work correctly')",
    )
    build_parser.add_argument(
        "--app-type", "-t",
        dest="app_type",
        default="",
        help="Force app type (fastapi/cli/library/generic)",
    )
    build_parser.add_argument(
        "--docker",
        action="store_true",
        default=False,
        help="Run Docker-based verification after build",
    )
    build_parser.add_argument(
        "--output-dir", "-o",
        dest="output_dir",
        default="",
        help="Directory to write the generated app (default: auto-generated temp dir)",
    )
    build_parser.set_defaults(func=cmd_build)


def _default_output_dir(project_id: str) -> str:
    """
    Build a default output path when --output-dir is not supplied.
    Format: ./outputs/<project_id>
    The directory is created by write_output_dir, not here.
    """
    return str(Path("outputs") / project_id)


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # re-apply even if already configured
    )


async def _async_list_projects():
    sm = StateManager()
    try:
        projects = await sm.list_projects()
        if not projects:
            print("No saved projects.")
        else:
            print(f"{'ID':<15} {'Status':<20} {'Updated'}")
            print("-" * 55)
            for p in projects:
                from datetime import datetime
                updated = datetime.fromtimestamp(p["updated_at"]).strftime("%Y-%m-%d %H:%M")
                print(f"{p['project_id']:<15} {p['status']:<20} {updated}")
    finally:
        await sm.close()


async def _async_resume(args):
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency)
    existing = await orch.state_mgr.load_project(args.resume)
    if not existing:
        print(f"Project {args.resume} not found.")
        sys.exit(1)
    print(f"Resuming project {args.resume}...")
    state = await orch.run_project(
        existing.project_description,
        existing.success_criteria,
        project_id=args.resume,
    )
    _print_results(state)
    output_dir = args.output_dir or _default_output_dir(args.resume)
    path = write_output_dir(state, output_dir, project_id=args.resume)
    print(f"\nOutput written to: {path}")


async def _async_file_project(args):
    try:
        result = load_project_file(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    spec = result.spec

    # CLI flags override file values when explicitly provided
    concurrency = args.concurrency if args.concurrency != 3 else result.concurrency
    # Re-apply logging with file's verbose setting merged with CLI flag
    setup_logging(args.verbose or result.verbose)
    budget = spec.budget

    # Print banner BEFORE Orchestrator init so it appears before WARNING logs
    print(f"Loading project from: {args.file}")
    print(f"Project: {spec.project_description[:80]}")
    print(f"Budget: ${budget.max_usd} / {budget.max_time_seconds}s")
    print("-" * 60)

    orch = Orchestrator(budget=budget, max_concurrency=concurrency)

    state = await orch.run_project(
        project_description=spec.project_description,
        success_criteria=spec.success_criteria,
        project_id=result.project_id,
    )
    _print_results(state)
    # CLI --output-dir > YAML output_dir > auto default
    output_dir = args.output_dir or result.output_dir or _default_output_dir(orch._project_id)
    path = write_output_dir(state, output_dir, project_id=orch._project_id)
    print(f"\nOutput written to: {path}")


async def _async_new_project(args):
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency)

    print(f"Starting project (budget: ${args.budget}, time: {args.time}s)")
    print(f"Project: {args.project}")
    print(f"Criteria: {args.criteria}")
    print("-" * 60)

    state = await orch.run_project(
        project_description=args.project,
        success_criteria=args.criteria,
        project_id=args.project_id,
    )
    _print_results(state)
    output_dir = args.output_dir or _default_output_dir(orch._project_id)
    path = write_output_dir(state, output_dir, project_id=orch._project_id)
    print(f"\nOutput written to: {path}")


def main():
    # ── Top-level parser ─────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Multi-LLM Orchestrator — Local AI Project Runner"
    )

    # ── Subcommands (e.g. 'build') ────────────────────────────────────────────
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    _build_subparsers(subparsers)

    # ── Legacy flat flags (kept for backwards compatibility) ──────────────────
    parser.add_argument("--project", "-p", type=str, help="Project description")
    parser.add_argument("--criteria", "-c", type=str, help="Success criteria")
    parser.add_argument("--budget", "-b", type=float, default=8.0,
                        help="Max budget in USD (default: 8.0)")
    parser.add_argument("--time", type=float, default=5400,
                        help="Max time in seconds (default: 5400)")
    parser.add_argument("--project-id", type=str, default="",
                        help="Project ID (auto-generated if empty)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume a previous project by ID")
    parser.add_argument("--list-projects", action="store_true",
                        help="List all saved projects")
    parser.add_argument("--concurrency", type=int, default=3,
                        help="Max simultaneous API calls (default: 3)")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--file", "-f", type=str, default="",
                        help="Load project spec from a YAML file")
    parser.add_argument("--output-dir", "-o", type=str, default="",
                        help="Write structured output files to this directory")

    args = parser.parse_args()
    setup_logging(getattr(args, "verbose", False))

    # Dispatch subcommand if present
    if args.subcommand is not None:
        func = getattr(args, "func", None)
        if func is not None:
            func(args)
        return

    if args.list_projects:
        asyncio.run(_async_list_projects())
        return

    if args.resume:
        asyncio.run(_async_resume(args))
        return

    if args.file:
        asyncio.run(_async_file_project(args))
        return

    if not args.project:
        parser.error("--project is required for new projects (or use --file <yaml>)")
    if not args.criteria:
        parser.error("--criteria is required for new projects (or use --file <yaml>)")

    asyncio.run(_async_new_project(args))


def _print_results(state):
    print("\n" + "=" * 60)
    print(f"STATUS: {state.status.value}")
    print(f"Budget spent: ${state.budget.spent_usd:.4f} / ${state.budget.max_usd}")
    print(f"Time elapsed: {state.budget.elapsed_seconds:.1f}s / {state.budget.max_time_seconds}s")
    print("-" * 60)

    for tid, result in state.results.items():
        emoji = "OK" if result.status.value == "completed" else "FAIL" if result.status.value == "failed" else "~"
        print(
            f"  {emoji} {tid}: score={result.score:.3f} "
            f"[{result.model_used.value}] "
            f"iters={result.iterations} "
            f"cost=${result.cost_usd:.4f}"
        )

    print("=" * 60)

    print("\n--- TASK OUTPUTS ---")
    for tid, result in state.results.items():
        if result.output:
            print(f"\n[{tid}] ({result.score:.3f}):")
            print(result.output[:500])
            if len(result.output) > 500:
                print(f"  ... ({len(result.output)} chars total)")


if __name__ == "__main__":
    main()
