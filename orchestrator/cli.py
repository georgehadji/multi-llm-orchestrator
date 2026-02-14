#!/usr/bin/env python3
"""
CLI Entry Point — run orchestrator from terminal
=================================================
Usage:
    python -m orchestrator.cli --project "Build a FastAPI auth service" \
                               --criteria "All tests pass, docs complete" \
                               --budget 8.0 --time 5400

    python -m orchestrator.cli --resume <project_id>
    python -m orchestrator.cli --list-projects
"""

import argparse
import asyncio
import logging
import sys

from .models import Budget
from .engine import Orchestrator
from .state import StateManager


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Multi-LLM Orchestrator — Local AI Project Runner"
    )
    parser.add_argument("--project", "-p", type=str, help="Project description")
    parser.add_argument("--criteria", "-c", type=str, help="Success criteria")
    parser.add_argument("--budget", "-b", type=float, default=8.0,
                        help="Max budget in USD (default: 8.0)")
    parser.add_argument("--time", "-t", type=float, default=5400,
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

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("orchestrator.cli")

    # List projects
    if args.list_projects:
        sm = StateManager()
        projects = sm.list_projects()
        if not projects:
            print("No saved projects.")
        else:
            print(f"{'ID':<15} {'Status':<20} {'Updated'}")
            print("-" * 55)
            for p in projects:
                from datetime import datetime
                updated = datetime.fromtimestamp(p["updated_at"]).strftime("%Y-%m-%d %H:%M")
                print(f"{p['project_id']:<15} {p['status']:<20} {updated}")
        sm.close()
        return

    # Resume
    if args.resume:
        budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
        orch = Orchestrator(budget=budget, max_concurrency=args.concurrency)
        sm = StateManager()
        existing = sm.load_project(args.resume)
        if not existing:
            print(f"Project {args.resume} not found.")
            sys.exit(1)
        print(f"Resuming project {args.resume}...")
        state = asyncio.run(orch.run_project(
            existing.project_description,
            existing.success_criteria,
            project_id=args.resume,
        ))
        _print_results(state)
        return

    # New project
    if not args.project:
        parser.error("--project is required for new projects")
    if not args.criteria:
        parser.error("--criteria is required for new projects")

    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency)

    print(f"Starting project (budget: ${args.budget}, time: {args.time}s)")
    print(f"Project: {args.project}")
    print(f"Criteria: {args.criteria}")
    print("-" * 60)

    state = asyncio.run(orch.run_project(
        project_description=args.project,
        success_criteria=args.criteria,
        project_id=args.project_id,
    ))

    _print_results(state)


def _print_results(state):
    print("\n" + "=" * 60)
    print(f"STATUS: {state.status.value}")
    print(f"Budget spent: ${state.budget.spent_usd:.4f} / ${state.budget.max_usd}")
    print(f"Time elapsed: {state.budget.elapsed_seconds:.1f}s / {state.budget.max_time_seconds}s")
    print("-" * 60)

    for tid, result in state.results.items():
        emoji = "✓" if result.status.value == "completed" else "✗" if result.status.value == "failed" else "~"
        print(
            f"  {emoji} {tid}: score={result.score:.3f} "
            f"[{result.model_used.value}] "
            f"iters={result.iterations} "
            f"cost=${result.cost_usd:.4f}"
        )

    print("=" * 60)

    # Output artifacts
    print("\n--- TASK OUTPUTS ---")
    for tid, result in state.results.items():
        if result.output:
            print(f"\n[{tid}] ({result.score:.3f}):")
            print(result.output[:500])
            if len(result.output) > 500:
                print(f"  ... ({len(result.output)} chars total)")


if __name__ == "__main__":
    main()
