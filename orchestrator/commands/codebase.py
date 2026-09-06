"""codebase command module."""

from __future__ import annotations


def execute(args) -> None:
    """Handle the 'modify' subcommand."""
    import asyncio
    from pathlib import Path

    from ._path_utils import resolve_allowed_path

    result = asyncio.run(
        _run_modify(
            repo=resolve_allowed_path(args.repo),
            objective=args.objective,
            dry_run=getattr(args, "dry_run", False),
            budget=getattr(args, "budget", 10.0),
        )
    )
    print(result)


async def _run_modify(repo, objective: str, dry_run: bool, budget: float = 10.0) -> str:
    """Execute the codebase modification flow."""
    from ..budget import Budget
    from ..engine import Orchestrator

    try:
        orch = Orchestrator(budget=Budget(max_usd=budget))
        state = await orch.modify_codebase(
            repo_path=repo,
            objective=objective,
            dry_run=dry_run,
        )
        return f"Modification complete.\nState keys: {list(state.keys()) if state else 'none'}"
    except Exception as exc:
        import traceback

        return f"Modification failed: {exc}\n{traceback.format_exc()}"


def register(subparsers) -> None:
    # Register the modify subcommand for codebase-aware operations.
    mp = subparsers.add_parser(
        "modify",
        help="Modify an existing codebase using AI reasoning",
    )
    mp.add_argument("--repo", required=True, help="Path to codebase root")
    mp.add_argument("--objective", required=True, help="What to do")
    mp.add_argument("--budget", type=float, default=10.0, help="Max LLM budget USD")
    mp.add_argument("--dry-run", action="store_true", help="Plan only")
    mp.set_defaults(func=execute)
