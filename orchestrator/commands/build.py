"""
Build app command handler — extracted from cli.py.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from ..app_builder import AppBuilder
from ..budget import Budget


def register(subparsers) -> None:
    """Register the 'build' subcommand arguments."""
    bp = subparsers.add_parser("build", help="Build a complete app from a description")
    bp.add_argument("--description", "-d", required=True, help="App description")
    bp.add_argument("--criteria", "-c", default="The app must work correctly", help="Success criteria")
    bp.add_argument("--app-type", "-t", dest="app_type", default="", help="Force app type")
    bp.add_argument("--docker", action="store_true", default=False, help="Docker verification")
    bp.add_argument("--output-dir", "-o", dest="output_dir", default="", help="Output directory")
    bp.set_defaults(func=execute)


def execute(args) -> None:
    """Build a complete app from a description using the AppBuilder pipeline."""
    output_dir = args.output_dir or tempfile.mkdtemp(prefix="app-builder-")

    builder = AppBuilder()
    _budget = getattr(args, "budget", None)
    _time = getattr(args, "time", None)
    result = asyncio.run(
        builder.build(
            description=args.description,
            criteria=args.criteria,
            output_dir=Path(output_dir),
            app_type_override=args.app_type or None,
            docker=args.docker,
            budget=Budget(max_usd=_budget, max_time_seconds=_time if _time is not None else 5400.0)
            if _budget is not None
            else None,
        )
    )

    if result.success:
        print(f"Build successful: {result.output_dir}")
    else:
        errors = ", ".join(result.errors) if result.errors else "unknown error"
        print(f"Build failed: {errors}")
