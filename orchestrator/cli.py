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
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv
load_dotenv(override=True)  # override=True: .env values win over empty system env vars

from .models import Budget
from .engine import Orchestrator
from .tracing import TracingConfig
from .state import StateManager
from .project_file import load_project_file
from .output_writer import write_output_dir
from .assembler import assemble_project
from .progress import ProgressRenderer
from .streaming import ProjectCompleted as _ProjectCompleted
from .visualization import DagRenderer
from .resume_detector import (
    ResumeCandidate,
    _extract_keywords,
    _is_exact_match,
    _recency_factor,
    _score_candidates,
)


def cmd_analyze(args) -> None:
    """
    Handle the 'analyze' subcommand: read a codebase and produce an analysis report.

    Uses CodebaseReader to scan files and CodebaseAnalyzer to run multi-LLM analysis.
    """
    from orchestrator.analyzer import CodebaseAnalyzer
    from pathlib import Path

    path = Path(args.path).resolve()
    if not path.exists():
        print(f"ERROR: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    focus = [f.strip() for f in args.focus.split(",")] if args.focus else None
    include_exts = (
        {ext if ext.startswith(".") else f".{ext}" for ext in args.extensions.split(",")}
        if args.extensions else None
    )

    analyzer = CodebaseAnalyzer(
        max_context_tokens=args.context_tokens,
        max_concurrency=args.concurrency,
    )

    print(f"Analyzing: {path}")
    if focus:
        print(f"Focus areas: {', '.join(focus)}")
    print(f"Budget: ${args.budget:.2f} | Context limit: {args.context_tokens:,} tokens")
    print("-" * 60)

    report = asyncio.run(analyzer.analyze(
        path=path,
        focus=focus,
        budget_usd=args.budget,
        include_exts=include_exts,
        max_tokens_per_section=args.section_tokens,
    ))

    # Print summary
    print(f"\nAnalysis complete: {len(report.sections)} sections | "
          f"${report.total_cost:.4f} | {report.elapsed_s:.1f}s")
    print(f"Files analyzed: {report.files_analyzed} | "
          f"Languages: {', '.join(sorted(report.languages))}")

    # Write report
    output_path = Path(args.output) if args.output else path / "ANALYSIS_REPORT.md"
    output_path.write_text(report.markdown, encoding="utf-8")
    print(f"\nReport written to: {output_path}")

    # Print preview
    if not args.quiet:
        print("\n" + "=" * 60)
        preview = report.markdown[:2000]
        print(preview)
        if len(report.markdown) > 2000:
            print(f"\n... ({len(report.markdown):,} chars total — see {output_path})")


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


def _analyze_subparsers(subparsers) -> None:
    """Register the 'analyze' subcommand on the given subparsers action."""
    ap = subparsers.add_parser(
        "analyze",
        help="Analyze a codebase and produce an improvement report",
    )
    ap.add_argument(
        "--path", "-p",
        required=True,
        help="Root directory of the codebase to analyze",
    )
    ap.add_argument(
        "--focus", "-f",
        default="",
        help=(
            "Comma-separated focus areas: architecture, quality, security, "
            "performance, improvements (default: all)"
        ),
    )
    ap.add_argument(
        "--extensions", "-e",
        default="",
        help="Comma-separated file extensions to include (e.g. .py,.ts). Default: all code files",
    )
    ap.add_argument(
        "--budget", "-b",
        type=float,
        default=3.0,
        help="Max API spend in USD (default: 3.0)",
    )
    ap.add_argument(
        "--context-tokens",
        dest="context_tokens",
        type=int,
        default=60_000,
        help="Max tokens for the codebase context passed to each LLM (default: 60000)",
    )
    ap.add_argument(
        "--section-tokens",
        dest="section_tokens",
        type=int,
        default=4096,
        help="Max output tokens per analysis section (default: 4096)",
    )
    ap.add_argument(
        "--concurrency",
        type=int,
        default=2,
        help="Max simultaneous API calls (default: 2)",
    )
    ap.add_argument(
        "--output", "-o",
        default="",
        help="Output file path for the report (default: <path>/ANALYSIS_REPORT.md)",
    )
    ap.add_argument(
        "--quiet", "-q",
        action="store_true",
        default=False,
        help="Suppress report preview in terminal",
    )
    ap.set_defaults(func=cmd_analyze)


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


def cmd_agent(args) -> None:
    """
    Handle the 'agent' subcommand: NL intent → draft specs → submit to ControlPlane.
    """
    from orchestrator.orchestration_agent import OrchestrationAgent
    from orchestrator.control_plane import ControlPlane

    agent = OrchestrationAgent()
    draft = asyncio.run(agent.draft(args.intent))

    print("\n=== Draft Job Spec ===")
    import json as _json
    from dataclasses import asdict
    print(_json.dumps(asdict(draft.job), indent=2, default=str))
    print("\n=== Draft Policy Spec ===")
    print(_json.dumps(asdict(draft.policy), indent=2, default=str))
    print(f"\nRationale: {draft.rationale}")

    if not args.interactive:
        return

    while True:
        feedback = input("\nFeedback (or 'submit' to run, 'quit' to exit): ").strip()
        if feedback.lower() == "quit":
            return
        if feedback.lower() == "submit":
            break
        draft = asyncio.run(agent.refine(draft, feedback))
        print("\n=== Revised Job Spec ===")
        print(_json.dumps(asdict(draft.job), indent=2, default=str))
        print(f"\nRationale: {draft.rationale}")

    print("\nSubmitting to ControlPlane...")
    cp = ControlPlane()
    state = asyncio.run(cp.submit(draft.job, draft.policy))
    print(f"Status: {state.status.value}")


def _agent_subparsers(subparsers) -> None:
    """Register the 'agent' subcommand."""
    ap = subparsers.add_parser(
        "agent",
        help="Convert NL intent to typed specs and optionally run via ControlPlane",
    )
    ap.add_argument(
        "--intent", "-i",
        required=True,
        help="Natural language description of the job to run",
    )
    ap.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Enter interactive refine loop before submitting",
    )
    ap.set_defaults(func=cmd_agent)


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


def _build_tracing_cfg(args) -> "TracingConfig | None":
    """Return a TracingConfig when --tracing is set, otherwise None."""
    if getattr(args, "tracing", False):
        return TracingConfig(
            enabled=True,
            otlp_endpoint=getattr(args, "otlp_endpoint", None),
        )
    return None


async def _async_resume(args):
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency,
                        tracing_cfg=_build_tracing_cfg(args))
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

    orch = Orchestrator(budget=budget, max_concurrency=concurrency,
                        tracing_cfg=_build_tracing_cfg(args))

    renderer = ProgressRenderer(quiet=getattr(args, "quiet", False))
    async for event in orch.run_project_streaming(
        project_description=spec.project_description,
        success_criteria=spec.success_criteria,
        project_id=result.project_id,
    ):
        renderer.handle(event)

    state = await orch.state_mgr.load_project(orch._project_id)
    _print_results(state)
    # CLI --output-dir > YAML output_dir > auto default
    output_dir = args.output_dir or result.output_dir or _default_output_dir(orch._project_id)
    path = write_output_dir(state, output_dir, project_id=orch._project_id)
    print(f"\nOutput written to: {path}")

    # Assembly: place files into declared target_path locations
    if state and (result.assemble or result.task_paths):
        resolved_paths = _resolve_task_paths(result.task_paths, state)
        assembly_dir = str(Path(output_dir) / "app")
        assembly = assemble_project(
            state,
            assembly_dir,
            task_paths=resolved_paths,
            verify_cmd=result.verify_cmd,
        )
        print(f"\nAssembled project: {assembly.output_dir}")
        for f in assembly.files_written:
            print(f"  + {f}")
        if assembly.verify_returncode is not None:
            status = "OK" if assembly.verify_returncode == 0 else "FAILED"
            print(f"\nVerification [{status}] (exit {assembly.verify_returncode})")
            if assembly.verify_output:
                print(assembly.verify_output[:500])
        if assembly.errors:
            for err in assembly.errors:
                print(f"  ! {err}", file=sys.stderr)

    if getattr(args, "dependency_report", False) and state:
        renderer = DagRenderer(state.tasks, results=state.results)
        print("\n" + renderer.dependency_report())


async def _check_resume(
    description: str,
    state_mgr,
    new_project: bool = False,
    _input_fn: Optional[Callable[[str], str]] = None,
) -> Optional[str]:
    """Gate that detects and offers to resume a previous project.

    Algorithm:
    1. Return None immediately if new_project=True (bypass flag).
    2. Extract keywords from description.
    3. Call state_mgr.find_resumable(keywords) with a 200 ms timeout.
    4. Convert DB rows to ResumeCandidate objects and score them.
    5. Exact keyword match → auto-resume (print message, return project_id).
    6. Single fuzzy match → prompt Y/n.
    7. Multiple fuzzy matches → show numbered list, user picks.
    8. No matches or timeout → return None (start fresh).

    Parameters
    ----------
    description:
        The new project description supplied by the user.
    state_mgr:
        An object with an async ``find_resumable(keywords)`` method.
    new_project:
        When True, skip all detection and return None immediately.
    _input_fn:
        Callable used to read user input (defaults to built-in ``input``).
        Injected during tests so the function is testable without mocking
        builtins.

    Returns
    -------
    str | None
        project_id to resume, or None to start a fresh project.
    """
    if new_project:
        return None

    if _input_fn is None:
        _input_fn = input

    keywords = _extract_keywords(description)
    if not keywords:
        return None

    # ── Fetch candidates with a hard timeout ────────────────────────────────
    try:
        rows: list[dict] = await asyncio.wait_for(
            state_mgr.find_resumable(keywords), timeout=0.2
        )
    except (asyncio.TimeoutError, Exception):
        return None

    if not rows:
        return None

    # ── Convert DB rows to ResumeCandidate objects ──────────────────────────
    now = datetime.utcnow()
    candidates: list[ResumeCandidate] = []
    for row in rows:
        # updated_at is stored as a Unix timestamp float in the DB
        updated_ts = row.get("updated_at") or 0.0
        try:
            updated_dt = datetime.utcfromtimestamp(float(updated_ts))
        except (ValueError, OSError, OverflowError):
            updated_dt = now

        recency = _recency_factor(updated_dt, reference_time=now)
        candidates.append(
            ResumeCandidate(
                project_id=row["project_id"],
                description=row.get("description", ""),
                keywords=row.get("keywords", []),
                recency_score=recency,
                similarity_score=0.0,  # computed by _score_candidates
                overall_score=0.0,     # computed by _score_candidates
            )
        )

    # ── Score and filter candidates ──────────────────────────────────────────
    scored = _score_candidates(keywords, candidates)
    if not scored:
        return None

    # ── Check for exact description match (auto-resume) ──────────────────────
    for candidate in scored:
        if _is_exact_match(keywords, candidate.keywords):
            print(
                f"\nResuming previous project (exact match): {candidate.project_id}\n"
                f"  {candidate.description[:80]}"
            )
            return candidate.project_id

    # ── Single fuzzy match ───────────────────────────────────────────────────
    if len(scored) == 1:
        c = scored[0]
        try:
            answer = _input_fn(
                f"\nFound a resumable project:\n"
                f"  [{c.project_id}] {c.description[:80]}\n"
                f"Resume it? [Y/n]: "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if answer in ("y", "yes", ""):
            return c.project_id
        return None

    # ── Multiple fuzzy matches ────────────────────────────────────────────────
    print("\nFound multiple resumable projects:")
    for i, c in enumerate(scored, start=1):
        print(f"  {i}. [{c.project_id}] {c.description[:70]}")
    print(f"  n. Start a new project")
    try:
        answer = _input_fn("Pick a number to resume, or 'n' to start fresh: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if answer == "n" or answer == "":
        return None
    try:
        idx = int(answer) - 1
        if 0 <= idx < len(scored):
            return scored[idx].project_id
    except ValueError:
        pass
    return None


async def _async_new_project(args):
    # ── Resume detection gate ────────────────────────────────────────────────
    if not getattr(args, "new_project", False):
        state_mgr = StateManager()
        try:
            project_id_to_resume = await _check_resume(
                description=args.project,
                state_mgr=state_mgr,
                new_project=getattr(args, "new_project", False),
            )
        finally:
            await state_mgr.close()
        if project_id_to_resume:
            args.resume = project_id_to_resume
            await _async_resume(args)
            return
    # ── existing code continues unchanged ────────────────────────────────────
    raw_tasks = getattr(args, "raw_tasks", False)

    if not raw_tasks:
        # Route through AppBuilder (detects app_type automatically)
        import tempfile
        from orchestrator.app_builder import AppBuilder
        output_dir = args.output_dir or tempfile.mkdtemp(prefix="app-builder-")
        print(f"Starting app build (budget: ${args.budget})")
        print(f"Project: {args.project}")
        print(f"Criteria: {args.criteria}")
        print("-" * 60)
        builder = AppBuilder()
        result = await builder.build(
            description=args.project,
            criteria=args.criteria,
            output_dir=Path(output_dir),
        )
        if result.success:
            print(f"Build successful: {result.output_dir}")
        else:
            errors = ", ".join(result.errors) if result.errors else "unknown error"
            print(f"Build failed: {errors}")
        return

    # --raw-tasks: legacy flat-file path (opt-in)
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency,
                        tracing_cfg=_build_tracing_cfg(args))

    print(f"Starting project (budget: ${args.budget}, time: {args.time}s) [raw-tasks mode]")
    print(f"Project: {args.project}")
    print(f"Criteria: {args.criteria}")
    print("-" * 60)

    renderer = ProgressRenderer(quiet=getattr(args, "quiet", False))
    async for event in orch.run_project_streaming(
        project_description=args.project,
        success_criteria=args.criteria,
        project_id=args.project_id,
    ):
        renderer.handle(event)

    state = await orch.state_mgr.load_project(orch._project_id)
    _print_results(state)
    output_dir = args.output_dir or _default_output_dir(orch._project_id)
    path = write_output_dir(state, output_dir, project_id=orch._project_id)
    print(f"\nOutput written to: {path}")
    if getattr(args, "dependency_report", False) and state:
        renderer = DagRenderer(state.tasks, results=state.results)
        print("\n" + renderer.dependency_report())


async def _async_visualize(args):
    """
    Decompose the project (without running tasks) and print the requested
    visualization (--visualize mermaid|ascii) or critical path (--critical-path),
    then exit.  Requires either --project + --criteria or --file.
    """
    if args.file:
        try:
            result = load_project_file(args.file)
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(1)
        spec = result.spec
        project_description = spec.project_description
        success_criteria = spec.success_criteria
        budget = spec.budget
    else:
        budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
        project_description = args.project
        success_criteria = getattr(args, "criteria", "") or ""

    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency,
                        tracing_cfg=_build_tracing_cfg(args))
    tasks = await orch._decompose(project_description, success_criteria)

    renderer = DagRenderer(tasks)
    if args.visualize == "mermaid":
        print(renderer.to_mermaid())
    elif args.visualize == "ascii":
        print(renderer.to_ascii())
    if args.critical_path:
        path = renderer.critical_path()
        print("Critical path: " + " -> ".join(path) if path else "Critical path: (empty)")


def _resolve_task_paths(
    task_paths: dict[str, str],
    state,
) -> dict[str, str]:
    """
    Resolve a ``task_paths`` dict from the YAML file into a
    ``{task_id: target_path}`` mapping.

    Keys may be:
    - 1-based integer index ("1", "2", …) → resolved against execution_order
    - Exact task_id string ("task_001", …) → used as-is
    """
    if not task_paths:
        return {}
    order = state.execution_order or list(state.results.keys())
    resolved: dict[str, str] = {}
    for key, target in task_paths.items():
        # Try numeric index first
        try:
            idx = int(key) - 1  # convert 1-based to 0-based
            if 0 <= idx < len(order):
                resolved[order[idx]] = target
            else:
                resolved[key] = target  # keep as-is, assembler will skip if unknown
        except ValueError:
            resolved[key] = target  # already a task_id string
    return resolved


def main():
    # ── Top-level parser ─────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Multi-LLM Orchestrator — Local AI Project Runner"
    )

    # ── Subcommands (e.g. 'build') ────────────────────────────────────────────
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    _analyze_subparsers(subparsers)
    _build_subparsers(subparsers)
    _agent_subparsers(subparsers)

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
    parser.add_argument(
        "--visualize",
        choices=["mermaid", "ascii"],
        metavar="FORMAT",
        default=None,
        help="Print task dependency graph (mermaid or ascii) after decomposition, then exit",
    )
    parser.add_argument(
        "--critical-path",
        action="store_true",
        default=False,
        help="Print the critical path through the task DAG, then exit",
    )
    parser.add_argument(
        "--dependency-report",
        action="store_true",
        default=False,
        help="After run, print dependency context-size report",
    )
    parser.add_argument(
        "--aggregate-metrics",
        action="store_true",
        help="Print cross-run model performance aggregation and exit",
    )
    parser.add_argument(
        "--reuse-profiles",
        action="store_true",
        help="Seed routing from historical run profiles (future feature)",
    )
    parser.add_argument(
        "--tracing",
        action="store_true",
        default=False,
        help="Enable OpenTelemetry distributed tracing. Requires: pip install -e '.[tracing]'"
    )
    parser.add_argument(
        "--otlp-endpoint",
        type=str,
        default=None,
        metavar="URL",
        help="OTLP gRPC endpoint for tracing export (e.g. http://localhost:4317). "
             "If --tracing is set but this is omitted, spans are printed to console."
    )
    parser.add_argument(
        "--raw-tasks",
        action="store_true",
        default=False,
        help=(
            "Skip AppBuilder pipeline and write raw task output files directly "
            "(legacy behaviour, opt-in)"
        ),
    )
    parser.add_argument(
        "--new-project", "-N",
        action="store_true",
        default=False,
        help="Skip resume detection and always start a fresh project",
    )

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

    if args.aggregate_metrics:
        print("Cross-run model performance aggregation (no historical data loaded yet).")
        print("Use ProfileAggregator from orchestrator.aggregator to record runs and query stats.")
        return

    if args.resume:
        asyncio.run(_async_resume(args))
        return

    # --visualize / --critical-path: decompose-only dry-run, then exit.
    # Works with either --file or --project + --criteria.
    if args.visualize or args.critical_path:
        if not args.file and not args.project:
            parser.error(
                "--visualize/--critical-path require --file <yaml> or --project + --criteria"
            )
        asyncio.run(_async_visualize(args))
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
