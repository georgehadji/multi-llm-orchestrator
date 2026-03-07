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
import re
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
from .output_organizer import (
    organize_project_output,
    suppress_cache_messages,
    CacheMessageSuppressor,
)
from .progress import ProgressRenderer
try:
    from .unified_events import ProjectCompletedEvent as _ProjectCompleted
except ImportError:
    from .events import ProjectCompletedEvent as _ProjectCompleted
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
    from orchestrator.secure_execution import InputValidator, SecurityContext, PathTraversalError
    from pathlib import Path

    # SECURITY FIX: Validate input path to prevent path traversal
    try:
        # Sanitize and resolve path
        base_path = Path.cwd()
        path = (base_path / args.path).resolve()
        
        # Verify path is within allowed base directory
        try:
            path.relative_to(base_path)
        except ValueError:
            print(f"ERROR: Path traversal detected: {args.path}", file=sys.stderr)
            sys.exit(1)
            
    except Exception as e:
        print(f"ERROR: Invalid path: {e}", file=sys.stderr)
        sys.exit(1)
    
    if not path.exists():
        print(f"ERROR: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    # SECURITY FIX: Validate focus areas and extensions
    focus = None
    if args.focus:
        focus = []
        for f in args.focus.split(","):
            f = f.strip()
            if f:  # Only add non-empty values
                # Basic validation: alphanumeric and common separators only
                if not re.match(r'^[\w\s\-_,]+$', f):
                    print(f"ERROR: Invalid focus area: {f}", file=sys.stderr)
                    sys.exit(1)
                focus.append(f)
    
    include_exts = None
    if args.extensions:
        include_exts = set()
        for ext in args.extensions.split(","):
            ext = ext.strip()
            if ext:
                # Validate extension format
                if not re.match(r'^[\w.]+$', ext):
                    print(f"ERROR: Invalid extension: {ext}", file=sys.stderr)
                    sys.exit(1)
                # Ensure extension starts with dot
                if not ext.startswith("."):
                    ext = f".{ext}"
                include_exts.add(ext)

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
    # SECURITY FIX: Validate output path
    if args.output:
        output_path = Path(args.output).resolve()
        # Ensure output path is safe (not traversing outside working dir)
        try:
            output_path.relative_to(Path.cwd())
        except ValueError:
            print(f"ERROR: Output path must be within current directory", file=sys.stderr)
            sys.exit(1)
        # Validate filename
        safe_filename = InputValidator.sanitize_filename(output_path.name)
        if safe_filename != output_path.name:
            print(f"WARNING: Output filename sanitized to: {safe_filename}", file=sys.stderr)
            output_path = output_path.parent / safe_filename
    else:
        output_path = path / "ANALYSIS_REPORT.md"
    
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
    from orchestrator.secure_execution import CommandInjectionError
    import re

    # SECURITY FIX: Validate intent input length and content
    intent = args.intent.strip()
    if len(intent) > 10000:
        print("ERROR: Intent description too long (max 10000 chars)", file=sys.stderr)
        sys.exit(1)
    
    # Basic check for potential injection patterns
    dangerous_patterns = [
        r'`.*?`',  # Backtick execution
        r'\$\(',  # Command substitution
        r'\$\{',  # Variable expansion
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, intent):
            print(f"WARNING: Intent contains potentially dangerous characters", file=sys.stderr)
            # Don't block, just warn - natural language can contain backticks

    agent = OrchestrationAgent()
    draft = asyncio.run(agent.draft(intent))

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
        try:
            feedback = input("\nFeedback (or 'submit' to run, 'quit' to exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            return
        
        # SECURITY FIX: Validate feedback length
        if len(feedback) > 5000:
            print("ERROR: Feedback too long (max 5000 chars)", file=sys.stderr)
            continue
        
        if feedback.lower() == "quit":
            return
        if feedback.lower() == "submit":
            break
        
        # SECURITY FIX: Additional validation for feedback
        try:
            draft = asyncio.run(agent.refine(draft, feedback))
        except CommandInjectionError as e:
            print(f"ERROR: Security violation in feedback: {e}", file=sys.stderr)
            continue
            
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


def _slash_subparsers(subparsers) -> None:
    """Register the 'slash' subcommand for interactive agent commands."""
    sp = subparsers.add_parser(
        "slash",
        help="Interactive slash commands (/analyst, /architect, /implement, etc.)",
    )
    sp.add_argument(
        "command",
        nargs="?",
        default="",
        help="Slash command to execute (e.g., 'analyst', 'architect', 'help')",
    )
    sp.add_argument(
        "--args", "-a",
        default="",
        help="Arguments for the slash command",
    )
    sp.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./slash_outputs",
        help="Directory for progressive output",
    )
    sp.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Enter interactive REPL mode",
    )
    sp.set_defaults(func=cmd_slash)


def cmd_slash(args) -> None:
    """Handle the 'slash' subcommand."""
    import asyncio
    from pathlib import Path
    from .slash_commands import get_slash_registry, SlashCommandContext
    from .api_clients import UnifiedClient
    from .cache import DiskCache
    
    registry = get_slash_registry()
    cache = DiskCache()
    client = UnifiedClient(cache=cache)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    ctx = SlashCommandContext(
        client=client,
        output_dir=output_dir,
        project_id=f"slash_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    
    if args.interactive or not args.command:
        # Interactive REPL mode
        print("╔══════════════════════════════════════════════════════════╗")
        print("║     Multi-LLM Orchestrator - Slash Command Mode          ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print("\nType /help for available commands, or /quit to exit\n")
        
        while True:
            try:
                user_input = input("orchestrator> ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
                    print("Goodbye!")
                    break
                if not user_input.startswith("/"):
                    user_input = "/" + user_input
                
                result = asyncio.run(registry.execute(user_input, ctx))
                print(f"\n{result}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}\n")
    else:
        # Single command mode
        cmd_line = f"/{args.command} {args.args}"
        result = asyncio.run(registry.execute(cmd_line, ctx))
        print(result)


def cmd_dashboard(args) -> None:
    """Handle the 'dashboard' subcommand: render persistent cross-run learning."""
    from .telemetry_store import TelemetryStore
    from .metrics import render_dashboard

    store = TelemetryStore()
    output = asyncio.run(render_dashboard(store, days=args.days))
    print(output)


def _dashboard_subparsers(subparsers) -> None:
    """Register the 'dashboard' subcommand."""
    dp = subparsers.add_parser(
        "dashboard",
        help="Show persistent cross-run model rankings, task leaders, and recommendations",
    )
    dp.add_argument(
        "--days",
        type=int,
        default=30,
        metavar="N",
        help="Lookback window in days (default: 30)",
    )
    dp.set_defaults(func=cmd_dashboard)


def _default_output_dir(project_id: str | None) -> str:
    """
    Build a default output path when --output-dir is not supplied.
    Format: ./outputs/<project_id> or ./outputs/app_<timestamp> if no project_id
    The directory is created by write_output_dir, not here.
    """
    if project_id:
        return str(Path("outputs") / project_id)
    # Generate timestamp-based directory for AppBuilder projects
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("outputs") / f"app_{timestamp}")


def setup_logging(verbose: bool = False, suppress_cache: bool = True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # re-apply even if already configured
    )
    
    # Suppress verbose cache messages unless in verbose mode
    if suppress_cache and not verbose:
        suppress_cache_messages()


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
    
    # Organize output: move tasks to tasks/, generate/run tests
    print("\n📁 Organizing project output...")
    org_report = await organize_project_output(path, auto_generate_tests=True, run_tests=True)
    print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
    if org_report.tests_run:
        passed = sum(1 for r in org_report.tests_run if r.passed)
        print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")


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

    # CLI --output-dir > YAML output_dir > auto default
    output_dir = args.output_dir or result.output_dir or _default_output_dir(result.project_id)

    renderer = ProgressRenderer(quiet=getattr(args, "quiet", False))
    project_id = result.project_id or ""
    try:
        event_count = 0
        async for event in orch.run_project_streaming(
            project_description=spec.project_description,
            success_criteria=spec.success_criteria,
            project_id=project_id,
        ):
            event_count += 1
            renderer.handle(event)
    except Exception as e:
        print(f"\n❌ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        return

    # Get the actual project_id that was used (orchestrator may have generated one)
    actual_project_id = getattr(orch, '_project_id', None) or project_id
    
    try:
        state = await orch.state_mgr.load_project(actual_project_id)
    except Exception as e:
        print(f"\nError loading state: {e}")
        state = None
    
    if state:
        _print_results(state)
    
    if state:
        path = write_output_dir(state, output_dir, project_id=actual_project_id)
        print(f"\nOutput written to: {path}")
        
        # Organize output: move tasks to tasks/, generate/run tests
        print("\n📁 Organizing project output...")
        org_report = await organize_project_output(
            path, 
            auto_generate_tests=True, 
            run_tests=True,
            fix_tests=getattr(args, 'fix_tests', True),
            max_fix_iterations=getattr(args, 'max_fix_iterations', 3),
            min_pass_rate=getattr(args, 'min_pass_rate', 0.7),
        )
        print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
        if org_report.tests_run:
            passed = sum(1 for r in org_report.tests_run if r.passed)
            print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")
    else:
        print("\n⚠️ No state available - skipping output writing")

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

        recency = _recency_factor(updated_dt.timestamp(), reference_time=now.timestamp())
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


async def _async_dry_run(args):
    """Handle --dry-run: show execution plan without running tasks."""
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency)

    print(f"DRY-RUN for project: {args.project}")
    print(f"Criteria: {args.criteria}")
    print("-" * 60)

    plan = await orch.dry_run(args.project, args.criteria)
    print(plan.render())
    await orch.state_mgr.close()
    await orch.cache.close()


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
    # ── Enhancement pass (spec improvement before decomposition) ────────────
    description = args.project
    criteria = args.criteria
    no_enhance = getattr(args, "no_enhance", False)

    if not no_enhance:
        from .enhancer import ProjectEnhancer, _present_enhancements, _apply_enhancements
        enhancer = ProjectEnhancer()
        suggestions = await enhancer.analyze(description, criteria)
        if suggestions:
            accepted = _present_enhancements(suggestions)
            description, criteria = _apply_enhancements(description, criteria, accepted)
    # ─────────────────────────────────────────────────────────────────────────
    raw_tasks = getattr(args, "raw_tasks", False)

    if not raw_tasks:
        # Route through AppBuilder (detects app_type automatically)
        from orchestrator.app_builder import AppBuilder
        output_dir = args.output_dir or _default_output_dir(None)
        print(f"Starting app build (budget: ${args.budget})")
        print(f"Project: {description}")
        print(f"Criteria: {criteria}")
        print("-" * 60)
        builder = AppBuilder()
        result = await builder.build(
            description=description,
            criteria=criteria,
            output_dir=Path(output_dir),
        )
        if result.success:
            print(f"Build successful: {result.output_dir}")
            # Debug: show state info
            if result.state:
                print(f"  State tasks: {len(result.state.tasks)}, results: {len(result.state.results)}")
                print(f"  Execution order: {result.state.execution_order}")
            # Also write individual task files
            if result.state:
                from .output_writer import write_output_dir
                tasks_dir = Path(output_dir) / "tasks"
                project_id = getattr(result.state, 'project_id', '')
                print(f"  Writing task files to: {tasks_dir}")
                path = write_output_dir(result.state, tasks_dir, project_id=project_id)
                print(f"Task files written to: {path}")
            
            # Organize output: move tasks to tasks/, generate/run tests
            print("\n📁 Organizing project output...")
            org_report = await organize_project_output(
                Path(output_dir), 
                auto_generate_tests=True, 
                run_tests=True,
                fix_tests=getattr(args, 'fix_tests', True),
                max_fix_iterations=getattr(args, 'max_fix_iterations', 3),
                min_pass_rate=getattr(args, 'min_pass_rate', 0.7),
            )
            print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
            if org_report.tests_run:
                passed = sum(1 for r in org_report.tests_run if r.passed)
                print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")
            print(f"\n📂 Output directory: {output_dir}")
        else:
            errors = ", ".join(result.errors) if result.errors else "unknown error"
            print(f"Build failed: {errors}")
        return

    # --raw-tasks: legacy flat-file path (opt-in)
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency,
                        tracing_cfg=_build_tracing_cfg(args))

    print(f"Starting project (budget: ${args.budget}, time: {args.time}s) [raw-tasks mode]")
    print(f"Project: {description}")
    print(f"Criteria: {criteria}")
    print("-" * 60)

    renderer = ProgressRenderer(quiet=getattr(args, "quiet", False))
    async for event in orch.run_project_streaming(
        project_description=description,
        success_criteria=criteria,
        project_id=args.project_id,
    ):
        renderer.handle(event)

    state = await orch.state_mgr.load_project(orch._project_id)
    _print_results(state)
    output_dir = args.output_dir or _default_output_dir(orch._project_id)
    path = write_output_dir(state, output_dir, project_id=orch._project_id)
    print(f"\nOutput written to: {path}")
    
    # Organize output: move tasks to tasks/, generate/run tests
    print("\n📁 Organizing project output...")
    org_report = await organize_project_output(path, auto_generate_tests=True, run_tests=True)
    print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
    if org_report.tests_run:
        passed = sum(1 for r in org_report.tests_run if r.passed)
        print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")
    
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


def _nash_subparsers(subparsers):
    """Add Nash stability subcommands."""
    nash_parser = subparsers.add_parser("nash", help="Nash stability management")
    nash_subparsers = nash_parser.add_subparsers(dest="nash_command", metavar="COMMAND")
    
    # nash status
    status_parser = nash_subparsers.add_parser("status", help="Show Nash stability status")
    status_parser.add_argument("--format", choices=["table", "json"], default="table")
    status_parser.add_argument("--watch", action="store_true", help="Watch mode")
    status_parser.set_defaults(func=_cmd_nash_status)
    
    # nash backup
    backup_parser = nash_subparsers.add_parser("backup", help="Backup/restore accumulated knowledge")
    backup_parser.add_argument("--list", action="store_true", help="List backups")
    backup_parser.add_argument("--restore", type=str, help="Restore from backup file")
    backup_parser.add_argument("--value", action="store_true", help="Show estimated value")
    backup_parser.set_defaults(func=_cmd_nash_backup)
    
    # nash tuning
    tuning_parser = nash_subparsers.add_parser("tuning", help="Auto-tuning control")
    tuning_parser.add_argument("--status", action="store_true", help="Show tuning status")
    tuning_parser.add_argument("--tune", type=str, help="Parameter to tune")
    tuning_parser.add_argument("--value", type=float, help="New value for parameter")
    tuning_parser.set_defaults(func=_cmd_nash_tuning)
    
    # nash compare
    compare_parser = nash_subparsers.add_parser("compare", help="Compare two models")
    compare_parser.add_argument("model_a", help="First model to compare")
    compare_parser.add_argument("model_b", help="Second model to compare")
    compare_parser.add_argument("--task-type", default="CODE_GEN", help="Task type")
    compare_parser.set_defaults(func=_cmd_nash_compare)


def _cmd_nash_status(args):
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
                os.system('cls' if os.name == 'nt' else 'clear')
                asyncio.run(show())
                print("\n[Press Ctrl+C to exit]")
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


def _cmd_nash_backup(args):
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


def _cmd_nash_tuning(args):
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


def _cmd_nash_compare(args):
    """Handle nash compare command."""
    import asyncio
    from orchestrator.pareto_frontier import get_cost_quality_frontier
    from orchestrator.models import Model, TaskType
    
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
            print(f"  {'Quality':<15} {data_a.get('quality', 0):<12.3f} {data_b.get('quality', 0):<12.3f}")
            print(f"  {'Cost':<15} ${data_a.get('cost', 0):<11.4f} ${data_b.get('cost', 0):<11.4f}")
            print(f"  {'Efficiency':<15} {data_a.get('efficiency', 0):<12.1f} {data_b.get('efficiency', 0):<12.1f}")
            
            print(f"\n  {comparison.get('recommendation', '')}")
            print("=" * 70 + "\n")
            
        except ValueError as e:
            print(f"Error: {e}")
    
    asyncio.run(run())


def cmd_cache_stats(args) -> None:
    """Show cache statistics."""
    from orchestrator.cache_optimizer import get_cache_optimizer
    
    optimizer = get_cache_optimizer()
    
    if args.clear:
        level = args.level if args.level else None
        asyncio.run(optimizer.clear(level))
        print(f"✓ Cache cleared (level: {level or 'all'})")
        return
    
    if args.cleanup:
        stats = asyncio.run(optimizer.cleanup())
        print(f"✓ Cleanup complete: {stats['l2_deleted']} expired entries removed")
        return
    
    # Print statistics
    stats = optimizer.get_stats()
    
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                    CACHE STATISTICS                              ║
╠══════════════════════════════════════════════════════════════════╣""")
    print(f"║ Total Requests:     {stats['total_requests']:>10,}                               ║")
    print(f"║ Total Hits:         {stats['total_hits']:>10,}  ({stats['overall_hit_rate']:.1%})                        ║")
    print(f"║ Total Misses:       {stats['total_misses']:>10,}                               ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ By Level:                                                        ║")
    print(f"║   L1 (Memory):      {stats['l1_hits']:>10,} hits                              ║")
    print(f"║   L2 (Disk):        {stats['l2_hits']:>10,} hits                              ║")
    print(f"║   L3 (Semantic):    {stats['l3_hits']:>10,} hits                              ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║ Savings:                                                         ║")
    print(f"║   Tokens Saved:     {stats['tokens_saved']:>10,}                               ║")
    print(f"║   Cost Saved:       ${stats['cost_saved']:>9.2f}                               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # L1 detailed stats
    if stats.get('l1_stats'):
        l1 = stats['l1_stats']
        print(f"\nL1 Memory Cache:")
        print(f"  Entries: {l1['entries']}/{l1['max_size']} ({100*l1['entries']/l1['max_size']:.1f}%)")
        print(f"  Hit Rate: {l1['hit_rate']:.1%}")


def cmd_cache_stats(args: argparse.Namespace) -> int:
    """Handle cache-stats subcommand."""
    import asyncio
    
    async def _run():
        from .cache_optimizer import get_cache_optimizer
        
        optimizer = get_cache_optimizer()
        
        if args.clear:
            level = args.level
            if level:
                print(f"🗑️  Clearing {level.upper()} cache...")
                if level == "l1":
                    optimizer.l1_cache.clear()
                elif level == "l2":
                    await optimizer.l2_cache.clear()
                elif level == "l3":
                    optimizer.l3_cache.clear()
                print(f"✅ {level.upper()} cache cleared")
            else:
                print("🗑️  Clearing all cache levels...")
                optimizer.l1_cache.clear()
                await optimizer.l2_cache.clear()
                optimizer.l3_cache.clear()
                print("✅ All caches cleared")
            return 0
        
        if args.cleanup:
            print("🧹 Cleaning up expired entries...")
            optimizer.l1_cache.cleanup()
            await optimizer.l2_cache.cleanup()
            optimizer.l3_cache.cleanup()
            print("✅ Cleanup complete")
            return 0
        
        # Show statistics
        optimizer.print_stats()
        return 0
    
    return asyncio.run(_run())


def _cache_stats_subparsers(subparsers) -> None:
    """Register the 'cache-stats' subcommand."""
    parser = subparsers.add_parser(
        "cache-stats",
        help="Show cache statistics and manage cache",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cache levels",
    )
    parser.add_argument(
        "--level",
        choices=["l1", "l2", "l3"],
        help="Specific cache level to clear (default: all)",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Remove expired cache entries",
    )
    parser.set_defaults(func=cmd_cache_stats)


def main():
    # ── Suppress specific warnings ───────────────────────────────────────────
    import warnings
    warnings.filterwarnings("ignore", 
                            message="urllib3.*doesn't match a supported version",
                            category=Warning,
                            module="requests")

    # ── Top-level parser ─────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Multi-LLM Orchestrator — Local AI Project Runner"
    )

    # ── Subcommands (e.g. 'build') ────────────────────────────────────────────
    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    _analyze_subparsers(subparsers)
    _build_subparsers(subparsers)
    _agent_subparsers(subparsers)
    _slash_subparsers(subparsers)
    _dashboard_subparsers(subparsers)
    _cache_stats_subparsers(subparsers)

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
        "--fix-tests",
        action="store_true",
        default=True,
        help="Iteratively fix failing tests (default: True)",
    )
    parser.add_argument(
        "--no-fix-tests",
        action="store_false",
        dest="fix_tests",
        help="Disable iterative test fixing",
    )
    parser.add_argument(
        "--max-fix-iterations",
        type=int,
        default=3,
        help="Maximum iterations for test fixing (default: 3)",
    )
    parser.add_argument(
        "--min-pass-rate",
        type=float,
        default=0.7,
        help="Minimum pass rate to stop fixing (default: 0.7)",
    )
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
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        default=False,
        help=(
            "Skip LLM spec enhancement pass and run original project description directly"
        ),
    )
    parser.add_argument("--dry-run", "-n", action="store_true", default=False,
                        help="Show execution plan without running any tasks")
    
    # ── Nash Stability commands ───────────────────────────────────────────────
    _nash_subparsers(subparsers)

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

    if args.dry_run:
        asyncio.run(_async_dry_run(args))
        return

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


# Click-based CLI for codebase analysis feature
try:
    import click

    @click.command()
    @click.option('--analyze-codebase', type=click.Path(exists=True),
                  help='Analyze an existing codebase')
    def cli(analyze_codebase):
        """AI Orchestrator: Codebase Analysis"""
        if analyze_codebase:
            import asyncio
            from orchestrator.codebase_understanding import CodebaseUnderstanding
            from orchestrator.improvement_suggester import ImprovementSuggester

            async def run_analysis():
                understanding = CodebaseUnderstanding()
                profile = await understanding.analyze(analyze_codebase)

                # Display analysis results
                print("\n" + "="*60)
                print("CODEBASE ANALYSIS COMPLETE")
                print("="*60)
                print(profile)

                # Generate and display improvement suggestions
                suggester = ImprovementSuggester()
                improvements = suggester.suggest(profile)

                if improvements:
                    print("\n" + "="*60)
                    print("IMPROVEMENT RECOMMENDATIONS")
                    print("="*60)
                    total_effort = sum(i.effort_hours for i in improvements)
                    print(f"\n{len(improvements)} recommendations | {total_effort}h total effort\n")

                    for i, imp in enumerate(improvements, 1):
                        print(f"{i}. {imp}")
                        print(f"   Description: {imp.description}")
                        print(f"   Impact: {imp.impact}")
                        print()

            asyncio.run(run_analysis())
        else:
            click.echo("Specify --analyze-codebase")

except ImportError:
    # Click not available, define a no-op cli function
    def cli(*args, **kwargs):
        print("Click not installed")


if __name__ == "__main__":
    main()
