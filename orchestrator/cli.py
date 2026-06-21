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
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Note: override=True makes .env values override existing system env vars.
# However, system env vars set before Python startup may take precedence
# depending on the python-dotenv version and platform.
load_dotenv(override=True)

# Setup logging
logger = logging.getLogger("orchestrator.cli")

from .assembler import assemble_project
from .budget import Budget
from .engine import Orchestrator
from .output_organizer import (
    organize_project_output,
    suppress_cache_messages,
)
from .output_writer import write_output_dir
from .progress import ProgressRenderer
from .project_file import load_project_file
from .state import StateManager
from .tracing import TracingConfig

try:
    from .unified_events import ProjectCompletedEvent as _ProjectCompleted
except ImportError:
    pass
from .command_registry import commands_by_category, resolve_command as resolve_slash_command
from .resume_detector import (
    ResumeCandidate,
    _extract_keywords,
    _is_exact_match,
    _recency_factor,
    _score_candidates,
)
from .visualization import DagRenderer


def cmd_analyze(args) -> None:
    """Analyze a codebase — delegates to commands.analyze."""
    from .commands.analyze import execute
    execute(args)
def cmd_build(args) -> None:
    """Build a complete app — delegates to commands.build."""
    from .commands.build import execute

    execute(args)







def cmd_agent(args) -> None:
    """Agent subcommand — delegates to commands.agent."""
    from .commands.agent import execute
    execute(args)






def cmd_slash(args) -> None:
    """Slash agent subcommand — delegates to commands.slash."""
    from .commands.slash import execute
    execute(args)
def cmd_dashboard(args) -> None:
    """Dashboard — delegates to commands.dashboard."""
    from .commands.dashboard import execute
    execute(args)



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
    orch = Orchestrator(
        budget=budget, max_concurrency=args.concurrency, tracing_cfg=_build_tracing_cfg(args)
    )
    # Apply agent profile if specified (Wave 1: W1 Agent Profiles)
    if getattr(args, "agent_profile", None):
        from .application.model_profile_builder import build_default_profiles

        profile_map = {
            "standard": {"quality_mode": "standard", "iteration_cap": 3},
            "max": {"quality_mode": "production", "iteration_cap": 5},
            "creative": {"quality_mode": "standard", "iteration_cap": 4, "temperature": 0.8},
            "conservative": {"quality_mode": "production", "iteration_cap": 2, "temperature": 0.3},
            "research": {"quality_mode": "standard", "iteration_cap": 6},
        }
        cfg = profile_map.get(args.agent_profile, {})
        for profile_name, profile in orch._profiles.items():
            if hasattr(profile, "quality_mode") and "quality_mode" in cfg:
                profile.quality_mode = cfg["quality_mode"]
            if hasattr(profile, "temperature") and "temperature" in cfg:
                profile.temperature = cfg.get("temperature", 0.7)
        logger.info(
            f"Agent profile '{args.agent_profile}': quality_mode={cfg.get('quality_mode')}, iteration_cap={cfg.get('iteration_cap')}"
        )
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
    _print_results(state, orch)
    output_dir = args.output_dir or _default_output_dir(args.resume)
    path = write_output_dir(state, output_dir, project_id=args.resume)
    print(f"\nOutput written to: {path}")

    # Organize output: move tasks to tasks/, generate/run tests
    print("\n[ORG] Organizing project output...")
    org_report = await organize_project_output(path, auto_generate_tests=True, run_tests=True)
    safe_print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
    if org_report.tests_run:
        passed = sum(1 for r in org_report.tests_run if r.passed)
        safe_print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")


async def _async_file_project(args):
    try:
        result = load_project_file(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    spec = result.spec

    # ═══════════════════════════════════════════════════════
    # TDD Configuration — YAML values are baseline, CLI flags override
    # ═══════════════════════════════════════════════════════
    # Determine effective TDD settings:
    #   - YAML tdd_first defaults to True (on by default)
    #   - CLI --tdd-first forces True; absence does NOT force False (YAML wins)
    cli_tdd_first = getattr(args, "tdd_first", False)
    effective_tdd_first = result.tdd_first or cli_tdd_first
    effective_tdd_quality = args.tdd_quality if cli_tdd_first else result.tdd_quality

    if effective_tdd_first:
        from .cost_optimization import (
            get_optimization_config,
            update_config,
        )

        config = get_optimization_config()
        config.enable_tdd_first = True
        config.tdd_quality_tier = effective_tdd_quality
        config.tdd_max_iterations = args.tdd_max_iterations
        config.tdd_min_test_coverage = args.tdd_min_coverage
        update_config(config)

        logger.info(
            f"TDD enabled: quality={effective_tdd_quality}, "
            f"max_iterations={args.tdd_max_iterations}, "
            f"min_coverage={args.tdd_min_coverage}"
        )

    # CLI flags override file values when explicitly provided
    concurrency = args.concurrency if args.concurrency != 3 else result.concurrency
    # Re-apply logging with file's verbose setting merged with CLI flag
    setup_logging(args.verbose or result.verbose)
    budget = spec.budget
    # Apply agent profile if specified (Wave 1: W1 Agent Profiles)
    if getattr(args, "agent_profile", None):
        from .application.model_profile_builder import build_default_profiles

        profile_map = {
            "standard": {"quality_mode": "standard", "iteration_cap": 3},
            "max": {"quality_mode": "production", "iteration_cap": 5},
            "creative": {"quality_mode": "standard", "iteration_cap": 4, "temperature": 0.8},
            "conservative": {"quality_mode": "production", "iteration_cap": 2, "temperature": 0.3},
            "research": {"quality_mode": "standard", "iteration_cap": 6},
        }
        cfg = profile_map.get(args.agent_profile, {})

    # Print banner BEFORE Orchestrator init so it appears before WARNING logs
    print(f"Loading project from: {args.file}")
    print(f"Project: {spec.project_description[:80]}")
    print(f"Budget: ${budget.max_usd} / {budget.max_time_seconds}s")
    print("-" * 60)

    orch = Orchestrator(
        budget=budget, max_concurrency=concurrency, tracing_cfg=_build_tracing_cfg(args)
    )

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
        safe_print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return

    # Get the actual project_id that was used (orchestrator may have generated one)
    actual_project_id = getattr(orch, "_project_id", None) or project_id

    try:
        state = await orch.state_mgr.load_project(actual_project_id)
    except Exception as e:
        print(f"\nError loading state: {e}")
        state = None

    if state:
        _print_results(state, orch)

    if state:
        path = write_output_dir(state, output_dir, project_id=actual_project_id)
        print(f"\nOutput written to: {path}")

        # Organize output: move tasks to tasks/, generate/run tests
        print("\n[ORG] Organizing project output...")
        org_report = await organize_project_output(
            path,
            auto_generate_tests=True,
            run_tests=True,
            fix_tests=getattr(args, "fix_tests", True),
            max_fix_iterations=getattr(args, "max_fix_iterations", 3),
            min_pass_rate=getattr(args, "min_pass_rate", 0.7),
        )
        safe_print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
        if org_report.tests_run:
            passed = sum(1 for r in org_report.tests_run if r.passed)
            safe_print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")

        # EXTRA: Run final test execution with detailed reporting
        print("\n[TEST] Running final test validation...")
        # Tests already ran in organize_project_output - just report results
        if org_report.tests_run:
            passed = sum(1 for r in org_report.tests_run if r.passed)
            total = len(org_report.tests_run)
            if passed == total:
                safe_print("\n✅ All tests passed!")
            else:
                safe_print(f"\n[WARN] {passed}/{total} tests passed - check output for details")
        else:
            safe_print("\nℹ️ No tests were executed")
    else:
        print("\n[WARN] No state available - skipping output writing")

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
    _input_fn: Callable[[str], str] | None = None,
) -> str | None:
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
        rows: list[dict] = await asyncio.wait_for(state_mgr.find_resumable(keywords), timeout=0.2)
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
                overall_score=0.0,  # computed by _score_candidates
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
            answer = (
                _input_fn(
                    f"\nFound a resumable project:\n"
                    f"  [{c.project_id}] {c.description[:80]}\n"
                    f"Resume it? [Y/n]: "
                )
                .strip()
                .lower()
            )
        except (EOFError, KeyboardInterrupt):
            return None
        if answer in ("y", "yes", ""):
            return c.project_id
        return None

    # ── Multiple fuzzy matches ────────────────────────────────────────────────
    print("\nFound multiple resumable projects:")
    for i, c in enumerate(scored, start=1):
        print(f"  {i}. [{c.project_id}] {c.description[:70]}")
    print("  n. Start a new project")
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
    # ═══════════════════════════════════════════════════════
    # TDD Configuration (NEW v3.0)
    # ═══════════════════════════════════════════════════════
    if getattr(args, "tdd_first", False):
        from .cost_optimization import (
            get_optimization_config,
            update_config,
        )

        config = get_optimization_config()
        config.enable_tdd_first = True
        config.tdd_quality_tier = args.tdd_quality
        config.tdd_max_iterations = args.tdd_max_iterations
        config.tdd_min_test_coverage = args.tdd_min_coverage
        update_config(config)

        logger.info(
            f"TDD enabled: quality={args.tdd_quality}, "
            f"max_iterations={args.tdd_max_iterations}, "
            f"min_coverage={args.tdd_min_coverage}"
        )

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
        from .enhancer import ProjectEnhancer, _apply_enhancements, _present_enhancements

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
            # Propagate the CLI --budget/--time; without this the build silently
            # used AppBuilder's default $8.00 despite the printed budget above.
            budget=Budget(max_usd=args.budget, max_time_seconds=args.time),
        )
        if result.success:
            print(f"Build successful: {result.output_dir}")
            # Debug: show state info
            if result.state:
                print(
                    f"  State tasks: {len(result.state.tasks)}, results: {len(result.state.results)}"
                )
                print(f"  Execution order: {result.state.execution_order}")
            # Also write individual task files
            if result.state:
                from .output_writer import write_output_dir

                tasks_dir = Path(output_dir) / "tasks"
                project_id = getattr(result.state, "project_id", "")
                print(f"  Writing task files to: {tasks_dir}")
                path = write_output_dir(result.state, tasks_dir, project_id=project_id)
                print(f"Task files written to: {path}")

            # Organize output: move tasks to tasks/, generate/run tests
            print("\n[ORG] Organizing project output...")
            org_report = await organize_project_output(
                Path(output_dir),
                auto_generate_tests=True,
                run_tests=True,
                fix_tests=getattr(args, "fix_tests", True),
                max_fix_iterations=getattr(args, "max_fix_iterations", 3),
                min_pass_rate=getattr(args, "min_pass_rate", 0.7),
            )
            safe_print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
            if org_report.tests_run:
                passed = sum(1 for r in org_report.tests_run if r.passed)
                safe_print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")
            print(f"\n[DIR] Output directory: {output_dir}")
        else:
            errors = ", ".join(result.errors) if result.errors else "unknown error"
            print(f"Build failed: {errors}")
        return

    # --raw-tasks: legacy flat-file path (opt-in)
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(
        budget=budget, max_concurrency=args.concurrency, tracing_cfg=_build_tracing_cfg(args)
    )
    # Apply agent profile if specified (Wave 1: W1 Agent Profiles)
    if getattr(args, "agent_profile", None):
        from .application.model_profile_builder import build_default_profiles

        profile_map = {
            "standard": {"quality_mode": "standard", "iteration_cap": 3},
            "max": {"quality_mode": "production", "iteration_cap": 5},
            "creative": {"quality_mode": "standard", "iteration_cap": 4, "temperature": 0.8},
            "conservative": {"quality_mode": "production", "iteration_cap": 2, "temperature": 0.3},
            "research": {"quality_mode": "standard", "iteration_cap": 6},
        }
        cfg = profile_map.get(args.agent_profile, {})
        for profile_name, profile in orch._profiles.items():
            if hasattr(profile, "quality_mode") and "quality_mode" in cfg:
                profile.quality_mode = cfg["quality_mode"]
            if hasattr(profile, "temperature") and "temperature" in cfg:
                profile.temperature = cfg.get("temperature", 0.7)
        logger.info(
            f"Agent profile '{args.agent_profile}': quality_mode={cfg.get('quality_mode')}, iteration_cap={cfg.get('iteration_cap')}"
        )

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
    _print_results(state, orch)
    output_dir = args.output_dir or _default_output_dir(orch._project_id)
    path = write_output_dir(state, output_dir, project_id=orch._project_id)
    print(f"\nOutput written to: {path}")

    # Organize output: move tasks to tasks/, generate/run tests
    print("\n[ORG] Organizing project output...")
    org_report = await organize_project_output(path, auto_generate_tests=True, run_tests=True)
    safe_print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
    if org_report.tests_run:
        passed = sum(1 for r in org_report.tests_run if r.passed)
        safe_print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")

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

    # Apply agent profile if specified (Wave 1: W1 Agent Profiles)
    if getattr(args, "agent_profile", None):
        profile_map = {
            "standard": {"quality_mode": "standard", "iteration_cap": 3},
            "max": {"quality_mode": "production", "iteration_cap": 5},
            "creative": {"quality_mode": "standard", "iteration_cap": 4, "temperature": 0.8},
            "conservative": {"quality_mode": "production", "iteration_cap": 2, "temperature": 0.3},
            "research": {"quality_mode": "standard", "iteration_cap": 6},
        }
        cfg = profile_map.get(args.agent_profile, {})

    orch = Orchestrator(
        budget=budget, max_concurrency=args.concurrency, tracing_cfg=_build_tracing_cfg(args)
    )
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





def _cmd_nash_status(args):
    """Delegates to commands.nash.status."""
    from .commands.nash import status
    status(args)



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



def cmd_cache_stats(args) -> None:
    """Cache stats — delegates to commands.cache_stats."""
    from .commands.cache_stats import execute
    execute(args)


def cmd_cache_stats(args: argparse.Namespace) -> int:
    """Cache stats (int return) — delegates to commands.cache_stats."""
    from .commands.cache_stats import execute_stats
    return execute_stats(args)








def _nexus_search_cmd(args) -> int:
    """Execute Nexus search."""
    from .nexus_cli import cmd_search

    return asyncio.run(cmd_search(args))


def _nexus_research_cmd(args) -> int:
    """Execute Nexus research."""
    from .nexus_cli import cmd_research

    return asyncio.run(cmd_research(args))


def _nexus_status_cmd(args) -> int:
    """Execute Nexus status."""
    from .nexus_cli import cmd_status

    return asyncio.run(cmd_status(args))


def _nexus_classify_cmd(args) -> int:
    """Execute Nexus classify."""
    from .nexus_cli import cmd_classify

    return asyncio.run(cmd_classify(args))


def print_help() -> None:
    """Print categorized help using the CommandRegistry."""
    from .command_registry import commands_by_category as _cmds_by_cat, resolve_command as _resolve

    by_cat = _cmds_by_cat()
    print("Available commands:")
    print()
    for category in ("Project", "Configuration", "Tools & Skills", "Info", "Exit"):
        cmds = by_cat.get(category, [])
        if not cmds:
            continue
        print(f"  {category}:")
        for cmd in cmds:
            args = f" {cmd.args_hint}" if cmd.args_hint else ""
            aliases = f" ({', '.join(cmd.aliases)})" if cmd.aliases else ""
            print(f"    {cmd.name}{args}{aliases}  — {cmd.description}")
        print()
    print("Type /help <command> for details on a specific command.")


def cmd_gateway(args) -> None:
    """Delegates to commands.gateway."""
    from .commands.gateway import execute
    execute(args)


def cmd_kanban(args) -> None:
    """Delegates to commands.kanban."""
    from .commands.kanban import execute
    execute(args)







def _cmd_website(args):
    """Generate a website — delegates to commands.website."""
    from .commands.website import execute

    execute(args)





def main():
    # ── Suppress specific warnings ───────────────────────────────────────────
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="urllib3.*doesn't match a supported version",
        category=Warning,
        module="requests",
    )

    # ── Top-level parser ─────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Multi-LLM Orchestrator — Local AI Project Runner")

    # ── Subcommands — dynamically discovered from commands/ package ──────────
    from importlib import import_module

    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    from .commands import discover_command_modules

    for cmd_mod in discover_command_modules():
        try:
            mod = import_module(f".commands.{cmd_mod}", __package__)
            mod.register(subparsers)
        except Exception as exc:
            import logging

            logger = logging.getLogger(__name__)
            logger.warning("Could not register command '%s': %s", cmd_mod, exc)

    # ── Legacy flat flags (kept for backwards compatibility) ──────────────────
    parser.add_argument("--project", "-p", type=str, help="Project description")
    parser.add_argument("--criteria", "-c", type=str, help="Success criteria")
    parser.add_argument(
        "--budget", "-b", type=float, default=8.0, help="Max budget in USD (default: 8.0)"
    )
    parser.add_argument(
        "--time", type=float, default=5400, help="Max time in seconds (default: 5400)"
    )
    parser.add_argument(
        "--project-id", type=str, default="", help="Project ID (auto-generated if empty)"
    )
    parser.add_argument("--resume", type=str, default="", help="Resume a previous project by ID")
    parser.add_argument("--list-projects", action="store_true", help="List all saved projects")
    parser.add_argument(
        "--concurrency", type=int, default=3, help="Max simultaneous API calls (default: 3)"
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--file", "-f", type=str, default="", help="Load project spec from a YAML file"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="",
        help="Write structured output files to this directory",
    )
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

    # ═══════════════════════════════════════════════════════
    # TDD-First Generation (NEW v3.0)
    # ═══════════════════════════════════════════════════════
    parser.add_argument(
        "--tdd-first",
        action="store_true",
        help="Enable Test-First Generation (TDD) for code tasks",
    )
    parser.add_argument(
        "--tdd-quality",
        choices=["budget", "balanced", "premium"],
        default="balanced",
        help="TDD model quality tier (default: balanced)",
    )
    parser.add_argument(
        "--tdd-max-iterations",
        type=int,
        default=3,
        help="Maximum TDD iterations before fallback (default: 3)",
    )
    parser.add_argument(
        "--tdd-min-coverage",
        type=float,
        default=0.8,
        help="Minimum test coverage required (default: 0.8)",
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
        "--profile",
        action="store_true",
        default=False,
        help="Enable NexusScope profiling for this run",
    )
    parser.add_argument(
        "--profile-output",
        default=None,
        metavar="PATH",
        help="Write profile report to PATH on exit",
    )
    parser.add_argument(
        "--profile-format", choices=["text", "html", "json", "speedscope"], default="text"
    )
    parser.add_argument(
        "--tracing",
        action="store_true",
        default=False,
        help="Enable OpenTelemetry distributed tracing. Requires: pip install -e '.[tracing]'",
    )
    parser.add_argument(
        "--otlp-endpoint",
        type=str,
        default=None,
        metavar="URL",
        help="OTLP gRPC endpoint for tracing export (e.g. http://localhost:4317). "
        "If --tracing is set but this is omitted, spans are printed to console.",
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
        "--new-project",
        "-N",
        action="store_true",
        default=False,
        help="Skip resume detection and always start a fresh project",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        default=False,
        help=("Skip LLM spec enhancement pass and run original project description directly"),
    )
    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        default=False,
        help="Show execution plan without running any tasks",
    )
    parser.add_argument(
        "--mode",
        choices=["query", "build"],
        default=None,
        help="Execution mode: query (no code gen, plan only) or build (full project)",
    )
    parser.add_argument(
        "--agent-profile",
        choices=["standard", "max", "creative", "conservative", "research"],
        default=None,
        help="Agent profile strategy: standard/max/creative/conservative/research",
    )
    parser.add_argument(
        "--autonomy",
        choices=["lite", "standard", "auto", "max"],
        default=None,
        help="Autonomy level: lite/standard/auto/max (controls iterations, repairs, model tier)",
    )

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

    # --mode query is the same as --dry-run (Ask Mode equivalent)
    if args.dry_run or args.mode == "query":
        asyncio.run(_async_dry_run(args))
        return

    asyncio.run(_async_new_project(args))


def safe_print(msg: str, **kwargs) -> None:
    """Print message with UTF-8 to ASCII fallback for restricted terminals."""

    try:
        print(msg, **kwargs)
    except UnicodeEncodeError:
        # Fallback for Windows consoles (cp1252, cp1253, etc.)
        replacements = {
            "✅": "[OK]",
            "❌": "[FAIL]",
            "⚠️": "[WARN]",
            "ℹ️": "[INFO]",
            "🚀": "[START]",
            "📁": "[DIR]",
            "📊": "[STATS]",
            "✓": "v",
            "✗": "x",
            "█": "#",
            "░": ".",
        }
        safe_msg = msg
        for char, repl in replacements.items():
            safe_msg = safe_msg.replace(char, repl)
        # Final safety pass: encode/decode as ASCII ignoring errors
        safe_msg = safe_msg.encode("ascii", "ignore").decode("ascii")
        try:
            print(safe_msg, **kwargs)
        except Exception:
            pass  # Silent failure if even this fails


def _print_results(state, orch=None):
    print("\n" + "=" * 60)
    print(f"STATUS: {state.status.value}")
    print(f"Budget spent: ${state.budget.spent_usd:.4f} / ${state.budget.max_usd}")
    print(f"Time elapsed: {state.budget.elapsed_seconds:.1f}s / {state.budget.max_time_seconds}s")
    print("-" * 60)

    for tid, result in state.results.items():
        emoji = (
            "OK"
            if result.status.value == "completed"
            else "FAIL" if result.status.value == "failed" else "~"
        )
        safe_print(
            f"  {emoji} {tid}: score={result.score:.3f} "
            f"[{result.model_used.value}] "
            f"iters={result.iterations} "
            f"cost=${result.cost_usd:.4f}"
        )

    print("=" * 60)

    # NEW: Show meta-optimization status if available
    if orch is not None and hasattr(orch, "meta_v2") and orch.meta_v2:
        from .meta_integration import get_meta_status

        meta_status = get_meta_status(orch.meta_v2)
        print("\n--- META-OPTIMIZATION STATUS ---")
        print(f"  Enabled: {meta_status.get('enabled', False)}")
        print(f"  Optimizations run: {meta_status.get('optimization_count', 0)}")
        if "archive_stats" in meta_status:
            print(f"  Projects in archive: {meta_status['archive_stats'].get('total_projects', 0)}")
            print(f"  Total executions: {meta_status['archive_stats'].get('total_executions', 0)}")


# ─────────────────────────────────────────────
# Meta-Optimization CLI Commands
# ─────────────────────────────────────────────


def cmd_meta(args) -> None:
    """Delegates to commands.meta."""
    from .commands.meta import execute
    execute(args)





def _handle_modify_command(args):
    # Handle the modify subcommand.
    import asyncio
    from pathlib import Path

    result = asyncio.run(
        _run_modify(
            repo=Path(args.repo).resolve(),
            objective=args.objective,
            dry_run=getattr(args, "dry_run", False),
        )
    )
    print(result)


async def _run_modify(repo, objective: str, dry_run: bool) -> str:
    # Execute the codebase modification flow.
    from orchestrator.engine import Orchestrator
    from orchestrator.budget import Budget

    try:
        orch = Orchestrator(budget=Budget(max_usd=10.0))
        state = await orch.modify_codebase(
            repo_path=repo,
            objective=objective,
            dry_run=dry_run,
        )
        return f"Modification complete.\nState keys: {list(state.keys()) if state else 'none'}"
    except Exception as exc:
        import traceback

        return f"Modification failed: {exc}\n{traceback.format_exc()}"





# Click-based CLI for codebase analysis feature
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

            from orchestrator.codebase_understanding import CodebaseUnderstanding
            from orchestrator.improvement_suggester import ImprovementSuggester

            async def run_analysis():
                understanding = CodebaseUnderstanding()
                profile = await understanding.analyze(analyze_codebase)

                # Display analysis results
                print("\n" + "=" * 60)
                print("CODEBASE ANALYSIS COMPLETE")
                print("=" * 60)
                print(profile)

                # Generate and display improvement suggestions
                suggester = ImprovementSuggester()
                improvements = suggester.suggest(profile)

                if improvements:
                    print("\n" + "=" * 60)
                    print("IMPROVEMENT RECOMMENDATIONS")
                    print("=" * 60)
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





def _cmd_nexusscope_sessions(args):
    """Delegates to commands.nexusscope.sessions."""
    from .commands.nexusscope import sessions
    sessions(args)



def _cmd_nexusscope_report(args):
    """Delegates to commands.nexusscope.report."""
    from .commands.nexusscope import report
    report(args)



# ─────────────────────────────────────────────────────────────────────────────
# chat — Interactive spec-gathering mode
# ─────────────────────────────────────────────────────────────────────────────





def cmd_chat(args) -> None:
    """Delegates to commands.chat."""
    from .commands.chat import execute
    execute(args)


if __name__ == "__main__":
    main()
