"""
CLI Dispatch — extracted from orchestrator/cli.py main()
===========================================================
Phase 5 Strangler Fig extraction: the argument parser, dispatch
routing, and all async handler functions.

Usage:
    from orchestrator.application.cli_dispatch import run
    run()
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from orchestrator.budget import Budget
from orchestrator.engine import Orchestrator
from orchestrator.output_organizer import organize_project_output
from orchestrator.output_writer import write_output_dir
from orchestrator.progress import ProgressRenderer
from orchestrator.project_file import load_project_file
from orchestrator.state import StateManager
from orchestrator.application.cli_helpers import (
    _build_tracing_cfg,
    _default_output_dir,
    _print_results,
    _resolve_task_paths,
    safe_print,
)

from orchestrator.state_mgmt.resume_detector import (
    ResumeCandidate,
    _extract_keywords,
    _is_exact_match,
    _recency_factor,
    _score_candidates,
)
from orchestrator.visualization import DagRenderer

logger = logging.getLogger("orchestrator.cli")


def run() -> None:
    """Entry point: build parser, parse args, dispatch to async handler."""
    import warnings

    warnings.filterwarnings(
        "ignore",
        message="urllib3.*doesn't match a supported version",
        category=Warning,
        module="requests",
    )

    # ── Parser ────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Multi-LLM Orchestrator — Local AI Project Runner")

    # Subcommands — dynamically discovered from commands/ package
    from importlib import import_module

    subparsers = parser.add_subparsers(dest="subcommand", metavar="SUBCOMMAND")
    from orchestrator.commands import discover_command_modules

    for cmd_mod in discover_command_modules():
        try:
            mod = import_module(f"..commands.{cmd_mod}", __package__)
            mod.register(subparsers)
        except Exception as exc:
            logger.warning("Could not register command '%s': %s", cmd_mod, exc)

    # Legacy flat flags (kept for backwards compatibility)
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

    # TDD-First Generation
    parser.add_argument(
        "--tdd-first",
        action="store_true",
        help="Enable Test-First Generation (TDD) for code tasks",
    )
    parser.add_argument(
        "--tdd-quality",
        type=str,
        default="standard",
        choices=["standard", "high", "maximum"],
        help="TDD quality tier (default: standard)",
    )
    parser.add_argument(
        "--tdd-max-iterations",
        type=int,
        default=3,
        help="Maximum iterations for TDD (default: 3)",
    )
    parser.add_argument(
        "--tdd-min-coverage",
        type=float,
        default=0.0,
        help="Minimum test coverage threshold (default: 0.0)",
    )

    # Visualization
    parser.add_argument(
        "--visualize",
        type=str,
        choices=["mermaid", "ascii"],
        help="Visualize task dependency graph",
    )
    parser.add_argument(
        "--critical-path",
        action="store_true",
        help="Show critical path",
    )
    parser.add_argument(
        "--dependency-report",
        action="store_true",
        help="Show dependency report after execution",
    )

    # Misc
    parser.add_argument("--dry-run", action="store_true", help="Dry run — plan only, no execution")
    parser.add_argument(
        "--mode",
        type=str,
        default="build",
        choices=["build", "query"],
        help="Execution mode (default: build)",
    )
    parser.add_argument(
        "--tracing",
        action="store_true",
        help="Enable OpenTelemetry tracing",
    )
    parser.add_argument(
        "--otlp-endpoint",
        type=str,
        default=None,
        help="OTLP endpoint for tracing",
    )
    parser.add_argument(
        "--agent-profile",
        type=str,
        default=None,
        help="Agent profile name (standard, max, creative, conservative, research)",
    )
    parser.add_argument(
        "--new-project",
        action="store_true",
        help="Skip resume detection and start fresh",
    )
    parser.add_argument(
        "--no-enhance",
        action="store_true",
        help="Skip project description enhancement",
    )
    parser.add_argument(
        "--raw-tasks",
        action="store_true",
        help="Use raw task mode (skip AppBuilder)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress progress output",
    )
    # Aggregate metrics — kept as no-op for backwards compat
    parser.add_argument(
        "--aggregate-metrics",
        action="store_true",
        help="Aggregate metrics across runs (NYI)",
    )
    # Spec-Kit ingestion
    parser.add_argument(
        "--from-speckit",
        type=str,
        default="",
        help="Path to a Spec-Kit output directory (tasks.md, spec.md, plan.md)",
    )
    # Nash subcommand (already registered via dynamic discovery)
    # Removed dead `_nash_subparsers(subparsers)` call (function never defined)

    args = parser.parse_args()

    # ── Dispatch ──────────────────────────────────────────────────────────────

    if args.subcommand is not None:
        func = getattr(args, "func", None)
        if func:
            func(args)
        return

    if args.list_projects:
        asyncio.run(_async_list_projects())
        return

    if args.aggregate_metrics:
        print("Aggregate metrics: feature not yet implemented")
        return

    if args.resume:
        asyncio.run(_async_resume(args))
        return

    if args.visualize or args.critical_path:
        asyncio.run(_async_visualize(args))
        return

    if args.file:
        asyncio.run(_async_file_project(args))
        return

    if args.from_speckit:
        asyncio.run(_async_speckit_project(args))
        return

    if not args.project or not args.criteria:
        parser.error("--project and --criteria are required")

    if args.dry_run or args.mode == "query":
        asyncio.run(_async_dry_run(args))
        return

    asyncio.run(_async_new_project(args))


# ─────────────────────────────────────────────────────────────────────────────
# Async Handlers
# ─────────────────────────────────────────────────────────────────────────────


async def _async_list_projects() -> None:
    sm = StateManager()
    try:
        projects = await sm.list_projects()
        if not projects:
            print("No saved projects.")
        else:
            print(f"{'ID':<15} {'Status':<20} {'Updated'}")
            print("-" * 55)
            for p in projects:
                updated = datetime.fromtimestamp(p["updated_at"]).strftime("%Y-%m-%d %H:%M")
                print(f"{p['project_id']:<15} {p['status']:<20} {updated}")
    finally:
        await sm.close()


async def _async_resume(args: Any) -> None:
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(
        budget=budget, max_concurrency=args.concurrency, tracing_cfg=_build_tracing_cfg(args)
    )
    if getattr(args, "agent_profile", None):

        profile_map = {
            "standard": {"quality_mode": "standard", "iteration_cap": 3},
            "max": {"quality_mode": "production", "iteration_cap": 5},
            "creative": {"quality_mode": "standard", "iteration_cap": 4, "temperature": 0.8},
            "conservative": {"quality_mode": "production", "iteration_cap": 2, "temperature": 0.3},
            "research": {"quality_mode": "standard", "iteration_cap": 6},
        }
        cfg = profile_map.get(args.agent_profile, {})
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
    org_report = await organize_project_output(path, auto_generate_tests=True, run_tests=True)
    safe_print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
    if org_report.tests_run:
        passed = sum(1 for r in org_report.tests_run if r.passed)
        safe_print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")


async def _async_file_project(args: Any) -> None:
    try:
        result = load_project_file(args.file)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    spec = result.spec

    cli_tdd_first = getattr(args, "tdd_first", False)
    effective_tdd_first = result.tdd_first or cli_tdd_first
    effective_tdd_quality = args.tdd_quality if cli_tdd_first else result.tdd_quality

    if effective_tdd_first:
        from orchestrator.cost_optimization import get_optimization_config, update_config

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

    concurrency = args.concurrency if args.concurrency != 3 else result.concurrency
    from orchestrator.application.cli_helpers import setup_logging

    setup_logging(args.verbose or result.verbose)
    budget = spec.budget

    if getattr(args, "agent_profile", None):

        profile_map = {
            "standard": {"quality_mode": "standard", "iteration_cap": 3},
            "max": {"quality_mode": "production", "iteration_cap": 5},
            "creative": {"quality_mode": "standard", "iteration_cap": 4, "temperature": 0.8},
            "conservative": {"quality_mode": "production", "iteration_cap": 2, "temperature": 0.3},
            "research": {"quality_mode": "standard", "iteration_cap": 6},
        }
        cfg = profile_map.get(args.agent_profile, {})

    print(f"Loading project from: {args.file}")
    print(f"Project: {spec.project_description[:80]}")
    print(f"Budget: ${budget.max_usd} / {budget.max_time_seconds}s")
    print("-" * 60)

    orch = Orchestrator(
        budget=budget, max_concurrency=concurrency, tracing_cfg=_build_tracing_cfg(args)
    )

    output_dir = args.output_dir or result.output_dir or _default_output_dir(result.project_id)

    renderer = ProgressRenderer(quiet=getattr(args, "quiet", False))
    project_id = result.project_id or ""
    try:
        async for event in orch.run_project_streaming(
            project_description=spec.project_description,
            success_criteria=spec.success_criteria,
            project_id=project_id,
        ):
            renderer.handle(event)
    except Exception as e:
        safe_print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return

    actual_project_id = getattr(orch, "_project_id", None) or project_id

    try:
        state = await orch.state_mgr.load_project(actual_project_id)
    except Exception:
        state = None

    if state:
        _print_results(state, orch)

    if state:
        path = write_output_dir(state, output_dir, project_id=actual_project_id)
        print(f"\nOutput written to: {path}")

        from orchestrator.assembler import assemble_project

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

        # Assembly
        if result.assemble or result.task_paths:
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
        dag_renderer = DagRenderer(state.tasks, results=state.results)
        print("\n" + dag_renderer.dependency_report())


async def _async_dry_run(args: Any) -> None:
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency)

    print(f"DRY-RUN for project: {args.project}")
    print(f"Criteria: {args.criteria}")
    print("-" * 60)

    plan = await orch.dry_run(args.project, args.criteria)
    print(plan.render())
    await orch.state_mgr.close()
    await orch.cache.close()


async def _async_new_project(args: Any) -> None:
    if getattr(args, "tdd_first", False):
        from orchestrator.cost_optimization import get_optimization_config, update_config

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

    # ── Enhancement pass ─────────────────────────────────────────────────────
    description = args.project
    criteria = args.criteria
    no_enhance = getattr(args, "no_enhance", False)

    if not no_enhance:
        from orchestrator.enhancer import (
            ProjectEnhancer,
            _apply_enhancements,
            _present_enhancements,
        )

        enhancer = ProjectEnhancer()
        suggestions = await enhancer.analyze(description, criteria)
        if suggestions:
            accepted = _present_enhancements(suggestions)
            description, criteria = _apply_enhancements(description, criteria, accepted)

    raw_tasks = getattr(args, "raw_tasks", False)

    if not raw_tasks:
        # Route through AppBuilder
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
            budget=Budget(max_usd=args.budget, max_time_seconds=args.time),
        )
        if result.success:
            print(f"Build successful: {result.output_dir}")
            if result.state:
                print(
                    f"  State tasks: {len(result.state.tasks)}, results: {len(result.state.results)}"
                )
                print(f"  Execution order: {result.state.execution_order}")
            if result.state:
                from orchestrator.output_writer import write_output_dir as _write_out

                tasks_dir = Path(output_dir) / "tasks"
                project_id = getattr(result.state, "project_id", "")
                print(f"  Writing task files to: {tasks_dir}")
                path = _write_out(result.state, tasks_dir, project_id=project_id)
                print(f"Task files written to: {path}")

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

    # --raw-tasks: legacy flat-file path
    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(
        budget=budget, max_concurrency=args.concurrency, tracing_cfg=_build_tracing_cfg(args)
    )

    if getattr(args, "agent_profile", None):

        profile_map = {
            "standard": {"quality_mode": "standard", "iteration_cap": 3},
            "max": {"quality_mode": "production", "iteration_cap": 5},
            "creative": {"quality_mode": "standard", "iteration_cap": 4, "temperature": 0.8},
            "conservative": {"quality_mode": "production", "iteration_cap": 2, "temperature": 0.3},
            "research": {"quality_mode": "standard", "iteration_cap": 6},
        }
        cfg = profile_map.get(args.agent_profile, {})

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

    org_report = await organize_project_output(path, auto_generate_tests=True, run_tests=True)
    safe_print(f"  ✅ Tasks moved: {len(org_report.tasks_moved)}")
    if org_report.tests_run:
        passed = sum(1 for r in org_report.tests_run if r.passed)
        safe_print(f"  ✅ Tests: {passed}/{len(org_report.tests_run)} passed")

    if getattr(args, "dependency_report", False) and state:
        dag_renderer = DagRenderer(state.tasks, results=state.results)
        print("\n" + dag_renderer.dependency_report())


async def _async_visualize(args: Any) -> None:
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


async def _async_speckit_project(args: Any) -> None:
    """Load a Spec-Kit directory and execute its tasks."""
    from orchestrator.ingest import SpecKitAdapter
    from orchestrator.infrastructure.file_reader import FileReader

    # Show available tracing profile
    tracing_cfg = _build_tracing_cfg(args)
    spec_dir = Path(args.from_speckit)
    if not spec_dir.is_dir():
        print(f"ERROR: --from-speckit path is not a directory: {spec_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading Spec-Kit artifacts from: {spec_dir}")
    print("-" * 60)

    # Load artifacts
    reader = FileReader()
    adapter = SpecKitAdapter(file_reader=reader)
    try:
        artifacts = await adapter.load(str(spec_dir))
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    tasks = artifacts.tasks
    constitution = artifacts.constitution
    routing_hints = artifacts.routing_hints

    if not tasks:
        print("ERROR: No tasks parsed from tasks.md", file=sys.stderr)
        sys.exit(1)

    print(f"Parsed {len(tasks)} tasks from tasks.md")
    print(
        f"Constitution: {'loaded' if constitution.protect_paths or constitution.forbidden_imports else 'empty defaults'}"
    )
    if routing_hints:
        print(f"Routing hints: {routing_hints}")
    print("-" * 60)

    # Build project description and criteria from spec
    description = f"Spec-Kit project from {spec_dir}"
    criteria = (
        " ".join(artifacts.raw_spec_criteria)
        if artifacts.raw_spec_criteria
        else "All tasks complete"
    )

    budget = Budget(max_usd=args.budget, max_time_seconds=args.time)
    orch = Orchestrator(budget=budget, max_concurrency=args.concurrency, tracing_cfg=tracing_cfg)

    renderer = ProgressRenderer(quiet=getattr(args, "quiet", False))
    project_id = args.project_id or ""

    print(f"Starting execution (budget: ${args.budget})...")
    try:
        state = await orch.run_project_with_tasks(
            project_description=description,
            success_criteria=criteria,
            tasks=tasks,
            project_id=project_id,
            constitution=constitution,
        )
    except Exception as e:
        safe_print(f"\n❌ Error during execution: {e}")
        import traceback

        traceback.print_exc()
        return

    actual_project_id = getattr(orch, "_project_id", None) or project_id

    try:
        state = state if state else await orch.state_mgr.load_project(actual_project_id)
    except Exception:
        pass

    if state:
        _print_results(state, orch)

    output_dir = args.output_dir or _default_output_dir(actual_project_id)
    path = write_output_dir(state, output_dir, project_id=actual_project_id)
    print(f"\nOutput written to: {path}")

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


async def _check_resume(
    description: str,
    state_mgr: Any,
    new_project: bool = False,
    _input_fn: Any = None,
) -> str | None:
    """Gate that detects and offers to resume a previous project."""
    import asyncio

    if new_project:
        return None

    if _input_fn is None:
        _input_fn = input

    keywords = _extract_keywords(description)
    if not keywords:
        return None

    try:
        rows = await asyncio.wait_for(state_mgr.find_resumable(keywords), timeout=0.2)
    except asyncio.TimeoutError:
        return None
    except Exception as exc:
        logger.warning("Failed to check resumable projects: %s", exc)
        return None

    if not rows:
        return None

    now = datetime.utcnow()
    candidates = []
    for row in rows:
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
                similarity_score=0.0,
                overall_score=0.0,
            )
        )

    scored = _score_candidates(keywords, candidates)
    if not scored:
        return None

    for candidate in scored:
        if _is_exact_match(keywords, candidate.keywords):
            print(
                f"\nResuming previous project (exact match): {candidate.project_id}\n"
                f"  {candidate.description[:80]}"
            )
            return candidate.project_id

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
