"""
CLI Helper Utilities — extracted from orchestrator/cli.py
===========================================================
Phase 5 Strangler Fig extraction: small standalone functions
used by the CLI dispatch layer.

safe_print, _print_results, _default_output_dir, setup_logging,
_build_tracing_cfg, _resolve_task_paths.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


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
        safe_msg = safe_msg.encode("ascii", "ignore").decode("ascii")
        try:
            print(safe_msg, **kwargs)
        except Exception:
            pass


def _default_output_dir(project_id: str | None) -> str:
    """
    Build a default output path when --output-dir is not supplied.
    Format: ./outputs/<project_id> or ./outputs/app_<timestamp> if no project_id
    The directory is created by write_output_dir, not here.
    """
    if project_id:
        return str(Path("outputs") / project_id)
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path("outputs") / f"app_{timestamp}")


def setup_logging(verbose: bool = False, suppress_cache: bool = True):
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    if suppress_cache and not verbose:
        try:
            from ..output_organizer import suppress_cache_messages

            suppress_cache_messages()
        except ImportError:
            pass


def _build_tracing_cfg(args) -> Any:
    """Return a TracingConfig when --tracing is set, otherwise None."""
    try:
        from ..tracing import TracingConfig
    except ImportError:
        return None
    if getattr(args, "tracing", False):
        return TracingConfig(
            enabled=True,
            otlp_endpoint=getattr(args, "otlp_endpoint", None),
        )
    return None


def _print_results(state, orch=None):
    """Print execution summary for a completed project."""
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

    if orch is not None and hasattr(orch, "meta_v2") and orch.meta_v2:
        try:
            from ..meta_integration import get_meta_status

            meta_status = get_meta_status(orch.meta_v2)
            print("\n--- META-OPTIMIZATION STATUS ---")
            print(f"  Enabled: {meta_status.get('enabled', False)}")
            print(f"  Optimizations run: {meta_status.get('optimization_count', 0)}")
            if "archive_stats" in meta_status:
                print(
                    f"  Projects in archive: {meta_status['archive_stats'].get('total_projects', 0)}"
                )
                print(
                    f"  Total executions: {meta_status['archive_stats'].get('total_executions', 0)}"
                )
        except ImportError:
            pass


def _resolve_task_paths(task_paths: dict[str, str], state) -> dict[str, str]:
    """
    Resolve a task_paths dict from the YAML file into a {task_id: target_path} mapping.

    Keys may be:
    - 1-based integer index ("1", "2", ...) → resolved against execution_order
    - Exact task_id string ("task_001", ...) → used as-is
    """
    if not task_paths:
        return {}
    order = state.execution_order or list(state.results.keys())
    resolved: dict[str, str] = {}
    for key, target in task_paths.items():
        try:
            idx = int(key) - 1
            if 0 <= idx < len(order):
                resolved[order[idx]] = target
            else:
                resolved[key] = target
        except ValueError:
            resolved[key] = target
    return resolved
