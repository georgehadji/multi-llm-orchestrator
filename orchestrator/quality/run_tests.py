"""
Run tests for a generated project — delegates to TestingService via container.
============================================================================
Author: Implementation Plan (Autonomous Testing Engine)

F-5 replacement: no longer a silent stub. Delegates to the configured
test runner via TestingService. Raises TestRunnerUnavailableError when
no runner resolves — never returns [] silently.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class TestRunnerUnavailableError(RuntimeError):
    """Raised when no test runner can be resolved for a project."""


def run_project_tests(
    project_dir: str,
    *,
    fixtures: list[str] | None = None,
    framework: str = "pytest",
    timeout_s: float = 120.0,
) -> list[dict[str, Any]]:
    """Run project tests and return structured results.

    Args:
        project_dir: Path to the project directory.
        fixtures: Optional list of fixture paths to install.
        framework: Test framework to use (default: pytest).
        timeout_s: Maximum execution time in seconds.

    Returns:
        List of test outcome dicts with keys: node_id, status, duration_ms, message.

    Raises:
        TestRunnerUnavailableError: If no runner can resolve for the framework.
    """
    try:
        from orchestrator.infrastructure.test_runners import get_runner
    except ImportError:
        logger.error(
            "Test runner infrastructure not available. " "Install orchestrator with dev extras."
        )
        raise TestRunnerUnavailableError("Test runner infrastructure not available") from None

    try:
        runner = get_runner(framework)
    except KeyError:
        raise TestRunnerUnavailableError(
            f"No runner registered for framework: {framework}"
        ) from None

    root = Path(project_dir).resolve()

    from orchestrator.domain.testing_models import Workspace

    workspace = Workspace(
        root=root,
        framework=framework,
        manifest=root / "pyproject.toml" if (root / "pyproject.toml").exists() else None,
    )

    # In a full async context this would be awaited, but this sync stub
    # bridges to the new runner architecture synchronously for backward compat.
    import os as _os
    import subprocess

    # Scrub sensitive env vars before passing to test subprocess
    clean_env = dict(_os.environ)
    for key in list(clean_env):
        if "API_KEY" in key.upper() or "SECRET" in key.upper() or "TOKEN" in key.upper():
            del clean_env[key]
    # Match the async runner: plugin autoload disabled, explicit loads only
    # (prevents third-party plugin side effects and double-loading).
    clean_env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    try:
        cmd = runner.build_command(workspace)
        result = subprocess.run(
            cmd,
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=clean_env,
        )
        stdout = result.stdout
        stderr = result.stderr
        exit_code = result.returncode
    except subprocess.TimeoutExpired:
        logger.warning("Test execution timed out after %ss", timeout_s)
        return [
            {
                "node_id": "__timeout__",
                "status": "error",
                "duration_ms": timeout_s * 1000,
                "message": f"Timeout after {timeout_s}s",
            }
        ]
    except FileNotFoundError:
        raise TestRunnerUnavailableError(
            f"Test runner command not found for framework: {framework}"
        ) from None

    report = runner.parse_report(
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        duration_ms=0.0,
    )

    results = [
        {
            "node_id": o.node_id,
            "status": o.status.value,
            "duration_ms": o.duration_ms,
            "message": o.message,
        }
        for o in report.outcomes
    ]

    # Include collection errors as pseudo-outcomes
    for err in report.collection_errors:
        results.append(
            {
                "node_id": "__collection__",
                "status": "error",
                "duration_ms": 0,
                "message": err,
            }
        )

    return results
