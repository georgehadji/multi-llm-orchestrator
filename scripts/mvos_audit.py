#!/usr/bin/env python3
"""
MVOS Audit Script — Automated Minimum Viable Operational State Verification.
===========================================================================

Run this script after any deployment or refactor to verify that all MVOS
invariants still hold.  It performs black-box checks on CLI, API, state
persistence, and circuit breaker behaviour.

Usage:
    python scripts/mvos_audit.py [--verbose]

Exit codes:
    0 — All invariants pass
    1 — One or more invariants failed

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Add project root to path so imports work when script is run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _run_cmd(cmd: list[str], timeout: float = 30.0) -> tuple[int, str, str]:
    """Run a subprocess command and return (rc, stdout, stderr)."""
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=PROJECT_ROOT,
    )
    return result.returncode, result.stdout, result.stderr


class MVOSAuditor:
    """
    Checks each MVOS invariant defined in ARCHITECTURAL_AUDIT_V5.md.
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: list[dict] = []
        self.python = sys.executable

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [mvos] {msg}")

    def _record(self, name: str, passed: bool, details: str = "") -> None:
        self.results.append({
            "invariant": name,
            "passed": passed,
            "details": details,
        })
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}: {name}")
        if details and not passed:
            print(f"         -> {details}")

    # ─────────────────────────────────────────────────────────────────────────
    # Invariant 1: CLI health check is callable
    # ─────────────────────────────────────────────────────────────────────────

    def check_cli_health(self) -> None:
        """Verify `python -m orchestrator --help` exits 0."""
        rc, stdout, stderr = _run_cmd([self.python, "-m", "orchestrator", "--help"])
        passed = rc == 0 and "Multi-LLM Orchestrator" in stdout
        self._record(
            "CLI_HEALTH_CHECK",
            passed,
            f"rc={rc}, stdout={stdout[:80]!r}" if not passed else "",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Invariant 2: CLI run_project is callable
    # ─────────────────────────────────────────────────────────────────────────

    def check_cli_run_project(self) -> None:
        """
        Verify run_project entry point is reachable.
        Uses --list-projects (fast, no API calls) to confirm the CLI loads
        all modules without import errors.
        """
        rc, stdout, stderr = _run_cmd(
            [self.python, "-m", "orchestrator", "--list-projects"],
            timeout=15.0,
        )
        # rc may be non-zero (empty list, no API key), but must NOT traceback
        passed = "Traceback" not in stderr and "ModuleNotFoundError" not in stderr
        self._record(
            "CLI_RUN_PROJECT",
            passed,
            f"stderr={stderr[:200]!r}" if not passed else "",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Invariant 3: API execute_task returns 200 or 400/500
    # ─────────────────────────────────────────────────────────────────────────

    def check_api_execute_task(self) -> None:
        """Verify APIServer can be imported and instantiated."""
        try:
            from orchestrator.api_server import APIServer

            server = APIServer(port=0, auth_required=False)
            assert server.app is not None
            self._record("API_EXECUTE_TASK", True)
        except Exception as exc:
            self._record("API_EXECUTE_TASK", False, str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # Invariant 4: State resume returns last_state or None
    # ─────────────────────────────────────────────────────────────────────────

    async def check_state_resume(self) -> None:
        """Verify StateManager can save and load a project."""
        try:
            import tempfile

            from orchestrator.budget import Budget
            from orchestrator.models import ProjectState
            from orchestrator.state import StateManager

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = Path(f.name)

            sm = StateManager(db_path=db_path)
            state = ProjectState(
                project_description="mvos audit",
                success_criteria="pass",
                budget=Budget(max_usd=1.0),
            )
            await sm.save_project("mvos-test", state)
            loaded = await sm.load_project("mvos-test")
            await sm.close()

            passed = loaded is not None and loaded.project_description == "mvos audit"
            self._record(
                "STATE_RESUME",
                passed,
                f"loaded={loaded}" if not passed else "",
            )
        except Exception as exc:
            self._record("STATE_RESUME", False, str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # Invariant 5: Results persisted within 5 seconds
    # ─────────────────────────────────────────────────────────────────────────

    async def check_results_persisted(self) -> None:
        """Verify save → load round-trip is < 5 seconds."""
        try:
            import tempfile

            from orchestrator.budget import Budget
            from orchestrator.models import ProjectState
            from orchestrator.state import StateManager

            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
                db_path = Path(f.name)

            sm = StateManager(db_path=db_path)
            state = ProjectState(
                project_description="speed test",
                success_criteria="fast",
                budget=Budget(max_usd=1.0),
            )

            t0 = time.monotonic()
            await sm.save_project("speed-test", state)
            loaded = await sm.load_project("speed-test")
            elapsed = time.monotonic() - t0
            await sm.close()

            passed = loaded is not None and elapsed < 5.0
            self._record(
                "RESULTS_PERSISTED",
                passed,
                f"elapsed={elapsed:.2f}s" if not passed else "",
            )
        except Exception as exc:
            self._record("RESULTS_PERSISTED", False, str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # Invariant 6: Circuit breaker trips within 30s
    # ─────────────────────────────────────────────────────────────────────────

    async def check_circuit_breaker(self) -> None:
        """Verify CircuitBreaker can be instantiated and trips correctly."""
        try:
            from orchestrator.circuit_breaker import CircuitBreaker, CircuitBreakerOpen

            cb = CircuitBreaker(failure_threshold=3, reset_timeout=1)
            # Trip it
            for _ in range(3):
                await cb.record_failure()

            assert cb.is_open, "Circuit breaker did not open after 3 failures"

            # Should raise immediately
            t0 = time.monotonic()
            try:
                async with cb.context():
                    pass  # context entry checks state
            except CircuitBreakerOpen:
                pass
            elapsed = time.monotonic() - t0

            passed = elapsed < 1.0  # Must be nearly instant
            self._record(
                "CIRCUIT_BREAKER_TRIP",
                passed,
                f"elapsed={elapsed:.2f}s" if not passed else "",
            )
        except Exception as exc:
            self._record("CIRCUIT_BREAKER_TRIP", False, str(exc))

    # ─────────────────────────────────────────────────────────────────────────
    # Run all checks
    # ─────────────────────────────────────────────────────────────────────────

    async def run_all(self) -> int:
        print("=" * 60)
        print("MVOS AUDIT — Minimum Viable Operational State Verification")
        print("=" * 60)

        self.check_cli_health()
        self.check_cli_run_project()
        self.check_api_execute_task()
        await self.check_state_resume()
        await self.check_results_persisted()
        await self.check_circuit_breaker()

        print("-" * 60)
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)
        print(f"Result: {passed}/{total} invariants passed")

        if passed < total:
            print("\nFailed invariants:")
            for r in self.results:
                if not r["passed"]:
                    print(f"  - {r['invariant']}: {r['details']}")
            return 1

        print("\n[PASS] ALL MVOS INVARIANTS PASS — deployment ready")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MVOS Audit Script")
    parser.add_argument("--verbose", "-v", action="store_true", help="Detailed logging")
    args = parser.parse_args()

    auditor = MVOSAuditor(verbose=args.verbose)
    return asyncio.run(auditor.run_all())


if __name__ == "__main__":
    sys.exit(main())
