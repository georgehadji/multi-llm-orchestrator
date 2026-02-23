"""
AppVerifier — verifies a generated app by running local tests and Docker checks.
Author: Georgios-Chrysovalantis Chatzivantsidis

verify_local():
  1. pip install -r requirements.txt (if requirements.txt exists)
  2. Run test_command (pytest by default)
  3. Try to start app with run_command (Popen), check it doesn't crash immediately

verify_docker():
  1. Check docker is available (docker info)
  2. Generate a minimal Dockerfile if one doesn't exist
  3. docker build
  4. docker run (with --rm and timeout)
"""
from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from orchestrator.app_detector import AppProfile

logger = logging.getLogger(__name__)

_DOCKER_TIMEOUT = 30  # seconds for docker build/run


@dataclass
class VerifyReport:
    """Report from AppVerifier verification steps."""

    local_install_ok: bool = False
    tests_passed: bool = False
    startup_ok: bool = True   # default True; only set False if startup actually fails
    docker_build_ok: bool = False
    docker_run_ok: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True if the relevant verification checks passed.

        verify_local() only sets local_install_ok/tests_passed/startup_ok.
        verify_docker() only sets docker_build_ok/docker_run_ok.
        This property checks whichever fields were actually exercised:
        - If docker fields were set → docker report → check docker flags only.
        - Otherwise → local report → check local flags.
        """
        if self.docker_build_ok or self.docker_run_ok:
            # Docker verify report — only docker flags matter here.
            # Local flags are checked separately via the local_verify report.
            return self.docker_build_ok and self.docker_run_ok
        return self.local_install_ok and self.tests_passed and self.startup_ok


class AppVerifier:
    """
    Verifies a generated app by running local tests and optional Docker checks.

    All subprocess calls are isolated through subprocess.run / subprocess.Popen
    so they can be mocked in tests.
    """

    _JS_APP_TYPES = frozenset({"nextjs", "react-fastapi", "html"})

    def verify_local(self, output_dir: Path, profile: AppProfile) -> VerifyReport:
        """
        Run local verification:
        1. Install dependencies (npm or pip depending on app_type)
        2. Run test_command (npm test or pytest)
        3. For JS/TS apps: npm run build instead of startup check
        4. For Python apps: start app with run_command (startup check)
        """
        output_dir = Path(output_dir)
        report = VerifyReport()

        is_js = profile.app_type in self._JS_APP_TYPES

        # ── Step 1: Install dependencies ──────────────────────────────────────
        if is_js:
            package_json = output_dir / "package.json"
            if package_json.exists():
                result = subprocess.run(
                    ["npm", "install", "--legacy-peer-deps"],
                    capture_output=True,
                    text=True,
                    cwd=str(output_dir),
                )
                if result.returncode == 0:
                    report.local_install_ok = True
                    logger.debug("npm install succeeded")
                else:
                    report.local_install_ok = False
                    report.errors.append(f"npm install failed: {result.stderr[:200]}")
                    logger.warning("npm install failed: %s", result.stderr[:100])
                    return report
            else:
                report.local_install_ok = True  # no package.json yet — skip
        else:
            req_file = output_dir / "requirements.txt"
            if req_file.exists():
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                    capture_output=True,
                    text=True,
                    cwd=str(output_dir),
                )
                if result.returncode == 0:
                    report.local_install_ok = True
                    logger.debug("pip install succeeded")
                else:
                    report.local_install_ok = False
                    report.errors.append(f"pip install failed: {result.stderr[:200]}")
                    logger.warning("pip install failed: %s", result.stderr[:100])
                    return report  # can't test if install failed
            else:
                # No requirements.txt → treat install as OK
                report.local_install_ok = True

        # ── Step 2: Run tests ──────────────────────────────────────────────────
        if is_js:
            test_cmd = "npm test -- --passWithNoTests"
        else:
            test_cmd = profile.test_command or "pytest"
        test_parts = test_cmd.split()
        result = subprocess.run(
            test_parts,
            capture_output=True,
            text=True,
            cwd=str(output_dir),
        )
        if result.returncode == 0:
            report.tests_passed = True
            logger.debug("Tests passed")
        else:
            report.tests_passed = False
            report.errors.append(f"Tests failed: {result.stderr[:200]}")
            logger.warning("Tests failed: %s", result.stderr[:100])

        # ── Step 3: Build check (JS) or startup check (Python) ────────────────
        if is_js:
            build_result = subprocess.run(
                ["npm", "run", "build"],
                capture_output=True,
                text=True,
                cwd=str(output_dir),
            )
            if build_result.returncode == 0:
                report.startup_ok = True
                logger.debug("npm run build succeeded")
            else:
                report.startup_ok = False
                report.errors.append(f"npm run build failed: {build_result.stderr[:200]}")
                logger.warning("npm run build failed: %s", build_result.stderr[:100])
            return report

        run_cmd = profile.run_command
        if not run_cmd:
            # No run command — skip startup check
            report.startup_ok = True
            return report

        proc = None
        try:
            proc = subprocess.Popen(
                run_cmd.split(),
                cwd=str(output_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Wait briefly to see if it crashes immediately
            time.sleep(0.5)
            if proc.poll() is None:
                # Still running → startup OK
                report.startup_ok = True
                logger.debug("Startup check passed (process still running)")
            else:
                report.startup_ok = False
                report.errors.append(f"App exited immediately with code {proc.returncode}")
                logger.warning("App exited immediately")
        except OSError as exc:
            report.startup_ok = False
            report.errors.append(f"Could not start app: {exc}")
            logger.warning("Could not start app: %s", exc)
        finally:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=3)
                except Exception:
                    proc.kill()

        return report

    def verify_docker(self, output_dir: Path, profile: AppProfile) -> VerifyReport:
        """
        Run Docker verification:
        1. Check docker is available
        2. Generate Dockerfile if not present
        3. docker build
        4. docker run
        """
        output_dir = Path(output_dir)
        report = VerifyReport()

        # ── Step 1: Check Docker is available ─────────────────────────────────
        docker_info = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
        )
        if docker_info.returncode != 0:
            report.errors.append(
                f"Docker not available: {docker_info.stderr[:100]}"
            )
            logger.warning("Docker not available — skipping Docker verification")
            return report

        # ── Step 2: Generate Dockerfile if not present ────────────────────────
        dockerfile = output_dir / "Dockerfile"
        if not dockerfile.exists():
            self._generate_dockerfile(output_dir, profile)

        # ── Step 3: docker build ───────────────────────────────────────────────
        image_tag = "app-builder-verify:latest"
        build_result = subprocess.run(
            ["docker", "build", "-t", image_tag, str(output_dir)],
            capture_output=True,
            text=True,
            timeout=_DOCKER_TIMEOUT,
        )
        if build_result.returncode == 0:
            report.docker_build_ok = True
            logger.debug("Docker build succeeded")
        else:
            report.docker_build_ok = False
            report.errors.append(f"Docker build failed: {build_result.stderr[:200]}")
            logger.warning("Docker build failed")
            return report

        # ── Step 4: docker run ─────────────────────────────────────────────────
        run_result = subprocess.run(
            ["docker", "run", "--rm", "--name", "app-builder-verify-run", image_tag],
            capture_output=True,
            text=True,
            timeout=_DOCKER_TIMEOUT,
        )
        if run_result.returncode == 0:
            report.docker_run_ok = True
            logger.debug("Docker run succeeded")
        else:
            report.docker_run_ok = False
            report.errors.append(f"Docker run failed: {run_result.stderr[:200]}")
            logger.warning("Docker run failed")

        return report

    def _generate_dockerfile(self, output_dir: Path, profile: AppProfile) -> None:
        """Generate a minimal Dockerfile for the app."""
        run_cmd = profile.run_command or "python main.py"
        dockerfile_content = (
            "FROM python:3.11-slim\n"
            "WORKDIR /app\n"
            "COPY requirements.txt* ./\n"
            "RUN pip install --no-cache-dir -r requirements.txt || true\n"
            "COPY . .\n"
            f'CMD {json.dumps(run_cmd.split())}\n'
        )
        (output_dir / "Dockerfile").write_text(dockerfile_content, encoding="utf-8")
        logger.debug("Generated Dockerfile")
