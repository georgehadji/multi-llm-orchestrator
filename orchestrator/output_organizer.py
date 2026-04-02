"""
Output Organizer & Test Runner
==============================
Post-project organization and test automation.

Features:
- Organizes task files into tasks/ folder
- Auto-generates missing tests based on code analysis
- Runs tests and captures results
- Moves test files to tests/ folder
- Suppresses verbose cache messages

Usage:
    from orchestrator.output_organizer import OutputOrganizer

    organizer = OutputOrganizer(output_dir)
    await organizer.organize_project()
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .log_config import get_logger

logger = get_logger(__name__)

# Import test fixer for iterative fixing
try:
    from .test_fixer import TestFixer, TestFixReport
except ImportError:
    TestFixer = None
    TestFixReport = None


@dataclass
class TestResult:
    """Result of a test run."""

    test_file: str
    passed: bool
    duration_ms: float
    output: str
    error_message: str = ""
    coverage_percent: float = 0.0


@dataclass
class OrganizationReport:
    """Report of the organization process."""

    tasks_moved: list[str] = field(default_factory=list)
    tests_created: list[str] = field(default_factory=list)
    tests_run: list[TestResult] = field(default_factory=list)
    tests_moved: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "tasks_moved": self.tasks_moved,
            "tests_created": self.tests_created,
            "tests_run": [
                {
                    "test_file": r.test_file,
                    "passed": r.passed,
                    "duration_ms": r.duration_ms,
                    "coverage_percent": r.coverage_percent,
                    "error_message": r.error_message if not r.passed else None,
                }
                for r in self.tests_run
            ],
            "tests_moved": self.tests_moved,
            "errors": self.errors,
            "summary": {
                "total_tests": len(self.tests_run),
                "passed_tests": sum(1 for r in self.tests_run if r.passed),
                "failed_tests": sum(1 for r in self.tests_run if not r.passed),
                "coverage_avg": (
                    sum(r.coverage_percent for r in self.tests_run) / len(self.tests_run)
                    if self.tests_run
                    else 0
                ),
            },
        }


class OutputOrganizer:
    """
    Organizes project output and manages tests.

    Args:
        output_dir: Directory containing project output
        auto_generate_tests: Whether to auto-generate missing tests
        run_tests: Whether to run tests after organization
        min_coverage: Minimum required coverage percentage
        fix_tests: Whether to iteratively fix failing tests
        max_fix_iterations: Maximum iterations for test fixing
        min_pass_rate: Minimum pass rate to stop fixing
    """

    def __init__(
        self,
        output_dir: Path,
        auto_generate_tests: bool = True,
        run_tests: bool = True,
        min_coverage: float = 80.0,
        fix_tests: bool = True,
        max_fix_iterations: int = 3,
        min_pass_rate: float = 0.7,
    ):
        self.output_dir = Path(output_dir)
        self.auto_generate_tests = auto_generate_tests
        self.run_tests = run_tests
        self.min_coverage = min_coverage
        self.fix_tests = fix_tests
        self.max_fix_iterations = max_fix_iterations
        self.min_pass_rate = min_pass_rate

        # Directories
        self.tasks_dir = self.output_dir / "tasks"
        self.tests_dir = self.output_dir / "tests"
        self.src_dir = self.output_dir / "src"
        self.app_dir = self.output_dir / "app"

        # Track what we've done
        self.report = OrganizationReport()
        self.fix_report: TestFixReport | None = None

    async def organize_project(self) -> OrganizationReport:
        """
        Main entry point: organize the entire project.

        Steps:
        1. Move task files to tasks/
        2. Detect source files
        3. Generate missing tests
        4. Run tests
        5. Move tests to tests/
        """
        logger.info("=" * 60)
        logger.info("📁 Organizing project output...")
        logger.info("=" * 60)

        try:
            # Step 1: Organize task files
            await self._organize_task_files()

            # Step 2: Detect source files
            source_files = self._detect_source_files()

            # Step 3: Generate tests if needed
            if self.auto_generate_tests and source_files:
                await self._generate_missing_tests(source_files)

            # Step 4: Run tests
            if self.run_tests:
                await self._run_all_tests()

            # Step 4b: Iteratively fix failing tests
            if self.run_tests and self.fix_tests and TestFixer is not None:
                await self._fix_failing_tests()

            # Step 5: Organize test files
            await self._organize_test_files()

        except Exception as e:
            logger.error(f"Organization failed: {e}")
            self.report.errors.append(str(e))

        # Print summary
        self._print_summary()

        # Save report
        self._save_report()

        return self.report

    async def _organize_task_files(self):
        """Move task_*.py/md/json files to tasks/ folder."""
        logger.info("📂 Moving task files to tasks/...")

        # Create tasks directory
        self.tasks_dir.mkdir(exist_ok=True)

        # Find task files
        task_patterns = ["task_*.py", "task_*.md", "task_*.json"]
        task_files = []
        for pattern in task_patterns:
            task_files.extend(self.output_dir.glob(pattern))

        # Move each file
        for task_file in task_files:
            dest = self.tasks_dir / task_file.name
            try:
                shutil.move(str(task_file), str(dest))
                self.report.tasks_moved.append(task_file.name)
                logger.debug(f"  Moved: {task_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to move {task_file.name}: {e}")

        logger.info(f"  ✅ Moved {len(self.report.tasks_moved)} task files")

    def _detect_source_files(self) -> list[Path]:
        """Detect Python source files in the project."""
        source_files = []

        # Check common source directories
        for src_dir in [self.src_dir, self.app_dir, self.output_dir]:
            if not src_dir.exists():
                continue

            # Find Python files (excluding tests, __pycache__)
            for py_file in src_dir.rglob("*.py"):
                if self._is_test_file(py_file):
                    continue
                if "__pycache__" in str(py_file):
                    continue
                if py_file.name.startswith("test_"):
                    continue
                source_files.append(py_file)

        logger.info(f"📄 Detected {len(source_files)} source files")
        return source_files

    def _is_test_file(self, path: Path) -> bool:
        """Check if a file is a test file."""
        name = path.name
        return name.startswith("test_") or name.endswith("_test.py") or path.parent.name == "tests"

    async def _generate_missing_tests(self, source_files: list[Path]):
        """Generate tests for source files that lack them."""
        logger.info("🧪 Checking for missing tests...")

        # Find existing tests
        existing_tests = set()
        for test_dir in [self.tests_dir, self.output_dir]:
            if not test_dir.exists():
                continue
            for test_file in test_dir.rglob("test_*.py"):
                # Extract module name from test file
                test_name = test_file.stem.replace("test_", "")
                existing_tests.add(test_name)

        # Ensure conftest.py exists in tests dir to add output_dir to sys.path
        await self._ensure_conftest()

        # Generate tests for files without them
        generated = []
        for src_file in source_files:
            module_name = src_file.stem
            if module_name in existing_tests:
                continue

            # Check if there's already a test file
            test_file = self.tests_dir / f"test_{module_name}.py"
            if test_file.exists():
                continue

            # Generate test
            try:
                test_content = await self._generate_test_for_file(src_file)
                if test_content:
                    self.tests_dir.mkdir(exist_ok=True)
                    test_file.write_text(test_content, encoding="utf-8")
                    generated.append(test_file.name)
                    logger.info(f"  ✨ Generated: {test_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to generate test for {src_file.name}: {e}")

        self.report.tests_created = generated
        if generated:
            logger.info(f"  ✅ Generated {len(generated)} test files")
        else:
            logger.info("  ✓ All source files already have tests")

    async def _ensure_conftest(self) -> None:
        """Create a conftest.py in tests/ that adds output_dir to sys.path."""
        self.tests_dir.mkdir(exist_ok=True)
        conftest_path = self.tests_dir / "conftest.py"
        if not conftest_path.exists():
            conftest_content = (
                '"""Auto-generated conftest — adds project root to sys.path."""\n'
                "import sys\n"
                "from pathlib import Path\n"
                "\n"
                "# Add project root so generated test imports resolve correctly\n"
                "sys.path.insert(0, str(Path(__file__).parent.parent))\n"
            )
            conftest_path.write_text(conftest_content, encoding="utf-8")
            logger.debug("  Created conftest.py with sys.path fix")

    async def _generate_test_for_file(self, src_file: Path) -> str | None:
        """Generate a test file for a source file."""
        try:
            content = src_file.read_text(encoding="utf-8")
        except Exception:
            return None

        # Extract classes and functions
        classes = re.findall(r"^class\s+(\w+)", content, re.MULTILINE)
        functions = re.findall(r"^def\s+(\w+)\s*\(", content, re.MULTILINE)
        public_functions = [f for f in functions if not f.startswith("_")]

        if not classes and not public_functions:
            return None

        # Generate test content
        module_path = self._get_module_import_path(src_file)

        test_lines = [
            '"""',
            f"Auto-generated tests for {src_file.name}",
            '"""',
            "import pytest",
            f'from {module_path} import {", ".join(classes + public_functions[:5])}',
            "",
            "",
        ]

        # Generate test for each class
        for cls in classes:
            test_lines.extend(
                [
                    f"class Test{cls}:",
                    f'    """Tests for {cls} class."""',
                    "",
                    f"    def test_{cls.lower()}_initialization(self):",
                    f'        """Test {cls} can be instantiated."""',
                    "        # TODO: Add proper initialization parameters",
                    f"        # instance = {cls}()",
                    "        # assert instance is not None",
                    "        pass",
                    "",
                ]
            )

        # Generate test for each function
        for func in public_functions[:5]:  # Limit to first 5 functions
            test_lines.extend(
                [
                    f"def test_{func}():",
                    f'    """Test {func} function."""',
                    "    # TODO: Add proper test parameters and assertions",
                    f"    # result = {func}()",
                    "    # assert result is not None",
                    "    pass",
                    "",
                ]
            )

        return "\n".join(test_lines)

    def _get_module_import_path(self, src_file: Path) -> str:
        """Get the Python import path for a source file."""
        # Try to find relative to src/ or app/
        for base in [self.src_dir, self.app_dir]:
            if base.exists():
                try:
                    rel_path = src_file.relative_to(base)
                    parts = list(rel_path.parts[:-1])  # Exclude filename
                    module_name = src_file.stem
                    if parts:
                        return ".".join(parts) + "." + module_name
                    return module_name
                except ValueError:
                    continue

        # Try relative to output_dir (most common case)
        try:
            rel_path = src_file.relative_to(self.output_dir)
            parts = list(rel_path.parts[:-1])  # Exclude filename
            module_name = src_file.stem
            if parts:
                return ".".join(parts) + "." + module_name
            return module_name
        except ValueError:
            pass

        # Last resort: bare module name
        return src_file.stem

    async def _run_all_tests(self):
        """Run all tests in the project."""
        logger.info("🚀 Running tests...")

        # Find test files
        test_files = []
        for test_dir in [self.tests_dir, self.output_dir]:
            if not test_dir.exists():
                continue
            test_files.extend(test_dir.rglob("test_*.py"))

        if not test_files:
            logger.info("  ℹ No tests found to run")
            return

        # Run each test file
        for test_file in test_files:
            result = await self._run_single_test(test_file)
            self.report.tests_run.append(result)

        # Print summary
        passed = sum(1 for r in self.report.tests_run if r.passed)
        total = len(self.report.tests_run)
        logger.info(f"  ✅ Tests: {passed}/{total} passed")

    async def _run_single_test(self, test_file: Path) -> TestResult:
        """Run a single test file."""
        import os
        import time

        start_time = time.time()

        try:
            # Run pytest with coverage
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                str(test_file),
                "-v",
                "--tb=short",
                "--no-header",
                "-q",
            ]

            # Try to add coverage if available
            try:
                import pytest_cov  # noqa: F401

                cmd.extend(["--cov=.", "--cov-report=term-missing"])
            except ImportError:
                pass

            # Ensure output_dir is on PYTHONPATH so generated imports resolve
            env = os.environ.copy()
            existing_pythonpath = env.get("PYTHONPATH", "")
            output_dir_str = str(self.output_dir)
            if existing_pythonpath:
                env["PYTHONPATH"] = output_dir_str + os.pathsep + existing_pythonpath
            else:
                env["PYTHONPATH"] = output_dir_str

            result = subprocess.run(
                cmd,
                cwd=str(self.output_dir),
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )

            duration = (time.time() - start_time) * 1000
            passed = result.returncode == 0

            # Parse coverage
            coverage = 0.0
            coverage_match = re.search(r"(\d+)%", result.stdout)
            if coverage_match:
                coverage = float(coverage_match.group(1))

            return TestResult(
                test_file=test_file.name,
                passed=passed,
                duration_ms=duration,
                output=result.stdout + result.stderr,
                error_message=result.stderr if not passed else "",
                coverage_percent=coverage,
            )

        except subprocess.TimeoutExpired:
            return TestResult(
                test_file=test_file.name,
                passed=False,
                duration_ms=60000,
                output="",
                error_message="Test timeout (60s)",
            )
        except Exception as e:
            return TestResult(
                test_file=test_file.name,
                passed=False,
                duration_ms=0,
                output="",
                error_message=str(e),
            )

    async def _fix_failing_tests(self):
        """Iteratively fix failing tests using TestFixer."""
        # Check if there are failed tests
        failed_count = sum(1 for r in self.report.tests_run if not r.passed)
        total_count = len(self.report.tests_run)

        if failed_count == 0 or total_count == 0:
            return

        pass_rate = (total_count - failed_count) / total_count
        if pass_rate >= self.min_pass_rate:
            logger.info(f"✅ Pass rate {pass_rate:.1%} >= {self.min_pass_rate:.1%}, skipping fixes")
            return

        logger.info("=" * 60)
        logger.info("🔧 Iterative Test Fixing")
        logger.info("=" * 60)
        logger.info(f"Initial: {total_count - failed_count}/{total_count} passed ({pass_rate:.1%})")

        try:
            fixer = TestFixer()
            self.fix_report = await fixer.fix_failing_tests(
                project_path=str(self.output_dir),
                max_iterations=self.max_fix_iterations,
                min_pass_rate=self.min_pass_rate,
            )

            # Re-run tests after fixing
            logger.info("🔄 Re-running tests after fixes...")
            self.report.tests_run = []  # Clear previous results
            await self._run_all_tests_no_fix()  # Run without triggering fix again

            # Log results
            if self.fix_report.success:
                logger.info("✅ Test fixing completed successfully!")
            else:
                logger.info("⚠️  Test fixing completed with partial success")

            if self.fix_report.iterations:
                logger.info(f"   Iterations: {len(self.fix_report.iterations)}")
                logger.info(f"   Total fixes: {self.fix_report.total_fixes_applied}")

        except Exception as e:
            logger.error(f"Test fixing failed: {e}")

    async def _run_all_tests_no_fix(self):
        """Run tests without triggering fix cycle (internal use)."""
        original_fix = self.fix_tests
        self.fix_tests = False
        try:
            await self._run_all_tests()
        finally:
            self.fix_tests = original_fix

    async def _organize_test_files(self):
        """Move test files to tests/ folder."""
        logger.info("📂 Organizing test files...")

        # Create tests directory
        self.tests_dir.mkdir(exist_ok=True)

        # Find test files in root
        test_files = list(self.output_dir.glob("test_*.py"))

        # Move each file
        for test_file in test_files:
            # Skip if already in tests/
            if str(test_file.parent) == str(self.tests_dir):
                continue

            dest = self.tests_dir / test_file.name
            try:
                # If destination exists, remove it first
                if dest.exists():
                    dest.unlink()
                shutil.move(str(test_file), str(dest))
                self.report.tests_moved.append(test_file.name)
                logger.debug(f"  Moved: {test_file.name}")
            except Exception as e:
                logger.warning(f"  Failed to move {test_file.name}: {e}")

        logger.info(f"  ✅ Organized {len(self.report.tests_moved)} test files")

    def _print_summary(self):
        """Print organization summary."""
        logger.info("=" * 60)
        logger.info("📊 Organization Summary")
        logger.info("=" * 60)

        # Tasks
        logger.info(f"  📁 Task files moved: {len(self.report.tasks_moved)}")

        # Tests created
        if self.report.tests_created:
            logger.info(f"  ✨ Tests generated: {len(self.report.tests_created)}")
            for test in self.report.tests_created:
                logger.info(f"     - {test}")

        # Test results
        if self.report.tests_run:
            passed = sum(1 for r in self.report.tests_run if r.passed)
            total = len(self.report.tests_run)
            failed = total - passed

            logger.info(f"  🧪 Tests run: {total}")
            logger.info(f"     ✅ Passed: {passed}")
            if failed > 0:
                logger.info(f"     ❌ Failed: {failed}")

            # Coverage
            avg_coverage = sum(r.coverage_percent for r in self.report.tests_run) / len(
                self.report.tests_run
            )
            logger.info(f"     📈 Coverage: {avg_coverage:.1f}%")

        # Tests organized
        logger.info(f"  📂 Test files organized: {len(self.report.tests_moved)}")

        if self.report.errors:
            logger.warning(f"  ⚠️  Errors: {len(self.report.errors)}")

        logger.info("=" * 60)

    def _save_report(self):
        """Save organization report to JSON."""
        report_file = self.output_dir / "organization_report.json"
        try:
            report_file.write_text(json.dumps(self.report.to_dict(), indent=2), encoding="utf-8")
            logger.info(f"📄 Report saved: {report_file}")
        except Exception as e:
            logger.warning(f"Failed to save report: {e}")


# Cache message suppression utilities
class CacheMessageSuppressor:
    """
    Suppresses verbose cache-related log messages.

    Usage:
        with CacheMessageSuppressor():
            # Cache messages will be suppressed
            await orchestrator.run_project(...)
    """

    def __init__(self, logger_names: list[str] | None = None):
        self.logger_names = logger_names or [
            "orchestrator.cache",
            "orchestrator.api_clients",
        ]
        self.original_levels: dict[str, int] = {}

    def __enter__(self):
        """Suppress cache messages."""
        for name in self.logger_names:
            logger = logging.getLogger(name)
            self.original_levels[name] = logger.level
            logger.setLevel(logging.WARNING)  # Only show warnings and above
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original log levels."""
        for name, level in self.original_levels.items():
            logging.getLogger(name).setLevel(level)


def suppress_cache_messages():
    """
    Globally suppress cache hit messages by setting appropriate log levels.

    Call this once at application startup.
    """
    # Set cache loggers to WARNING level (suppress INFO/DEBUG)
    logging.getLogger("orchestrator.cache").setLevel(logging.WARNING)
    logging.getLogger("orchestrator.api_clients").setLevel(logging.WARNING)

    # Also suppress specific performance cache messages
    logging.getLogger("orchestrator.performance").setLevel(logging.WARNING)


# Convenience function for engine integration
async def organize_project_output(
    output_dir: Path,
    auto_generate_tests: bool = True,
    run_tests: bool = True,
    fix_tests: bool = True,
    max_fix_iterations: int = 3,
    min_pass_rate: float = 0.7,
) -> OrganizationReport:
    """
    Convenience function to organize project output.

    Args:
        output_dir: Directory containing project output
        auto_generate_tests: Whether to auto-generate missing tests
        run_tests: Whether to run tests
        fix_tests: Whether to iteratively fix failing tests
        max_fix_iterations: Maximum iterations for test fixing
        min_pass_rate: Minimum pass rate to stop fixing

    Returns:
        OrganizationReport with details of what was done
    """
    organizer = OutputOrganizer(
        output_dir=output_dir,
        auto_generate_tests=auto_generate_tests,
        run_tests=run_tests,
        fix_tests=fix_tests,
        max_fix_iterations=max_fix_iterations,
        min_pass_rate=min_pass_rate,
    )
    return await organizer.organize_project()
