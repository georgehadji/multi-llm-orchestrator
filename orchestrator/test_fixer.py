"""
Test Fixer — Iterative Test Failure Resolution
================================================
Automatically fixes failing tests by analyzing failures
and generating code fixes via LLM.

Usage:
    fixer = TestFixer(orchestrator)
    report = await fixer.fix_failing_tests(
        project_path="outputs/my_project",
        max_iterations=3
    )
"""

import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .api_clients import UnifiedClient
from .budget import Budget
from .tracing import get_tracer


class BudgetManager:
    """Simple budget manager for test fixing."""

    def __init__(self, max_cost: float = 2.0, max_time: float = 300):
        self._budget = Budget(max_usd=max_cost, max_time_seconds=max_time)

    def is_exhausted(self) -> bool:
        return self._budget.spent_usd >= self._budget.max_usd


@dataclass
class TestFailure:
    """Represents a single test failure."""

    test_file: str
    test_name: str
    error_type: str
    error_message: str
    stack_trace: str
    source_file: str | None = None
    line_number: int | None = None


@dataclass
class FixIteration:
    """Results from one fix iteration."""

    iteration: int
    tests_before: int
    failures_before: int
    fix_attempts: list[dict]
    tests_after: int
    failures_after: int
    success: bool


@dataclass
class TestFixReport:
    """Complete report from test fixing process."""

    project_path: str
    iterations: list[FixIteration] = field(default_factory=list)
    total_fixes_applied: int = 0
    final_pass_rate: float = 0.0
    success: bool = False
    error: str | None = None


class TestFixer:
    """
    Fixes failing tests iteratively using LLM analysis.
    """

    def __init__(self, orchestrator=None):
        self.orchestrator = orchestrator
        self.tracer = get_tracer()
        self._budget = BudgetManager(max_cost=2.0, max_time=300)  # $2, 5min for fixes

    async def fix_failing_tests(
        self, project_path: str, max_iterations: int = 3, min_pass_rate: float = 0.8
    ) -> TestFixReport:
        """
        Main entry point: fix failing tests iteratively.

        Args:
            project_path: Path to project output directory
            max_iterations: Maximum fix attempts
            min_pass_rate: Stop when this pass rate is achieved

        Returns:
            TestFixReport with complete results
        """
        with self.tracer.start_as_current_span("fix_failing_tests") as span:
            span.set_attribute("project_path", project_path)
            span.set_attribute("max_iterations", max_iterations)

            report = TestFixReport(project_path=project_path)
            project_path_obj = Path(project_path)

            if not project_path_obj.exists():
                report.error = f"Project path not found: {project_path}"
                return report

            # Run initial test suite
            for iteration in range(1, max_iterations + 1):
                if self._budget.is_exhausted():
                    print("⚠️  Fix budget exhausted, stopping")
                    break

                # Run tests
                test_result = await self._run_tests(project_path_obj)
                failures = self._parse_failures(test_result)

                current_passed = test_result.get("passed", 0)
                current_failed = test_result.get("failed", 0)
                current_total = current_passed + current_failed

                if current_total == 0:
                    print("ℹ️  No tests found to run")
                    break

                pass_rate = current_passed / current_total
                print(
                    f"\n📊 Iteration {iteration}: {current_passed}/{current_total} passed ({pass_rate:.1%})"
                )

                # Check if we've reached target
                if pass_rate >= min_pass_rate or current_failed == 0:
                    print("✅ Target pass rate achieved!")
                    report.success = True
                    report.final_pass_rate = pass_rate
                    break

                # Attempt fixes
                if not failures:
                    print("ℹ️  No specific failures to fix")
                    break

                fix_attempts = await self._attempt_fixes(project_path_obj, failures, iteration)

                # Record iteration
                iteration_result = FixIteration(
                    iteration=iteration,
                    tests_before=current_total,
                    failures_before=current_failed,
                    fix_attempts=fix_attempts,
                    tests_after=current_total,  # Will update in next iteration
                    failures_after=current_failed,  # Will update in next iteration
                    success=len(fix_attempts) > 0,
                )
                report.iterations.append(iteration_result)
                report.total_fixes_applied += len(fix_attempts)

                if not fix_attempts:
                    print("⚠️  No fixes could be applied")
                    break

            # Final test run
            final_result = await self._run_tests(project_path_obj)
            total = final_result.get("passed", 0) + final_result.get("failed", 0)
            if total > 0:
                report.final_pass_rate = final_result.get("passed", 0) / total

            return report

    async def _run_tests(self, project_path: Path) -> dict:
        """Run pytest and return results."""
        try:
            # Use pytest with JSON output if available
            result = subprocess.run(
                ["python", "-m", "pytest", "-v", "--tb=short", "--no-header"],
                cwd=project_path,
                capture_output=True,
                text=True,
                timeout=60,
            )

            # Parse output
            return self._parse_pytest_output(result.stdout + result.stderr)
        except subprocess.TimeoutExpired:
            return {"passed": 0, "failed": 0, "error": "Timeout"}
        except Exception as e:
            return {"passed": 0, "failed": 0, "error": str(e)}

    def _parse_pytest_output(self, output: str) -> dict:
        """Parse pytest output to extract results."""
        result = {"passed": 0, "failed": 0, "output": output}

        # Look for summary line: "X passed, Y failed"
        match = re.search(r"(\d+) passed.*? (\d+) failed", output)
        if match:
            result["passed"] = int(match.group(1))
            result["failed"] = int(match.group(2))
        else:
            # Try different patterns
            passed_match = re.search(r"(\d+) passed", output)
            failed_match = re.search(r"(\d+) failed", output)
            if passed_match:
                result["passed"] = int(passed_match.group(1))
            if failed_match:
                result["failed"] = int(failed_match.group(1))

        return result

    def _parse_failures(self, test_result: dict) -> list[TestFailure]:
        """Parse test output into TestFailure objects."""
        failures = []
        output = test_result.get("output", "")

        # Parse pytest failure blocks
        failure_blocks = re.split(r"={3,}\s*FAILURES\s*={3,}", output)
        if len(failure_blocks) > 1:
            failure_section = failure_blocks[-1]

            # Split individual test failures
            test_failures = re.split(r"_{3,}\s*([^_]+)\s*_{3,}", failure_section)

            for i in range(1, len(test_failures), 2):
                if i < len(test_failures):
                    test_name = test_failures[i].strip()
                    failure_content = test_failures[i + 1] if i + 1 < len(test_failures) else ""

                    # Extract error type and message
                    error_match = re.search(r"(\w+Error):\s*(.+?)(?:\n|$)", failure_content)
                    error_type = error_match.group(1) if error_match else "UnknownError"
                    error_message = error_match.group(2) if error_match else "Unknown error"

                    # Extract stack trace
                    stack_lines = []
                    for line in failure_content.split("\n"):
                        if 'File "' in line or line.strip().startswith("def "):
                            stack_lines.append(line)

                    failures.append(
                        TestFailure(
                            test_file="",
                            test_name=test_name,
                            error_type=error_type,
                            error_message=error_message,
                            stack_trace="\n".join(stack_lines),
                        )
                    )

        return failures

    async def _attempt_fixes(
        self, project_path: Path, failures: list[TestFailure], iteration: int
    ) -> list[dict]:
        """Attempt to fix failures using LLM."""
        fix_attempts = []

        # Group failures by source file
        failures_by_file = {}
        for failure in failures:
            # Try to determine source file from test name
            source_file = self._guess_source_file(project_path, failure.test_name)
            if source_file:
                if source_file not in failures_by_file:
                    failures_by_file[source_file] = []
                failures_by_file[source_file].append(failure)

        # Fix each file
        for source_file, file_failures in failures_by_file.items():
            if self._budget.is_exhausted():
                break

            print(f"  🔧 Fixing {source_file} ({len(file_failures)} failures)...")

            fix = await self._generate_fix(project_path, source_file, file_failures)
            if fix:
                success = await self._apply_fix(project_path, source_file, fix)
                fix_attempts.append(
                    {
                        "file": source_file,
                        "fix_type": fix.get("type", "unknown"),
                        "success": success,
                    }
                )

        return fix_attempts

    def _guess_source_file(self, project_path: Path, test_name: str) -> str | None:
        """Guess the source file from test name."""
        # Convert test_task_003.py -> task_003_code_generation.py
        match = re.search(r"test_(task_\d+)", test_name)
        if match:
            task_id = match.group(1)
            # Look for files in tasks/ directory
            tasks_dir = project_path / "tasks"
            if tasks_dir.exists():
                for f in tasks_dir.iterdir():
                    if task_id in f.name and f.suffix == ".py":
                        return str(f.relative_to(project_path))

        # Check main.py for test_main
        if "test_main" in test_name:
            main_file = project_path / "main.py"
            if main_file.exists():
                return "main.py"

        # Look in src/ directory
        src_dir = project_path / "src"
        if src_dir.exists():
            for py_file in src_dir.rglob("*.py"):
                # Check if test name contains part of filename
                file_stem = py_file.stem.replace("_", "")
                test_clean = test_name.replace("_", "").replace("test", "")
                if file_stem in test_clean or test_clean in file_stem:
                    return str(py_file.relative_to(project_path))

        return None

    async def _generate_fix(
        self, project_path: Path, source_file: str, failures: list[TestFailure]
    ) -> dict | None:
        """Generate a fix using LLM."""
        source_path = project_path / source_file
        if not source_path.exists():
            return None

        source_code = source_path.read_text(encoding="utf-8")

        # Build failure context
        failure_context = "\n\n".join(
            [
                f"Test: {f.test_name}\n"
                f"Error: {f.error_type}: {f.error_message}\n"
                f"Trace:\n{f.stack_trace[:500]}"
                for f in failures[:3]  # Limit to first 3 failures
            ]
        )

        prompt = f"""You are an expert Python developer. Fix the code to make the failing tests pass.

## Source File: {source_file}

```python
{source_code}
```

## Test Failures

{failure_context}

## Instructions

1. Analyze the failures and identify the root cause
2. Fix ONLY the source code, not the tests
3. Preserve the original functionality
4. Ensure the code is valid Python

## Response Format

Provide ONLY the fixed code in a code block:

```python
# Fixed code here
```

If no fix is needed, respond with: NO_FIX_NEEDED
"""

        try:
            # Use UnifiedClient for API calls
            client = UnifiedClient()
            response = await client.call_model(
                model="gpt-4o-mini",  # Use cheaper model for fixes
                prompt=prompt,
                max_tokens=4000,
                temperature=0.2,
            )

            content = response.text

            # Extract code block
            code_match = re.search(r"```python\n(.*?)\n```", content, re.DOTALL)
            if code_match:
                fixed_code = code_match.group(1).strip()
                return {"type": "code_fix", "original": source_code, "fixed": fixed_code}

            if "NO_FIX_NEEDED" in content:
                return None

            return None

        except Exception as e:
            print(f"  ⚠️  Error generating fix: {e}")
            return None

    async def _apply_fix(self, project_path: Path, source_file: str, fix: dict) -> bool:
        """Apply the fix to the source file."""
        try:
            source_path = project_path / source_file

            # Backup original
            backup_path = source_path.with_suffix(".py.backup")
            source_path.rename(backup_path)

            # Write fixed code
            source_path.write_text(fix["fixed"], encoding="utf-8")

            # Verify syntax
            try:
                compile(fix["fixed"], source_file, "exec")
                print("    ✅ Fix applied successfully")
                return True
            except SyntaxError as e:
                # Restore backup
                backup_path.rename(source_path)
                print(f"    ❌ Fix has syntax errors: {e}")
                return False

        except Exception as e:
            print(f"    ❌ Error applying fix: {e}")
            return False
