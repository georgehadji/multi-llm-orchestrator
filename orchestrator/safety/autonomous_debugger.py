"""
Autonomous Debugger — Self-Healing Test Fixer
==============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

When tests fail, this module:
1. Analyzes test failures to find root causes
2. Evaluates the failure context
3. Creates a plan to fix the problems
4. Implements the fixes
5. Re-runs tests
6. Iterates up to N times until tests pass or max iterations reached

Usage:
    from orchestrator.autonomous_debugger import AutonomousDebugger

    debugger = AutonomousDebugger(output_dir)
    report = await debugger.debug_and_fix(max_iterations=3)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# FIXED: from ..log_config import get_logger
from ..log_config import get_logger

if TYPE_CHECKING:
# FIXED: from ..output_organizer import TestResult
    from ...output_organizer import TestResult

logger = get_logger(__name__)


@dataclass
class FailureAnalysis:
    """Analysis of a single test failure."""

    test_file: str
    test_name: str
    error_type: str
    error_message: str
    stack_trace: str
    source_file: str | None = None
    line_number: int | None = None
    root_cause: str | None = None
    suggested_fix: str | None = None


@dataclass
class FixIteration:
    """Results from one fix iteration."""

    iteration: int
    failures_before: list[FailureAnalysis]
    fix_plan: dict[str, list[str]]  # file -> list of fixes to apply
    fixes_applied: list[dict]
    failures_after: list[FailureAnalysis]
    tests_passed: bool
    duration_seconds: float


@dataclass
class DebugReport:
    """Complete report from autonomous debugging process."""

    project_path: str
    iterations: list[FixIteration] = field(default_factory=list)
    total_fixes_applied: int = 0
    final_success: bool = False
    error: str | None = None
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "project_path": self.project_path,
            "final_success": self.final_success,
            "total_iterations": len(self.iterations),
            "total_fixes_applied": self.total_fixes_applied,
            "error": self.error,
            "iterations": [
                {
                    "iteration": i.iteration,
                    "failures_before_count": len(i.failures_before),
                    "fixes_applied_count": len(i.fixes_applied),
                    "failures_after_count": len(i.failures_after),
                    "tests_passed": i.tests_passed,
                    "duration_seconds": i.duration_seconds,
                }
                for i in self.iterations
            ],
            "summary": self.summary,
        }


class AutonomousDebugger:
    """
    Autonomous debugging system that fixes failing tests iteratively.

    The debugging loop:
    1. Run tests and collect failures
    2. Analyze each failure to find root cause
    3. Create a fix plan
    4. Implement fixes
    5. Re-run tests
    6. If tests still fail and iterations < max, go to step 2

    Args:
        output_dir: Directory containing the project to debug
        max_iterations: Maximum number of debug/fix iterations
        min_pass_rate: Minimum test pass rate to consider successful
    """

    def __init__(
        self,
        output_dir: Path,
        max_iterations: int = 3,
        min_pass_rate: float = 0.9,
    ):
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.min_pass_rate = min_pass_rate
        self.report = DebugReport(project_path=str(output_dir))

    async def debug_and_fix(self, test_results: list[TestResult] | None = None) -> DebugReport:
        """
        Main entry point: find and fix failing tests autonomously.

        Args:
            test_results: Optional pre-existing test results to analyze

        Returns:
            DebugReport with complete debugging history and results
        """
        logger.info("=" * 70)
        logger.info("🔬 AUTONOMOUS DEBUGGER STARTING")
        logger.info("=" * 70)
        logger.info(f"Project: {self.output_dir}")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"Min pass rate: {self.min_pass_rate:.0%}")

        try:
            # Get test results if not provided
            if test_results is None:
                test_results = await self._run_tests()

            # Check if there are any failures
            failed_tests = [r for r in test_results if not r.passed]
            if not failed_tests:
                logger.info("✅ All tests passed - no debugging needed")
                self.report.final_success = True
                return self.report

            logger.info(f"🐛 Found {len(failed_tests)} failing tests")

            # Start iterative debugging loop
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"\n{'='*70}")
                logger.info(f"🔁 ITERATION {iteration}/{self.max_iterations}")
                logger.info(f"{'='*70}")

                iteration_result = await self._run_debug_iteration(iteration, test_results)
                self.report.iterations.append(iteration_result)

                # Check if we've achieved success
                if iteration_result.tests_passed:
                    logger.info(f"\n✅ DEBUGGING SUCCESSFUL after {iteration} iteration(s)")
                    self.report.final_success = True
                    break

                # Prepare for next iteration
                test_results = await self._run_tests()

            else:
                # Max iterations reached without success
                logger.warning(
                    f"\n⚠️ Max iterations ({self.max_iterations}) reached without full success"
                )
                # Calculate final pass rate
                final_results = await self._run_tests()
                passed = sum(1 for r in final_results if r.passed)
                total = len(final_results)
                pass_rate = passed / total if total > 0 else 0

                if pass_rate >= self.min_pass_rate:
                    logger.info(f"✅ Acceptable pass rate achieved: {pass_rate:.1%}")
                    self.report.final_success = True
                else:
                    logger.error(
                        f"❌ Final pass rate too low: {pass_rate:.1%} (min: {self.min_pass_rate:.0%})"
                    )

            # Generate summary
            self._generate_summary()

        except Exception as e:
            logger.exception(f"Autonomous debugger failed: {e}")
            self.report.error = str(e)

        return self.report

    async def _run_debug_iteration(
        self, iteration: int, test_results: list[TestResult]
    ) -> FixIteration:
        """Run one complete debug iteration."""
        import time

        start_time = time.time()

        # Step 1: Analyze failures
        logger.info("\n📊 STEP 1: Analyzing test failures...")
        failures = self._analyze_failures(test_results)
        logger.info(f"   Found {len(failures)} distinct failure(s)")

        # Step 2: Create fix plan
        logger.info("\n📝 STEP 2: Creating fix plan...")
        fix_plan = self._create_fix_plan(failures)
        logger.info(f"   Fix plan covers {len(fix_plan)} file(s)")

        # Step 3: Implement fixes
        logger.info("\n🔧 STEP 3: Implementing fixes...")
        fixes_applied = await self._implement_fixes(fix_plan, failures)
        logger.info(f"   Applied {len(fixes_applied)} fix(es)")

        # Step 4: Re-run tests
        logger.info("\n🧪 STEP 4: Re-running tests...")
        new_results = await self._run_tests()

        # Step 5: Analyze results
        new_failures = [r for r in new_results if not r.passed]
        tests_passed = len(new_failures) == 0

        if tests_passed:
            logger.info("   ✅ All tests now passing!")
        else:
            logger.info(f"   ⚠️ {len(new_failures)} test(s) still failing")

        duration = time.time() - start_time

        return FixIteration(
            iteration=iteration,
            failures_before=failures,
            fix_plan=fix_plan,
            fixes_applied=fixes_applied,
            failures_after=self._analyze_failures(new_results),
            tests_passed=tests_passed,
            duration_seconds=duration,
        )

    def _analyze_failures(self, test_results: list[TestResult]) -> list[FailureAnalysis]:
        """Analyze test failures to find root causes."""
        analyses = []

        for result in test_results:
            if result.passed:
                continue

            # Parse pytest output to extract failure details
            analysis = self._parse_failure_output(
                result.test_file, result.output, result.error_message
            )
            analyses.extend(analysis)

        return analyses

    def _parse_failure_output(
        self, test_file: str, output: str, error_message: str
    ) -> list[FailureAnalysis]:
        """Parse pytest output to extract failure details."""
        analyses = []

        # Combine output and error_message for analysis
        full_output = f"{output}\n{error_message}" if error_message else output

        # Pattern 1: Parse FAILED test sections (various formats)
        # Format: FAILED test_file.py::TestClass::test_name - ErrorType: message
        failed_patterns = [
            # Standard pytest format
            r"FAILED\s+([\w/\.]+)::?(\w+)(?:::\w+)?\s*-\s*([\w\s]+?)(?::\s*|\n)(.*?)(?=FAILED|PASSED|ERROR|=+|$)",
            # Short format
            r"FAILED\s+([\w/\.]+)\s*-\s*([\w\s]+?)(?::\s*|\n)(.*?)(?=FAILED|PASSED|ERROR|=+|$)",
        ]

        for pattern in failed_patterns:
            for match in re.finditer(pattern, full_output, re.DOTALL):
                if len(match.groups()) >= 4:
                    test_path = match.group(1)
                    test_name = match.group(2)
                    error_type = match.group(3).strip()
                    error_detail = match.group(4).strip()
                else:
                    test_path = match.group(1)
                    test_name = "unknown"
                    error_type = match.group(2).strip()
                    error_detail = match.group(3).strip()

                # Determine root cause
                root_cause = self._determine_root_cause(error_type, error_detail)
                suggested_fix = self._suggest_fix(root_cause, error_detail)

                analysis = FailureAnalysis(
                    test_file=test_file,
                    test_name=test_name,
                    error_type=error_type,
                    error_message=error_detail[:500] if error_detail else error_type,
                    stack_trace=self._extract_stack_trace(full_output, test_name),
                    source_file=self._find_source_file(test_path),
                    root_cause=root_cause,
                    suggested_fix=suggested_fix,
                )
                if not any(a.test_name == test_name and a.test_file == test_file for a in analyses):
                    analyses.append(analysis)

        # Pattern 2: Collection errors (tests that couldn't even be collected)
        if "ERROR collecting" in full_output or "ImportError while importing" in full_output:
            # Extract the error details
            collection_match = re.search(
                r"ERROR collecting.*?\n([\s\S]*?)(?=(ERROR collecting|=====|$))", full_output
            )
            if collection_match:
                error_detail = collection_match.group(1).strip()
                analyses.append(
                    FailureAnalysis(
                        test_file=test_file,
                        test_name="collection",
                        error_type="CollectionError",
                        error_message=error_detail[:500],
                        stack_trace="",
                        root_cause=(
                            "missing_import"
                            if "No module named" in error_detail
                            else "syntax_error"
                        ),
                        suggested_fix=(
                            "Fix import or syntax error in test file"
                            if "No module named" in error_detail
                            else "Fix syntax error in test file"
                        ),
                    )
                )

        # Pattern 3: Import errors
        if "ImportError" in full_output or "ModuleNotFoundError" in full_output:
            import_matches = re.findall(r"No module named ['\"](\w+)['\"]", full_output)
            for module_name in set(import_matches):  # Use set to deduplicate
                analyses.append(
                    FailureAnalysis(
                        test_file=test_file,
                        test_name="import",
                        error_type="ImportError",
                        error_message=f"Missing module: {module_name}",
                        stack_trace="",
                        root_cause="missing_dependency",
                        suggested_fix=f"Add '{module_name}' to dependencies or install it",
                    )
                )

        # Pattern 4: Syntax errors
        if "SyntaxError" in full_output:
            syntax_matches = re.findall(r"SyntaxError:\s*(.+?)(?:\n|\Z)", full_output)
            for syntax_msg in set(syntax_matches):
                analyses.append(
                    FailureAnalysis(
                        test_file=test_file,
                        test_name="syntax",
                        error_type="SyntaxError",
                        error_message=syntax_msg,
                        stack_trace="",
                        root_cause="syntax_error",
                        suggested_fix="Fix syntax error in source file",
                    )
                )

        # Pattern 5: Attribute/Name errors
        if "AttributeError" in full_output:
            attr_matches = re.findall(r"AttributeError:\s*(.+?)(?:\n|\Z)", full_output)
            for attr_msg in set(attr_matches):
                analyses.append(
                    FailureAnalysis(
                        test_file=test_file,
                        test_name="attribute",
                        error_type="AttributeError",
                        error_message=attr_msg,
                        stack_trace="",
                        root_cause="attribute_error",
                        suggested_fix="Ensure the attribute/method exists and is spelled correctly",
                    )
                )

        # Pattern 6: If no specific patterns matched but we know it failed
        if not analyses and error_message:
            analyses.append(
                FailureAnalysis(
                    test_file=test_file,
                    test_name="unknown",
                    error_type="UnknownError",
                    error_message=error_message[:500],
                    stack_trace="",
                    root_cause="unknown",
                    suggested_fix="Review the error message and fix accordingly",
                )
            )

        return analyses

    def _determine_root_cause(self, error_type: str, error_detail: str) -> str:
        """Determine the root cause of a failure."""
        error_lower = (error_type + " " + error_detail).lower()

        cause_patterns = {
            "missing_import": ["importerror", "modulenotfound", "no module named"],
            "syntax_error": ["syntaxerror", "invalid syntax", "unexpected indent"],
            "attribute_error": ["attributeerror", "has no attribute", "object has no"],
            "type_error": ["typeerror", "takes", "positional argument", "missing", "required"],
            "name_error": ["nameerror", "is not defined", "undefined"],
            "assertion_error": ["assertionerror", "assert false"],
            "key_error": ["keyerror", "key not found"],
            "index_error": ["indexerror", "index out of range", "list index"],
            "value_error": ["valueerror", "invalid value"],
            "file_not_found": ["filenotfound", "no such file"],
            "empty_file": ["no code", "empty", "no content"],
            "placeholder_code": ["code generation failed", "placeholder", "not implemented"],
        }

        for cause, patterns in cause_patterns.items():
            if any(pattern in error_lower for pattern in patterns):
                return cause

        return "unknown"

    def _suggest_fix(self, root_cause: str, error_detail: str) -> str:
        """Suggest a fix based on root cause."""
        fixes = {
            "missing_import": "Add missing import statement or dependency",
            "syntax_error": "Fix syntax error in the source code",
            "attribute_error": "Ensure the attribute/method exists and is spelled correctly",
            "type_error": "Fix function signature or call arguments",
            "name_error": "Define the missing variable/function or fix the reference",
            "assertion_error": "Fix the logic to match expected behavior",
            "key_error": "Check if key exists before accessing or use .get()",
            "index_error": "Add bounds checking before indexing",
            "value_error": "Validate input values before processing",
            "file_not_found": "Create the missing file or fix the path",
            "empty_file": "Add actual implementation to the empty file",
            "placeholder_code": "Regenerate the file with actual implementation",
        }
        return fixes.get(root_cause, "Review and fix the source code")

    def _extract_stack_trace(self, output: str, test_name: str) -> str:
        """Extract the stack trace for a specific test failure."""
        # Pattern 1: Standard pytest traceback format
        # Look for "File " lines after the test name
        patterns = [
            # Standard pytest format
            rf"{re.escape(test_name)}.*?(File\s+\".+?)(?=\n\n|\Z|=====)",
            # Error details section
            r"(E\s+\nE\s+File\s+\".+?)(?=\n\n|\Z|=====)",
            # Any File references
            r"(File\s+\".+?)(?=\n\n|\Z|=====)",
        ]

        for pattern in patterns:
            match = re.search(pattern, output, re.DOTALL)
            if match:
                trace = match.group(1).strip()
                # Clean up the trace (remove E   prefix from pytest)
                trace = re.sub(r"^E\s+", "", trace, flags=re.MULTILINE)
                return trace[:1000]  # Limit length

        return ""

    def _find_source_file(self, test_path: str) -> str | None:
        """Find the source file corresponding to a test."""
        # Convert test path to source path
        # e.g., tests/test_config.py -> src/config.py
        if "test_" in test_path:
            source_name = test_path.replace("test_", "").replace("tests/", "src/")
            potential_path = self.output_dir / source_name
            if potential_path.exists():
                return str(potential_path.relative_to(self.output_dir))
        return None

    def _create_fix_plan(self, failures: list[FailureAnalysis]) -> dict[str, list[str]]:
        """Create a plan of fixes needed per file."""
        plan: dict[str, list[str]] = {}

        for failure in failures:
            source_file = failure.source_file or failure.test_file
            if source_file not in plan:
                plan[source_file] = []

            fix_description = f"{failure.root_cause}: {failure.suggested_fix}"
            if fix_description not in plan[source_file]:
                plan[source_file].append(fix_description)

        return plan

    async def _implement_fixes(
        self, fix_plan: dict[str, list[str]], failures: list[FailureAnalysis]
    ) -> list[dict]:
        """Implement the planned fixes."""
        fixes_applied = []

        for file_path, fixes in fix_plan.items():
            full_path = self.output_dir / file_path

            for fix_description in fixes:
                logger.info(f"   🔧 Fixing {file_path}: {fix_description[:60]}...")

                try:
                    # Get the root cause from the failure
                    related_failures = [f for f in failures if f.source_file == file_path]
                    if not related_failures:
                        continue

                    root_cause = related_failures[0].root_cause

                    # Apply appropriate fix based on root cause
                    fix_result = await self._apply_fix(full_path, root_cause, related_failures)

                    if fix_result["success"]:
                        fixes_applied.append(
                            {
                                "file": file_path,
                                "fix": fix_description,
                                "type": root_cause,
                            }
                        )
                        logger.info(f"      ✅ Fix applied successfully")
                    else:
                        logger.warning(f"      ⚠️ Fix failed: {fix_result.get('error')}")

                except Exception as e:
                    logger.error(f"      ❌ Error applying fix: {e}")

        return fixes_applied

    async def _apply_fix(
        self, file_path: Path, root_cause: str, failures: list[FailureAnalysis]
    ) -> dict:
        """Apply a specific fix based on root cause."""

        # Fix 1: Empty file - add placeholder implementation
        if root_cause == "empty_file":
            return await self._fix_empty_file(file_path)

        # Fix 2: Missing import - add import statement
        elif root_cause == "missing_import":
            return await self._fix_missing_import(file_path, failures)

        # Fix 3: Syntax error - try to fix common issues
        elif root_cause == "syntax_error":
            return await self._fix_syntax_error(file_path, failures)

        # Fix 4: Name error - add missing definition
        elif root_cause == "name_error":
            return await self._fix_name_error(file_path, failures)

        # Fix 5: Placeholder code - mark for regeneration
        elif root_cause == "placeholder_code":
            return await self._fix_placeholder_code(file_path)

        # Default: return failure for unhandled causes
        else:
            return {
                "success": False,
                "error": f"No automated fix available for {root_cause}",
            }

    async def _fix_empty_file(self, file_path: Path) -> dict:
        """Fix an empty file by adding placeholder implementation."""
        try:
            module_name = file_path.stem

            content = f'''"""
{module_name.replace('_', ' ').title()}
{"=" * len(module_name)}
Auto-generated module - implementation pending.
"""
from __future__ import annotations

# TODO: Implement actual functionality
# This is a placeholder that was added by the autonomous debugger
# to prevent import errors. Replace with actual implementation.

def placeholder_function():
    """Placeholder function - replace with actual implementation."""
    pass


class PlaceholderClass:
    """Placeholder class - replace with actual implementation."""
    
    def __init__(self):
        pass
'''
            file_path.write_text(content, encoding="utf-8")
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _fix_missing_import(self, file_path: Path, failures: list[FailureAnalysis]) -> dict:
        """Fix missing import by adding it to the file."""
        try:
            # Extract missing module from error
            for failure in failures:
                match = re.search(r"No module named '(\w+)'", failure.error_message)
                if match:
                    module_name = match.group(1)

                    # Read existing content
                    content = file_path.read_text(encoding="utf-8") if file_path.exists() else ""

                    # Add import at the top
                    import_line = f"import {module_name}\n"
                    if content:
                        # Find position after docstring if present
                        lines = content.split("\n")
                        insert_pos = 0
                        in_docstring = False
                        for i, line in enumerate(lines):
                            if '"""' in line or "'''" in line:
                                in_docstring = not in_docstring
                            if not in_docstring and line.strip() and not line.startswith("#"):
                                insert_pos = i
                                break
                        lines.insert(insert_pos, import_line)
                        new_content = "\n".join(lines)
                    else:
                        new_content = import_line

                    file_path.write_text(new_content, encoding="utf-8")
                    return {"success": True}

            return {"success": False, "error": "Could not extract module name from error"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _fix_syntax_error(self, file_path: Path, failures: list[FailureAnalysis]) -> dict:
        """Attempt to fix common syntax errors."""
        try:
            content = file_path.read_text(encoding="utf-8")
            original_content = content

            # Fix 1: Remove line numbers from pasted code (e.g., "    1: def foo()")
            content = re.sub(r"^\s*\d+[:\.]\s*", "", content, flags=re.MULTILINE)

            # Fix 2: Fix common indentation issues
            lines = content.split("\n")
            fixed_lines = []
            for line in lines:
                # Replace tabs with 4 spaces
                line = line.replace("\t", "    ")
                fixed_lines.append(line)
            content = "\n".join(fixed_lines)

            if content != original_content:
                file_path.write_text(content, encoding="utf-8")
                return {"success": True}
            else:
                return {
                    "success": False,
                    "error": "No automatic fix available for this syntax error",
                }

        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _fix_name_error(self, file_path: Path, failures: list[FailureAnalysis]) -> dict:
        """Fix undefined name by adding a placeholder definition."""
        try:
            for failure in failures:
                match = re.search(r"name '(\w+)' is not defined", failure.error_message)
                if match:
                    name = match.group(1)
                    content = file_path.read_text(encoding="utf-8")

                    # Add placeholder definition at end of file
                    placeholder = f"\n\n# TODO: Implement {name}\n{name} = None  # Placeholder\n"
                    content += placeholder

                    file_path.write_text(content, encoding="utf-8")
                    return {"success": True}

            return {"success": False, "error": "Could not extract name from error"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _fix_placeholder_code(self, file_path: Path) -> dict:
        """Mark placeholder code for regeneration by renaming the file."""
        try:
            # Rename the file to indicate it needs regeneration
            backup_path = file_path.with_suffix(file_path.suffix + ".needs_regeneration")
            file_path.rename(backup_path)

            # Create empty file
            file_path.write_text("# This file needs to be regenerated\n", encoding="utf-8")

            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _run_tests(self) -> list[TestResult]:
        """Run tests and return results."""
# FIXED: from ..output_organizer import OutputOrganizer
        from ....output_organizer import OutputOrganizer

        organizer = OutputOrganizer(self.output_dir, run_tests=True, fix_tests=False)
        await organizer._run_all_tests()
        return organizer.report.tests_run

    def _generate_summary(self) -> None:
        """Generate final summary of debugging process."""
        total_iterations = len(self.report.iterations)
        total_fixes = sum(len(i.fixes_applied) for i in self.report.iterations)

        if self.report.iterations:
            final_iteration = self.report.iterations[-1]
            final_failures = len(final_iteration.failures_after)
            initial_failures = len(self.report.iterations[0].failures_before)

            self.report.summary = {
                "initial_failures": initial_failures,
                "final_failures": final_failures,
                "failures_resolved": initial_failures - final_failures,
                "total_iterations": total_iterations,
                "total_fixes_applied": total_fixes,
                "success_rate": (
                    (initial_failures - final_failures) / initial_failures * 100
                    if initial_failures > 0
                    else 0
                ),
            }
        else:
            self.report.summary = {
                "initial_failures": 0,
                "final_failures": 0,
                "total_iterations": 0,
                "total_fixes_applied": 0,
                "success_rate": 100,
            }