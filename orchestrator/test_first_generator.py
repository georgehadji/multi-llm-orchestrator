"""
Test-First Generation (TDD Inversion)
======================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Date: 2026-03-30

Paradigm Shift: Generate tests FIRST → Generate code to pass tests → Verify

Instead of: Generate code → Verify → Fix (score-based, heuristic)
We do: Generate tests → Generate code → Run tests → Fix to pass (deterministic)

Usage:
    from orchestrator.test_first_generator import TestFirstGenerator
    from orchestrator.cost_optimization import get_tdd_profile

    # Get TDD profile (budget, balanced, or premium)
    tdd_config = get_tdd_profile("balanced")

    tdd = TestFirstGenerator(client, sandbox, model_config=tdd_config)
    result = await tdd.generate_with_tests(task, project_context)

    # Result includes:
    # - Implementation code
    # - Test suite
    # - Test results (pass/fail)
    # - Coverage metrics
    # - Cost breakdown
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

from .log_config import get_logger
from .models import Model, Task, TaskType

if TYPE_CHECKING:
    from .tdd_config import TDDModelConfig

logger = get_logger(__name__)


class TestingFramework(Enum):
    """Supported testing frameworks."""

    PYTEST = "pytest"
    UNITTEST = "unittest"
    JEST = "jest"
    VITEST = "vitest"
    MOCHA = "mocha"
    GO_TEST = "go_test"
    CARGO_TEST = "cargo_test"
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════
# Testing Framework Detection
# ═══════════════════════════════════════════════════════


def detect_testing_framework(
    task_prompt: str,
    project_context: str = "",
    file_extension: str = "",
) -> TestingFramework:
    """
    Detect testing framework from task/project context.

    Args:
        task_prompt: Task description
        project_context: Optional project context
        file_extension: File extension (.py, .js, .ts, etc.)

    Returns:
        TestingFramework enum value

    Examples:
        >>> detect_testing_framework("Build a Python API with pytest tests", "", ".py")
        <TestingFramework.PYTEST: 'pytest'>
        >>> detect_testing_framework("Create React components", "", ".js")
        <TestingFramework.JEST: 'jest'>
    """
    # Check task prompt and project context for framework keywords FIRST
    # (keywords are more specific than file extensions)
    prompt_lower = (task_prompt + " " + project_context).lower()

    # Framework-specific keywords (check these first for accuracy)
    framework_keywords = {
        TestingFramework.PYTEST: ["pytest", "py.test"],
        TestingFramework.UNITTEST: ["unittest", "python unittest"],
        TestingFramework.JEST: ["jest", "javascript test", "react test"],
        TestingFramework.VITEST: ["vitest", "typescript test", "vue test", "svelte test"],
        TestingFramework.MOCHA: ["mocha", "chai"],
        TestingFramework.GO_TEST: ["go test", "golang test"],
        TestingFramework.CARGO_TEST: ["cargo test", "rust test"],
    }

    # Check for explicit framework keywords first
    for framework, keywords in framework_keywords.items():
        if any(kw in prompt_lower for kw in keywords):
            return framework

    # If no explicit keywords, check file extension
    ext_framework_map = {
        ".py": TestingFramework.PYTEST,  # Default to pytest for Python
        ".js": TestingFramework.JEST,
        ".jsx": TestingFramework.JEST,
        ".ts": TestingFramework.VITEST,
        ".tsx": TestingFramework.VITEST,
        ".go": TestingFramework.GO_TEST,
        ".rs": TestingFramework.CARGO_TEST,
    }

    if file_extension.lower() in ext_framework_map:
        return ext_framework_map[file_extension.lower()]

    # Default based on language detection
    lang_framework_map = {
        "python": TestingFramework.PYTEST,
        "javascript": TestingFramework.JEST,
        "typescript": TestingFramework.VITEST,
        "react": TestingFramework.JEST,
        "vue": TestingFramework.VITEST,
        "svelte": TestingFramework.VITEST,
        "go": TestingFramework.GO_TEST,
        "rust": TestingFramework.CARGO_TEST,
        "node": TestingFramework.JEST,
    }

    for lang, framework in lang_framework_map.items():
        if lang in prompt_lower:
            return framework

    # Default to pytest (most common)
    return TestingFramework.PYTEST


def get_framework_config(framework: TestingFramework) -> dict:
    """
    Get testing framework configuration.

    Args:
        framework: Testing framework enum

    Returns:
        Dictionary with framework configuration
    """
    configs = {
        TestingFramework.PYTEST: {
            "test_file_prefix": "test_",
            "test_file_suffix": ".py",
            "run_command": "pytest -v",
            "prompt_template": "pytest",
        },
        TestingFramework.UNITTEST: {
            "test_file_prefix": "test_",
            "test_file_suffix": ".py",
            "run_command": "python -m unittest -v",
            "prompt_template": "unittest",
        },
        TestingFramework.JEST: {
            "test_file_prefix": "",
            "test_file_suffix": ".test.js",
            "run_command": "npm test",
            "prompt_template": "jest",
        },
        TestingFramework.VITEST: {
            "test_file_prefix": "",
            "test_file_suffix": ".test.ts",
            "run_command": "npm test",
            "prompt_template": "vitest",
        },
        TestingFramework.MOCHA: {
            "test_file_prefix": "",
            "test_file_suffix": ".test.js",
            "run_command": "npm test",
            "prompt_template": "mocha",
        },
        TestingFramework.GO_TEST: {
            "test_file_prefix": "",
            "test_file_suffix": "_test.go",
            "run_command": "go test -v",
            "prompt_template": "go_test",
        },
        TestingFramework.CARGO_TEST: {
            "test_file_prefix": "",
            "test_file_suffix": ".rs",
            "run_command": "cargo test",
            "prompt_template": "cargo_test",
        },
        TestingFramework.UNKNOWN: {
            "test_file_prefix": "test_",
            "test_file_suffix": ".py",
            "run_command": "pytest -v",
            "prompt_template": "pytest",
        },
    }

    return configs.get(framework, configs[TestingFramework.UNKNOWN])


@dataclass
class TestSpec:
    """Specification for generated tests."""

    test_file_name: str
    test_code: str
    test_framework: str = "pytest"
    test_count: int = 0
    edge_cases_covered: list[str] = field(default_factory=list)


@dataclass
class TestExecutionResult:
    """Result of running tests against implementation."""

    passed: bool
    tests_run: int
    tests_passed: int
    tests_failed: int
    errors: list[str] = field(default_factory=list)
    coverage_percent: float = 0.0
    output: str = ""


@dataclass
class TDDResult:
    """Complete result of TDD generation."""

    implementation_code: str
    test_spec: TestSpec
    test_result: TestExecutionResult
    iterations: int = 1
    success: bool = True
    # Cost tracking (NEW v3.0)
    cost_usd: float = 0.0
    test_generation_cost: float = 0.0
    implementation_cost: float = 0.0
    review_cost: float = 0.0
    # Model info (NEW v3.0)
    test_model_used: str = ""
    implementation_model_used: str = ""
    review_model_used: str = ""


class TestFirstGenerator:
    """
    Test-Driven Development generator.

    Flow:
    1. Generate comprehensive test suite from requirements
    2. Generate implementation code to pass tests
    3. Run tests against implementation
    4. Self-heal if tests fail (repair to pass tests)

    This provides machine-verifiable success criteria:
    - Instead of "score: 0.85" → "17/17 tests passed"
    - Tests serve as executable specifications

    Updated v3.0:
    - Optimal model selection per TDD phase
    - Cost tracking per phase
    - Quality tier support (budget, balanced, premium)
    """

    def __init__(
        self,
        client,  # UnifiedClient
        sandbox,  # DockerSandbox
        max_test_iterations: int = 3,
        model_config: TDDModelConfig | None = None,
        quality_tier: str = "balanced",
        language: str | None = None,
    ):
        """
        Initialize TDD generator.

        Args:
            client: LLM client for generation
            sandbox: Sandbox for running tests safely
            max_test_iterations: Maximum repair iterations
            model_config: TDD model configuration (optional, uses default if None)
            quality_tier: Quality tier (budget, balanced, premium)
            language: Language-specific optimizations (python, javascript, etc.)
        """
        from .cost_optimization import get_tdd_profile

        self.client = client
        self.sandbox = sandbox
        self.max_test_iterations = max_test_iterations

        # Load TDD configuration
        self.model_config = model_config or get_tdd_profile(quality_tier, language)
        self.quality_tier = quality_tier
        self.language = language

        # Cost tracking
        self._cost_tracker: dict[str, float] = {
            "test_generation": 0.0,
            "implementation": 0.0,
            "review": 0.0,
        }

    def _get_model_for_phase(self, phase: str) -> Model:
        """
        Get optimal model for TDD phase.

        Args:
            phase: Phase name (test_generation, implementation, test_review, refactoring)

        Returns:
            Model instance for the phase
        """
        model_id = self.model_config.get_model(phase, self.quality_tier, self.language)
        return Model(model_id)

    def _track_cost(self, phase: str, cost: float) -> None:
        """
        Track cost for a TDD phase.

        Args:
            phase: Phase name
            cost: Cost in USD
        """
        if phase in self._cost_tracker:
            self._cost_tracker[phase] += cost

    def _get_total_cost(self) -> float:
        """Get total cost across all phases."""
        return sum(self._cost_tracker.values())

    async def generate_with_tests(
        self,
        task: Task,
        project_context: str = "",
        model: Model | None = None,
    ) -> TDDResult:
        """
        Generate code using TDD approach.

        Args:
            task: Task to execute
            project_context: Optional project context
            model: Model to use (default: task's model)

        Returns:
            TDDResult with implementation, tests, and test results
        """
        # Use caller-supplied model, or fall back to the TDD config's implementation model.
        # Note: Task has no model_used attribute — routing is handled by the TDD config.
        target_model = model or self._get_model_for_phase("implementation")

        logger.info(f"Starting TDD generation for task {task.id}")

        # ═══════════════════════════════════════════════════════
        # Detect Testing Framework (NEW v3.0)
        # ═══════════════════════════════════════════════════════
        framework = detect_testing_framework(
            task_prompt=task.prompt,
            project_context=project_context,
            file_extension="",  # Could extract from task if available
        )
        framework_config = get_framework_config(framework)

        logger.info(
            f"  Detected testing framework: {framework.value} "
            f"({framework_config['run_command']})"
        )

        # ═══════════════════════════════════════════════════════
        # Phase 1: Generate Test Specification
        # ═══════════════════════════════════════════════════════
        logger.info("  Phase 1: Generating test suite...")
        test_spec = await self._generate_test_spec(
            requirement=task.prompt,
            project_context=project_context,
            task_type=task.type,
            model=target_model,
        )

        if not test_spec or not test_spec.test_code:
            logger.warning(
                f"  {task.id}: Test generation failed, falling back to standard generation"
            )
            # Fallback to standard generation without tests
            return await self._fallback_standard_generation(task, project_context, target_model)

        logger.info(f"  {task.id}: Generated {test_spec.test_count} tests")

        # ═══════════════════════════════════════════════════════
        # Phase 2: Generate Implementation to Pass Tests
        # ═══════════════════════════════════════════════════════
        logger.info("  Phase 2: Generating implementation...")
        implementation_code = await self._generate_code_to_pass_tests(
            tests=test_spec.test_code,
            requirement=task.prompt,
            project_context=project_context,
            task_type=task.type,
            model=target_model,
        )

        if not implementation_code:
            logger.error(f"  {task.id}: Implementation generation failed")
            return TDDResult(
                implementation_code="",
                test_spec=test_spec,
                test_result=TestExecutionResult(
                    passed=False,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    errors=["Implementation generation failed"],
                ),
                success=False,
            )

        # ═══════════════════════════════════════════════════════
        # Phase 3: Run Tests Against Implementation
        # ═══════════════════════════════════════════════════════
        logger.info("  Phase 3: Running tests...")
        test_result = await self._run_tests_and_collect_results(
            test_code=test_spec.test_code,
            implementation_code=implementation_code,
            task_type=task.type,
        )

        logger.info(
            f"  {task.id}: Tests complete - "
            f"{test_result.tests_passed}/{test_result.tests_run} passed "
            f"({test_result.coverage_percent:.0f}% coverage)"
        )

        # ═══════════════════════════════════════════════════════
        # Phase 4: Self-Heal if Tests Fail
        # ═══════════════════════════════════════════════════════
        iterations = 1
        if not test_result.passed:
            logger.info(f"  {task.id}: Tests failed, attempting self-heal...")
            implementation_code, test_result, iterations = await self._repair_to_pass_tests(
                tests=test_spec.test_code,
                implementation=implementation_code,
                errors=test_result.errors,
                requirement=task.prompt,
                model=target_model,
            )

        # Build final result
        tdd_result = TDDResult(
            implementation_code=implementation_code,
            test_spec=test_spec,
            test_result=test_result,
            iterations=iterations,
            success=test_result.passed,
        )

        if tdd_result.success:
            logger.info(f"  {task.id}: TDD complete - ALL TESTS PASSED")
        else:
            logger.warning(
                f"  {task.id}: TDD complete - "
                f"{test_result.tests_failed}/{test_result.tests_run} tests still failing"
            )

        return tdd_result

    async def _generate_test_spec(
        self,
        requirement: str,
        project_context: str,
        task_type: TaskType,
        model: Model,
    ) -> TestSpec | None:
        """
        Phase 1: Generate comprehensive test specification.

        Args:
            requirement: Task requirement/prompt
            project_context: Project context
            task_type: Type of task
            model: Model to use

        Returns:
            TestSpec with generated tests
        """
        # Build test generation prompt
        # Key insight: Don't say "do TDD" — just give tests as natural language spec
        prompt = (
            f"Write comprehensive pytest tests for the following requirement.\n\n"
            f"Requirement: {requirement}\n\n"
        )

        if project_context:
            prompt += f"Project Context:\n{project_context}\n\n"

        prompt += (
            "Requirements for tests:\n"
            "1. Include edge cases (empty inputs, max values, error conditions)\n"
            "2. Test error handling and exception cases\n"
            "3. Include type checking where applicable\n"
            "4. Cover happy path and at least 3 edge cases\n"
            "5. Use descriptive test names (test_should_... pattern)\n"
            "6. Include assertions for expected behavior\n\n"
            "Output ONLY test code. Do NOT write implementation.\n"
            "Use pytest framework. Include all necessary imports.\n\n"
            "```python\n"
        )

        try:
            response = await self.client.call(
                model=model,
                prompt=prompt,
                system="You are an expert software tester writing comprehensive pytest tests. "
                "Focus on edge cases, error handling, and clear assertions. "
                "Output ONLY test code, no explanations.",
                max_tokens=3000,
                temperature=0.3,  # Slightly higher for test creativity
                timeout=120,
            )

            test_code = response.text.strip()

            # Clean up markdown fences if present
            test_code = test_code.replace("```python\n", "").replace("```", "")

            # Count tests
            test_count = test_code.count("def test_")

            # Extract edge cases mentioned
            edge_cases = self._extract_edge_cases(test_code)

            return TestSpec(
                test_file_name="test_main.py",
                test_code=test_code,
                test_framework="pytest",
                test_count=test_count,
                edge_cases_covered=edge_cases,
            )

        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            return None

    async def _generate_code_to_pass_tests(
        self,
        tests: str,
        requirement: str,
        project_context: str,
        task_type: TaskType,
        model: Model,
    ) -> str:
        """
        Phase 2: Generate implementation that passes all tests.

        Args:
            tests: Generated test code
            requirement: Original requirement
            project_context: Project context
            task_type: Type of task
            model: Model to use

        Returns:
            Implementation code
        """
        prompt = (
            f"Write the implementation code that passes ALL of these tests.\n\n"
            f"Tests:\n```python\n{tests}\n```\n\n"
            f"Original requirement: {requirement}\n\n"
        )

        if project_context:
            prompt += f"Project Context:\n{project_context}\n\n"

        prompt += (
            f"Requirements:\n"
            f"1. Every test must pass\n"
            f"2. Follow best practices for {task_type.value}\n"
            f"3. Include comprehensive comments\n"
            f"4. Handle all edge cases covered in tests\n"
            f"5. Production-ready quality\n\n"
            f"Output ONLY implementation code. No explanations.\n\n"
            f"```python\n"
        )

        try:
            response = await self.client.call(
                model=model,
                prompt=prompt,
                system="You are an expert software engineer. "
                "Write clean, production-ready code that passes all tests. "
                "Output ONLY implementation code.",
                max_tokens=4000,
                temperature=0.0,  # Deterministic for code
                timeout=180,
            )

            code = response.text.strip()
            code = code.replace("```python\n", "").replace("```", "")

            return code

        except Exception as e:
            logger.error(f"Implementation generation failed: {e}")
            return ""

    async def _run_tests_and_collect_results(
        self,
        test_code: str,
        implementation_code: str,
        task_type: TaskType,
    ) -> TestExecutionResult:
        """
        Phase 3: Run tests against implementation.

        Args:
            test_code: Test suite
            implementation_code: Implementation to test
            task_type: Type of task

        Returns:
            TestExecutionResult with pass/fail info
        """
        if task_type != TaskType.CODE_GEN:
            # Non-code tasks don't have executable tests
            return TestExecutionResult(
                passed=True,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
            )

        # Prepare files for test execution
        code_files = {
            "main.py": implementation_code,
            "test_main.py": test_code,
        }

        # Run pytest
        command = "python -m pytest test_main.py -v --tb=short"

        try:
            result = await self.sandbox.execute(
                code_files=code_files,
                command=command,
                timeout=60,
            )

            # Parse pytest output
            passed = result.return_code == 0
            output = result.output + result.error

            # Extract test counts from pytest output
            tests_run, tests_passed = self._parse_pytest_output(output)
            tests_failed = tests_run - tests_passed

            # Estimate coverage (simplified - would need coverage.py for real metrics)
            coverage_percent = self._estimate_coverage(
                implementation_code, test_code, tests_passed, tests_run
            )

            # Extract errors
            errors = []
            if not passed:
                errors = self._extract_test_errors(output)

            return TestExecutionResult(
                passed=passed,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                errors=errors,
                coverage_percent=coverage_percent,
                output=output,
            )

        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=[str(e)],
            )

    async def _repair_to_pass_tests(
        self,
        tests: str,
        implementation: str,
        errors: list[str],
        requirement: str,
        model: Model,
    ) -> tuple[str, TestExecutionResult, int]:
        """
        Phase 4: Self-heal implementation to pass tests.

        Args:
            tests: Test suite
            implementation: Current implementation
            errors: Test errors to fix
            requirement: Original requirement
            model: Model to use

        Returns:
            Tuple of (fixed code, test result, iterations)
        """
        current_code = implementation
        iterations = 1

        for iteration in range(self.max_test_iterations):
            logger.info(f"  Repair iteration {iteration + 1}/{self.max_test_iterations}")

            # Build repair prompt with error context
            prompt = (
                f"Fix this code so all tests pass.\n\n"
                f"Tests:\n```python\n{tests}\n```\n\n"
                f"Current implementation:\n```python\n{current_code}\n```\n\n"
                f"Test errors:\n"
            )

            for i, error in enumerate(errors, 1):
                prompt += f"{i}. {error}\n"

            prompt += (
                "\nRequirements:\n"
                "1. Fix ALL test errors\n"
                "2. Preserve working functionality\n"
                "3. Output ONLY the fixed implementation code\n\n"
                "```python\n"
            )

            try:
                response = await self.client.call(
                    model=model,
                    prompt=prompt,
                    system="You are debugging code to make all tests pass. "
                    "Fix the specific errors mentioned. Output ONLY fixed code.",
                    max_tokens=4000,
                    temperature=0.1,  # Low temp for focused fixes
                    timeout=180,
                )

                current_code = response.text.strip()
                current_code = current_code.replace("```python\n", "").replace("```", "")

                # Re-run tests
                test_result = await self._run_tests_and_collect_results(
                    test_code=tests,
                    implementation_code=current_code,
                    task_type=TaskType.CODE_GEN,
                )

                if test_result.passed:
                    logger.info(f"  All tests passed after {iteration + 1} iterations")
                    return current_code, test_result, iteration + 1

                # Update errors for next iteration
                errors = test_result.errors
                iterations = iteration + 1

            except Exception as e:
                logger.error(f"Repair iteration failed: {e}")
                errors = [str(e)]

        # Max iterations reached, return best effort
        logger.warning("Max repair iterations reached, returning best effort")
        final_result = await self._run_tests_and_collect_results(
            test_code=tests,
            implementation_code=current_code,
            task_type=TaskType.CODE_GEN,
        )
        return current_code, final_result, iterations

    def _fallback_standard_generation(
        self,
        task: Task,
        project_context: str,
        model: Model,
    ) -> TDDResult:
        """
        Fallback to standard generation if TDD fails.

        Returns:
            TDDResult with empty tests
        """
        # This would call the standard generation path
        # For now, return a failure result
        return TDDResult(
            implementation_code="",
            test_spec=TestSpec(
                test_file_name="",
                test_code="",
                test_count=0,
            ),
            test_result=TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=["TDD generation failed, standard fallback needed"],
            ),
            success=False,
        )

    def _extract_edge_cases(self, test_code: str) -> list[str]:
        """Extract edge cases covered by tests."""
        edge_cases = []

        # Look for common edge case patterns
        patterns = [
            ("empty", "Empty input handling"),
            ("none", "None/null handling"),
            ("max", "Maximum value handling"),
            ("min", "Minimum value handling"),
            ("invalid", "Invalid input handling"),
            ("error", "Error condition handling"),
            ("exception", "Exception handling"),
            ("boundary", "Boundary condition"),
        ]

        for pattern, description in patterns:
            if pattern.lower() in test_code.lower():
                edge_cases.append(description)

        return edge_cases

    def _parse_pytest_output(self, output: str) -> tuple[int, int]:
        """Parse pytest output to extract test counts."""
        import re

        # Look for pattern: "X passed, Y failed" or "X passed"
        passed_match = re.search(r"(\d+) passed", output)
        failed_match = re.search(r"(\d+) failed", output)

        tests_passed = int(passed_match.group(1)) if passed_match else 0
        tests_failed = int(failed_match.group(1)) if failed_match else 0

        tests_run = tests_passed + tests_failed

        # If no summary found, try counting test functions
        if tests_run == 0:
            tests_passed = output.count("PASSED")
            tests_failed = output.count("FAILED")
            tests_run = tests_passed + tests_failed

        return tests_run, tests_passed

    def _estimate_coverage(
        self,
        implementation: str,
        tests: str,
        passed: int,
        total: int,
    ) -> float:
        """
        Estimate test coverage (simplified).

        Real coverage would require coverage.py integration.
        This is a rough estimate based on test pass rate.
        """
        if total == 0:
            return 0.0

        # Base coverage on test pass rate
        pass_rate = passed / total

        # Adjust based on test-to-code ratio
        test_lines = len(tests.split("\n"))
        code_lines = len(implementation.split("\n"))

        # Good ratio is ~1:2 to 1:1 (test:code)
        ratio = test_lines / max(1, code_lines)
        ratio_factor = min(1.0, ratio * 2)  # Cap at 1.0

        coverage = pass_rate * ratio_factor * 100
        return min(100.0, coverage)

    def _extract_test_errors(self, output: str) -> list[str]:
        """Extract error messages from pytest output."""
        errors = []

        # Look for FAILED tests and their error messages
        import re

        failed_tests = re.findall(
            r"FAILED ([^\s]+) - (.+?)(?=\nFAILED|\nPASSED|\n=|$)", output, re.DOTALL
        )

        for test_name, error_msg in failed_tests:
            errors.append(f"{test_name}: {error_msg.strip()}")

        # If no structured errors, take first few error lines
        if not errors:
            error_lines = [
                line for line in output.split("\n") if "Error" in line or "FAILED" in line
            ]
            errors = error_lines[:5]  # Limit to first 5 errors

        return errors


__all__ = [
    "TestFirstGenerator",
    "TestSpec",
    "TestExecutionResult",
    "TDDResult",
]
