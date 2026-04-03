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

import asyncio
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
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


# ═══════════════════════════════════════════════════════
# Module-level helpers
# ═══════════════════════════════════════════════════════


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from LLM output."""
    import re

    # Remove opening fence (```lang or ```) and closing fence (```)
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text.strip())
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _count_tests(test_code: str, framework_template: str) -> int:
    """
    Count the number of tests in generated test code.

    Different frameworks use different patterns:
    - pytest/unittest: ``def test_``
    - jest/vitest/mocha: ``test(`` or ``it(``
    - go_test: ``func Test``
    - cargo_test: ``#[test]``
    """
    if framework_template in ("pytest", "unittest"):
        return test_code.count("def test_")
    elif framework_template in ("jest", "vitest", "mocha"):
        import re

        return len(re.findall(r"\b(?:test|it)\s*\(", test_code))
    elif framework_template == "go_test":
        return test_code.count("func Test")
    elif framework_template == "cargo_test":
        return test_code.count("#[test]")
    # fallback
    return test_code.count("def test_")


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
        file_extension = Path(task.target_path).suffix if task.target_path else ""
        framework = detect_testing_framework(
            task_prompt=task.prompt,
            project_context=project_context,
            file_extension=file_extension,
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
            framework_config=framework_config,
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
            framework_config=framework_config,
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
            framework=framework,
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
                framework=framework,
            )

        # Build final result with accumulated cost
        tdd_result = TDDResult(
            implementation_code=implementation_code,
            test_spec=test_spec,
            test_result=test_result,
            iterations=iterations,
            success=test_result.passed,
            cost_usd=self._get_total_cost(),
            test_generation_cost=self._cost_tracker["test_generation"],
            implementation_cost=self._cost_tracker["implementation"],
            review_cost=self._cost_tracker["review"],
        )

        if test_result.tests_run == 0:
            # No tests actually executed — could be no sandbox, non-code task, or
            # framework detection couldn't find test functions to run.
            logger.info(f"  {task.id}: TDD complete — tests skipped (no runnable tests)")
        elif tdd_result.success:
            logger.info(f"  {task.id}: TDD complete — ALL TESTS PASSED")
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
        framework_config: dict | None = None,
    ) -> TestSpec | None:
        """
        Phase 1: Generate comprehensive test specification.

        Args:
            requirement: Task requirement/prompt
            project_context: Project context
            task_type: Type of task
            model: Model to use
            framework_config: Framework configuration dict from get_framework_config()

        Returns:
            TestSpec with generated tests
        """
        cfg = framework_config or get_framework_config(TestingFramework.PYTEST)
        fw_template = cfg.get("prompt_template", "pytest")
        code_lang = cfg.get("test_file_suffix", ".py").lstrip(".")

        # Map framework template name to human-readable description for prompt
        fw_descriptions = {
            "pytest": "pytest (Python)",
            "unittest": "unittest (Python)",
            "jest": "Jest (JavaScript)",
            "vitest": "Vitest (TypeScript/JavaScript)",
            "mocha": "Mocha/Chai (JavaScript)",
            "go_test": "Go testing package",
            "cargo_test": "Rust/Cargo tests",
        }
        fw_label = fw_descriptions.get(fw_template, fw_template)

        # Framework-specific test conventions for the prompt
        fw_conventions = {
            "pytest": (
                "Use pytest. Name test functions `def test_<name>`. "
                "Include all necessary imports."
            ),
            "unittest": (
                "Use unittest. Subclass TestCase, name methods `def test_<name>`. "
                "Include all necessary imports."
            ),
            "jest": (
                "Use Jest. Use describe()/it() or test() blocks. "
                "Include all necessary imports/requires."
            ),
            "vitest": (
                "Use Vitest. Use describe()/it() or test() blocks with TypeScript types. "
                "Import from 'vitest'."
            ),
            "mocha": (
                "Use Mocha + Chai. Use describe()/it() blocks with expect() assertions. "
                "Include all necessary imports."
            ),
            "go_test": (
                "Use Go testing package. Name functions `func Test<Name>(t *testing.T)`. "
                "Include package declaration."
            ),
            "cargo_test": (
                "Use Rust #[cfg(test)] module with #[test] functions. "
                "Include use super::*; inside test module."
            ),
        }
        fw_convention = fw_conventions.get(fw_template, fw_conventions["pytest"])

        prompt = (
            f"Write comprehensive {fw_label} tests for the following requirement.\n\n"
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
            "5. Use descriptive test names\n"
            "6. Include assertions for expected behavior\n\n"
            f"Output ONLY test code. Do NOT write implementation.\n"
            f"{fw_convention}\n\n"
            f"```{code_lang}\n"
        )

        try:
            response = await self.client.call(
                model=model,
                prompt=prompt,
                system=f"You are an expert software tester writing comprehensive {fw_label} tests. "
                "Focus on edge cases, error handling, and clear assertions. "
                "Output ONLY test code, no explanations.",
                max_tokens=3000,
                temperature=0.3,
                timeout=120,
            )

            # Track cost for this phase
            self._track_cost("test_generation", response.cost_usd)

            test_code = response.text.strip()

            # Clean up markdown fences
            test_code = _strip_code_fences(test_code)

            # Count tests — pattern depends on framework
            test_count = _count_tests(test_code, fw_template)

            # Extract edge cases mentioned
            edge_cases = self._extract_edge_cases(test_code)

            return TestSpec(
                test_file_name=f"test_main{cfg['test_file_suffix']}",
                test_code=test_code,
                test_framework=fw_template,
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
        framework_config: dict | None = None,
    ) -> str:
        """
        Phase 2: Generate implementation that passes all tests.

        Args:
            tests: Generated test code
            requirement: Original requirement
            project_context: Project context
            task_type: Type of task
            model: Model to use
            framework_config: Framework configuration dict from get_framework_config()

        Returns:
            Implementation code
        """
        cfg = framework_config or get_framework_config(TestingFramework.PYTEST)
        code_lang = cfg.get("test_file_suffix", ".py").lstrip(".")

        prompt = (
            f"Write the implementation code that passes ALL of these tests.\n\n"
            f"Tests:\n```{code_lang}\n{tests}\n```\n\n"
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
            f"```{code_lang}\n"
        )

        try:
            response = await self.client.call(
                model=model,
                prompt=prompt,
                system="You are an expert software engineer. "
                "Write clean, production-ready code that passes all tests. "
                "Output ONLY implementation code.",
                max_tokens=4000,
                temperature=0.0,
                timeout=180,
            )

            # Track cost for this phase
            self._track_cost("implementation", response.cost_usd)

            code = response.text.strip()
            code = _strip_code_fences(code)

            return code

        except Exception as e:
            logger.error(f"Implementation generation failed: {e}")
            return ""

    def _check_pytest_available(self) -> bool:
        """Check if pytest is installed and available."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def _fix_test_imports(self, test_code: str) -> str:
        """
        Fix test imports to use 'main' module instead of original module names.

        This allows tests to import from the implementation file (main.py)
        when running in the temporary test directory.

        Args:
            test_code: Original test code

        Returns:
            Modified test code with imports fixed
        """
        import re

        # Pattern 1: Replace common import patterns
        # from mymodule import something -> from main import something
        fixed_code = re.sub(
            r"^from\s+[\w\.]+\s+import",
            "from main import",
            test_code,
            flags=re.MULTILINE,
        )

        # Pattern 2: Replace import module patterns
        # import mymodule -> import main as mymodule
        # This is trickier - we need to preserve the alias
        lines = fixed_code.split("\n")
        for i, line in enumerate(lines):
            match = re.match(r"^import\s+([\w\.]+)(?:\s+as\s+(\w+))?", line)
            if match and not line.strip().startswith("#"):
                module = match.group(1)
                alias = match.group(2)
                if module != "main":
                    if alias:
                        lines[i] = f"import main as {alias}"
                    else:
                        # Try to extract the last part of dotted module name
                        last_part = module.split(".")[-1]
                        lines[i] = f"import main as {last_part}"

        fixed_code = "\n".join(lines)

        return fixed_code

    async def _run_tests_locally(
        self,
        test_code: str,
        implementation_code: str,
        framework: TestingFramework = TestingFramework.PYTEST,
    ) -> TestExecutionResult:
        """
        Run tests locally using subprocess when no sandbox is available.

        SECURITY WARNING: This runs AI-generated code on your local system.
        Only use in development/trusted environments. For production, configure
        a Docker sandbox.

        Args:
            test_code: Test suite code
            implementation_code: Implementation code to test
            framework: Testing framework to use

        Returns:
            TestExecutionResult with pass/fail info
        """
        logger.info("Running tests locally (no sandbox configured)")
        logger.warning(
            "SECURITY: Running AI-generated code without sandbox isolation. "
            "Configure Docker sandbox for production use."
        )

        # Check framework availability
        if framework == TestingFramework.PYTEST:
            if not self._check_pytest_available():
                return TestExecutionResult(
                    passed=False,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    errors=["pytest not installed. Install with: pip install pytest"],
                    output="pytest not available",
                )
        elif framework in [TestingFramework.JEST, TestingFramework.VITEST, TestingFramework.MOCHA]:
            if not self._check_npm_available():
                return TestExecutionResult(
                    passed=False,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    errors=["npm not installed. Node.js required for JavaScript tests."],
                    output="npm not available",
                )

        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Run tests based on framework
            if framework == TestingFramework.PYTEST:
                # Write implementation and test files
                main_file = temp_path / "main.py"
                test_file = temp_path / "test_main.py"

                main_file.write_text(implementation_code)

                # Fix imports in test code to use 'main' module
                fixed_test_code = self._fix_test_imports(test_code)
                test_file.write_text(fixed_test_code)

                return await self._run_pytest_locally(
                    temp_dir, implementation_code, fixed_test_code
                )
            elif framework in [TestingFramework.JEST, TestingFramework.VITEST, TestingFramework.MOCHA]:
                return await self._run_npm_tests_locally(
                    temp_dir, implementation_code, test_code, framework
                )
            elif framework == TestingFramework.GO_TEST:
                return await self._run_go_tests_locally(
                    temp_dir, implementation_code, test_code
                )
            elif framework == TestingFramework.CARGO_TEST:
                return await self._run_cargo_tests_locally(
                    temp_dir, implementation_code, test_code
                )
            else:
                return TestExecutionResult(
                    passed=True,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    errors=[f"Framework {framework.value} not supported for local execution"],
                    output=f"Unsupported framework: {framework.value}",
                )

    async def _run_pytest_locally(
        self, temp_dir: str, implementation_code: str, test_code: str
    ) -> TestExecutionResult:
        """Run pytest tests locally."""
        try:
            result = subprocess.run(
                ["python", "-m", "pytest", "test_main.py", "-v", "--tb=short"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=120,  # Increased timeout for complex tests
            )

            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Extract test counts from pytest output
            tests_run, tests_passed = self._parse_pytest_output(output)
            tests_failed = tests_run - tests_passed

            # Calculate test quality score (not real coverage)
            quality_score = self._calculate_test_quality(
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
                coverage_percent=quality_score,
                output=output,
            )

        except subprocess.TimeoutExpired:
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=["Test execution timed out after 120 seconds"],
                output="Test execution timed out",
            )
        except Exception as e:
            logger.error(f"Local test execution failed: {e}")
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=[str(e)],
            )

    def _check_npm_available(self) -> bool:
        """Check if npm is installed and available."""
        try:
            result = subprocess.run(
                ["npm", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    async def _run_npm_tests_locally(
        self,
        temp_dir: str,
        implementation_code: str,
        test_code: str,
        framework: TestingFramework,
    ) -> TestExecutionResult:
        """Run npm-based tests (Jest/Vitest/Mocha) locally."""
        import json

        try:
            # Determine file extensions based on framework
            if framework == TestingFramework.VITEST:
                impl_ext = ".ts"
                test_ext = ".test.ts"
                test_runner = "vitest"
                dev_deps = {"vitest": "^1.0.0", "typescript": "^5.0.0"}
            elif framework == TestingFramework.MOCHA:
                impl_ext = ".js"
                test_ext = ".test.js"
                test_runner = "mocha"
                dev_deps = {"mocha": "^10.0.0", "chai": "^4.3.0"}
            else:  # JEST
                impl_ext = ".js"
                test_ext = ".test.js"
                test_runner = "jest"
                dev_deps = {"jest": "^29.0.0"}

            # Create package.json with test script and dependencies
            package_json = {
                "name": "test-project",
                "version": "1.0.0",
                "type": "module",
                "scripts": {
                    "test": f"{test_runner} --colors"
                },
                "devDependencies": dev_deps,
            }

            (Path(temp_dir) / "package.json").write_text(json.dumps(package_json, indent=2))

            # Write implementation file
            impl_file = Path(temp_dir) / f"main{impl_ext}"
            impl_file.write_text(implementation_code)

            # Write test file with proper extension
            test_file = Path(temp_dir) / f"main{test_ext}"
            test_file.write_text(test_code)

            # Install dependencies first
            install_result = subprocess.run(
                ["npm", "install"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=180,
            )

            if install_result.returncode != 0:
                return TestExecutionResult(
                    passed=False,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    errors=[f"npm install failed: {install_result.stderr}"],
                    output=install_result.stdout + install_result.stderr,
                )

            # Run npm test
            result = subprocess.run(
                ["npm", "test"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )

            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Parse npm test output
            tests_run, tests_passed = self._parse_npm_test_output(output)
            tests_failed = tests_run - tests_passed

            # Calculate test quality score
            quality_score = self._calculate_test_quality(
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
                coverage_percent=quality_score,
                output=output,
            )

        except subprocess.TimeoutExpired:
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=["Test execution timed out after 120 seconds"],
                output="Test execution timed out",
            )
        except Exception as e:
            logger.error(f"Local npm test execution failed: {e}")
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=[str(e)],
            )

    def _parse_npm_test_output(self, output: str) -> tuple[int, int]:
        """
        Parse npm test output to extract test counts from Jest/Vitest/Mocha.

        Handles various output formats:
        - Jest: "Tests: 5 passed, 2 failed"
        - Vitest: "Test Files  3 passed (3)"
        - Mocha: "passing (5)" / "failing (2)"
        """
        import re

        tests_passed = 0
        tests_failed = 0

        # Jest patterns
        jest_passed = re.search(r"Tests:\s*(\d+)\s*passed", output)
        jest_failed = re.search(r"Tests:\s*(\d+)\s*failed", output)
        if jest_passed:
            tests_passed = int(jest_passed.group(1))
        if jest_failed:
            tests_failed = int(jest_failed.group(1))

        # Vitest patterns
        if tests_passed == 0 and tests_failed == 0:
            vitest_passed = re.search(r"Test Files\s+(\d+)\s+passed", output)
            vitest_failed = re.search(r"Test Files\s+(\d+)\s+failed", output)
            if vitest_passed:
                tests_passed = int(vitest_passed.group(1))
            if vitest_failed:
                tests_failed = int(vitest_failed.group(1))

        # Mocha patterns
        if tests_passed == 0 and tests_failed == 0:
            mocha_passing = re.search(r"passing\s*\((\d+)\)", output)
            mocha_failing = re.search(r"failing\s*\((\d+)\)", output)
            if mocha_passing:
                tests_passed = int(mocha_passing.group(1))
            if mocha_failing:
                tests_failed = int(mocha_failing.group(1))

        # Fallback: count PASS/FAIL lines for Jest
        if tests_passed == 0 and tests_failed == 0:
            tests_passed = output.count(" PASS ")
            tests_failed = output.count(" FAIL ")

        # Count individual test results from Jest/Vitest detail lines
        if tests_passed == 0 and tests_failed == 0:
            tests_passed = len(re.findall(r"\u2713\s+|\u2714\s+|PASS\s+|✓\s+|✔\s+", output))
            tests_failed = len(re.findall(r"\u2717\s+|\u2718\s+|FAIL\s+|✗\s+|✘\s+", output))

        tests_run = tests_passed + tests_failed
        return tests_run, tests_passed

    async def _run_go_tests_locally(
        self, temp_dir: str, implementation_code: str, test_code: str
    ) -> TestExecutionResult:
        """Run Go tests locally."""
        try:
            # Check if Go is available
            result = subprocess.run(
                ["go", "version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return TestExecutionResult(
                    passed=False,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    errors=["Go not installed or not available"],
                    output="Go not available",
                )

            # Write go.mod
            (Path(temp_dir) / "go.mod").write_text("module test-project\n\ngo 1.21\n")

            # Write implementation and test files
            (Path(temp_dir) / "main.go").write_text(implementation_code)
            (Path(temp_dir) / "main_test.go").write_text(test_code)

            # Run tests
            result = subprocess.run(
                ["go", "test", "-v"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )

            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Parse Go test output
            tests_passed = output.count("PASS:")
            tests_failed = output.count("FAIL:")
            tests_run = tests_passed + tests_failed

            # Fallback: parse "--- PASS:" and "--- FAIL:" patterns
            if tests_run == 0:
                tests_passed = len(re.findall(r"---\s+PASS:", output))
                tests_failed = len(re.findall(r"---\s+FAIL:", output))
                tests_run = tests_passed + tests_failed

            quality_score = self._calculate_test_quality(
                implementation_code, test_code, tests_passed, tests_run
            )

            errors = []
            if not passed:
                errors = self._extract_test_errors(output)

            return TestExecutionResult(
                passed=passed,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                errors=errors,
                coverage_percent=quality_score,
                output=output,
            )

        except subprocess.TimeoutExpired:
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=["Go test execution timed out"],
                output="Test execution timed out",
            )
        except Exception as e:
            logger.error(f"Go test execution failed: {e}")
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=[str(e)],
            )

    async def _run_cargo_tests_locally(
        self, temp_dir: str, implementation_code: str, test_code: str
    ) -> TestExecutionResult:
        """Run Rust/Cargo tests locally."""
        try:
            # Check if Cargo is available
            result = subprocess.run(
                ["cargo", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return TestExecutionResult(
                    passed=False,
                    tests_run=0,
                    tests_passed=0,
                    tests_failed=0,
                    errors=["Cargo not installed or not available"],
                    output="Cargo not available",
                )

            # Create Cargo.toml
            cargo_toml = """[package]
name = "test-project"
version = "0.1.0"
edition = "2021"
"""
            (Path(temp_dir) / "Cargo.toml").write_text(cargo_toml)

            # Create src directory and write files
            src_dir = Path(temp_dir) / "src"
            src_dir.mkdir(exist_ok=True)

            (src_dir / "lib.rs").write_text(implementation_code + "\n\n" + test_code)

            # Run tests
            result = subprocess.run(
                ["cargo", "test"],
                cwd=temp_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )

            output = result.stdout + result.stderr
            passed = result.returncode == 0

            # Parse Cargo test output
            # Pattern: "test result: ok. 5 passed; 2 failed;"
            match = re.search(r"test result:\s+\w+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed;", output)
            if match:
                tests_passed = int(match.group(1))
                tests_failed = int(match.group(2))
            else:
                tests_passed = output.count("test ... ok") if passed else 0
                tests_failed = output.count("test ... FAILED") if not passed else 0

            tests_run = tests_passed + tests_failed

            quality_score = self._calculate_test_quality(
                implementation_code, test_code, tests_passed, tests_run
            )

            errors = []
            if not passed:
                errors = self._extract_test_errors(output)

            return TestExecutionResult(
                passed=passed,
                tests_run=tests_run,
                tests_passed=tests_passed,
                tests_failed=tests_failed,
                errors=errors,
                coverage_percent=quality_score,
                output=output,
            )

        except subprocess.TimeoutExpired:
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=["Cargo test execution timed out"],
                output="Test execution timed out",
            )
        except Exception as e:
            logger.error(f"Cargo test execution failed: {e}")
            return TestExecutionResult(
                passed=False,
                tests_run=0,
                tests_passed=0,
                tests_failed=0,
                errors=[str(e)],
            )

    async def _run_tests_and_collect_results(
        self,
        test_code: str,
        implementation_code: str,
        task_type: TaskType,
        framework: TestingFramework = TestingFramework.PYTEST,
    ) -> TestExecutionResult:
        """
        Phase 3: Run tests against implementation.

        Args:
            test_code: Test suite
            implementation_code: Implementation to test
            task_type: Type of task
            framework: Testing framework to use

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

        # Try sandbox first if available, otherwise use local test runner
        if self.sandbox is not None:
            # Use sandbox for test execution
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

                # Calculate test quality score (simplified - would need coverage.py for real metrics)
                quality_score = self._calculate_test_quality(
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
                    coverage_percent=quality_score,
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
        else:
            # No sandbox available - run tests locally
            return await self._run_tests_locally(test_code, implementation_code, framework)

    async def _repair_to_pass_tests(
        self,
        tests: str,
        implementation: str,
        errors: list[str],
        requirement: str,
        model: Model,
        framework: TestingFramework = TestingFramework.PYTEST,
    ) -> tuple[str, TestExecutionResult, int]:
        """
        Phase 4: Self-heal implementation to pass tests.

        Args:
            tests: Test suite
            implementation: Current implementation
            errors: Test errors to fix
            requirement: Original requirement
            model: Model to use
            framework: Testing framework to use

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
                    framework=framework,
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
            framework=framework,
        )
        return current_code, final_result, iterations

    async def _fallback_standard_generation(
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

    def _calculate_test_quality(
        self,
        implementation: str,
        tests: str,
        passed: int,
        total: int,
    ) -> float:
        """
        Calculate test quality score (NOT real code coverage).

        This is a heuristic score based on:
        - Test pass rate
        - Test-to-code line ratio

        Real code coverage requires coverage.py or similar tools.
        This score is meant to give a rough indication of test quality.

        Args:
            implementation: Implementation code
            tests: Test code
            passed: Number of tests passed
            total: Total number of tests

        Returns:
            Quality score from 0.0 to 100.0
        """
        if total == 0:
            return 0.0

        # Base score on test pass rate
        pass_rate = passed / total

        # Adjust based on test-to-code ratio
        # Good ratio is ~1:2 to 1:1 (test:code)
        test_lines = len(tests.split("\n"))
        code_lines = len(implementation.split("\n"))
        ratio = test_lines / max(1, code_lines)
        ratio_factor = min(1.0, ratio * 2)  # Cap at 1.0

        # Quality score = pass rate * ratio factor * 100
        quality = pass_rate * ratio_factor * 100
        return min(100.0, quality)

    def _parse_pytest_output(self, output: str) -> tuple[int, int]:
        """
        Parse pytest output to extract test counts.

        Handles various pytest output formats:
        - "1 passed, 2 failed"
        - "1 passed, 1 warning"
        - "1 failed, 1 error"
        - "1 passed, 1 skipped"
        - "no tests ran"
        - Individual PASSED/FAILED lines

        Args:
            output: pytest output text

        Returns:
            Tuple of (tests_run, tests_passed)
        """
        import re

        # Try to find the summary line first
        # Pattern: "X passed, Y failed, Z skipped, W errors"
        summary_patterns = [
            r"(\d+)\s+passed,\s*(\d+)\s+failed",
            r"(\d+)\s+passed\s+in",  # "1 passed in 0.1s"
            r"(\d+)\s+failed\s+in",  # "1 failed in 0.1s"
            r"(\d+)\s+passed,\s*(\d+)\s+warning",
            r"(\d+)\s+passed,\s*(\d+)\s+skipped",
            r"(\d+)\s+error",
        ]

        tests_passed = 0
        tests_failed = 0

        for pattern in summary_patterns:
            match = re.search(pattern, output)
            if match:
                tests_passed = int(match.group(1))
                if len(match.groups()) > 1:
                    tests_failed = int(match.group(2))
                break

        tests_run = tests_passed + tests_failed

        # If no summary found, try counting individual test results
        if tests_run == 0:
            # Count PASSED and FAILED lines (more reliable for some output formats)
            passed_lines = re.findall(r"^.*\sPASSED\s*$", output, re.MULTILINE)
            failed_lines = re.findall(r"^.*\sFAILED\s*$", output, re.MULTILINE)

            tests_passed = len(passed_lines)
            tests_failed = len(failed_lines)
            tests_run = tests_passed + tests_failed

        # Handle "no tests ran" or "collected 0 items"
        if tests_run == 0:
            if "no tests ran" in output.lower() or "collected 0 items" in output:
                return 0, 0

        return tests_run, tests_passed

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
