"""
Tests for Paradigm Shift Enhancements
======================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_paradigm_shifts.py -v
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.test_first_generator import (
    TestFirstGenerator,
    TestSpec,
    TestExecutionResult,
    TDDResult,
)
from orchestrator.diff_generator import (
    DiffGenerator,
    DiffResult,
    apply_unified_diff,
)
from orchestrator.models import Task, TaskType, Model


# ─────────────────────────────────────────────
# Test TestFirstGenerator
# ─────────────────────────────────────────────

class TestTestFirstGenerator:
    """Test TDD-first generation."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.call = AsyncMock()
        return client

    @pytest.fixture
    def mock_sandbox(self):
        """Create mock Docker sandbox."""
        sandbox = AsyncMock()
        sandbox.execute = AsyncMock()
        return sandbox

    @pytest.fixture
    def tdd_generator(self, mock_client, mock_sandbox):
        """Create TestFirstGenerator instance."""
        return TestFirstGenerator(
            client=mock_client,
            sandbox=mock_sandbox,
            max_test_iterations=3,
        )

    @pytest.mark.asyncio
    async def test_generate_test_spec(self, tdd_generator, mock_client):
        """Test Phase 1: Test specification generation."""
        # Mock LLM response
        mock_client.call.return_value = MagicMock(
            text="""
import pytest

def test_should_return_only_valid_emails():
    emails = ['test@example.com', 'invalid', 'another@valid.org']
    result = filter_valid_emails(emails)
    assert result == ['test@example.com', 'another@valid.org']

def test_should_handle_empty_list():
    result = filter_valid_emails([])
    assert result == []

def test_should_handle_invalid_emails():
    result = filter_valid_emails(['not-an-email', 'also-not'])
    assert result == []
"""
        )

        # Generate test spec
        test_spec = await tdd_generator._generate_test_spec(
            requirement="Create a function that filters valid emails from a list",
            project_context="",
            task_type=TaskType.CODE_GEN,
            model=Model.GPT_4O,
        )

        # Verify
        assert test_spec is not None
        assert test_spec.test_count == 3
        assert "test_main.py" in test_spec.test_file_name
        assert "def test_" in test_spec.test_code

    @pytest.mark.asyncio
    async def test_generate_code_to_pass_tests(self, tdd_generator, mock_client):
        """Test Phase 2: Implementation generation."""
        # Mock LLM response
        mock_client.call.return_value = MagicMock(
            text="""
import re

def filter_valid_emails(emails):
    '''Filter and return only valid email addresses.'''
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return [email for email in emails if re.match(email_pattern, email)]
"""
        )

        # Generate implementation
        code = await tdd_generator._generate_code_to_pass_tests(
            tests="def test_should_return_only_valid_emails(): ...",
            requirement="Create a function that filters valid emails",
            project_context="",
            task_type=TaskType.CODE_GEN,
            model=Model.GPT_4O,
        )

        # Verify
        assert code is not None
        assert "def filter_valid_emails" in code
        assert "import re" in code

    @pytest.mark.asyncio
    async def test_run_tests_success(self, tdd_generator, mock_sandbox):
        """Test Phase 3: Test execution with passing tests."""
        # Mock sandbox response (all tests pass)
        mock_sandbox.execute.return_value = MagicMock(
            return_code=0,
            output="""
============================= test session starts =============================
collected 3 items

test_main.py::test_should_return_only_valid_emails PASSED
test_main.py::test_should_handle_empty_list PASSED
test_main.py::test_should_handle_invalid_emails PASSED

============================== 3 passed in 0.05s ==============================
""",
            error="",
        )

        # Run tests
        result = await tdd_generator._run_tests_and_collect_results(
            test_code="def test_...: ...",
            implementation_code="def filter_valid_emails(...): ...",
            task_type=TaskType.CODE_GEN,
        )

        # Verify
        assert result.passed is True
        assert result.tests_run == 3
        assert result.tests_passed == 3
        assert result.tests_failed == 0

    @pytest.mark.asyncio
    async def test_run_tests_failure(self, tdd_generator, mock_sandbox):
        """Test Phase 3: Test execution with failing tests."""
        # Mock sandbox response (some tests fail)
        mock_sandbox.execute.return_value = MagicMock(
            return_code=1,
            output="""
============================= test session starts =============================
collected 3 items

test_main.py::test_should_return_only_valid_emails PASSED
test_main.py::test_should_handle_empty_list FAILED
test_main.py::test_should_handle_invalid_emails PASSED

=================================== FAILURES ===================================
________________________ test_should_handle_empty_list _________________________
    def test_should_handle_empty_list():
>       assert filter_valid_emails([]) == []
E       AssertionError: assert None == []
=========================== 1 failed, 2 passed in 0.05s ===========================
""",
            error="",
        )

        # Run tests
        result = await tdd_generator._run_tests_and_collect_results(
            test_code="def test_...: ...",
            implementation_code="def filter_valid_emails(...): ...",
            task_type=TaskType.CODE_GEN,
        )

        # Verify
        assert result.passed is False
        assert result.tests_run == 3
        assert result.tests_passed == 2
        assert result.tests_failed == 1
        assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_repair_to_pass_tests(self, tdd_generator, mock_client, mock_sandbox):
        """Test Phase 4: Self-healing to pass tests."""
        # Mock LLM responses (first fails, second succeeds)
        mock_client.call.side_effect = [
            # First repair attempt
            MagicMock(text="def filter_valid_emails(emails):\n    return [e for e in emails if '@' in e]"),
        ]

        # Mock sandbox (tests pass after repair)
        mock_sandbox.execute.return_value = MagicMock(
            return_code=0,
            output="3 passed",
            error="",
        )

        # Repair
        code, result, iterations = await tdd_generator._repair_to_pass_tests(
            tests="def test_...: ...",
            implementation="def filter_valid_emails(emails):\n    return None",
            errors=["AssertionError: assert None == []"],
            requirement="Filter valid emails",
            model=Model.GPT_4O,
        )

        # Verify
        assert iterations >= 1
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_full_tdd_cycle(self, tdd_generator, mock_client, mock_sandbox):
        """Test complete TDD cycle: tests → code → run → pass."""
        # Mock test generation
        mock_client.call.side_effect = [
            # Phase 1: Test generation
            MagicMock(text="def test_valid_emails(): assert True"),
            # Phase 2: Implementation
            MagicMock(text="def filter_valid_emails(emails): return emails"),
            # Phase 3: Test execution (mocked via sandbox)
        ]

        mock_sandbox.execute.return_value = MagicMock(
            return_code=0,
            output="1 passed",
            error="",
        )

        # Create test task
        task = Task(
            id="test-task-001",
            prompt="Create a function to filter valid emails",
            type=TaskType.CODE_GEN,
            max_iterations=3,
        )

        # Run full TDD cycle
        result = await tdd_generator.generate_with_tests(
            task=task,
            project_context="",
            model=Model.GPT_4O,
        )

        # Verify
        assert result.success is True
        assert result.implementation_code != ""
        assert result.test_spec.test_count > 0
        assert result.test_result.passed is True

    def test_parse_pytest_output(self, tdd_generator):
        """Test pytest output parsing."""
        output = """
============================= test session starts =============================
collected 5 items

test_main.py::test_1 PASSED
test_main.py::test_2 PASSED
test_main.py::test_3 FAILED
test_main.py::test_4 PASSED
test_main.py::test_5 PASSED

=========================== 4 passed, 1 failed in 0.10s ===========================
"""
        tests_run, tests_passed = tdd_generator._parse_pytest_output(output)

        assert tests_run == 5
        assert tests_passed == 4

    def test_extract_edge_cases(self, tdd_generator):
        """Test edge case extraction from test code."""
        test_code = """
def test_should_handle_empty_input():
    pass

def test_should_handle_none_values():
    pass

def test_should_handle_max_boundary():
    pass

def test_should_handle_invalid_input():
    pass
"""
        edge_cases = tdd_generator._extract_edge_cases(test_code)

        assert len(edge_cases) >= 3
        assert any("Empty" in ec for ec in edge_cases)
        assert any("None" in ec for ec in edge_cases)


# ─────────────────────────────────────────────
# Test DiffGenerator
# ─────────────────────────────────────────────

class TestDiffGenerator:
    """Test diff-based revisions."""

    @pytest.fixture
    def mock_client(self):
        """Create mock LLM client."""
        client = AsyncMock()
        client.call = AsyncMock()
        return client

    @pytest.fixture
    def diff_generator(self, mock_client):
        """Create DiffGenerator instance."""
        return DiffGenerator(client=mock_client)

    def test_validate_diff_format_valid(self, diff_generator):
        """Test valid diff format validation."""
        valid_diff = """--- a/main.py
+++ b/main.py
@@ -10,7 +10,8 @@
 def existing_function():
     # Existing code
-    old_line()
+    new_line()
+    additional_line()
     # More existing code
"""
        assert diff_generator._validate_diff_format(valid_diff) is True

    def test_validate_diff_format_invalid(self, diff_generator):
        """Test invalid diff format detection."""
        invalid_diff = """
This is just text, not a diff.
No headers, no hunks, no changes.
"""
        assert diff_generator._validate_diff_format(invalid_diff) is False

    def test_count_diff_changes(self, diff_generator):
        """Test diff change counting."""
        diff = """--- a/main.py
+++ b/main.py
@@ -10,3 +10,4 @@
 def func():
     pass
+    new_line()
+    another_line()
-    old_line()
"""
        added, removed = diff_generator._count_diff_changes(diff)

        assert added == 2
        assert removed == 1

    def test_apply_unified_diff_simple(self):
        """Test applying simple unified diff."""
        original = """def hello():
    print("Hello")
    return True
"""
        diff = """--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def hello():
-    print("Hello")
+    print("Hello, World!")
     return True
"""
        patched, error = apply_unified_diff(original, diff)

        assert error is None
        assert "Hello, World!" in patched
        assert "Hello" not in patched or "Hello, World!" in patched

    def test_apply_unified_diff_add_lines(self):
        """Test applying diff that adds lines."""
        original = """def calculate(a, b):
    return a + b
"""
        diff = """--- a/calc.py
+++ b/calc.py
@@ -1,2 +1,5 @@
 def calculate(a, b):
     return a + b
+
+def multiply(a, b):
+    return a * b
"""
        patched, error = apply_unified_diff(original, diff)

        assert error is None
        assert "def multiply" in patched

    @pytest.mark.asyncio
    async def test_generate_diff(self, diff_generator, mock_client):
        """Test diff generation."""
        # Mock LLM response
        mock_client.call.return_value = MagicMock(
            text="""--- a/main.py
+++ b/main.py
@@ -10,7 +10,8 @@
 def validate_email(email):
-    return '@' in email
+    import re
+    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$
+    return bool(re.match(pattern, email))
"""
        )

        # Create test task
        task = Task(
            id="test-task-002",
            prompt="Improve email validation",
            type=TaskType.CODE_GEN,
            max_iterations=3,
        )

        # Generate diff
        result = await diff_generator.generate_diff(
            current_code="def validate_email(email):\n    return '@' in email",
            critique="Use regex for proper email validation",
            task=task,
            model=Model.GPT_4O,
        )

        # Verify
        assert result.success is True
        assert result.diff_text != ""
        assert result.patched_code != ""
        assert result.lines_added >= 2

    @pytest.mark.asyncio
    async def test_generate_diff_fallback(self, diff_generator, mock_client):
        """Test diff generation with invalid format fallback."""
        # Mock LLM response with invalid format
        mock_client.call.return_value = MagicMock(
            text="Just change the return statement to use regex instead"
        )

        task = Task(
            id="test-task-003",
            prompt="Improve email validation",
            type=TaskType.CODE_GEN,
            max_iterations=3,
        )

        # Generate diff (should cleanup and try to apply)
        result = await diff_generator.generate_diff(
            current_code="def validate_email(email):\n    return '@' in email",
            critique="Use regex",
            task=task,
            model=Model.GPT_4O,
        )

        # Should either succeed or return error gracefully
        assert result is not None


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────

class TestParadigmShiftIntegration:
    """Test integration of paradigm shifts."""

    def test_config_flags_exist(self):
        """Test that config flags for paradigm shifts exist."""
        from orchestrator.cost_optimization import OptimizationConfig

        config = OptimizationConfig()

        assert hasattr(config, 'enable_tdd_first')
        assert hasattr(config, 'enable_diff_revisions')
        assert config.enable_tdd_first is False  # Opt-in by default
        assert config.enable_diff_revisions is True  # On by default

    def test_modules_importable(self):
        """Test that paradigm shift modules are importable."""
        from orchestrator import test_first_generator
        from orchestrator import diff_generator

        assert hasattr(test_first_generator, 'TestFirstGenerator')
        assert hasattr(diff_generator, 'DiffGenerator')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
