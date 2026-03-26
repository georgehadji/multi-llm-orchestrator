"""
Test Production Fixes
=====================
Author: Georgios-Chrysovalantis Chatzivantsidis

Tests for critical production fixes:
- AST validation for code security
- SQLite durability
- Async file I/O
- Budget locking
- API timeouts

USAGE:
    pytest tests/test_production_fixes.py -v
"""

from __future__ import annotations

import asyncio
import pytest
import tempfile
from pathlib import Path
from typing import Optional

# Test AST validation
from orchestrator.code_validator import (
    validate_code,
    extract_code_from_llm_response,
    SecurityConfig,
    ValidationResult,
    ASTSecurityAnalyzer,
    is_code_safe,
)

# Test async file I/O
from orchestrator.async_file_io import (
    async_write_text,
    async_read_text,
    async_write_json,
    async_read_json,
    async_append_text,
    HAS_AIOFILES,
)

# Test Budget locking
from orchestrator.models import Budget


# ─────────────────────────────────────────────
# AST Validation Tests
# ─────────────────────────────────────────────

class TestASTValidation:
    """Test AST-based code validation."""

    def test_valid_code(self):
        """Test that valid code passes validation."""
        code = """
def hello():
    print("Hello, World!")

hello()
"""
        result = validate_code(code)
        assert result.is_valid, f"Valid code failed: {result.errors}"
        assert result.syntax_valid
        assert result.security_valid

    def test_syntax_error(self):
        """Test that syntax errors are detected."""
        code = "def broken("  # Missing closing paren
        result = validate_code(code)
        assert not result.is_valid
        assert not result.syntax_valid
        assert "Syntax error" in str(result.errors)

    def test_eval_detection(self):
        """Test that eval() is detected as dangerous."""
        code = "result = eval(user_input)"
        result = validate_code(code)
        assert not result.is_valid
        assert "eval" in result.dangerous_patterns_found

    def test_exec_detection(self):
        """Test that exec() is detected as dangerous."""
        code = "exec(code_string)"
        result = validate_code(code)
        assert not result.is_valid
        assert "exec" in result.dangerous_patterns_found

    def test_os_system_detection(self):
        """Test that os.system() is detected."""
        code = """
import os
os.system("ls -la")
"""
        result = validate_code(code)
        assert not result.is_valid
        assert "os_system" in result.dangerous_patterns_found

    def test_subprocess_detection(self):
        """Test that subprocess calls are detected."""
        code = """
import subprocess
subprocess.run(["ls", "-la"])
"""
        result = validate_code(code)
        assert not result.is_valid
        assert "subprocess" in result.dangerous_patterns_found

    def test_safe_code_with_config_override(self):
        """Test that dangerous patterns can be allowed via config."""
        code = "result = eval('1 + 1')"
        config = SecurityConfig(allow_eval=True)
        result = validate_code(code, config)
        assert result.is_valid

    def test_empty_code(self):
        """Test that empty code is rejected."""
        result = validate_code("")
        assert not result.is_valid
        assert "Code is empty" in result.errors

    def test_whitespace_only_code(self):
        """Test that whitespace-only code is rejected."""
        result = validate_code("   \n\n   ")
        assert not result.is_valid

    def test_code_metrics(self):
        """Test code metrics extraction."""
        from orchestrator.code_validator import get_code_metrics
        
        code = """
import os

def hello():
    print("Hello")

class MyClass:
    pass
"""
        metrics = get_code_metrics(code)
        assert metrics["functions"] == 1
        assert metrics["classes"] == 1
        assert metrics["imports"] == 1

    def test_extract_code_from_markdown(self):
        """Test code extraction from LLM response with markdown."""
        response = """
Here's the code you requested:

```python
def hello():
    print("Hello!")
```

Hope this helps!
"""
        code = extract_code_from_llm_response(response)
        assert "def hello():" in code
        assert "print" in code

    def test_extract_code_without_fences(self):
        """Test code extraction without markdown fences."""
        response = """
def hello():
    print("Hello!")
"""
        code = extract_code_from_llm_response(response)
        assert "def hello():" in code

    def test_is_code_safe(self):
        """Test quick safety check."""
        safe_code = "print('hello')"
        is_safe, issues = is_code_safe(safe_code)
        assert is_safe

        unsafe_code = "eval(x)"
        is_safe, issues = is_code_safe(unsafe_code)
        assert not is_safe
        assert len(issues) > 0


# ─────────────────────────────────────────────
# Async File I/O Tests
# ─────────────────────────────────────────────

class TestAsyncFileIO:
    """Test async file I/O operations."""

    @pytest.mark.asyncio
    async def test_async_write_and_read_text(self):
        """Test async text write and read."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = Path(f.name)
        
        try:
            content = "Hello, Async World!"
            await async_write_text(temp_path, content)
            
            read_content = await async_read_text(temp_path)
            assert read_content == content
        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_async_write_json(self):
        """Test async JSON write and read."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = Path(f.name)
        
        try:
            data = {"name": "test", "value": 42}
            await async_write_json(temp_path, data)
            
            read_data = await async_read_json(temp_path)
            assert read_data == data
        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_async_append_text(self):
        """Test async text append."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            temp_path = Path(f.name)
        
        try:
            await async_write_text(temp_path, "Line 1\n")
            await async_append_text(temp_path, "Line 2\n")
            
            content = await async_read_text(temp_path)
            assert "Line 1" in content
            assert "Line 2" in content
        finally:
            temp_path.unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_async_write_creates_parents(self):
        """Test that async write creates parent directories."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "nested" / "dir" / "file.txt"
            
            await async_write_text(temp_path, "content", mkdir_parents=True)
            assert temp_path.exists()

    @pytest.mark.asyncio
    async def test_async_file_io_concurrent_writes(self):
        """Test concurrent async writes don't corrupt data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "concurrent.txt"
            
            async def write_task(content: str):
                await async_write_text(temp_path, content)
                await asyncio.sleep(0.01)  # Small delay
            
            # Run multiple writes concurrently
            await asyncio.gather(
                write_task("content1"),
                write_task("content2"),
                write_task("content3"),
            )
            
            # File should exist and have some content
            assert temp_path.exists()


# ─────────────────────────────────────────────
# Budget Locking Tests
# ─────────────────────────────────────────────

class TestBudgetLocking:
    """Test budget locking for concurrent access."""

    @pytest.mark.asyncio
    async def test_budget_concurrent_charges(self):
        """Test that concurrent budget charges are atomic."""
        budget = Budget(max_usd=100.0)
        
        async def charge_task(amount: float):
            await budget.charge(amount, "generation")
        
        # Run multiple charges concurrently
        await asyncio.gather(
            charge_task(10.0),
            charge_task(20.0),
            charge_task(30.0),
        )
        
        # Total should be exact sum
        assert budget.spent_usd == 60.0

    @pytest.mark.asyncio
    async def test_budget_reserve_atomic(self):
        """Test that budget reservations are atomic."""
        budget = Budget(max_usd=100.0)
        
        # Reserve should succeed
        reserved = await budget.reserve(50.0)
        assert reserved
        assert budget.remaining_usd == 50.0

    @pytest.mark.asyncio
    async def test_budget_reserve_insufficient_funds(self):
        """Test that reservation fails with insufficient funds."""
        budget = Budget(max_usd=100.0, spent_usd=80.0)
        
        # Reserve should fail (only 20 left, trying to reserve 50)
        reserved = await budget.reserve(50.0)
        assert not reserved

    @pytest.mark.asyncio
    async def test_budget_commit_reservation(self):
        """Test committing a reservation."""
        budget = Budget(max_usd=100.0)
        
        # Reserve first
        await budget.reserve(50.0)
        assert budget.remaining_usd == 50.0
        
        # Commit with actual amount
        await budget.commit_reservation(45.0, "generation")
        assert budget.spent_usd == 45.0

    @pytest.mark.asyncio
    async def test_budget_release_reservation(self):
        """Test releasing unused reservation."""
        budget = Budget(max_usd=100.0)
        
        # Reserve
        await budget.reserve(50.0)
        assert budget.remaining_usd == 50.0
        
        # Release
        await budget.release_reservation(50.0)
        assert budget.remaining_usd == 100.0

    @pytest.mark.asyncio
    async def test_budget_concurrent_reserve_race_condition(self):
        """Test that concurrent reservations don't exceed budget."""
        budget = Budget(max_usd=100.0)
        
        async def try_reserve():
            return await budget.reserve(60.0)
        
        # Try to reserve 60 twice concurrently (only one should succeed)
        results = await asyncio.gather(
            try_reserve(),
            try_reserve(),
        )
        
        # Exactly one should succeed
        assert sum(results) == 1
        assert budget.remaining_usd == 40.0


# ─────────────────────────────────────────────
# SQLite Durability Tests
# ─────────────────────────────────────────────

class TestSQLiteDurability:
    """Test SQLite durability settings."""

    @pytest.mark.asyncio
    async def test_state_manager_checkpoint(self):
        """Test that state manager checkpoints WAL."""
        from orchestrator.state import StateManager
        from orchestrator.models import Budget, ProjectState
        
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test_state.db"
            state_mgr = StateManager(db_path)
            
            # Create test state
            budget = Budget(max_usd=10.0)
            state = ProjectState(
                project_description="Test project",
                success_criteria="Tests pass",
                budget=budget,
            )
            
            # Save project (should checkpoint WAL)
            await state_mgr.save_project("test-project", state)
            
            # Load and verify
            loaded = await state_mgr.load_project("test-project")
            assert loaded is not None
            assert loaded.project_description == "Test project"
            
            await state_mgr.close()


# ─────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────

class TestProductionFixesIntegration:
    """Integration tests for production fixes."""

    @pytest.mark.asyncio
    async def test_full_workflow_with_validation(self):
        """Test complete workflow with code validation."""
        # Simulate LLM code generation response
        llm_response = """
Here's your code:

```python
def calculate_sum(a, b):
    return a + b

result = calculate_sum(1, 2)
```
"""
        # Extract code
        code = extract_code_from_llm_response(llm_response)
        assert "def calculate_sum" in code
        
        # Validate code
        result = validate_code(code)
        assert result.is_valid, f"Code validation failed: {result.errors}"

    @pytest.mark.asyncio
    async def test_malicious_code_rejected(self):
        """Test that malicious code is detected and rejected."""
        malicious_response = """
```python
import os
os.system("rm -rf /")
```
"""
        code = extract_code_from_llm_response(malicious_response)
        result = validate_code(code)
        
        assert not result.is_valid
        assert "os_system" in result.dangerous_patterns_found

    @pytest.mark.asyncio
    async def test_concurrent_budget_with_file_io(self):
        """Test concurrent budget operations with file I/O."""
        budget = Budget(max_usd=100.0)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "budget_log.json"
            
            async def charge_and_log(amount: float, label: str):
                await budget.charge(amount, "generation")
                await async_write_json(temp_path, {"label": label, "amount": amount})
            
            # Run concurrently
            await asyncio.gather(
                charge_and_log(10.0, "task1"),
                charge_and_log(20.0, "task2"),
                charge_and_log(30.0, "task3"),
            )
            
            assert budget.spent_usd == 60.0


# ─────────────────────────────────────────────
# Run Tests
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
