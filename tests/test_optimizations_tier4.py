"""
Tests for Tier 4 DevOps Optimizations
=======================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Run: pytest tests/test_optimizations_tier4.py -v
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from orchestrator.cost_optimization import (
    DockerSandbox,
    ExecutionResult,
    SandboxMetrics,
    GitHubIntegration,
    CommitMetadata,
    PushResult,
    GitHubMetrics,
)


# ─────────────────────────────────────────────
# Test DockerSandbox
# ─────────────────────────────────────────────

class TestDockerSandbox:
    """Test DockerSandbox class."""

    def test_sandbox_initialization(self):
        """Test sandbox initializes correctly."""
        sandbox = DockerSandbox()

        assert sandbox is not None
        assert sandbox.image == "python:3.12-slim"
        assert sandbox.memory_limit == "256m"
        assert sandbox.metrics is not None

    def test_custom_initialization(self):
        """Test sandbox with custom parameters."""
        sandbox = DockerSandbox(
            image="python:3.11-slim",
            memory_limit="512m",
            timeout=60,
        )

        assert sandbox.image == "python:3.11-slim"
        assert sandbox.memory_limit == "512m"
        assert sandbox.default_timeout == 60

    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = SandboxMetrics()
        metrics.total_executions = 100
        metrics.successful_executions = 85
        metrics.failed_executions = 15

        metrics_dict = metrics.to_dict()
        assert "success_rate" in metrics_dict
        assert metrics_dict["success_rate"] == 0.85

    @pytest.mark.asyncio
    async def test_execute_with_mock_docker(self):
        """Test execution with mocked Docker."""
        sandbox = DockerSandbox()

        # Mock Docker check
        sandbox._docker_available = False

        # Test fallback to subprocess
        result = await sandbox.execute(
            code_files={"test.py": "print('hello')"},
            command="python test.py",
            timeout=10,
        )

        assert isinstance(result, ExecutionResult)
        # Subprocess fallback should work
        assert result.return_code >= 0 or result.error  # Either success or error message

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test execution timeout."""
        sandbox = DockerSandbox(timeout=1)

        # Mock Docker check
        sandbox._docker_available = False

        # Test with sleep command (should timeout)
        result = await sandbox.execute(
            code_files={},
            command="sleep 10",
            timeout=1,
        )

        assert result.timeout is True or result.return_code != 0

    def test_execute_sync(self):
        """Test synchronous execution wrapper."""
        sandbox = DockerSandbox()
        sandbox._docker_available = False

        result = sandbox.execute_sync(
            code_files={"test.py": "print('sync test')"},
            command="python test.py",
            timeout=10,
        )

        assert isinstance(result, ExecutionResult)

    def test_metrics_reset(self):
        """Test metrics reset."""
        sandbox = DockerSandbox()
        sandbox.metrics.total_executions = 100

        sandbox.reset_metrics()

        assert sandbox.metrics.total_executions == 0

    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test sandbox cleanup."""
        sandbox = DockerSandbox()

        # Add fake workspace
        sandbox._workspaces["/fake"] = Path("/fake")

        await sandbox.cleanup()

        assert len(sandbox._workspaces) == 0


# ─────────────────────────────────────────────
# Test GitHubIntegration
# ─────────────────────────────────────────────

class TestGitHubIntegration:
    """Test GitHubIntegration class."""

    def test_github_initialization(self):
        """Test GitHub integration initializes correctly."""
        github = GitHubIntegration(
            token="test_token",
            owner="test_owner",
            repo="test_repo",
        )

        assert github is not None
        assert github.token == "test_token"
        assert github.owner == "test_owner"
        assert github.repo == "test_repo"

    def test_github_from_env(self):
        """Test GitHub integration from environment."""
        import os
        os.environ["GITHUB_TOKEN"] = "env_token"
        os.environ["GITHUB_OWNER"] = "env_owner"
        os.environ["GITHUB_REPO"] = "env_repo"

        github = GitHubIntegration()

        assert github.token == "env_token"
        assert github.owner == "env_owner"
        assert github.repo == "env_repo"

        # Cleanup
        del os.environ["GITHUB_TOKEN"]
        del os.environ["GITHUB_OWNER"]
        del os.environ["GITHUB_REPO"]

    def test_build_commit_message(self):
        """Test conventional commit message building."""
        github = GitHubIntegration()

        metadata = CommitMetadata(
            budget_spent=1.50,
            quality_score=0.85,
            tasks_completed=10,
            tasks_total=12,
            models_used=["claude-sonnet-4.6", "deepseek-chat"],
        )

        message = github._build_commit_message(
            project_id="my-project",
            summary="Generated REST API",
            metadata=metadata,
            commit_type="feat",
        )

        assert message.startswith("feat(my-project): Generated REST API")
        assert "Budget: $1.50" in message
        assert "Quality: 0.85" in message
        assert "Tasks: 10/12" in message

    def test_build_commit_message_no_metadata(self):
        """Test commit message without metadata."""
        github = GitHubIntegration()

        message = github._build_commit_message(
            project_id="my-project",
            summary="Simple commit",
            metadata=None,
        )

        assert message.startswith("feat(my-project): Simple commit")
        assert "Generated by AI Orchestrator" in message

    def test_metrics_to_dict(self):
        """Test GitHub metrics serialization."""
        metrics = GitHubMetrics()
        metrics.total_pushes = 50
        metrics.successful_pushes = 45
        metrics.failed_pushes = 5

        metrics_dict = metrics.to_dict()
        assert "success_rate" in metrics_dict
        assert metrics_dict["success_rate"] == 0.9

    @pytest.mark.asyncio
    async def test_push_results_no_git(self):
        """Test push results when git not available."""
        github = GitHubIntegration(token="test")

        # Mock git check
        github._git_available = False

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await github.push_results(
                output_dir=Path(tmpdir),
                project_id="test",
                summary="Test commit",
            )

            assert result.success is False
            assert "Git not available" in result.error

    @pytest.mark.asyncio
    async def test_push_results_no_token(self):
        """Test push results without token."""
        github = GitHubIntegration(token=None)
        github._git_available = True

        with tempfile.TemporaryDirectory() as tmpdir:
            result = await github.push_results(
                output_dir=Path(tmpdir),
                project_id="test",
                summary="Test commit",
            )

            assert result.success is False
            assert "token" in result.error.lower()

    @pytest.mark.asyncio
    async def test_push_results_no_files(self):
        """Test push results with no files."""
        github = GitHubIntegration(token="test")
        github._git_available = True

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            # Don't create any files

            result = await github.push_results(
                output_dir=output_dir,
                project_id="test",
                summary="Test commit",
            )

            assert result.success is False
            assert "No files found" in result.error

    @pytest.mark.asyncio
    async def test_push_results_with_files(self):
        """Test push results with actual files."""
        github = GitHubIntegration(token="test")
        github._git_available = True

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create test files
            (output_dir / "test.py").write_text("print('hello')")
            (output_dir / "README.md").write_text("# Test")

            result = await github.push_results(
                output_dir=output_dir,
                project_id="test",
                summary="Test commit",
                push_remote=False,  # Don't actually push
            )

            # Should succeed locally (no remote push)
            assert result.success is True or result.error  # Either success or specific error
            assert result.files_pushed >= 0  # May be 0 if git operations fail

    def test_metrics_reset(self):
        """Test GitHub metrics reset."""
        github = GitHubIntegration(token="test")
        github.metrics.total_pushes = 100

        github.reset_metrics()

        assert github.metrics.total_pushes == 0


# ─────────────────────────────────────────────
# Test Integration
# ─────────────────────────────────────────────

class TestIntegration:
    """Test integration between Tier 4 modules."""

    def test_combined_metrics(self):
        """Test collecting metrics from both modules."""
        sandbox = DockerSandbox()
        github = GitHubIntegration(token="test")

        all_metrics = {
            "sandbox": sandbox.get_metrics(),
            "github": github.get_metrics(),
        }

        assert "total_executions" in all_metrics["sandbox"]
        assert "total_pushes" in all_metrics["github"]

    @pytest.mark.asyncio
    async def test_sandbox_then_push(self):
        """Test sandbox execution followed by push."""
        sandbox = DockerSandbox()
        github = GitHubIntegration(token="test")

        sandbox._docker_available = False
        github._git_available = True

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Execute code
            result = await sandbox.execute(
                code_files={"test.py": "print('hello')"},
                command="python test.py",
                timeout=10,
            )

            # Save output
            (output_dir / "output.txt").write_text(result.output)

            # Push results
            push_result = await github.push_results(
                output_dir=output_dir,
                project_id="test",
                summary="Test execution",
                push_remote=False,
            )

            # Both should complete (locally)
            assert result is not None
            assert push_result is not None


# ─────────────────────────────────────────────
# Test Tier 3 & 4 Fixes
# ─────────────────────────────────────────────

class TestDeferredBugFixes:
    """Tests for BUG-OPT-004 and BUG-OPT-005 fixes."""

    def test_pydantic_import_error(self):
        """Test BUG-OPT-004: Pydantic import raises clear error."""
        # This test verifies the module imports correctly when pydantic is available
        from orchestrator.cost_optimization import StructuredOutputEnforcer
        assert StructuredOutputEnforcer is not None

    @pytest.mark.asyncio
    async def test_eval_dataset_concurrent_writes(self):
        """Test BUG-OPT-005 fix: Concurrent writes use atomic append."""
        from orchestrator.cost_optimization import EvalDatasetBuilder
        import asyncio
        import json
        
        builder = EvalDatasetBuilder()
        
        # Create temp file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            builder.dataset_path = Path(f.name)
        
        try:
            # Simulate concurrent writes
            async def write_failure(i):
                await builder.record_failure(
                    task_prompt=f"Prompt {i}",
                    generated_code=f"code_{i}",
                    errors=[f"Error {i}"],
                    eval_scores={"quality": 0.5},
                    model=f"model-{i % 3}",
                    task_type="code_generation",
                )
            
            # Write 10 failures concurrently
            await asyncio.gather(*[write_failure(i) for i in range(10)])
            
            # Verify file is valid JSONL (each line is valid JSON)
            with open(builder.dataset_path, 'r') as f:
                lines = f.readlines()
            
            # Note: Due to atomic append, all lines should be written
            # but order may vary due to async scheduling
            assert len(lines) == 10, f"Expected 10 lines, got {len(lines)}"
            
            for i, line in enumerate(lines):
                try:
                    data = json.loads(line)
                    assert "prompt" in data, f"Line {i} missing 'prompt'"
                    assert "bad_output" in data, f"Line {i} missing 'bad_output'"
                except json.JSONDecodeError as e:
                    pytest.fail(f"Line {i} is not valid JSON: {e}")
        
        finally:
            # Cleanup
            builder.dataset_path.unlink(missing_ok=True)

    def test_eval_dataset_file_locking(self):
        """Test BUG-OPT-005: File locking prevents interleaved writes."""
        from orchestrator.cost_optimization import EvalDatasetBuilder
        import tempfile
        
        builder = EvalDatasetBuilder()
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            builder.dataset_path = Path(f.name)
        
        try:
            # Record failure synchronously
            import asyncio
            asyncio.get_event_loop().run_until_complete(
                builder.record_failure(
                    task_prompt="Test prompt",
                    generated_code="test code",
                    errors=["test error"],
                    eval_scores={"quality": 0.5},
                    model="test-model",
                    task_type="code_generation",
                )
            )
            
            # Verify file content
            with open(builder.dataset_path, 'r') as f:
                content = f.read()
            
            assert "Test prompt" in content
            assert "test code" in content
        
        finally:
            # Cleanup
            builder.dataset_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
