import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from orchestrator.codebase_analyzer import CodebaseAnalyzer
from orchestrator.codebase_understanding import CodebaseUnderstanding
from orchestrator.codebase_profile import CodebaseProfile


class TestCodebaseUnderstanding:
    """Test CodebaseUnderstanding (LLM-powered analysis)"""

    @pytest.mark.asyncio
    async def test_analyze_returns_profile(self):
        """Analyze returns CodebaseProfile with basic info"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a simple Python project structure
            root = Path(tmpdir)
            (root / "main.py").write_text("# FastAPI app")

            understanding = CodebaseUnderstanding()

            # Mock the LLM call
            with patch.object(understanding, '_call_llm') as mock_llm:
                mock_llm.return_value = {
                    "purpose": "FastAPI microservice",
                    "patterns": ["REST API", "async"],
                    "anti_patterns": ["no error handling"],
                    "test_coverage": "low",
                }

                profile = await understanding.analyze(
                    codebase_path=str(root)
                )

                assert isinstance(profile, CodebaseProfile)
                assert "FastAPI" in profile.purpose
                assert "REST API" in profile.primary_patterns

    @pytest.mark.asyncio
    async def test_analyze_reads_key_files(self):
        """Analysis should read and include key file contents"""
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "README.md").write_text("# My FastAPI App\nProvides user management")
            (root / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")

            understanding = CodebaseUnderstanding()

            with patch.object(understanding, '_call_llm') as mock_llm:
                mock_llm.return_value = {
                    "purpose": "User management service",
                    "patterns": [],
                    "anti_patterns": [],
                    "test_coverage": "good",
                }

                profile = await understanding.analyze(str(root))

                # Verify LLM was called with file contents
                mock_llm.assert_called_once()
                call_args = mock_llm.call_args[0][0]  # First positional arg
                assert "main.py" in call_args or "FastAPI" in call_args
