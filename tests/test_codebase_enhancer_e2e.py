"""End-to-end test for Codebase Enhancer feature"""

import pytest
import tempfile
from pathlib import Path
from orchestrator import (
    CodebaseAnalyzer,
    CodebaseUnderstanding,
    ImprovementSuggester,
)


class TestCodebaseEnhancerE2E:
    """End-to-end tests for the Codebase Enhancer feature"""

    def test_static_analysis_workflow(self):
        """Test complete static analysis workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a simple FastAPI project structure
            (root / "main.py").write_text("from fastapi import FastAPI\napp = FastAPI()")
            (root / "requirements.txt").write_text("fastapi\nuvicorn\n")
            (root / "README.md").write_text("# My FastAPI App")
            (root / "tests").mkdir()
            (root / "tests" / "test_main.py").write_text("import pytest\n")

            # Run static analysis
            analyzer = CodebaseAnalyzer()
            codebase_map = analyzer.scan(str(root))

            # Verify static analysis results
            assert codebase_map.total_files == 4  # main.py, requirements.txt, README.md, test_main.py
            assert codebase_map.primary_language == "python"
            assert codebase_map.project_type == "fastapi"
            assert codebase_map.has_tests == True
            assert codebase_map.has_docs == True

    @pytest.mark.asyncio
    async def test_semantic_analysis_workflow(self):
        """Test complete semantic analysis workflow"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Create a project with known characteristics
            (root / "main.py").write_text(
                "from fastapi import FastAPI\n"
                "app = FastAPI()\n"
                "@app.get('/')\ndef read_root(): return {}"
            )
            (root / "requirements.txt").write_text("fastapi\n")

            # Run semantic understanding
            understanding = CodebaseUnderstanding()
            profile = await understanding.analyze(str(root))

            # Verify profile was created
            assert profile.purpose is not None
            assert profile.primary_language == "python"
            assert profile.project_type == "fastapi"

    def test_improvement_suggestions_workflow(self):
        """Test improvement suggestion workflow"""
        # Create a profile representing a low-test, minimal-docs FastAPI project
        from orchestrator import CodebaseProfile, ImprovementSuggester

        profile = CodebaseProfile(
            purpose="FastAPI REST API",
            test_coverage="low",
            documentation="minimal",
            anti_patterns=["no error handling", "no type hints"],
            primary_language="python",
            project_type="fastapi",
        )

        # Generate suggestions
        suggester = ImprovementSuggester()
        improvements = suggester.suggest(profile)

        # Verify suggestions
        assert len(improvements) > 0

        # Should include test suggestion (HIGH priority)
        high_priority = [i for i in improvements if i.priority == "HIGH"]
        assert len(high_priority) > 0
        assert any("test" in i.title.lower() for i in high_priority)

        # Should include documentation suggestion (MEDIUM priority)
        medium_priority = [i for i in improvements if i.priority == "MEDIUM"]
        assert len(medium_priority) > 0
        assert any("doc" in i.title.lower() for i in medium_priority)

        # All improvements should have effort estimates
        for improvement in improvements:
            assert improvement.effort_hours > 0
            assert improvement.category in ["testing", "refactoring", "documentation", "features"]

    @pytest.mark.asyncio
    async def test_full_pipeline(self):
        """Test the complete codebase enhancer pipeline"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)

            # Setup a realistic project
            (root / "main.py").write_text(
                "from fastapi import FastAPI\n"
                "import json\n"
                "app = FastAPI()\n\n"
                "@app.get('/users/{user_id}')\n"
                "def get_user(user_id: int):\n"
                "    return {'id': user_id, 'name': 'John'}\n"
            )
            (root / "requirements.txt").write_text("fastapi==0.104.0\nuvicorn==0.24.0\n")
            (root / "README.md").write_text("# User API\nA simple FastAPI user service")
            (root / "tests").mkdir()

            # Step 1: Static Analysis
            analyzer = CodebaseAnalyzer()
            codebase_map = analyzer.scan(str(root))
            assert codebase_map.project_type == "fastapi"

            # Step 2: Semantic Analysis
            understanding = CodebaseUnderstanding()
            profile = await understanding.analyze(str(root))
            assert profile.primary_language == "python"

            # Step 3: Generate Improvements
            suggester = ImprovementSuggester()
            improvements = suggester.suggest(profile)
            assert len(improvements) > 0

            # Verify the complete pipeline produced actionable recommendations
            total_effort = sum(i.effort_hours for i in improvements)
            assert total_effort > 0
