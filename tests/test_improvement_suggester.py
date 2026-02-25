import pytest
from orchestrator.codebase_profile import CodebaseProfile
from orchestrator.improvement_suggester import ImprovementSuggester, Improvement


class TestImprovementSuggester:
    """Test improvement suggestion generation"""

    def test_suggest_improvements_for_lowcoverage(self):
        """Suggest tests when coverage is low"""
        profile = CodebaseProfile(
            purpose="FastAPI app",
            test_coverage="low",
            anti_patterns=["no error handling"],
            primary_language="python",
            project_type="fastapi",
        )

        suggester = ImprovementSuggester()
        improvements = suggester.suggest(profile)

        assert len(improvements) > 0
        # Should suggest tests as high priority
        assert any("test" in i.title.lower() for i in improvements)

    def test_improvement_has_effort_estimate(self):
        """Each improvement should include effort estimate"""
        profile = CodebaseProfile(
            purpose="Python script",
            anti_patterns=["no type hints", "missing docs"],
            test_coverage="moderate",
            primary_language="python",
            project_type="generic",
        )

        suggester = ImprovementSuggester()
        improvements = suggester.suggest(profile)

        for improvement in improvements:
            assert improvement.effort_hours > 0
            assert improvement.priority in ["LOW", "MEDIUM", "HIGH"]
