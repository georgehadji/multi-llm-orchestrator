import pytest
from orchestrator.codebase_profile import CodebaseProfile


class TestCodebaseProfile:
    """Test CodebaseProfile dataclass"""

    def test_create_codebase_profile(self):
        """Create a profile with basic info"""
        profile = CodebaseProfile(
            purpose="FastAPI microservice for inventory management",
            primary_patterns=["layered", "REST API", "async"],
            anti_patterns=["no type hints"],
            test_coverage="low",
            documentation="minimal",
            primary_language="python",
            project_type="fastapi",
        )

        assert profile.purpose == "FastAPI microservice for inventory management"
        assert "layered" in profile.primary_patterns
        assert len(profile.anti_patterns) == 1

    def test_profile_repr(self):
        """Profile should have readable string representation"""
        profile = CodebaseProfile(
            purpose="Test project",
            primary_patterns=[],
            anti_patterns=[],
            test_coverage="moderate",
            documentation="README present",
            primary_language="python",
            project_type="python",
        )

        str_repr = str(profile)
        assert "Test project" in str_repr
        assert "python" in str_repr
