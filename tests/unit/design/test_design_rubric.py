"""Unit tests for DefaultDesignRubric."""

import pytest

from orchestrator.design.design_rubric import DefaultDesignRubric


@pytest.mark.unit
class TestDefaultDesignRubric:
    def test_build_contains_dimensions(self):
        rubric = DefaultDesignRubric()
        prompt = rubric.build("Build a landing page", "<div>Hello</div>")
        assert "Typography" in prompt
        assert "Colour" in prompt
        assert "Layout" in prompt
        assert "Interactivity" in prompt
        assert "Content Realism" in prompt
        assert "score" in prompt

    def test_build_includes_task_prompt(self):
        rubric = DefaultDesignRubric()
        prompt = rubric.build("Build a SaaS dashboard", "output")
        assert "Build a SaaS dashboard" in prompt

    def test_build_includes_output_snippet(self):
        rubric = DefaultDesignRubric()
        output = "<html><body>Test</body></html>"
        prompt = rubric.build("task", output)
        assert output in prompt

    def test_parse_score_from_json(self):
        rubric = DefaultDesignRubric()
        text = '{"score": 0.85, "dimensions": {}}'
        assert rubric.parse_score(text) == 0.85

    def test_parse_score_clamps_above_one(self):
        rubric = DefaultDesignRubric()
        text = '{"score": 1.5}'
        assert rubric.parse_score(text) == 1.0

    def test_parse_score_clamps_below_zero(self):
        rubric = DefaultDesignRubric()
        text = '{"score": -0.2}'
        assert rubric.parse_score(text) == 0.0

    def test_parse_score_fallback_regex(self):
        rubric = DefaultDesignRubric()
        text = 'Some prose\n"score": 0.72\nMore prose'
        assert rubric.parse_score(text) == 0.72

    def test_parse_score_no_match_defaults_half(self):
        rubric = DefaultDesignRubric()
        assert rubric.parse_score("no score here") == 0.5

    def test_system_prompt_non_empty(self):
        assert "design reviewer" in DefaultDesignRubric.SYSTEM_PROMPT.lower()
