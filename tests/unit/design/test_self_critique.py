"""Unit tests for self_critique parser."""

import pytest

from orchestrator.design.self_critique import SelfCritiqueParser, inject_self_critique


@pytest.mark.unit
class TestSelfCritiqueParser:
    def test_parse_valid_stamp(self):
        text = "Some code\n/* Hallmark · pre-emit critique: P4 H3 E5 S4 R2 V3 */\nMore code"
        scores = SelfCritiqueParser.parse_scores(text)
        assert scores == {"P": 4, "H": 3, "E": 5, "S": 4, "R": 2, "V": 3}

    def test_parse_no_stamp(self):
        assert SelfCritiqueParser.parse_scores("Just some code without any stamp") is None

    def test_parse_malformed_stamp(self):
        assert SelfCritiqueParser.parse_scores("/* Hallmark · pre-emit critique: P H E S R V */") is None

    def test_parse_partial_stamp(self):
        assert SelfCritiqueParser.parse_scores("/* Hallmark · pre-emit critique: P4 H3 */") is None

    def test_inject_adds_prompt(self):
        prefix = "Design a landing page"
        injected = inject_self_critique(prefix)
        assert "Before handing back any output, score it" in injected
        assert prefix in injected

    def test_min_score_with_stamp(self):
        text = "/* Hallmark · pre-emit critique: P4 H3 E5 S4 R2 V3 */"
        assert SelfCritiqueParser.min_score(text) == 2 / 5.0

    def test_min_score_without_stamp(self):
        assert SelfCritiqueParser.min_score("no stamp here") == 1.0
