"""parse_score must tolerate reasoning-model <think>...</think> output.

Dogfooding with minimax/minimax-m3 (a reasoning model) produced
`parse_score: could not parse from: <think>` — the evaluator could not extract a
score because the thinking block was not stripped before parsing.
"""

import pytest

from orchestrator.application.evaluator import EvaluatorService

pytestmark = pytest.mark.unit


def test_strips_closed_think_block_and_parses_trailing_json():
    out = '<think>The code looks decent, maybe 3 out of 5, hmm 0.2?</think>\n{"score": 0.85}'
    assert EvaluatorService.parse_score(out) == pytest.approx(0.85)


def test_strips_think_block_before_human_readable_score():
    out = "<think>let me weigh 7/10 vs 9/10 internally</think> score: 0.9"
    assert EvaluatorService.parse_score(out) == pytest.approx(0.9)


def test_truncated_open_think_block_does_not_parse_reasoning_numbers():
    # Reasoning truncated mid-thought, no answer emitted. Numbers inside the
    # thinking ("0.1 0.2 0.3") must NOT be mistaken for the score.
    out = "<think>still reasoning about edge cases and 0.1 0.2 0.3"
    assert EvaluatorService.parse_score(out) == pytest.approx(0.5)  # safe default


def test_plain_output_still_works():
    assert EvaluatorService.parse_score('{"score": 0.7}') == pytest.approx(0.7)
