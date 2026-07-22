"""CritiqueCycle._extract_score must ignore reasoning-model <think> content.

Dogfooding produced near-zero task scores (0.05) because the reviewer emitted
`<think>...maybe score: 0.05...</think> {"score": 0.85}` and _extract_score
grabbed the first 'score: N' it saw — the number inside the chain-of-thought —
instead of the real verdict. This is the same class of bug already fixed in
EvaluatorService.parse_score, but in the parser that actually drives best_score.
"""

import pytest

pytestmark = pytest.mark.unit

from orchestrator.application.critique_cycle import CritiqueCycle

pytestmark = pytest.mark.unit


def _score(text: str) -> float:
    # _extract_score does not use `self`; call it with a dummy instance.
    return CritiqueCycle._extract_score(object(), text)


def test_ignores_score_inside_think_block_uses_real_json_verdict():
    out = '<think>hmm this could be score: 0.05, or maybe higher</think>\n{"score": 0.85}'
    assert _score(out) == pytest.approx(0.85)


def test_ignores_think_block_before_human_readable_score():
    out = "<think>weighing 0.1 vs 0.2 internally</think>\nFinal score: 0.9"
    assert _score(out) == pytest.approx(0.9)


def test_truncated_think_with_score_phrase_does_not_grab_reasoning_number():
    # Reviewer is a reasoning model truncated mid-thought: no final JSON verdict,
    # but the chain-of-thought literally says "score: 0.05". That reasoning number
    # must NOT become the verdict.
    out = "<think>my initial score: 0.05 but I should reconsider after checking edge"
    assert _score(out) == pytest.approx(0.5)  # safe default, not 0.05


def test_plain_json_and_text_still_work():
    assert _score('{"score": 0.7}') == pytest.approx(0.7)
    assert _score("Overall score: 0.8") == pytest.approx(0.8)
