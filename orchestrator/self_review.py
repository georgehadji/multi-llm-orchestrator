"""
SelfReviewToggle - Same-model pre-pass before cross-model critique.
==================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Part of Category 10, Phase 3 (Replit-inspired).
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
import json
import re
import logging

if TYPE_CHECKING:
    from .infrastructure.llm_client import UnifiedClient

logger = logging.getLogger(__name__)


class SelfReviewResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNCERTAIN = "uncertain"


@dataclass
class SelfReviewConfig:
    enabled: bool = True
    pass_threshold: float = 0.7
    max_self_review_tokens: int = 500
    save_pct_estimate: float = 0.40


_SR_PROMPT = 'Review your own code output. Rate 0-1 for correctness, completeness, style. Return JSON: {"score": 0.8, "issues": [], "verdict": "pass|fail|uncertain"}'


class SelfReviewer:
    """Same-model self-review before cross-model critique."""

    def __init__(self, client=None):
        self._client = client

    def should_skip_cross_review(self, result_score, deterministic_issues, config=None):
        cfg = config or SelfReviewConfig()
        if not cfg.enabled or deterministic_issues > 0:
            return False
        return result_score >= cfg.pass_threshold

    async def review(self, code, task_description=""):
        if not self._client:
            return {"score": 0.7, "issues": [], "verdict": SelfReviewResult.UNCERTAIN.value}
        prompt = (
            _SR_PROMPT
            + "\n\nTask: "
            + task_description[:200]
            + "\n\nCode:\n```\n"
            + code[:2000]
            + "\n```"
        )
        try:
            response = await self._client.call(
                model=None,
                prompt=prompt,
                system="You are reviewing your own code. Be honest about issues.",
                max_tokens=300,
                temperature=0.1,
                timeout=30,
            )
            return self._parse(response.text)
        except Exception:
            return {
                "score": 0.5,
                "issues": ["Self-review failed"],
                "verdict": SelfReviewResult.UNCERTAIN.value,
            }

    @staticmethod
    def _parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"score": 0.5, "issues": [], "verdict": "uncertain"}
