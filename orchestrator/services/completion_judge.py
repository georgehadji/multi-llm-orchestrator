"""
CompletionJudge — maker-checker stop condition.

ENH-2 (Loop Engineering §VI): Generator and Evaluator must be different models.
A fresh-model judge gives an independent PASS/FAIL verdict before the loop exits,
preventing the generator from grading its own work (cognitive surrender).

Design:
- judge_model != generator_model enforced at construction (SameModelError)
- PASS → loop may exit; FAIL → force another iteration
- Unparseable response or LLM exception → conservative FAIL (fail-closed)
- from_models() factory: picks cheapest non-generator candidate; returns None
  when no candidates differ from the generator (caller skips judge gracefully)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("orchestrator.services.completion_judge")


class JudgeVerdict(Enum):
    """Binary verdict from the maker-checker judge."""

    PASS = "PASS"
    FAIL = "FAIL"

    def __bool__(self) -> bool:
        return self is JudgeVerdict.PASS

    def __str__(self) -> str:
        return self.value


class SameModelError(RuntimeError):
    """Raised when judge_model == generator_model — independence invariant violated."""


_JUDGE_SYSTEM_PROMPT = (
    "Independent code reviewer. "
    "You did NOT write this output. "
    "Verdict PASS only if output fully satisfies the task with no blockers. "
    "Verdict FAIL if anything is missing, wrong, or untested. "
    'Return ONLY JSON: {"verdict": "PASS"|"FAIL", "reason": "<one sentence>"}'
)


class CompletionJudge:
    """Maker-checker stop condition: asks a different model whether work is done.

    Usage:
        judge = CompletionJudge(client, judge_model, generator_model)
        verdict = await judge.judge(task_prompt, artifact)
        if verdict is JudgeVerdict.PASS:
            break  # exit iteration loop
    """

    def __init__(
        self,
        client: Any,
        judge_model: Any,
        generator_model: Any,
    ) -> None:
        judge_val = judge_model.value if hasattr(judge_model, "value") else str(judge_model)
        gen_val = generator_model.value if hasattr(generator_model, "value") else str(generator_model)
        if judge_model is generator_model or judge_val == gen_val:
            raise SameModelError(
                f"CompletionJudge: judge_model ({judge_val!r}) must differ from "
                f"generator_model ({gen_val!r}) — independence invariant violated"
            )
        self._client = client
        self._judge_model = judge_model
        self._generator_model = generator_model

    async def judge(self, task_prompt: str, artifact: str) -> JudgeVerdict:
        """Ask the judge model whether *artifact* satisfies *task_prompt*.

        Returns FAIL conservatively on any error (fail-closed).
        """
        prompt = (
            f"TASK:\n{task_prompt}\n\n"
            f"OUTPUT TO REVIEW:\n{artifact}\n\n"
            "Does this output fully satisfy the task with no blockers? "
            'Reply ONLY with JSON: {"verdict": "PASS"|"FAIL", "reason": "..."}'
        )
        try:
            response = await self._client.call(
                self._judge_model,
                prompt,
                system=_JUDGE_SYSTEM_PROMPT,
                max_tokens=150,
                temperature=0.0,
            )
            data = json.loads(response.text)
            raw = str(data.get("verdict", "")).upper()
            if raw == "PASS":
                return JudgeVerdict.PASS
            if raw == "FAIL":
                return JudgeVerdict.FAIL
            logger.warning("CompletionJudge: unknown verdict %r → FAIL", raw)
            return JudgeVerdict.FAIL
        except Exception as exc:
            logger.warning("CompletionJudge: error calling judge model → FAIL: %s", exc)
            return JudgeVerdict.FAIL

    @classmethod
    def from_models(
        cls,
        client: Any,
        judge_candidates: list[Any],
        generator_model: Any,
    ) -> "CompletionJudge | None":
        """Factory: pick cheapest candidate that differs from generator_model.

        Returns None when no suitable candidate exists (caller should skip judge).
        Judge candidates should be ordered cheapest-first.
        """
        gen_val = generator_model.value if hasattr(generator_model, "value") else str(generator_model)
        for candidate in judge_candidates:
            cand_val = candidate.value if hasattr(candidate, "value") else str(candidate)
            if candidate is not generator_model and cand_val != gen_val:
                try:
                    return cls(
                        client=client,
                        judge_model=candidate,
                        generator_model=generator_model,
                    )
                except SameModelError:
                    continue
        logger.debug("CompletionJudge.from_models: no valid judge candidate found")
        return None
