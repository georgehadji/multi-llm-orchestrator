"""
BrainstormingDecomposer — Ask clarifying questions before decomposing.
======================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Part of the Category 1 Enhancement Plan (Wave 2: N4 Brainstorming Mode).
Before creating a task plan, asks 3-5 clarifying questions to narrow scope
and reduce ambiguity. The answers are injected into the decomposition
context, producing more targeted task breakdowns.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .infrastructure.llm_client import UnifiedClient

logger = logging.getLogger(__name__)

# Prompt template for generating clarifying questions
_CLARIFYING_PROMPT = """You are a project planning assistant. Given a project description,
ask 3-5 clarifying questions that will help create a precise task breakdown.

Focus on:
- Scope boundaries (what's in/out)
- Technical constraints (language, framework, platform)
- Success criteria (how to verify completion)
- Architecture preferences (monolith vs microservices, etc.)
- Priority ordering

Project description:
{description}

Respond as a JSON array of questions. Each question is a string. Example:
["What framework should be used?", "Should this support mobile?"]

Output exactly this JSON (no other text):
{{"questions": ["question 1", "question 2", "question 3"]}}"""


@dataclass
class ClarifyingQuestions:
    """Bidirectional protocol: questions from the system, answers from the user."""

    questions: list[str] = field(default_factory=list)
    answers: list[str] = field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        return len(self.answers) == len(self.questions) and len(self.questions) > 0

    def build_context(self) -> str:
        """Build a context string from questions and answers for injection."""
        if not self.questions:
            return ""
        lines = ["## Clarifications", ""]
        for q, a in zip(self.questions, self.answers):
            lines.append(f"Q: {q}")
            lines.append(f"A: {a}")
            lines.append("")
        lines.append("---")
        return "\n".join(lines)


class BrainstormingDecomposer:
    """Generates clarifying questions, collects answers, feeds them into decomposition.

    Usage:
        decomposer = BrainstormingDecomposer(client)
        questions = await decomposer.ask("Build a task management app")
        # ... collect answers from user ...
        context = decomposer.build_context(answers)
    """

    def __init__(self, client: UnifiedClient | None = None):
        self._client = client
        self._pending: ClarifyingQuestions | None = None

    async def ask(
        self,
        description: str,
        criteria: str = "",
        max_questions: int = 5,
    ) -> list[str]:
        """Generate clarifying questions for a project description.

        Args:
            description: Project description
            criteria: Optional success criteria
            max_questions: Maximum questions to generate (3-5)

        Returns:
            List of clarifying questions
        """
        if not self._client:
            logger.warning("No client available for brainstorming — skipping questions")
            return []

        prompt = _CLARIFYING_PROMPT.format(description=description)

        try:
            response = await self._client.call(
                model=None,  # use default
                prompt=prompt,
                system="You are a precise planning assistant. Return only valid JSON.",
                max_tokens=400,
                temperature=0.3,
                timeout=30,
            )

            parsed = self._parse_response(response.text)
            questions = parsed.get("questions", [])[:max_questions]
            self._pending = ClarifyingQuestions(questions=questions)
            return questions

        except Exception as exc:
            logger.warning(f"Brainstorming question generation failed: {exc}")
            return []

    @staticmethod
    def _parse_response(text: str) -> dict:
        """Parse the LLM response into structured questions."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown
            import re

            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
        return {"questions": []}

    def build_context(self, answers: list[str]) -> str:
        """Build context string from previously asked questions and given answers.

        Args:
            answers: User's answers to the clarifying questions

        Returns:
            Context string for injection into decomposition prompt
        """
        if not self._pending or not self._pending.questions:
            return ""

        self._pending.answers = answers[: len(self._pending.questions)]
        return self._pending.build_context()

    def inject_into_description(self, description: str, answers: list[str]) -> str:
        """Return the original description augmented with Q&A context.

        Args:
            description: Original project description
            answers: User's answers

        Returns:
            Augmented description string
        """
        context = self.build_context(answers)
        if context:
            return f"{description}\n\n{context}"
        return description
