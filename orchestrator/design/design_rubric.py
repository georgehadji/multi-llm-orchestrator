"""
Default Design Rubric — Prompt builder for LLM-based design critique.
====================================================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Builds a structured critique prompt that asks an LLM to score frontend
output on five design dimensions: typography, colour, layout,
interactivity, and content realism.

Usage:
    rubric = DefaultDesignRubric()
    prompt = rubric.build(task_prompt, generated_output)
"""

from __future__ import annotations


class DefaultDesignRubric:
    """Generic design rubric for any frontend task."""

    SYSTEM_PROMPT: str = (
        "You are a senior design reviewer. Score the provided frontend code "
        "on five dimensions. Be specific, cite exact lines, and suggest fixes. "
        "Return ONLY JSON."
    )

    def build(self, task_prompt: str, output: str) -> str:
        """Build the critique prompt.

        Args:
            task_prompt: The original user prompt.
            output: The generated frontend code/HTML/CSS.

        Returns:
            A structured prompt for the LLM reviewer.
        """
        return (
            "## Design Review Request\n\n"
            "Original brief:\n"
            f"{task_prompt[:500]}\n\n"
            "Generated output (first 3000 chars):\n"
            "```\n"
            f"{output[:3000]}\n"
            "```\n\n"
            "Score each dimension 0.0–1.0 and list up to 3 specific issues:\n\n"
            "1. **Typography** — font choice, hierarchy, line-height, readability\n"
            "2. **Colour** — palette coherence, contrast, token discipline\n"
            "3. **Layout** — grid use, spacing scale, responsive behaviour\n"
            "4. **Interactivity** — hover/focus states, motion, feedback\n"
            "5. **Content Realism** — placeholder text, generic icons, Lorem ipsum\n\n"
            "Return JSON: {\n"
            '  "score": 0.0-1.0,\n'
            '  "dimensions": {\n'
            '    "typography": {"score": 0.0-1.0, "issues": ["..."]},\n'
            '    "colour": {"score": 0.0-1.0, "issues": ["..."]},\n'
            '    "layout": {"score": 0.0-1.0, "issues": ["..."]},\n'
            '    "interactivity": {"score": 0.0-1.0, "issues": ["..."]},\n'
            '    "content_realism": {"score": 0.0-1.0, "issues": ["..."]}\n'
            "  },\n"
            '  "top_fixes": ["..."]\n'
            "}"
        )

    @staticmethod
    def parse_score(critique_text: str) -> float:
        """Extract the overall score from critique JSON.

        Returns 0.5 if parsing fails.
        """
        import json
        import re

        try:
            # Find JSON block
            match = re.search(r"\{.*\}", critique_text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                score = float(data.get("score", 0.5))
                return max(0.0, min(1.0, score))
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: look for "score": N.N
        match = re.search(r'"score"\s*:\s*([0-9.]+)', critique_text)
        if match:
            try:
                return max(0.0, min(1.0, float(match.group(1))))
            except ValueError:
                pass

        return 0.5
