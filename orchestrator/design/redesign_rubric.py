"""redesign_rubric — structures the redesign-skill audit as a critique prompt.

When a task has ``design_variant == DesignVariant.REDESIGN``, ``CritiqueCycle``
uses this rubric instead of ``CritiquePrompt.build_score`` so the reviewer LLM
evaluates the output against a structured scan→diagnose→fix audit rather than
generic code quality.

The output format must preserve the ``{"score": x, "reasoning": ...}`` JSON
contract expected by ``CritiqueCycle._extract_score``.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_FALLBACK_SECTIONS = """
Typography: font choice, size contrast, letter-spacing, line-height, headings
Color: palette, saturation, gradients, shadows, background tones
Layout: symmetry, whitespace, grid structure, card hierarchy, container width
Interactivity: hover/focus/active states, transitions, loading/empty/error states
Content: realistic copy, no lorem ipsum, no generic placeholder names
"""


class RedesignRubric:
    """Builds a structured critique prompt based on the redesign-skill checklist."""

    def __init__(self, loader: object | None = None) -> None:
        self._loader = loader
        self._rubric_cache: str | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_score(
        self,
        original_prompt: str,
        generated_output: str,
        task_type_value: str,
    ) -> str:
        """Return a critique prompt that uses the redesign audit as its rubric.

        The prompt ends with an explicit instruction to produce
        ``{"score": <0.0-1.0>, "reasoning": "..."}`` JSON so that
        ``CritiqueCycle._extract_score`` can parse it unchanged.

        Falls back to a minimal inline rubric if the skill file is not bundled.
        """
        rubric = self._load_rubric()
        output_preview = generated_output[:3000]

        return (
            f"You are a senior UI/UX engineer performing a structured design audit.\n\n"
            f"## Original request\n{original_prompt}\n\n"
            f"## Generated output (preview)\n```\n{output_preview}\n```\n\n"
            f"## Redesign audit rubric\nEvaluate the output against each section below. "
            f"Identify every generic or weak pattern found. Be specific and actionable.\n\n"
            f"{rubric}\n\n"
            f"## Scoring instructions\n"
            f"Score 0.0-1.0 where:\n"
            f"  1.0 = passes all audit sections, no generic AI patterns detected\n"
            f"  0.8 = minor issues in 1-2 sections\n"
            f"  0.6 = noticeable issues in 3+ sections\n"
            f"  0.4 = significant generic patterns present\n"
            f"  0.0 = output is generic slop with no intentional design\n\n"
            f"Respond with ONLY valid JSON on the last line:\n"
            f'{{\"score\": <float 0.0-1.0>, \"reasoning\": \"<1-3 sentences>\"}}'
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load_rubric(self) -> str:
        if self._rubric_cache is not None:
            return self._rubric_cache

        if self._loader is not None:
            try:
                content = self._loader.load("redesign")  # type: ignore[attr-defined]
                if content:
                    self._rubric_cache = content
                    return self._rubric_cache
            except Exception as exc:
                logger.debug("redesign_rubric: loader failed: %s", exc)

        # Inline fallback — covers the five major audit sections
        self._rubric_cache = _FALLBACK_SECTIONS.strip()
        return self._rubric_cache
