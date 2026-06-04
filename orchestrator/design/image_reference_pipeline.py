"""image_reference_pipeline — optional pre-flight visual-context generator.

When ``flags.image_reference_pipeline`` is True and the task is a greenfield
frontend task, this pipeline calls a vision-capable LLM with the
``imagegen_web`` skill as its system prompt to produce a *text* visual-direction
description.  That description is then appended to the task prompt so the code
generator has richer design context before writing any HTML/CSS/JS.

No binary images are ever produced or required — this is a text-only pipeline
that generates a visual brief from the task description alone.

Usage (engine.py):
    pipeline = ImageReferencePipeline(loader, client, flags)
    visual_context = await pipeline.build_visual_context(task)
    if visual_context:
        task = dataclasses.replace(task, prompt=f"{task.prompt}\n\n{visual_context}")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_VISUAL_CONTEXT_PREFIX = "## Visual direction (auto-generated)\n"
_MAX_VISUAL_TOKENS = 400


class ImageReferencePipeline:
    """Generates a text visual-direction brief using the imagegen_web skill."""

    def __init__(
        self,
        loader: object,
        client: object,
        flags: object | None = None,
    ) -> None:
        self._loader = loader
        self._client = client
        self._flags = flags

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_visual_context(self, task: object) -> str:
        """Return a text visual-direction description for *task*, or "".

        Returns "" when:
        - ``flags.image_reference_pipeline`` is False (default)
        - the imagegen_web skill file is not bundled
        - the LLM call fails for any reason
        """
        if not getattr(self._flags, "image_reference_pipeline", False):
            return ""

        skill_content = self._loader.load("imagegen_web")  # type: ignore[attr-defined]
        if not skill_content:
            logger.debug("image_reference_pipeline: imagegen_web skill not bundled; skipping")
            return ""

        task_prompt = getattr(task, "prompt", "")
        user_message = (
            f"Based on the following project brief, describe the visual design direction "
            f"in 3-5 bullet points (typography, colour palette, layout style, key components). "
            f"Be specific and opinionated. Do NOT generate code.\n\n"
            f"## Brief\n{task_prompt[:1500]}"
        )

        try:
            response = await self._client.call(  # type: ignore[attr-defined]
                model=None,
                prompt=user_message,
                system=skill_content,
                max_tokens=_MAX_VISUAL_TOKENS,
                temperature=0.7,
            )
            if response and response.text:
                return f"{_VISUAL_CONTEXT_PREFIX}{response.text.strip()}"
        except Exception as exc:
            logger.debug("image_reference_pipeline: LLM call failed: %s", exc)

        return ""
