"""
GenerateStage — LLM code/text generation
==========================================
"""

from __future__ import annotations

import logging
from ..pipeline import PipelineContext
from ...domain.ports import LLMClient
from ...budget import Budget
from ...model_selector import ModelSelector
from ...models import Model, TaskType
from ...prompt_builder import SystemPrompt

logger = logging.getLogger("orchestrator.engine_core.stages.generate")


class GenerateStage:
    """Generate output for a task using the primary model.

    Handles model selection (respecting preferred_model override),
    prompt construction, and API call with timeout.
    """

    def __init__(
        self,
        client: LLMClient,
        budget: Budget,
        selector: ModelSelector,
        event_bus: object = None,
        hook_registry: object = None,
    ) -> None:
        self._client = client
        self._budget = budget
        self._selector = selector
        self._event_bus = event_bus
        self._hook_registry = hook_registry

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run generation for the current task.

        Selects model, builds prompt, calls LLM, stores output.
        """
        task = ctx.task
        model = task.preferred_model or self._selector.select(task.type)
        ctx.model = model

        system_prompt = SystemPrompt.build(task)
        # SkillOpt: prepend the injected skill document when available
        if ctx.skill_prefix:
            system_prompt = f"<skill>\n{ctx.skill_prefix}\n</skill>\n\n{system_prompt}"

        prompt_text = task.prompt
        if task.revision_context:
            prompt_text = f"{task.revision_context}\n\n{task.prompt}"

        response = await self._client.call(
            model=model,
            prompt=prompt_text,
            system=system_prompt,
            max_tokens=task.max_output_tokens,
            temperature=0.3,
            timeout=160,
            retries=2,
        )

        ctx.output = response.text
        ctx.cost_usd += response.cost_usd
        ctx.tokens_used["input"] += (
            getattr(response, "usage", None) and getattr(response.usage, "input_tokens", 0) or 0
        )
        ctx.tokens_used["output"] += (
            getattr(response, "usage", None) and getattr(response.usage, "output_tokens", 0) or 0
        )

        return ctx
