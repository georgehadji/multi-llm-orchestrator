"""
GenerateStage — LLM code/text generation
==========================================
"""

from __future__ import annotations

import logging

from ..pipeline import PipelineContext
from ...domain.ports import LLMClient
from ...budget import Budget
from ...crosscutting.config import flags
from ...model_selector import ModelSelector
from ...models import TaskType
from ...prompt_builder import SystemPrompt

# Lazy import for VerbalizedSampler (avoids circular dep at module level)
_VS_SAMPLER = None
_VS_IMPORT_LOCK = __import__("threading").Lock()


def _get_vs_sampler(client):
    global _VS_SAMPLER
    if _VS_SAMPLER is None:
        with _VS_IMPORT_LOCK:
            if _VS_SAMPLER is None:
                from ...application.verbalized_sampling import VerbalizedSampler as _VS

                _VS_SAMPLER = _VS
    return _VS_SAMPLER(client=client)


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

        # ── CodeWhale Phase 6: VS-first generation ──────────────────────────
        # When vs_generate flag is on, model tier allows it, and we're not
        # already in a retry, use VerbalizedSampler to generate k diverse
        # candidates, then pick the highest-probability one.
        if (
            flags.vs_generate
            and task.type == TaskType.CODE_GEN
            and not ctx.should_abort
            and "[VS_RETRY_ESCAPE]" not in prompt_text
        ):
            try:
                from ...models import vs_variant_for

                cfg = vs_variant_for(model, default_k=flags.vs_k)
                if cfg is not None:
                    sampler = _get_vs_sampler(self._client)
                    candidates = await sampler.sample(
                        prompt=prompt_text,
                        model=model,
                        cfg=cfg,
                        system_extra=system_prompt,
                        max_tokens=task.max_output_tokens,
                        timeout=160,
                    )
                    if candidates:
                        # Pick the text of the first (highest-prob) candidate
                        best_text = candidates[0].text
                        ctx.output = best_text
                        ctx.cost_usd += sum(getattr(c, "cost_usd", 0) for c in candidates[:1])
                        # Use existing token tracking fallback for VS calls
                        ctx.tokens_used["output"] += len(best_text.split())
                        logger.info(
                            "VS-first: task %s, %d candidates, " "top prob=%.3f, output_len=%d",
                            task.id,
                            len(candidates),
                            candidates[0].probability,
                            len(best_text),
                        )
                        return ctx
                    logger.info(
                        "VS-first returned no candidates for %s — "
                        "falling back to standard generation",
                        task.id,
                    )
            except Exception as exc:
                logger.warning("VS-first generation failed for %s: %s", task.id, exc)

        # Standard generation path (flag off, budget model, or VS failed)
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
