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
from ...models import FALLBACK_CHAIN, TaskType
from ...prompt_builder import SystemPrompt

from ...domain.ports import VSSamplerPort
from ...operations.resilience import STAGE_RETRY_GENERATE

logger = logging.getLogger("orchestrator.engine_core.stages.generate")


class GenerateStage:
    """Generate output for a task using the primary model.

    Handles model selection (respecting preferred_model override),
    prompt construction, and API call with timeout.
    """

    # Pipeline stage ordering — lower values run first
    priority: int = 100

    @classmethod
    def build_kwargs(cls, **deps):
        return {
            "client": deps["client"],
            "budget": deps["budget"],
            "selector": deps["selector"],
            "vs_sampler": deps.get("vs_sampler"),
            "record_failure_fn": deps.get("record_failure_fn"),
        }

    def __init__(
        self,
        client: LLMClient,
        budget: Budget,
        selector: ModelSelector,
        event_bus: object = None,
        hook_registry: object = None,
        vs_sampler: VSSamplerPort | None = None,
        vs_selector: Any | None = None,
        record_failure_fn: Any | None = None,
    ) -> None:
        self._client = client
        # async (model, error) -> None; the engine late-binds its health tracker.
        self._record_failure_fn = record_failure_fn
        self._vs_sampler = vs_sampler
        self._vs_selector = vs_selector
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

        target_lang = getattr(task, "target_language", "") or ""
        system_prompt = SystemPrompt.build(
            task_type=task.type.value,
            mode=task.mode or "production",
            target_language=target_lang,
        )
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
                    if self._vs_sampler is not None:
                        candidates = await self._vs_sampler.sample(
                            prompt=prompt_text,
                            model=model,
                            cfg=cfg,
                            system_extra=system_prompt,
                            max_tokens=task.max_output_tokens,
                            timeout=160,
                        )
                    else:
                        candidates = []
                    if candidates:
                        # Use CandidateSelector if reranking is enabled
                        if self._vs_selector is not None and flags.vs_reranking_enabled:
                            best = await self._vs_selector.select(task, candidates)
                            if best is not None:
                                best_text = best.text
                            else:
                                best_text = candidates[0].text
                        else:
                            # Default: highest-probability candidate
                            best_text = candidates[0].text
                        ctx.output = best_text
                        # Cost is tracked inside VerbalizedSampler.sample() via budget.charge;
                        # VSCandidate has no cost_usd so we do not double-count here.
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

        # Standard generation path (flag off, budget model, or VS failed).
        # A provider error gets exactly one hop down FALLBACK_CHAIN; a second
        # failure is a named abort, never an unbounded walk.
        try:
            response = await self._call(model, prompt_text, system_prompt, task)
        except Exception as exc:
            await self._record_failure(model, exc)
            fallback = FALLBACK_CHAIN.get(model)
            if fallback is None or fallback == model:
                return self._abort_provider_error(ctx, model, exc)
            logger.warning(
                "Generation on %s failed for task %s (%s); retrying once on %s",
                getattr(model, "value", model),
                task.id,
                exc,
                fallback.value,
            )
            try:
                response = await self._call(fallback, prompt_text, system_prompt, task)
            except Exception as exc2:
                await self._record_failure(fallback, exc2)
                return self._abort_provider_error(ctx, fallback, exc2)
            ctx.model = fallback

        ctx.output = response.text
        ctx.cost_usd += response.cost_usd
        # Charge at the call site, the way the VS path, the evaluator and the
        # validator's revision call all do. ctx.cost_usd is only a per-task
        # tally; without this the run's dominant spend never reaches Budget and
        # max_usd caps nothing (BudgetEnforcer.record_cost, the one method that
        # would have charged it downstream, has no production callers).
        await self._budget.charge(response.cost_usd, "generation")
        # APIResponse carries the counts itself; it has no `.usage`.
        ctx.tokens_used["input"] += getattr(response, "input_tokens", 0) or 0
        ctx.tokens_used["output"] += getattr(response, "output_tokens", 0) or 0

        return ctx

    async def _call(self, model: Any, prompt_text: str, system_prompt: str, task: Any) -> Any:
        return await self._client.call(
            model=model,
            prompt=prompt_text,
            system=system_prompt,
            max_tokens=task.max_output_tokens,
            temperature=0.3,
            timeout=160,
            retries=STAGE_RETRY_GENERATE,
        )

    async def _record_failure(self, model: Any, error: Exception) -> None:
        if self._record_failure_fn is None:
            return
        try:
            await self._record_failure_fn(model, error)
        except Exception as exc:
            logger.debug("Health tracker record_failure failed: %s", exc)

    @staticmethod
    def _abort_provider_error(
        ctx: PipelineContext, model: Any, error: Exception
    ) -> PipelineContext:
        logger.error(
            "Generation failed for task %s on %s: %s",
            ctx.task.id,
            getattr(model, "value", model),
            error,
        )
        ctx.should_abort = True
        ctx.abort_reason = "provider_error"
        return ctx
