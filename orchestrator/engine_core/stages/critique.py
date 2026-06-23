"""
CritiqueStage — Cross-model critique/review
==============================================
CodeWhale Phase 1: LSP diagnostics are injected into the critique prompt so the
reviewer model sees deterministic language-server errors and warnings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..pipeline import PipelineContext
from ...domain.ports import LLMClient
from ...crosscutting.config import flags
from ...models import ProbabilityFormat, TaskType, VSConfig

from ...domain.ports import VSSamplerPort

if TYPE_CHECKING:
    from ...domain.ports import LSPValidatorPort

logger = logging.getLogger("orchestrator.engine_core.stages.critique")


class CritiqueStage:
    """Run cross-model critique on generated output.

    Uses a different (often higher-quality) model to review the
    generated output and produce structured feedback.

    When an LSPValidatorPort is provided, deterministic language-server
    diagnostics are injected into the critique prompt so the reviewer
    model sees concrete errors and warnings alongside generated code.
    """

    def __init__(
        self,
        client: LLMClient,
        get_reviewer_fn: object = None,
        lsp_validator: LSPValidatorPort | None = None,
        vs_sampler: VSSamplerPort | None = None,
    ) -> None:
        self._client = client
        self._get_reviewer_fn = get_reviewer_fn
        self._lsp_validator = lsp_validator
        self._vs_sampler = vs_sampler

    async def process(self, ctx: PipelineContext) -> PipelineContext:
        """Run critique if a reviewer model is available."""
        if ctx.model is None:
            return ctx

        reviewer = self._resolve_reviewer(ctx)
        if reviewer is None or reviewer == ctx.model:
            return ctx

        ctx.reviewer_model = reviewer

        # Build critique prompt (enriched with LSP diagnostics if available)
        critique_prompt = await self._build_critique_prompt(ctx)

        try:
            if flags.vs_code_review and ctx.task and ctx.task.type == TaskType.CODE_GEN:
                if self._vs_sampler is not None:
                    candidates = await self._vs_sampler.sample(
                        prompt=critique_prompt,
                        model=reviewer,
                        cfg=VSConfig(k=3, temperature=0.2, fmt=ProbabilityFormat.CONFIDENCE),
                        system_extra="You are a code reviewer. Generate 3 independent review "
                        "hypotheses. Each must explore a different angle "
                        "(correctness, performance, security, style, edge cases). "
                        "Be specific and constructive.",
                        max_tokens=2048,
                        timeout=60,
                    )
                if candidates:
                    ctx.critique = "\n\n".join(
                        f"## Review {i+1} (confidence: {c.probability:.0%})\n{c.text[:1500]}"
                        for i, c in enumerate(candidates)
                    )
                else:
                    raise RuntimeError("VS returned no candidates")
            else:
                response = await self._client.call(
                    model=reviewer,
                    prompt=critique_prompt,
                    system=(
                        "You are a code reviewer. Provide constructive, "
                        "specific feedback. Be concise."
                    ),
                    max_tokens=2048,
                    temperature=0.3,
                    timeout=60,
                    retries=1,
                )
                ctx.critique = response.text[:2000]
        except Exception as e:
            logger.warning("Critique failed for task %s: %s", ctx.task.id, e)

        return ctx

    async def _build_critique_prompt(self, ctx: PipelineContext) -> str:
        """Build the critique prompt, enriched with LSP diagnostics if available."""
        base = (
            "Review the following generated code for correctness, "
            "performance, security, and style:\n\n"
            f"{ctx.output[:4000]}"
        )

        if self._lsp_validator is not None and ctx.task is not None:
            try:
                language = self._detect_language(ctx.task)
                diags = await self._lsp_validator.validate(ctx.output, language)
                if diags:
                    summary = self._build_diag_summary(diags)
                    if summary:
                        base += (
                            "\n\n## LSP Validation Results\n\n"
                            + summary
                            + "\n\n**Note:** Address all LSP errors in your review. "
                            "These are deterministic findings from the language server.\n"
                        )
                        logger.info(
                            "LSP: %d diagnostics injected into critique for %s",
                            len(diags),
                            getattr(ctx.task, "id", "?"),
                        )
            except Exception as exc:
                logger.debug("LSP validation in CritiqueStage: %s", exc)

        return base

    @staticmethod
    def _build_diag_summary(diags: list) -> str:
        """Build a concise diagnostic summary string."""
        errors = [d for d in diags if getattr(d, "severity", "") == "error"]
        warnings = [d for d in diags if getattr(d, "severity", "") == "warning"]

        parts: list[str] = []
        if errors:
            parts.append(f"### {len(errors)} Error(s)")
            for e in errors[:10]:
                code = f" ({e.code})" if getattr(e, "code", "") else ""
                parts.append(f"- L{e.line}:{e.column} {e.message}{code}")
            if len(errors) > 10:
                parts.append(f"- ... and {len(errors) - 10} more errors")

        if warnings:
            parts.append(f"### {len(warnings)} Warning(s)")
            for w in warnings[:10]:
                code = f" ({w.code})" if getattr(w, "code", "") else ""
                parts.append(f"- L{w.line}:{w.column} {w.message}{code}")
            if len(warnings) > 10:
                parts.append(f"- ... and {len(warnings) - 10} more warnings")

        return "\n".join(parts)

    @staticmethod
    def _detect_language(task) -> str:
        """Detect programming language from task metadata."""
        if hasattr(task, "language") and task.language:
            return task.language
        if hasattr(task, "target_path") and task.target_path:
            ext = Path(task.target_path).suffix.lower()
            ext_map = {
                ".py": "python",
                ".ts": "typescript",
                ".tsx": "typescript",
                ".js": "typescript",
                ".jsx": "typescript",
            }
            return ext_map.get(ext, "python")
        return "python"

    def _resolve_reviewer(self, ctx: PipelineContext):
        """Resolve the reviewer model."""
        if self._get_reviewer_fn is not None:
            return self._get_reviewer_fn(ctx.model, ctx.task.type)
        return None
