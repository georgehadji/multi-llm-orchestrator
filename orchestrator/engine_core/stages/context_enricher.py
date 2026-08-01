"""
TaskContextEnricher — Pre/post processing for task execution
=============================================================
Extracts skill-prefix building, visual-context enrichment,
anti-slop checking, and trajectory recording from engine.py.

Cluster 4 of the Strangler Fig extraction.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import Task

logger = logging.getLogger(__name__)


class TaskContextEnricher:
    """
    Pre/post processing for task execution.

    Responsibilities:
    1. Build combined skill prefix (SkillOpt + taste-skill)
    2. Enrich task prompt with image-reference visual context
    3. Check anti-slop patterns in generated output
    4. Record SkillOpt trajectories for optimizer learning

    Usage:
        enricher = TaskContextEnricher(skill_manager, taste_skill_service, client)

        task = await enricher.build_skill_prefix(task)
        task = await enricher.enrich_with_visual_context(task)
        enricher.check_anti_slop(task, ctx)
        ctx = await enricher.record_trajectory(task, ctx, background_tasks)
    """

    # Pipeline stage ordering — lower values run first
    priority: int = -50

    @classmethod
    def build_kwargs(cls, **deps):
        return {}

    def __init__(self, skill_manager=None, taste_skill_service=None, client=None):
        self._skill_manager = skill_manager
        self._taste_skill_service = taste_skill_service
        self._client = client

    async def build_prefix(self, task: Task) -> str:
        """Build combined skill prefix from SkillOpt + taste-skill."""
        prefix = ""
        if self._skill_manager is not None:
            try:
                prefix = await self._skill_manager.best_skill(task.type) or ""
            except Exception:
                pass
        if self._taste_skill_service is not None:
            try:
                taste = self._taste_skill_service.build_prefix(task)
                if taste:
                    prefix = f"{taste}\n\n{prefix}".strip()
            except Exception:
                pass
        return prefix

    async def enrich_with_visual_context(self, task: Task) -> Task:
        """Optionally enrich task prompt with image-reference visual context."""
        try:
            from ..crosscutting.config import flags as _ts_flags2
            from ..design.frontend_detect import is_web_frontend_task as _is_fe2

            if getattr(_ts_flags2, "image_reference_pipeline", False) and _is_fe2(
                task.prompt, getattr(task, "target_path", "")
            ):
                import dataclasses as _dc
                from ..design.image_reference_pipeline import ImageReferencePipeline as _IRP
                from ..design.taste_skill_loader import get_default_loader as _get_loader

                _irp = _IRP(loader=_get_loader(), client=self._client, flags=_ts_flags2)
                _visual_ctx = await _irp.build_visual_context(task)
                if _visual_ctx:
                    task = _dc.replace(task, prompt=f"{task.prompt}\n\n{_visual_ctx}")
        except Exception:
            pass
        return task

    def check_anti_slop(self, task: Task, ctx: Any) -> None:
        """Soft anti-slop WARN check for frontend tasks."""
        try:
            from ..crosscutting.config import flags as _ts_flags
            from ..design.frontend_detect import is_web_frontend_task as _is_frontend

            if (
                getattr(_ts_flags, "taste_skill_enabled", False)
                and _is_frontend(task.prompt, getattr(task, "target_path", ""))
                and getattr(ctx, "output", None)
                and not getattr(task, "target_path", "").startswith("components/")
            ):
                from ..quality.design_validators import validate_anti_slop as _anti_slop

                _slop_result = _anti_slop(ctx.output)
                if not _slop_result.passed:
                    logger.warning("taste-skill anti_slop [%s]: %s", task.id, _slop_result.details)
        except Exception:
            pass

    async def record_trajectory(self, task: Task, ctx: Any, background_tasks: set) -> None:
        """Record SkillOpt trajectory (fire-and-forget with reference storage)."""
        if self._skill_manager is None:
            return
        try:
            import time as _time
            from ..models_skill import Trajectory as _Trajectory

            _t = _Trajectory(
                task_id=task.id,
                task_type=task.type,
                prompt=task.prompt[:2000],
                output=ctx.output[:4000] if getattr(ctx, "output", None) else "",
                score=getattr(ctx, "score", 0.0),
                critique_text=(
                    getattr(ctx, "critique", "")[:1000] if getattr(ctx, "critique", None) else ""
                ),
                model_used=getattr(ctx.model, "value", "") if getattr(ctx, "model", None) else "",
                cost_usd=getattr(ctx, "cost_usd", 0.0),
                recorded_at=_time.time(),
            )
            _task_handle = asyncio.create_task(self._skill_manager.record_trajectory(_t))
            background_tasks.add(_task_handle)
            _task_handle.add_done_callback(background_tasks.discard)
        except Exception as _e:
            logger.debug("SkillOpt trajectory skipped: %s", _e)

    async def process(self, ctx: Any) -> Any:
        """No-op process method to satisfy pipeline stage execution."""
        return ctx
