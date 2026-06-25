"""
Cache warmup — proactive cache warming before parallel task execution.
======================================================================
Prevents cache miss storms when multiple parallel tasks start simultaneously.

Part of Application Layer. Extracted from engine.py._warm_cache_for_level.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any, Dict, List

logger = logging.getLogger(__name__)


async def warm_cache_for_level(
    context_service: Any,
    results: dict[str, Any],
    client: Any,
) -> None:
    """Warm the LLM response cache before executing a parallel batch.

    Builds system prompt + project context from the given services and
    issues a single cache-warming call.  This is a best-effort optimisation —
    failures are logged but never propagated.

    Args:
        context_service: Service that builds system prompts and project context.
        results: Current task results (used for project context construction).
        client: LLM client with cache integration.
    """
    try:
        system_prompt = context_service.build_system_prompt("")
        project_context = context_service.build_project_context(results)

        if not system_prompt and not project_context:
            logger.debug("Cache warming skipped: no system prompt or context")
            return

        from ..operations.cache_warmup import warm_prompt_cache

        await warm_prompt_cache(
            system_prompt=system_prompt,
            project_context=project_context,
            client=client,
        )
        logger.info("Cache warmed for parallel execution level")
    except Exception as e:
        logger.warning(f"Cache warming failed (non-critical): {e}")
