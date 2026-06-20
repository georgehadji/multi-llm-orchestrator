"""
Supervisor intake — normalise human/agent directives into engine job args.

Both channels collapse to a single ``Directive`` shape; this module turns that
shape into the arguments expected by ``Orchestrator.run_project_streaming()``.

Author: Georgios-Chrysovalantis Chatzivantsidis
"""

from __future__ import annotations

import logging
import re

from .models import Directive, JobArgs

logger = logging.getLogger("orchestrator.supervisor.intake")

_DEFAULT_CRITERIA = "All tasks complete, validated, and production-ready."


def normalize(directive: Directive) -> JobArgs:
    """Convert a directive into engine-ready job arguments.

    Phase 1 is pass-through with lightweight budget extraction.  Phase 3 will
    optionally route free-text directives through ``ConversationAgent`` for a
    clarification loop.
    """
    text = (directive.text or "").strip()
    criteria = (directive.criteria or "").strip() or _DEFAULT_CRITERIA
    project_id = (directive.project_id or "").strip()

    budget = directive.budget
    if budget is None:
        budget = _extract_budget(text)

    return JobArgs(
        project_description=text,
        success_criteria=criteria,
        budget=budget,
        project_id=project_id,
    )


def _extract_budget(text: str) -> float | None:
    """Look for a sane dollar amount in free text.

    Requires an explicit currency marker (``$5``, ``5 USD``, ``5 dollars``) so
    that an arbitrary number in the directive (e.g. "build a 5-page site") is
    never mistaken for a budget.
    """
    match = re.search(r"\$\s*(\d+(?:\.\d+)?)", text)
    if match is None:
        match = re.search(r"\b(\d+(?:\.\d+)?)\s*(?:USD|usd|dollars?)\b", text)
    if match:
        value = float(match.group(1))
        if 0 < value < 10000:
            return value
    return None
