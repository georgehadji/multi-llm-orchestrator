"""
Scope Detector — Component vs Page flow routing.
===============================================
Author: Georgios-Chrysovalantis Chatzivantsidis

Detects whether a frontend brief is component-scope or page-scope
and applies different rules accordingly.

Source: Hallmark design skill (component-scope section)
"""

from __future__ import annotations

import re
from typing import Literal

from orchestrator.models import DesignScope

_COMPONENT_SIGNALS: frozenset[str] = frozenset({
    "button", "input", "card", "modal", "dropdown", "tooltip",
    "select", "checkbox", "switch", "tab", "chip", "badge", "banner",
    "snackbar", "popover", "slider", "date picker", "avatar",
    "toggle", "radio", "textarea", "form", "field", "pill",
})

_SCOPE_PHRASES: frozenset[str] = frozenset({
    "just the", "only the", "this one element", "a single",
    "one component", "single component", "only component",
})

_SINGLE_FILE_EXTS: tuple[str, ...] = (
    ".tsx", ".jsx", ".vue", ".svelte", ".css", ".scss", ".less",
)


def detect_scope(prompt: str, target_path: str = "") -> DesignScope:
    """Detect whether *prompt* + *target_path* describes a component or a page.

    Returns ``DesignScope.COMPONENT`` when at least two component signals fire,
    otherwise ``DesignScope.PAGE``.
    """
    lower = prompt.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    signals = 0

    # Signal 1: names a single UI element
    if words & _COMPONENT_SIGNALS:
        # If the prompt is short (≤ 30 words) and names a component → strong signal
        if len(words) < 30:
            signals += 2
        else:
            signals += 1

    # Signal 2: target is a single component file
    if target_path:
        tp = target_path.replace("\\", "/")
        if any(tp.endswith(ext) for ext in _SINGLE_FILE_EXTS):
            parts = tp.split("/")
            # If the file sits directly in components/ or has no subdir → component
            if len(parts) <= 2 or "component" in tp.lower():
                signals += 2
            else:
                signals += 1

    # Signal 3: user explicitly scopes to one element
    if any(phrase in lower for phrase in _SCOPE_PHRASES):
        signals += 2

    # Signal 4: brief is very short and mentions a component
    if len(words) < 15 and (words & _COMPONENT_SIGNALS):
        signals += 1

    return DesignScope.COMPONENT if signals >= 2 else DesignScope.PAGE
