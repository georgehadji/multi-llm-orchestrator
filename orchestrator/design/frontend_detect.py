"""frontend_detect — shared helper for identifying web-frontend tasks.

Extracted from engine.py:2206-2214 so that engine.py and TasteSkillService
share a single source of truth rather than duplicating the heuristic.
"""

from __future__ import annotations

_FRONTEND_KEYWORDS = frozenset({"html", "css", "javascript", "js", "react", "vue", "svelte"})
_FRONTEND_EXTENSIONS = frozenset({".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte"})
_BACKEND_KEYWORDS = frozenset({"fastapi", "flask", "django", "backend"})


def is_web_frontend_task(prompt: str, target_path: str = "") -> bool:
    """Return True when a task targets web-frontend output.

    Mirrors the heuristic previously inline in engine.py:2206-2214.
    A task is frontend when its prompt or target_path mentions frontend
    keywords/extensions, AND it is not a backend-Python task.
    """
    prompt_lower = prompt.lower()
    path_lower = target_path.lower()

    is_backend = (
        any(kw in prompt_lower for kw in _BACKEND_KEYWORDS)
        and ".py" in path_lower
    )
    if is_backend:
        return False

    return (
        any(kw in prompt_lower for kw in _FRONTEND_KEYWORDS)
        or any(path_lower.endswith(ext) for ext in _FRONTEND_EXTENSIONS)
    )
