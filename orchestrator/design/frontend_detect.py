"""frontend_detect — shared helper for identifying web-frontend tasks.

Extracted from engine.py:2206-2214 so that engine.py and TasteSkillService
share a single source of truth rather than duplicating the heuristic.
"""

from __future__ import annotations

_FRONTEND_KEYWORDS = frozenset({"html", "css", "javascript", "js", "react", "vue", "svelte"})
_FRONTEND_EXTENSIONS = frozenset({".html", ".css", ".js", ".jsx", ".ts", ".tsx", ".vue", ".svelte"})
_BACKEND_KEYWORDS = frozenset({"fastapi", "flask", "django", "backend"})

# ── Animation detection signals (Emil Kowalski integration) ───────
_ANIMATION_LIBS = frozenset(
    {
        "framer-motion",
        "motion",
        "gsap",
        "lenis",
        "lottie-react",
        "@react-spring",
        "auto-animate",
        "react-spring",
        "popmotion",
        "animejs",
        "three",
        "@react-three/fiber",
        "rive-react",
    }
)

_ANIMATION_CODE_SIGNALS = frozenset(
    {
        "@keyframes",
        "animation:",
        "transition:",
        "cubic-bezier(",
        "transform:",
        "useSpring",
        "useMotion",
        "animate={{",
        "motion.",
        "ScrollTrigger",
    }
)
# ────────────────────────────────────────────────────────────────────


def is_web_frontend_task(prompt: str, target_path: str = "") -> bool:
    """Return True when a task targets web-frontend output.

    Mirrors the heuristic previously inline in engine.py:2206-2214.
    A task is frontend when its prompt or target_path mentions frontend
    keywords/extensions, AND it is not a backend-Python task.
    """
    prompt_lower = prompt.lower()
    path_lower = target_path.lower()

    is_backend = any(kw in prompt_lower for kw in _BACKEND_KEYWORDS) and ".py" in path_lower
    if is_backend:
        return False

    return any(kw in prompt_lower for kw in _FRONTEND_KEYWORDS) or any(
        path_lower.endswith(ext) for ext in _FRONTEND_EXTENSIONS
    )


def get_animation_weight(
    prompt: str = "",
    target_path: str = "",
    dependencies: list[str] | None = None,
    generated_code: str = "",
) -> float:
    """Return a 0.0–1.0 score indicating animation surface area.

    Used to decide whether to auto-inject animation-review variant,
    enable animation-specific validators, and route to animation-aware
    critique models.

    Weight factors (each capped, summed, then normalized):
      - Animation library in dependencies: +0.35 per lib (max 2 counted)
      - Animation keyword density in generated code: +0.05 per signal
        (max 10 counted)
      - Prompt mentions animation concepts: +0.1 per cue (max 3 counted)
    """
    weight = 0.0

    # 1. Dependencies
    if dependencies:
        lib_count = sum(
            1 for dep in dependencies if any(lib in dep.lower() for lib in _ANIMATION_LIBS)
        )
        weight += min(lib_count, 2) * 0.35

    # 2. Generated code signals
    code_lower = generated_code.lower()
    signal_count = sum(1 for sig in _ANIMATION_CODE_SIGNALS if sig.lower() in code_lower)
    weight += min(signal_count, 10) * 0.05

    # 3. Prompt cues
    prompt_lower = prompt.lower()
    _prompt_cues = frozenset(
        {
            "animation",
            "animate",
            "motion",
            "transition",
            "spring",
            "easing",
            "gesture",
            "drag",
            "swipe",
            "scroll-driven",
            "parallax",
            "stagger",
        }
    )
    cue_count = sum(1 for cue in _prompt_cues if cue in prompt_lower)
    weight += min(cue_count, 3) * 0.1

    return min(weight, 1.0)
