"""

Multi-LLM Orchestrator — Core Models & Types

=============================================

Author: Georgios-Chrysovalantis Chatzivantsidis

All data structures, enums, routing tables, cost tables, budget logic.

"""

from __future__ import annotations


import hashlib

import time

from dataclasses import dataclass, field

from enum import Enum

from typing import TYPE_CHECKING, Any
from .budget import Budget  # noqa: F401

# ─────────────────────────────────────────────

# Enums

# ─────────────────────────────────────────────


@dataclass
class ProviderStrategy:
    """Provider sorting strategy for OpenRouter model selection."""

    sort: str = "price"

    preferred_min_throughput: float | None = None

    preferred_max_latency: float | None = None


class ProjectType(str, Enum):
    """Project type classification for architecture routing."""

    BACKEND = "backend"

    FRONTEND = "frontend"

    FULLSTACK = "fullstack"

    CLI = "cli"

    LIBRARY = "library"

    MOBILE = "mobile"

    SCRIPT = "script"


class Language(str, Enum):
    """Programming language for architecture routing."""

    PYTHON = "python"

    TYPESCRIPT = "typescript"

    JAVASCRIPT = "javascript"

    GO = "go"

    RUST = "rust"

    DART = "dart"


class TaskType(str, Enum):

    CODE_GEN = "code_generation"

    CODE_REVIEW = "code_review"

    REASONING = "complex_reasoning"

    WRITING = "creative_writing"

    DATA_EXTRACT = "data_extraction"

    SUMMARIZE = "summarization"

    EVALUATE = "evaluation"

    IMAGE_GEN = "image_generation"


class DesignVariant(str, Enum):
    """Visual design direction for frontend code generation tasks.

    Passed as ``Task.design_variant`` to select the appropriate
    taste-skill SKILL.md prefix and critique rubric.
    """

    DEFAULT = "default"  # Anti-slop default (taste-skill v2)
    SOFT = "soft"  # Premium agency / Awwwards-tier
    MINIMALIST = "minimalist"  # Editorial / Notion-style
    BRUTALIST = "brutalist"  # Swiss / industrial mechanical
    REDESIGN = "redesign"  # Audit-first redesign of existing UI


class DesignScope(str, Enum):
    """Scope classification for frontend code generation.

    Distinguishes component-level tasks from full-page tasks.
    Used by scope_detector to route to the correct generation pipeline.
    """

    COMPONENT = "component"
    PAGE = "page"


class Genre(str, Enum):
    """Visual genre classification for design themes.

    Used by the design catalog to select appropriate themes,
    navigation patterns, and footer styles.
    """

    EDITORIAL = "editorial"
    MODERN_MINIMAL = "modern_minimal"
    ATMOSPHERIC = "atmospheric"
    PLAYFUL = "playful"
    TERMINAL = "terminal"


class Model(str, Enum):

    # ═══════════════════════════════════════════════════════

    # OPENROUTER MODELS - All models via OpenRouter

    # Format: vendor/model-name (see https://openrouter.ai/models)

    # Updated v3.0 with Xiaomi, Moonshot, StepFun, GLM models

    # ═══════════════════════════════════════════════════════

    # OpenAI Models

    GPT_4O = "openai/gpt-4o"

    GPT_4O_MINI = "openai/gpt-4o-mini"

    GPT_5 = "openai/gpt-5"

    GPT_5_MINI = "openai/gpt-5-mini"

    GPT_5_NANO = "openai/gpt-5-nano"

    GPT_5_4 = "openai/gpt-5.4"
    GPT_5_4_NANO = "openai/gpt-5.4-nano"  # $0.20/$1.25, 400K ctx, intel=38.2

    GPT_5_4_MINI = "openai/gpt-5.4-mini"

    GPT_5_4_CODEX = "openai/gpt-5.3-codex"

    O1 = "openai/o1"

    O3_MINI = "openai/o3-mini"

    O4_MINI = "openai/o4-mini"

    # Google Gemini Models

    GEMINI_FLASH = "google/gemini-3.5-flash"  # latest flash

    GEMINI_FLASH_LITE = "google/gemini-3.1-flash-lite"  # cost-effective lite

    # Anthropic Claude Models

    CLAUDE_3_5_SONNET = "anthropic/claude-sonnet-4.5"

    CLAUDE_3_OPUS = "anthropic/claude-opus-4"

    CLAUDE_3_HAIKU = "anthropic/claude-3-haiku"

    CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4-5"

    CLAUDE_SONNET_4_6 = "anthropic/claude-sonnet-4-6"

    CLAUDE_SONNET = CLAUDE_SONNET_4_6

    CLAUDE_OPUS_4_5 = "anthropic/claude-opus-4-5"

    CLAUDE_OPUS_4_6 = "anthropic/claude-opus-4-6"

    CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4-5"

    # DeepSeek Models — V4 series only

    DEEPSEEK_V4_PRO = "deepseek/deepseek-v4-pro"  # flagship reasoning + coding

    DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"  # fast + cost-effective

    # Meta LLaMA Models (OpenRouter)

    LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick"  # 400B MoE

    LLAMA_4_SCOUT = "meta-llama/llama-4-scout"  # 109B MoE

    LLAMA_3_3_70B = "meta-llama/llama-3.3-70b-instruct"  # 70B

    LLAMA_3_1_405B = "meta-llama/llama-3.3-70b-instruct"  # 405B

    # Microsoft Phi Models (OpenRouter)

    PHI_4 = "microsoft/phi-4"  # 14B

    PHI_4_REASONING = "openai/o3-mini"  # Use o3-mini for reasoning

    # Google Gemma Models (OpenRouter)

    GEMMA_3_27B = "google/gemma-3-27b-it"  # 27B
    GEMMA_4_31B = "google/gemma-4-31b-it"  # $0.12/$0.35, 262K ctx, coding=38.7

    # Nous Research Hermes (OpenRouter)

    HERMES_3_70B = "nousresearch/hermes-3-llama-3.1-70b"  # 70B fine-tuned

    # ═══════════════════════════════════════════════════════

    # XIAOMI MODELS (NEW v3.0) - GAME CHANGERS!

    # ═══════════════════════════════════════════════════════

    # NOTE: xiaomi/mimo-v2-{flash,pro,omni} were deprecated by OpenRouter (404,
    # "migrate to xiaomi/mimo-v2.5"). Repointed to the live v2.5 IDs; the legacy
    # enum names are retained as aliases so existing references keep resolving.
    XIAOMI_MIMO_V2_FLASH = "xiaomi/mimo-v2.5"  # was mimo-v2-flash (deprecated)

    XIAOMI_MIMO_V2_PRO = "xiaomi/mimo-v2.5-pro"  # was mimo-v2-pro (deprecated)

    XIAOMI_MIMO_V2_OMNI = "xiaomi/mimo-v2.5"  # was mimo-v2-omni (deprecated); v2.5 is omni-modal

    # Xiaomi Mimo V2.5 — best coding VFM
    XIAOMI_MIMO_V2_5 = "xiaomi/mimo-v2.5"  # $0.14/$0.28, 1M ctx, coding=42.1
    XIAOMI_MIMO_V2_5_PRO = "xiaomi/mimo-v2.5-pro"  # $0.44/$0.87, 1M ctx

    # ═══════════════════════════════════════════════════════

    # MOONSHOT KIMI MODELS (NEW v3.0)

    # ═══════════════════════════════════════════════════════

    MOONSHOT_KIMI_K2_7_CODE = "moonshotai/kimi-k2.7-code"  # $1.10/$4.50, 256K, code-optimized

    MOONSHOT_KIMI_K2_6 = "moonshotai/kimi-k2.6"  # $0.95/$4.00, 256K, reasoning SOTA

    MOONSHOT_KIMI_K2 = "moonshotai/kimi-k2"  # $0.50/$1.50

    # Backward compatibility aliases

    KIMI_K2_6 = MOONSHOT_KIMI_K2_6

    KIMI_K2 = MOONSHOT_KIMI_K2

    # ═══════════════════════════════════════════════════════

    # STEPFUN MODELS (NEW v3.0) - BEST VALUE!

    # ═══════════════════════════════════════════════════════

    STEPFUN_STEP_3_5_FLASH = "stepfun/step-3.5-flash"  # $0.10/$0.30, 196B MoE ⭐

    STEPFUN_STEP_3_5 = "stepfun/step-3.7-flash"  # $0.15/$0.45

    # ═══════════════════════════════════════════════════════

    # Z.AI GLM MODELS — three canonical models only

    # ═══════════════════════════════════════════════════════

    ZHIPU_GLM_5_1 = "z-ai/glm-5.1"  # balanced, 202K context
    ZHIPU_GLM_5_TURBO = "z-ai/glm-5-turbo"  # fast variant
    ZHIPU_GLM_5_2 = "z-ai/glm-5.2"  # latest model

    # ═══════════════════════════════════════════════════════

    # XAI GROK MODELS — grok-4.20 only

    # ═══════════════════════════════════════════════════════

    XAI_GROK_4_20 = "x-ai/grok-4.20"  # $2.00/$6.00, 2M context, lowest hallucination

    # ═══════════════════════════════════════════════════════

    # QWEN MODELS — two canonical models only

    # ═══════════════════════════════════════════════════════

    QWEN_3_7_MAX = "qwen/qwen3.7-max"  # flagship reasoning + coding
    QWEN_3_7_PLUS = "qwen/qwen3.7-plus"  # $0.32/$1.28, 1M ctx, coding=46.5

    QWEN_3_6_FLASH = "qwen/qwen3.6-flash"  # fast + cost-effective

    # ═══════════════════════════════════════════════════════

    # MINIMAX MODELS (NEW v3.0)

    # Note: Verified available 2026-04-01

    # ═══════════════════════════════════════════════════════

    MINIMAX_M2_7 = "minimax/minimax-m2.7"  # $0.30/$1.20, 205K, multi-agent
    # MiniMax M3 — best intelligence + agentic VFM
    MINIMAX_M3 = "minimax/minimax-m3"  # $0.30/$1.20, 1M ctx, intel=44.4, agentic=68.6

    # Backward compatibility alias

    MINIMAX_TEXT_01 = MINIMAX_M2_7

    # ═══════════════════════════════════════════════════════

    # NVIDIA MODELS (redirected via model_registry to fallback)

    # ═══════════════════════════════════════════════════════

    NVIDIA_NEMOTRON_3_SUPER = "nvidia/nemotron-3-super-120b-a12b"  # redirects → minimax-m2.7

    # InclusionAI Ring Models

    INCLUSION_RING_2_6_1T = "inclusionai/ring-2.6-1t"  # 1T params, strong reasoning

    # OpenRouter Auto-Router

    OPENROUTER_AUTO = "openrouter/auto"  # Dynamic routing

    # ═══════════════════════════════════════════════════════
    # IMAGE GENERATION MODELS
    # ═══════════════════════════════════════════════════════

    # Google Nano Banana series
    NANO_BANANA = "google/gemini-2.5-flash-image"  # $0.30/$2.50 img, 32K ctx
    NANO_BANANA_2 = "google/gemini-3.1-flash-image-preview"  # $0.50/$3 img, 131K ctx
    NANO_BANANA_PRO = "google/gemini-3-pro-image-preview"  # $2/$12 img, 65K ctx

    # OpenAI GPT Image series
    GPT_5_IMAGE = "openai/gpt-5-image"  # $10/$10 img, 400K ctx
    GPT_5_IMAGE_MINI = "openai/gpt-5-image-mini"  # $2.50/$2 img, 400K ctx
    GPT_54_IMAGE_2 = "openai/gpt-5.4-image-2"  # $8/$15 img, 272K ctx

    # Black Forest Labs FLUX series
    FLUX_2_KLEIN = "google/gemini-2.5-flash-image"  # $0.014/img, 40K ctx
    FLUX_2_MAX = "google/gemini-3-pro-image"  # $0.07/img, 46K ctx
    FLUX_2_FLEX = "google/gemini-3.1-flash-image"  # from $0.06/img, 67K ctx
    FLUX_2_PRO = "google/gemini-3-pro-image"  # $0.03/img, 46K ctx

    # Recraft V4 series
    RECRAFT_V4_UTILITY = "google/gemini-2.5-flash-image"  # $0.04/img, 65K ctx
    RECRAFT_V4_PRO = "google/gemini-3-pro-image"  # $0.25/img, 65K ctx
    RECRAFT_V4 = "google/gemini-3.1-flash-image"  # $0.04/img, 65K ctx
    RECRAFT_V4_PRO_VECTOR = "google/gemini-3-pro-image"  # $0.30/img, SVG
    RECRAFT_V4_VECTOR = "google/gemini-2.5-flash-image"  # $0.08/img, SVG
    RECRAFT_V4_1_PRO = "google/gemini-3-pro-image"  # $0.25/img, 65K ctx
    RECRAFT_V4_1 = "google/gemini-3.1-flash-image"  # $0.04/img, 65K ctx
    RECRAFT_V3 = "google/gemini-2.5-flash-image"  # $0.04/img, 65K ctx

    # Sourceful Riverflow series
    RIVERFLOW_V2_PRO = "google/gemini-3-pro-image"  # from $0.15/img, 8K ctx
    RIVERFLOW_V2_FAST = "google/gemini-2.5-flash-image"  # from $0.02/img, 8K ctx
    RIVERFLOW_V2_MAX = "google/gemini-3-pro-image"  # $0.075/img, 8K ctx
    RIVERFLOW_V2_STANDARD = "google/gemini-3.1-flash-image"  # $0.035/img, 8K ctx
    RIVERFLOW_V2_FAST_PREVIEW = "google/gemini-2.5-flash-image"  # $0.03/img, 8K ctx

    # ByteDance Seedream
    SEEDREAM_4_5 = "google/gemini-3.1-flash-image"  # $0.04/img, 4K ctx


class ProjectStatus(str, Enum):

    SUCCESS = "SUCCESS"

    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"

    COMPLETED_DEGRADED = "COMPLETED_DEGRADED"

    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"

    TIMEOUT = "TIMEOUT"

    SYSTEM_FAILURE = "SYSTEM_FAILURE"


class TaskStatus(str, Enum):

    PENDING = "pending"

    RUNNING = "running"

    COMPLETED = "completed"

    FAILED = "failed"

    DEGRADED = "degraded"


# ─────────────────────────────────────────────

# Provider detection

# ─────────────────────────────────────────────


from functools import lru_cache


@lru_cache(maxsize=256)
def get_provider(model: Model) -> str:
    """

    Get provider name for a model.



    All models now use OpenRouter exclusively.

    """

    return "openrouter"


# ─────────────────────────────────────────────

# Configuration Loaders

# ─────────────────────────────────────────────


import json
from pathlib import Path


def _load_static_config(filename: str) -> dict[str, Any]:
    """Load JSON config from orchestrator/config directory.

    Results are cached per filename via _build_* functions — this function
    itself is intentionally *not* called at module level.  Rule #2: models.py
    must not execute I/O at import time.
    """
    config_path = Path(__file__).parent / "config" / filename
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "orchestrator" / "config" / filename
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _build_cost_table() -> "dict[Model, dict[str, float]]":
    data = _load_static_config("costs.json")
    return {Model(k): v for k, v in data.items() if k in Model._value2member_map_}


def _build_routing_table() -> "dict[TaskType, list[Model]]":
    data = _load_static_config("routing.json")
    return {
        TaskType(k): [Model(m) for m in v if m in Model._value2member_map_]
        for k, v in data.items()
        if k in TaskType._value2member_map_
    }


def _build_fallback_chain() -> "dict[Model, Model]":
    data = _load_static_config("fallbacks.json")
    return {
        Model(k): Model(v)
        for k, v in data.items()
        if k in Model._value2member_map_ and v in Model._value2member_map_
    }


def _build_default_thresholds() -> "dict[TaskType, float]":
    data = _load_static_config("thresholds.json")
    return {TaskType(k): float(v) for k, v in data.items() if k in TaskType._value2member_map_}


def _build_max_output_tokens() -> "dict[TaskType, int]":
    data = _load_static_config("limits.json")
    return {TaskType(k): int(v) for k, v in data.items() if k in TaskType._value2member_map_}


# ─────────────────────────────────────────────
# Lazy-loaded tables — no disk I/O at import time (Rule #2).
# The five dicts below are populated on first access via __getattr__.
# After the first access the value is written into module globals so
# subsequent reads are O(1) dict lookups with no function call overhead.
# ─────────────────────────────────────────────

TASK_PROVIDER_STRATEGIES: "dict[TaskType, ProviderStrategy]" = {}

_LAZY_TABLES: dict[str, Any] = {
    "COST_TABLE": _build_cost_table,
    "ROUTING_TABLE": _build_routing_table,
    "FALLBACK_CHAIN": _build_fallback_chain,
    "DEFAULT_THRESHOLDS": _build_default_thresholds,
    "MAX_OUTPUT_TOKENS": _build_max_output_tokens,
}


def __getattr__(name: str) -> Any:
    """Lazy-initialise tables that require disk I/O on first access."""
    if name in _LAZY_TABLES:
        value = _LAZY_TABLES[name]()
        # Cache in module globals so subsequent `from models import X` is instant.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Model-specific max tokens limits (override MAX_OUTPUT_TOKENS)

MODEL_MAX_TOKENS: dict[Model, int] = {
    # Anthropic Claude models
    Model.CLAUDE_3_HAIKU: 4096,
    Model.CLAUDE_3_5_SONNET: 8192,
    Model.CLAUDE_3_OPUS: 4096,
    Model.CLAUDE_HAIKU_4_5: 4096,
    Model.CLAUDE_SONNET_4_5: 8192,
    Model.CLAUDE_SONNET_4_6: 8192,
    Model.CLAUDE_OPUS_4_5: 4096,
    Model.CLAUDE_OPUS_4_6: 4096,
    # Google Gemini models (high limits)
    Model.GEMINI_FLASH: 8192,
    Model.GEMINI_FLASH_LITE: 8192,
    # DeepSeek models
    Model.DEEPSEEK_V4_FLASH: 8192,
    Model.DEEPSEEK_V4_PRO: 8192,
    # Z.AI GLM models
    Model.ZHIPU_GLM_5_1: 16384,  # z-ai/glm-5.1 (balanced)
    Model.ZHIPU_GLM_5_TURBO: 16384,  # z-ai/glm-5-turbo (fast)
    Model.ZHIPU_GLM_5_2: 16384,  # z-ai/glm-5.2 (latest)
    # Meta LLaMA models
    Model.LLAMA_4_MAVERICK: 8192,
    Model.LLAMA_4_SCOUT: 8192,
    Model.LLAMA_3_3_70B: 8192,
    Model.LLAMA_3_1_405B: 8192,
    # Microsoft Phi models
    Model.PHI_4: 4096,
    Model.PHI_4_REASONING: 4096,
    # Image generation models
    Model.NANO_BANANA: 4096,
    Model.NANO_BANANA_2: 4096,
    Model.NANO_BANANA_PRO: 4096,
    Model.GPT_5_IMAGE: 4096,
    Model.GPT_5_IMAGE_MINI: 4096,
    Model.GPT_54_IMAGE_2: 4096,
    Model.FLUX_2_KLEIN: 4096,
    Model.FLUX_2_MAX: 4096,
    Model.FLUX_2_FLEX: 4096,
    Model.FLUX_2_PRO: 4096,
    Model.RECRAFT_V4_UTILITY: 4096,
    Model.RECRAFT_V4_PRO: 4096,
    Model.RECRAFT_V4: 4096,
    Model.RECRAFT_V4_PRO_VECTOR: 4096,
    Model.RECRAFT_V4_VECTOR: 4096,
    Model.RECRAFT_V4_1_PRO: 4096,
    Model.RECRAFT_V4_1: 4096,
    Model.RECRAFT_V3: 4096,
    Model.RIVERFLOW_V2_PRO: 4096,
    Model.RIVERFLOW_V2_FAST: 4096,
    Model.RIVERFLOW_V2_MAX: 4096,
    Model.RIVERFLOW_V2_STANDARD: 4096,
    Model.RIVERFLOW_V2_FAST_PREVIEW: 4096,
    Model.SEEDREAM_4_5: 4096,
    # Xiaomi Mimo V2.5
    Model.XIAOMI_MIMO_V2_5: 8192,
    Model.XIAOMI_MIMO_V2_5_PRO: 8192,
    # MiniMax M3
    Model.MINIMAX_M3: 8192,
    # Qwen 3.7 Plus
    Model.QWEN_3_7_PLUS: 8192,
    # GPT-5.4 Nano
    Model.GPT_5_4_NANO: 4096,
    # Google Gemma models
    Model.GEMMA_3_27B: 8192,
    # Nous Hermes models
    Model.HERMES_3_70B: 8192,
    # OpenAI models
    Model.GPT_4O: 8192,
    Model.GPT_4O_MINI: 4096,
    Model.GPT_5: 8192,
    Model.GPT_5_MINI: 4096,
    Model.GPT_5_NANO: 4096,
    Model.O1: 4096,
    Model.O3_MINI: 4096,
    Model.O4_MINI: 4096,
}


def get_max_iterations(task_type: TaskType) -> int:

    if task_type == TaskType.CODE_GEN:

        return 3  # heavy — needs generate + critique + revise

    if task_type == TaskType.CODE_REVIEW:

        return 4  # extra iteration: reviews depend on context quality and

        # often need a full second pass after critique improves framing

    if task_type == TaskType.REASONING:

        return 3

    return 2  # light


# ─────────────────────────────────────────────

from .budget import Budget  # noqa: F401

# Data classes

# ─────────────────────────────────────────────


@dataclass
class Task:

    id: str

    type: TaskType

    prompt: str

    context: str = ""

    dependencies: list[str] = field(default_factory=list)

    acceptance_threshold: float = 0.85

    max_iterations: int = 3

    max_output_tokens: int = 1500

    status: TaskStatus = TaskStatus.PENDING

    hard_validators: list[str] = field(default_factory=list)

    # App Builder fields (Improvement 8)

    target_path: str = ""  # e.g. "src/routes/auth.py"

    module_name: str = ""  # e.g. "src.routes.auth"

    tech_context: str = ""  # brief note on tech stack for this file

    preferred_model: object | None = None  # Model | None

    revision_context: str = ""

    mode: str = ""  # "" means STANDARD

    design_variant: "DesignVariant | None" = None  # taste-skill aesthetic direction

    # NOTE: type-specific defaults (thresholds, iterations, token limits) are

    # set by TaskFactory.create() in orchestrator/task_factory.py — not here.

    # models.py is pure data; no behavioral methods belong in dataclasses.


@dataclass
class TaskResult:

    task_id: str

    output: str

    score: float

    model_used: Model

    reviewer_model: Model | None = None

    tokens_used: dict[str, int] = field(default_factory=lambda: {"input": 0, "output": 0})

    iterations: int = 0

    cost_usd: float = 0.0

    status: TaskStatus = TaskStatus.COMPLETED

    critique: str = ""

    deterministic_check_passed: bool = True

    degraded_fallback_count: int = 0

    attempt_history: list[AttemptRecord] = field(default_factory=list)

    preflight_result: PreflightResult | None = None

    preflight_passed: bool = True

    task_type: str = ""

    # TDD artifacts (populated when TDD-first generation is used)

    test_files: dict = field(default_factory=dict)

    tests_passed: int = 0

    tests_total: int = 0

    metadata: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        """Backward compatibility: success = completed status and score > 0."""

        return self.status == TaskStatus.COMPLETED and self.score > 0.5


@dataclass
class AttemptRecord:
    """Records one failed iteration attempt so the next retry has failure context."""

    attempt_num: int  # 1-based

    model_used: str  # Model.value — str for easy serialization

    output_snippet: str  # first 200 chars of output

    failure_reason: str  # human-readable description

    validators_failed: list[str] = field(default_factory=list)


@dataclass
class ProjectState:
    """Full serializable state for resume capability."""

    project_description: str

    success_criteria: str

    budget: Budget | None  # Budget object or None

    tasks: dict[str, Task] = field(default_factory=dict)

    results: dict[str, TaskResult] = field(default_factory=dict)

    api_health: dict[str, bool] = field(default_factory=dict)

    status: ProjectStatus = ProjectStatus.PARTIAL_SUCCESS

    execution_order: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────

# JobSpec — App Builder job specification

# ─────────────────────────────────────────────


@dataclass
class JobSpec:
    """

    App Builder job specification.



    A lightweight spec for the App Builder pipeline, separate from the

    policy-oriented JobSpec in policy.py.



    Fields

    ------

    description      : Human-readable description of the app to build.

    success_criteria : Acceptance criteria for the build.

    app_type         : Optional override for the app type (e.g. "fastapi",

                       "cli", "library").  Empty string means auto-detect.

    docker           : Whether to run Docker-based verification.

    output_dir       : Where to write the generated app.  Empty string means

                       auto-generate a temp directory.

    """

    description: str

    success_criteria: str

    app_type: str = ""

    docker: bool = False

    output_dir: str = ""


# ─────────────────────────────────────────────

# Utilities

# ─────────────────────────────────────────────


def prompt_hash(
    model: str, prompt: str, max_tokens: int, system: str = "", temperature: float = 0.3
) -> str:

    payload = f"{model}||{system}||{prompt}||{max_tokens}||{temperature}"

    return hashlib.sha256(payload.encode()).hexdigest()


def estimate_cost(model: Model, input_tokens: int, output_tokens: int) -> float:

    costs = COST_TABLE.get(model, {"input": 5.0, "output": 20.0})

    return (input_tokens * costs["input"] + output_tokens * costs["output"]) / 1_000_000





# ─────────────────────────────────────────────

# Verbalized Sampling types (CodeWhale Phase 0)

# ─────────────────────────────────────────────


def vs_variant_for(model: Model, default_k: int = 5) -> VSConfig | None:
    """Choose VS variant based on model cost tier (Phase 5).

    PREMIUM models (named "pro", "opus", "o1", etc.) → full VS.
    STANDARD models (most others) → standard VS.
    BUDGET models (named "flash", "mini", etc.) → None (skip VS).

    Uses model name heuristics since all models route through OpenRouter.
    """
    name = model.value.lower()
    _budget = ("flash", "mini", "nano", "lite", "scout", "haiku", "tiny", "gemma", "phi")
    for pat in _budget:
        if pat in name:
            return None
    _premium = ("pro", "opus", "o1", "o3", "k2", "k3", "maverick", "sonnet-4-5", "max", "turbo")
    return VSConfig(k=default_k, temperature=0.9, fmt=ProbabilityFormat.EXPLICIT, top_p=0.95)


class ProbabilityFormat(str, Enum):
    """Format for verbalized probability in VS prompts.

    EXPLICIT — "the estimated probability from 0.0 to 1.0 of this response
               given the input prompt (relative to the full distribution)"
               Best for VS-Standard (paper H.3).

    CONFIDENCE — "the normalized likelihood score between 0.0 and 1.0 that
                  indicates how representative or typical this response is"
                  Best for VS-Multi (paper H.3).
    """

    EXPLICIT = "explicit"
    CONFIDENCE = "confidence"


@dataclass(frozen=True)
class VSConfig:
    """Configuration for a single Verbalized Sampling call.

    k: Number of candidates to generate (paper default 5; H.1 shows
       diminishing returns above).
    probability_threshold: None = no threshold. 0.10 = sample from the
       tail (probability < 0.10). Paper §Tail.
    fmt: Probability format string to inject (see ProbabilityFormat).
    temperature: Sampling temperature (VS is orthogonal; paper §5.3).
    top_p: Nucleus sampling (paper H.2 optimum 0.95).
    """

    k: int = 5
    probability_threshold: float | None = None
    fmt: ProbabilityFormat = ProbabilityFormat.EXPLICIT
    temperature: float = 0.9
    top_p: float = 0.95


# ─────────────────────────────────────────────

# App Store Assets

# ─────────────────────────────────────────────


@dataclass
class ProjectSpec:
    """

    Project specification for App Store asset generation.



    Attributes:

        name: Project/app name

        description: App description

        criteria: Success criteria

    """

    name: str

    description: str = ""

    criteria: str = ""
