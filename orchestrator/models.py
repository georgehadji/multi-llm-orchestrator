"""
Unified Model Registry
========================
Author: Georgios-Chrysovalantis Chatzivantsidis

Centralised enum for all LLM models used in the orchestrator.
Provides a single source of truth for model names, validation,
and metadata lookups (cost, provider, context size).

Usage:
    from orchestrator.models import Model, COST_TABLE, get_provider

    # Access model enum
    model = Model.GPT_4O

    # Look up cost
    cost = COST_TABLE[model]
    print(f"Input cost: ${cost['input']}/M tokens")

    # Get provider
    provider = get_provider(model) # "openai"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, TypedDict


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

    VIDEO_GEN = "video_generation"

    MECHANISM_RESEARCH = "mechanism_research"


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
    # ── Emil Kowalski animation skills (skills-main) ──────────────
    ANIMATION_VOCABULARY = "animation_vocabulary"  # Precise motion terminology
    APPLE_FLUID = "apple_fluid"  # Apple WWDC fluid-interface principles
    ANIMATION_REVIEW = "animation_review"  # 10-standard animation review bar
    # ───────────────────────────────────────────────────────────────


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


class Model(Enum):
    """Unified enum for all supported LLM models."""

    # ═══════════════════════════════════════════════════════
    # Free Tier (OpenRouter) — DEPRECATED, no longer routable
    # ═══════════════════════════════════════════════════════
    _QWEN_CODER_FREE = "qwen/qwen3-coder:free"  # deprecated
    _GPT_OSS_120B_FREE = "openai/gpt-oss-120b:free"  # deprecated
    _QWEN_NEXT_80B_FREE = "qwen/qwen3-next-80b-a3b-instruct:free"  # deprecated
    _LLAMA_3_3_70B_FREE = "meta-llama/llama-3.3-70b-instruct:free"  # deprecated
    _NEMOTRON_3_ULTRA_FREE = "nvidia/nemotron-3-ultra-550b-a55b:free"  # deprecated
    _NEMOTRON_3_SUPER_FREE = "nvidia/nemotron-3-super-120b-a12b:free"  # deprecated
    _NEMOTRON_NANO_9B_FREE = "nvidia/nemotron-nano-9b-v2:free"  # deprecated

    # ═══════════════════════════════════════════════════════
    # Open-Source Models (Tier 1 - Top Performers)
    # ═══════════════════════════════════════════════════════
    GPT_OSS_120B = "openai/gpt-oss-120b"
    GPT_OSS_20B = "openai/gpt-oss-20b"
    QWEN_NEXT_80B = "qwen/qwen3-next-80b-a3b-instruct"
    GLM_4_7_FLASH = "z-ai/glm-4.7-flash"
    MINIMAX_M2_5 = "minimax/minimax-m2.5"
    DEVSTRAL_2512 = "mistralai/devstral-2512"
    MISTRAL_LARGE_2512 = "mistralai/mistral-large-2512"
    GLM_5 = "z-ai/glm-5"

    # ═══════════════════════════════════════════════════════
    # Proprietary Models (Tier 1 - Flagship)
    # ═══════════════════════════════════════════════════════
    # Google
    GEMINI_3_FLASH_PREVIEW = "google/gemini-3-flash-preview"
    GEMINI_3_1_PRO_PREVIEW = "google/gemini-3.1-pro-preview"

    # OpenAI
    GPT_5_2 = "openai/gpt-5.2"
    GPT_4O = "openai/gpt-4o"
    GPT_4O_MINI = "openai/gpt-4o-mini"
    GPT_5 = "openai/gpt-5"
    GPT_5_MINI = "openai/gpt-5-mini"
    GPT_5_NANO = "openai/gpt-5-nano"
    O1 = "openai/o1"
    O3_MINI = "openai/o3-mini"
    O4_MINI = "openai/o4-mini"

    # Google
    GEMINI_FLASH = "google/gemini-3.5-flash"
    GEMINI_FLASH_LITE = "google/gemini-3.1-flash-lite"
    GEMINI_FLASH_LITE_IMAGE = "google/gemini-3.1-flash-lite-image"

    # Anthropic
    CLAUDE_FABLE_5 = "anthropic/claude-fable-5"
    CLAUDE_SONNET_4_5 = "anthropic/claude-sonnet-4-5"
    CLAUDE_SONNET_5 = "anthropic/claude-sonnet-5"
    CLAUDE_OPUS_4_5 = "anthropic/claude-opus-4-5"
    CLAUDE_OPUS_4_8 = "anthropic/claude-opus-4-8"
    CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4-5"

    # DeepSeek
    DEEPSEEK_V4_PRO = "deepseek/deepseek-v4-pro"
    DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"

    # Meta
    LLAMA_4_MAVERICK = "meta-llama/llama-4-maverick"
    LLAMA_4_SCOUT = "meta-llama/llama-4-scout"
    LLAMA_3_3_70B = "meta-llama/llama-3.3-70b-instruct"
    LLAMA_3_1_405B = "meta-llama/llama-3.1-405b-instruct"

    # Microsoft
    PHI_4 = "microsoft/phi-4"

    # Google (Open Source)
    GEMMA_3_27B = "google/gemma-3-27b-it"

    # Other High-Performers
    HERMES_3_LLAMA_3_1_70B = "nousresearch/hermes-3-llama-3.1-70b"
    MOONSHOT_KIMI_K2_7_CODE = "moonshotai/kimi-k2.7-code"
    MOONSHOT_KIMI_K2_6 = "moonshotai/kimi-k2.6"
    MOONSHOT_KIMI_K2 = "moonshotai/kimi-k2"
    STEPFUN_STEP_3_5_FLASH = "stepfun/step-3.5-flash"
    ZHIPU_GLM_5_2 = "z-ai/glm-5.2"
    ZHIPU_GLM_5_TURBO = "z-ai/glm-5-turbo"
    XAI_GROK_4_5 = "x-ai/grok-4.5"
    QWEN_3_7_MAX = "qwen/qwen3.7-max"
    QWEN_3_6_FLASH = "openai/gpt-4o-mini"
    MINIMAX_M2_7 = "minimax/minimax-m2.7"
    XIAOMI_MIMO_V2_FLASH = "xiaomi/mimo-v2.5"
    XIAOMI_MIMO_V2_5 = "xiaomi/mimo-v2.5"
    XIAOMI_MIMO_V2_5_PRO = "xiaomi/mimo-v2.5-pro"

    # ═══════════════════════════════════════════════════════
    # Proprietary Models (Tier 2 - Next-Gen & Experimental)
    # ═══════════════════════════════════════════════════════
    GPT_5_4 = "openai/gpt-5.4"
    GPT_5_4_MINI = "openai/gpt-5.4-mini"
    GPT_5_4_CODEX = "openai/gpt-5.3-codex"
    RING_2_6_1T = "inclusionai/ring-2.6-1t"
    MINIMAX_M3 = "minimax/minimax-m3"
    QWEN_3_7_PLUS = "qwen/qwen3.7-plus"
    GPT_5_4_NANO = "openai/gpt-5.4-nano"
    GEMMA_4_31B = "google/gemma-4-31b-it"
    STEPFUN_STEP_3_7_FLASH = "stepfun/step-3.7-flash"
    NEMOTRON_3_SUPER_120B = "nvidia/nemotron-3-super-120b-a12b"
    GPT_5_CODEX = "openai/gpt-5-codex"
    GPT_5_4_PRO = "openai/gpt-5.4-pro"
    XAI_GROK_4_MINI = "x-ai/grok-4-mini"
    QWEN_3_CODER = "qwen/qwen3-coder"
    QWEN_3_CODER_NEXT = "qwen/qwen3-coder-next"
    QWEN_3_5_397B = "qwen/qwen3.5-397b-a17b"
    QWEN_3_235B_THINKING = "qwen/qwen3-235b-a22b-thinking-2507"
    QWEN_3_MAX_THINKING = "qwen/qwen3-max-thinking"
    DEEPSEEK_R1 = "deepseek/deepseek-r1"
    DEEPSEEK_V3_2 = "deepseek/deepseek-v3.2"
    DEEPSEEK_V3_1_TERMINUS = "deepseek/deepseek-v3.1-terminus"
    MOONSHOT_KIMI_K2_5 = "moonshotai/kimi-k2.5"
    MOONSHOT_KIMI_K2_THINKING = "moonshotai/kimi-k2-thinking"
    MOONSHOT_KIMI_K2_0905 = "moonshotai/kimi-k2-0905"
    XAI_GROK_4_3 = "x-ai/grok-4.3"
    QWEN_3_MAX = "qwen/qwen3-max"
    QWEN_3_235B = "qwen/qwen3-235b-a22b"
    QWEN_3_CODER_FLASH = "qwen/qwen3-coder-flash"
    QWEN_3_CODER_PLUS = "qwen/qwen3-coder-plus"
    GPT_5_5 = "openai/gpt-5.5"
    GPT_5_5_PRO = "openai/gpt-5.5-pro"
    GPT_LATEST = "~openai/gpt-latest"
    GPT_MINI_LATEST = "~openai/gpt-mini-latest"
    XAI_GROK_BUILD_0_1 = "x-ai/grok-build-0.1"

    # ═══════════════════════════════════════════════════════
    # Alias Models
    # ═══════════════════════════════════════════════════════
    CLAUDE_OPUS = CLAUDE_OPUS_4_8
    CLAUDE_SONNET = CLAUDE_SONNET_5
    CLAUDE_HAIKU = CLAUDE_HAIKU_4_5
    LLAMA_3_1_405B_INSTRUCT = LLAMA_3_1_405B  # Alias for clarity
    OPENROUTER_AUTO = "openrouter/auto"  # Dynamic routing

    # ═══════════════════════════════════════════════════════
    # Image Generation Models
    # ═══════════════════════════════════════════════════════
    # Google
    GEMINI_3_1_FLASH_IMAGE_PREVIEW = "google/gemini-3.1-flash-image-preview"
    GEMINI_2_5_FLASH_IMAGE = "google/gemini-2.5-flash-image"
    GEMINI_3_PRO_IMAGE_PREVIEW = "google/gemini-3-pro-image-preview"

    # OpenAI
    GPT_5_IMAGE = "openai/gpt-5-image"
    GPT_5_IMAGE_MINI = "openai/gpt-5-image-mini"
    GPT_5_4_IMAGE_2 = "openai/gpt-5.4-image-2"

    # Black Forest Labs
    FLUX_2_KLEIN = "black-forest-labs/flux.2-klein-4b"
    FLUX_2_MAX = "black-forest-labs/flux.2-max"
    FLUX_2_FLEX = "black-forest-labs/flux.2-flex"
    FLUX_2_PRO = "black-forest-labs/flux.2-pro"

    # Recraft
    RECRAFT_V4_1_UTILITY = "recraft/recraft-v4.1-utility"
    RECRAFT_V4_1_PRO = "recraft/recraft-v4.1-pro"
    RECRAFT_V4_1 = "recraft/recraft-v4.1"
    RECRAFT_V4_PRO_VECTOR = "recraft/recraft-v4-pro-vector"
    RECRAFT_V4_VECTOR = "recraft/recraft-v4-vector"
    RECRAFT_V4_PRO = "recraft/recraft-v4-pro"
    RECRAFT_V4 = "recraft/recraft-v4"
    RECRAFT_V3 = "recraft/recraft-v3"

    # Sourceful
    RIVERFLOW_V2_PRO = "sourceful/riverflow-v2-pro"
    RIVERFLOW_V2_FAST = "sourceful/riverflow-v2-fast"
    RIVERFLOW_V2_MAX_PREVIEW = "sourceful/riverflow-v2-max-preview"
    RIVERFLOW_V2_STANDARD_PREVIEW = "sourceful/riverflow-v2-standard-preview"
    RIVERFLOW_V2_FAST_PREVIEW = "sourceful/riverflow-v2-fast-preview"

    # Bytedance
    SEEDREAM_4_5 = "bytedance-seed/seedream-4.5"

    # ═══════════════════════════════════════════════════════
    # Video Generation Models
    # ═══════════════════════════════════════════════════════
    SORA_2_PRO = "openai/sora-2-pro"
    VEO_3_1 = "google/veo-3.1"
    VEO_3_1_FAST = "google/veo-3.1-fast"
    VEO_3_1_LITE = "google/veo-3.1-lite"
    KLING_V3_0_PRO = "kwaivgi/kling-v3.0-pro"
    KLING_V3_0_STD = "kwaivgi/kling-v3.0-std"
    KLING_VIDEO_O1 = "kwaivgi/kling-video-o1"
    HAILUO_2_3 = "minimax/hailuo-2.3"
    SEEDANCE_2_0 = "bytedance/seedance-2.0"
    SEEDANCE_2_0_FAST = "bytedance/seedance-2.0-fast"
    SEEDANCE_1_5_PRO = "bytedance/seedance-1-5-pro"
    WAN_2_7 = "alibaba/wan-2.7"
    WAN_2_6 = "alibaba/wan-2.6"
    GROK_IMAGINE_VIDEO = "x-ai/grok-imagine-video"

    # ═══════════════════════════════════════════════════════
    # Deprecated Models (for reference)
    # ═══════════════════════════════════════════════════════
    # CLAUDE_OPUS_4_0 = "anthropic/claude-opus-4"
    # CLAUDE_OPUS_4_1 = "anthropic/claude-opus-4.1"
    # CLAUDE_SONNET_4_0 = "anthropic/claude-sonnet-4"

    # ═══════════════════════════════════════════════════════
    # Special-purpose / internal models
    # ═══════════════════════════════════════════════════════
    NANO_BANANA_2 = "internal/nano-banana-2"  # Example internal model


class CostDict(TypedDict):
    """Strongly typed dict for model costs."""

    input: float
    output: float


# ═══════════════════════════════════════════════════════════════════════════════
# COST TABLE
# ═══════════════════════════════════════════════════════════════════════════════
COST_TABLE: dict[Model, CostDict] = {
    # ------------------------------------------------
    # Free Tier Models
    # ------------------------------------------------
    Model._QWEN_CODER_FREE: {"input": 0.00, "output": 0.00},
    Model._GPT_OSS_120B_FREE: {"input": 0.00, "output": 0.00},
    Model._QWEN_NEXT_80B_FREE: {"input": 0.00, "output": 0.00},
    Model._LLAMA_3_3_70B_FREE: {"input": 0.00, "output": 0.00},
    Model._NEMOTRON_3_ULTRA_FREE: {"input": 0.00, "output": 0.00},
    Model._NEMOTRON_3_SUPER_FREE: {"input": 0.00, "output": 0.00},
    Model._NEMOTRON_NANO_9B_FREE: {"input": 0.00, "output": 0.00},
    # ------------------------------------------------
    # Open-Source Models (Tier 1)
    # ------------------------------------------------
    Model.GPT_OSS_120B: {"input": 0.04, "output": 0.18},
    Model.GPT_OSS_20B: {"input": 0.03, "output": 0.14},
    Model.QWEN_NEXT_80B: {"input": 0.09, "output": 1.10},
    Model.GLM_4_7_FLASH: {"input": 0.06, "output": 0.40},
    Model.MINIMAX_M2_5: {"input": 0.15, "output": 0.90},
    Model.DEVSTRAL_2512: {"input": 0.40, "output": 2.00},
    Model.MISTRAL_LARGE_2512: {"input": 0.50, "output": 1.50},
    Model.GLM_5: {"input": 0.60, "output": 1.92},
    # ------------------------------------------------
    # Proprietary Models (Tier 1)
    # ------------------------------------------------
    Model.GEMINI_3_FLASH_PREVIEW: {"input": 0.50, "output": 3.00},
    Model.GEMINI_3_1_PRO_PREVIEW: {"input": 2.00, "output": 12.00},
    Model.GPT_5_2: {"input": 1.75, "output": 14.00},
    Model.GPT_4O: {"input": 2.50, "output": 10.00},
    Model.GPT_4O_MINI: {"input": 0.15, "output": 0.60},
    Model.GPT_5: {"input": 1.25, "output": 10.00},
    Model.GPT_5_MINI: {"input": 0.25, "output": 2.00},
    Model.GPT_5_NANO: {"input": 0.05, "output": 0.40},
    Model.O1: {"input": 15.00, "output": 60.00},
    Model.O3_MINI: {"input": 1.10, "output": 4.40},
    Model.O4_MINI: {"input": 1.50, "output": 6.00},
    Model.GEMINI_FLASH: {"input": 0.15, "output": 0.60},
    Model.GEMINI_FLASH_LITE: {"input": 0.10, "output": 0.40},
    Model.CLAUDE_FABLE_5: {"input": 10.00, "output": 50.00},
    Model.CLAUDE_SONNET_4_5: {"input": 3.00, "output": 15.00},
    Model.CLAUDE_SONNET_5: {"input": 3.00, "output": 15.00},
    Model.CLAUDE_OPUS_4_5: {"input": 5.00, "output": 25.00},
    Model.CLAUDE_OPUS_4_8: {"input": 6.00, "output": 30.00},
    Model.CLAUDE_HAIKU_4_5: {"input": 1.00, "output": 5.00},
    Model.DEEPSEEK_V4_PRO: {"input": 1.50, "output": 6.00},
    Model.DEEPSEEK_V4_FLASH: {"input": 0.27, "output": 1.10},
    Model.LLAMA_4_MAVERICK: {"input": 0.17, "output": 0.17},
    Model.LLAMA_4_SCOUT: {"input": 0.11, "output": 0.34},
    Model.LLAMA_3_3_70B: {"input": 0.12, "output": 0.30},
    Model.LLAMA_3_1_405B: {"input": 2.00, "output": 2.00},
    Model.PHI_4: {"input": 0.07, "output": 0.14},
    Model.GEMMA_3_27B: {"input": 0.08, "output": 0.20},
    Model.HERMES_3_LLAMA_3_1_70B: {"input": 0.40, "output": 0.40},
    Model.MOONSHOT_KIMI_K2_7_CODE: {"input": 1.10, "output": 4.50},
    Model.MOONSHOT_KIMI_K2_6: {"input": 0.95, "output": 4.00},
    Model.MOONSHOT_KIMI_K2: {"input": 0.50, "output": 1.50},
    Model.STEPFUN_STEP_3_5_FLASH: {"input": 0.10, "output": 0.30},
    Model.ZHIPU_GLM_5_2: {"input": 0.50, "output": 2.00},
    Model.ZHIPU_GLM_5_TURBO: {"input": 1.20, "output": 4.00},
    Model.XAI_GROK_4_5: {"input": 1.50, "output": 4.00},
    Model.QWEN_3_7_MAX: {"input": 0.78, "output": 3.90},
    Model.QWEN_3_6_FLASH: {"input": 0.12, "output": 0.50},
    Model.MINIMAX_M2_7: {"input": 0.30, "output": 1.20},
    Model.XIAOMI_MIMO_V2_FLASH: {"input": 0.14, "output": 0.28},
    Model.XIAOMI_MIMO_V2_5_PRO: {"input": 0.44, "output": 0.87},
    # ------------------------------------------------
    # Proprietary Models (Tier 2)
    # ------------------------------------------------
    Model.GPT_5_4: {"input": 2.50, "output": 15.00},
    Model.GPT_5_4_MINI: {"input": 0.75, "output": 4.50},
    Model.GPT_5_4_CODEX: {"input": 1.75, "output": 14.00},
    Model.RING_2_6_1T: {"input": 0.50, "output": 2.00},
    Model.MINIMAX_M3: {"input": 0.30, "output": 1.20},
    Model.QWEN_3_7_PLUS: {"input": 0.32, "output": 1.28},
    Model.GPT_5_4_NANO: {"input": 0.20, "output": 1.25},
    Model.GEMMA_4_31B: {"input": 0.12, "output": 0.35},
    Model.STEPFUN_STEP_3_7_FLASH: {"input": 0.15, "output": 0.45},
    Model.NEMOTRON_3_SUPER_120B: {"input": 0.10, "output": 0.50},
    Model.GPT_5_CODEX: {"input": 1.25, "output": 10.00},
    Model.GPT_5_4_PRO: {"input": 30.00, "output": 180.00},
    Model.XAI_GROK_4_MINI: {"input": 0.30, "output": 0.60},
    Model.QWEN_3_CODER: {"input": 0.20, "output": 0.80},
    Model.QWEN_3_CODER_NEXT: {"input": 0.50, "output": 2.00},
    Model.QWEN_3_5_397B: {"input": 1.20, "output": 4.80},
    Model.QWEN_3_235B_THINKING: {"input": 2.50, "output": 10.00},
    Model.QWEN_3_MAX_THINKING: {"input": 3.50, "output": 14.00},
    Model.DEEPSEEK_R1: {"input": 0.70, "output": 2.50},
    Model.DEEPSEEK_V3_2: {"input": 0.229, "output": 0.343},
    Model.DEEPSEEK_V3_1_TERMINUS: {"input": 0.27, "output": 0.95},
    Model.MOONSHOT_KIMI_K2_5: {"input": 0.375, "output": 2.025},
    Model.MOONSHOT_KIMI_K2_THINKING: {"input": 0.60, "output": 2.50},
    Model.MOONSHOT_KIMI_K2_0905: {"input": 0.60, "output": 2.50},
    Model.XAI_GROK_4_3: {"input": 1.25, "output": 2.50},
    Model.QWEN_3_MAX: {"input": 0.78, "output": 3.90},
    Model.QWEN_3_235B: {"input": 2.00, "output": 6.00},
    Model.QWEN_3_CODER_FLASH: {"input": 0.12, "output": 0.50},
    Model.QWEN_3_CODER_PLUS: {"input": 0.50, "output": 2.00},
    Model.GPT_5_5: {"input": 5.00, "output": 30.00},
    Model.GPT_5_5_PRO: {"input": 30.00, "output": 180.00},
    Model.GPT_LATEST: {"input": 5.00, "output": 30.00},
    Model.GPT_MINI_LATEST: {"input": 0.75, "output": 4.50},
    Model.XAI_GROK_BUILD_0_1: {"input": 1.00, "output": 2.00},
    # ------------------------------------------------
    # Image Generation Models
    # ------------------------------------------------
    Model.GEMINI_3_1_FLASH_IMAGE_PREVIEW: {"input": 0.0005, "output": 0.003},
    Model.GEMINI_2_5_FLASH_IMAGE: {"input": 0.0003, "output": 0.0025},
    Model.GEMINI_3_PRO_IMAGE_PREVIEW: {"input": 0.002, "output": 0.012},
    Model.GEMINI_FLASH_LITE_IMAGE: {"input": 0.25, "output": 1.50},
    Model.GPT_5_IMAGE: {"input": 10.00, "output": 10.00},
    Model.GPT_5_IMAGE_MINI: {"input": 2.50, "output": 2.00},
    Model.GPT_5_4_IMAGE_2: {"input": 8.00, "output": 15.00},
    Model.FLUX_2_KLEIN: {"input": 0.014, "output": 0},
    Model.FLUX_2_MAX: {"input": 0.07, "output": 0},
    Model.FLUX_2_FLEX: {"input": 0.06, "output": 0},
    Model.FLUX_2_PRO: {"input": 0.03, "output": 0},
    Model.RECRAFT_V4_1_UTILITY: {"input": 0.04, "output": 0},
    Model.RECRAFT_V4_1_PRO: {"input": 0.25, "output": 0},
    Model.RECRAFT_V4_1: {"input": 0.04, "output": 0},
    Model.RECRAFT_V4_PRO_VECTOR: {"input": 0.30, "output": 0},
    Model.RECRAFT_V4_VECTOR: {"input": 0.08, "output": 0},
    Model.RECRAFT_V4_PRO: {"input": 0.25, "output": 0},
    Model.RECRAFT_V4: {"input": 0.04, "output": 0},
    Model.RECRAFT_V3: {"input": 0.04, "output": 0},
    Model.RIVERFLOW_V2_PRO: {"input": 0.15, "output": 0},
    Model.RIVERFLOW_V2_FAST: {"input": 0.02, "output": 0},
    Model.RIVERFLOW_V2_MAX_PREVIEW: {"input": 0.075, "output": 0},
    Model.RIVERFLOW_V2_STANDARD_PREVIEW: {"input": 0.035, "output": 0},
    Model.RIVERFLOW_V2_FAST_PREVIEW: {"input": 0.03, "output": 0},
    Model.SEEDREAM_4_5: {"input": 0.04, "output": 0},
    # ------------------------------------------------
    # Video Generation Models
    # ------------------------------------------------
    Model.SORA_2_PRO: {"input": 0.30, "output": 0},
    Model.VEO_3_1: {"input": 0.40, "output": 0},
    Model.VEO_3_1_FAST: {"input": 0.10, "output": 0},
    Model.VEO_3_1_LITE: {"input": 0.05, "output": 0},
    Model.KLING_V3_0_PRO: {"input": 0.168, "output": 0},
    Model.KLING_V3_0_STD: {"input": 0.126, "output": 0},
    Model.KLING_VIDEO_O1: {"input": 0.112, "output": 0},
    Model.HAILUO_2_3: {"input": 0.0817, "output": 0},
    Model.SEEDANCE_2_0: {"input": 0.06726, "output": 0},
    Model.SEEDANCE_2_0_FAST: {"input": 0.0538, "output": 0},
    Model.SEEDANCE_1_5_PRO: {"input": 0.02306, "output": 0},
    Model.WAN_2_7: {"input": 0.10, "output": 0},
    Model.WAN_2_6: {"input": 0.04, "output": 0},
    Model.GROK_IMAGINE_VIDEO: {"input": 0.05, "output": 0},
    # ------------------------------------------------
    # Special / Internal
    # ------------------------------------------------
    Model.OPENROUTER_AUTO: {"input": 0.00, "output": 0.00},
    Model.NANO_BANANA_2: {"input": 0.01, "output": 0.01},  # Example cost
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT WINDOW TABLE
# ═══════════════════════════════════════════════════════════════════════════════
CONTEXT_WINDOWS: dict[Model, int] = {
    # Free Tier
    Model._QWEN_CODER_FREE: 32768,
    Model._GPT_OSS_120B_FREE: 32768,
    Model._QWEN_NEXT_80B_FREE: 32768,
    Model._LLAMA_3_3_70B_FREE: 8192,
    Model._NEMOTRON_3_ULTRA_FREE: 4096,
    Model._NEMOTRON_3_SUPER_FREE: 4096,
    Model._NEMOTRON_NANO_9B_FREE: 4096,
    # Open Source (Tier 1)
    Model.GPT_OSS_120B: 32768,
    Model.GPT_OSS_20B: 32768,
    Model.QWEN_NEXT_80B: 32768,
    Model.GLM_4_7_FLASH: 128000,
    Model.MINIMAX_M2_5: 32768,
    Model.DEVSTRAL_2512: 32768,
    Model.MISTRAL_LARGE_2512: 32768,
    Model.GLM_5: 128000,
    # Proprietary (Tier 1)
    Model.GEMINI_3_FLASH_PREVIEW: 1048576,
    Model.GEMINI_3_1_PRO_PREVIEW: 1048576,
    Model.GPT_5_2: 131072,
    Model.GPT_4O: 131072,
    Model.GPT_4O_MINI: 131072,
    Model.GPT_5: 131072,
    Model.GPT_5_MINI: 131072,
    Model.GPT_5_NANO: 131072,
    Model.O1: 200000,
    Model.O3_MINI: 131072,
    Model.O4_MINI: 131072,
    Model.GEMINI_FLASH: 1048576,
    Model.GEMINI_FLASH_LITE: 1048576,
    Model.GEMINI_FLASH_LITE_IMAGE: 66000,
    Model.CLAUDE_FABLE_5: 200000,
    Model.CLAUDE_SONNET_4_5: 200000,
    Model.CLAUDE_SONNET_5: 200000,
    Model.CLAUDE_OPUS_4_5: 200000,
    Model.CLAUDE_OPUS_4_8: 200000,
    Model.CLAUDE_HAIKU_4_5: 200000,
    Model.DEEPSEEK_V4_PRO: 131072,
    Model.DEEPSEEK_V4_FLASH: 1048576,
    Model.LLAMA_4_MAVERICK: 131072,
    Model.LLAMA_4_SCOUT: 131072,
    Model.LLAMA_3_3_70B: 8192,
    Model.LLAMA_3_1_405B: 131072,
    Model.PHI_4: 131072,
    Model.GEMMA_3_27B: 8192,
    Model.HERMES_3_LLAMA_3_1_70B: 8192,
    Model.MOONSHOT_KIMI_K2_7_CODE: 200000,
    Model.MOONSHOT_KIMI_K2_6: 200000,
    Model.MOONSHOT_KIMI_K2: 200000,
    Model.STEPFUN_STEP_3_5_FLASH: 131072,
    Model.ZHIPU_GLM_5_2: 128000,
    Model.ZHIPU_GLM_5_TURBO: 128000,
    Model.XAI_GROK_4_5: 131072,
    Model.QWEN_3_7_MAX: 65536,
    Model.QWEN_3_6_FLASH: 32768,
    Model.MINIMAX_M2_7: 32768,
    Model.XIAOMI_MIMO_V2_FLASH: 131072,
    # Proprietary (Tier 2)
    Model.GPT_5_4: 131072,
    Model.GPT_5_4_MINI: 131072,
    Model.GPT_5_4_CODEX: 131072,
    Model.RING_2_6_1T: 1048576,
    Model.MINIMAX_M3: 200000,
    Model.QWEN_3_7_PLUS: 65536,
    Model.GPT_5_4_NANO: 131072,
    Model.GEMMA_4_31B: 8192,
    Model.STEPFUN_STEP_3_7_FLASH: 131072,
    Model.NEMOTRON_3_SUPER_120B: 4096,
    Model.GPT_5_CODEX: 131072,
    Model.GPT_5_4_PRO: 131072,
    Model.XAI_GROK_4_MINI: 131072,
    Model.QWEN_3_CODER: 65536,
    Model.QWEN_3_CODER_NEXT: 65536,
    Model.QWEN_3_5_397B: 65536,
    Model.QWEN_3_235B_THINKING: 65536,
    Model.QWEN_3_MAX_THINKING: 65536,
    Model.DEEPSEEK_R1: 131072,
    Model.DEEPSEEK_V3_2: 131072,
    Model.DEEPSEEK_V3_1_TERMINUS: 131072,
    Model.MOONSHOT_KIMI_K2_5: 200000,
    Model.MOONSHOT_KIMI_K2_THINKING: 200000,
    Model.MOONSHOT_KIMI_K2_0905: 200000,
    Model.XAI_GROK_4_3: 131072,
    Model.QWEN_3_MAX: 65536,
    Model.QWEN_3_235B: 65536,
    Model.QWEN_3_CODER_FLASH: 65536,
    Model.QWEN_3_CODER_PLUS: 65536,
    Model.GPT_5_5: 131072,
    Model.GPT_5_5_PRO: 131072,
    Model.GPT_LATEST: 131072,
    Model.GPT_MINI_LATEST: 131072,
    Model.XAI_GROK_BUILD_0_1: 131072,
    # Image/Video - Context is N/A
    Model.GEMINI_3_1_FLASH_IMAGE_PREVIEW: 0,
    Model.GEMINI_2_5_FLASH_IMAGE: 0,
    Model.GEMINI_3_PRO_IMAGE_PREVIEW: 0,
    Model.GPT_5_IMAGE: 0,
    Model.GPT_5_IMAGE_MINI: 0,
    Model.GPT_5_4_IMAGE_2: 0,
    Model.FLUX_2_KLEIN: 0,
    Model.FLUX_2_MAX: 0,
    Model.FLUX_2_FLEX: 0,
    Model.FLUX_2_PRO: 0,
    Model.RECRAFT_V4_1_UTILITY: 0,
    Model.RECRAFT_V4_1_PRO: 0,
    Model.RECRAFT_V4_1: 0,
    Model.RECRAFT_V4_PRO_VECTOR: 0,
    Model.RECRAFT_V4_VECTOR: 0,
    Model.RECRAFT_V4_PRO: 0,
    Model.RECRAFT_V4: 0,
    Model.RECRAFT_V3: 0,
    Model.RIVERFLOW_V2_PRO: 0,
    Model.RIVERFLOW_V2_FAST: 0,
    Model.RIVERFLOW_V2_MAX_PREVIEW: 0,
    Model.RIVERFLOW_V2_STANDARD_PREVIEW: 0,
    Model.RIVERFLOW_V2_FAST_PREVIEW: 0,
    Model.SEEDREAM_4_5: 0,
    Model.SORA_2_PRO: 0,
    Model.VEO_3_1: 0,
    Model.VEO_3_1_FAST: 0,
    Model.VEO_3_1_LITE: 0,
    Model.KLING_V3_0_PRO: 0,
    Model.KLING_V3_0_STD: 0,
    Model.KLING_VIDEO_O1: 0,
    Model.HAILUO_2_3: 0,
    Model.SEEDANCE_2_0: 0,
    Model.SEEDANCE_2_0_FAST: 0,
    Model.SEEDANCE_1_5_PRO: 0,
    Model.WAN_2_7: 0,
    Model.WAN_2_6: 0,
    Model.GROK_IMAGINE_VIDEO: 0,
    # Special / Internal
    Model.OPENROUTER_AUTO: 8192,  # Varies, use a safe default
    Model.NANO_BANANA_2: 4096,  # Example context size
}


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

Provider = Literal[
    "openai",
    "anthropic",
    "google",
    "deepseek",
    "meta-llama",
    "microsoft",
    "nousresearch",
    "moonshotai",
    "stepfun",
    "z-ai",
    "x-ai",
    "qwen",
    "minimax",
    "xiaomi",
    "inclusionai",
    "openrouter",
    "internal",
    "nvidia",
    "black-forest-labs",
    "recraft",
    "sourceful",
    "bytedance",
    "kwaivgi",
    "alibaba",
]

PROVIDER_MAP: dict[str, Provider] = {
    "openai/": "openai",
    "anthropic/": "anthropic",
    "google/": "google",
    "deepseek/": "deepseek",
    "meta-llama/": "meta-llama",
    "microsoft/": "microsoft",
    "nousresearch/": "nousresearch",
    "moonshotai/": "moonshotai",
    "stepfun/": "stepfun",
    "z-ai/": "z-ai",
    "x-ai/": "x-ai",
    "qwen/": "qwen",
    "minimax/": "minimax",
    "xiaomi/": "xiaomi",
    "inclusionai/": "inclusionai",
    "openrouter/": "openrouter",
    "internal/": "internal",
    "nvidia/": "nvidia",
    "black-forest-labs/": "black-forest-labs",
    "recraft/": "recraft",
    "sourceful/": "sourceful",
    "bytedance-seed/": "bytedance",
    "kwaivgi/": "kwaivgi",
    "alibaba/": "alibaba",
}


def get_provider(model: Model) -> Provider:
    """Extract provider from model enum value."""
    model_val = model.value
    for prefix, provider in PROVIDER_MAP.items():
        if model_val.startswith(prefix):
            return provider
    # Fallback for models without a standard prefix
    if "claude" in model_val:
        return "anthropic"
    if "gemini" in model_val:
        return "google"
    if "gpt" in model_val or "o1" in model_val:
        return "openai"
    return "openrouter"


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
_MODEL_MAX_TOKENS_RAW = {
    # Anthropic Claude models
    "CLAUDE_HAIKU_4_5": 4096,
    "CLAUDE_SONNET_4_5": 8192,
    "CLAUDE_SONNET_5": 8192,
    "CLAUDE_OPUS_4_5": 4096,
    "CLAUDE_OPUS_4_8": 4096,
    # Google Gemini models (high limits)
    "GEMINI_FLASH": 8192,
    "GEMINI_FLASH_LITE": 8192,
    # DeepSeek models
    "DEEPSEEK_V4_FLASH": 8192,
    "DEEPSEEK_V4_PRO": 8192,
    # Z.AI GLM models
    "ZHIPU_GLM_5_2": 16384,  # z-ai/glm-5.2 (canonical)
    "ZHIPU_GLM_5_TURBO": 16384,  # z-ai/glm-5-turbo (fast)
    # Meta LLaMA models
    "LLAMA_4_MAVERICK": 8192,
    "LLAMA_4_SCOUT": 8192,
    "LLAMA_3_3_70B": 8192,
    # Microsoft Phi models
    "PHI_4": 4096,
    "PHI_4_REASONING": 4096,
    # Image generation models
    "NANO_BANANA": 4096,
    "NANO_BANANA_2": 4096,
    "NANO_BANANA_PRO": 4096,
    "GPT_5_IMAGE": 4096,
    "GPT_5_IMAGE_MINI": 4096,
    "GPT_54_IMAGE_2": 4096,
    # Xiaomi Mimo V2.5
    "XIAOMI_MIMO_V2_5": 8192,
    "XIAOMI_MIMO_V2_5_PRO": 8192,
    # MiniMax M3
    "MINIMAX_M3": 8192,
    # Qwen 3.7 Plus
    "QWEN_3_7_PLUS": 8192,
    # GPT-5.4 Nano
    "GPT_5_4_NANO": 4096,
    # Google Gemma models
    "GEMMA_3_27B": 8192,
    # Nous Hermes models
    "HERMES_3_70B": 8192,
    # OpenAI models
    "GPT_4O": 8192,
    "GPT_4O_MINI": 4096,
    "GPT_5": 8192,
    "GPT_5_MINI": 4096,
    "GPT_5_NANO": 4096,
    "GPT_5_CODEX": 8192,
    "GPT_5_4_PRO": 8192,
    "O1": 4096,
    "O3_MINI": 4096,
    "O4_MINI": 4096,
    # xAI Grok 4 family
    "XAI_GROK_4_5": 131072,
    "XAI_GROK_4_3": 131072,
    # Moonshot Kimi K2 additions
    "MOONSHOT_KIMI_K2_5": 262144,
    "MOONSHOT_KIMI_K2_THINKING": 262144,
    "MOONSHOT_KIMI_K2_0905": 262144,
    # DeepSeek V3/R1 additions
    "DEEPSEEK_R1": 65536,
    "DEEPSEEK_V3_2": 65536,
    "DEEPSEEK_V3_1_TERMINUS": 65536,
    # Qwen 2026 extended lineup
    "QWEN_3_MAX": 131072,
    "QWEN_3_235B": 131072,
    "QWEN_3_CODER": 32768,
    "QWEN_3_CODER_FLASH": 32768,
    "QWEN_3_CODER_PLUS": 32768,
    "QWEN_3_CODER_NEXT": 32768,
    "QWEN_3_5_397B": 131072,
    "QWEN_3_235B_THINKING": 131072,
    "QWEN_3_MAX_THINKING": 131072,
    # Anthropic new additions
    "CLAUDE_OPUS_4": 8192,
    "CLAUDE_OPUS_4_1": 8192,
    "CLAUDE_SONNET_4": 8192,
}

MODEL_MAX_TOKENS: dict[Model, int] = {}
for name, val in _MODEL_MAX_TOKENS_RAW.items():
    if hasattr(Model, name):
        MODEL_MAX_TOKENS[getattr(Model, name)] = val


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


@dataclass(frozen=True)
class Verdict:
    """Binary verdict with score for objective verification.

    Returned by Verifier implementations to signal pass/fail.
    ``signals`` captures which sub-checks fired (for telemetry).
    ``detail`` contains a human-readable explanation.
    """

    passed: bool
    score: float  # 0..1, for cascade min_score comparison
    signals: tuple[str, ...] = ()  # which checks fired, for telemetry
    detail: str = ""


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

    target_language: str = ""  # "python", "html", "css", "javascript", "typescript", etc.

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

    def __post_init__(self) -> None:
        _max_desc = 10_000
        _max_criteria = 10_000
        _max_app_type = 64
        _max_output_dir = 512

        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("description must be a non-empty string")
        if len(self.description) > _max_desc:
            raise ValueError(f"description exceeds {_max_desc} characters")

        if not isinstance(self.success_criteria, str) or not self.success_criteria.strip():
            raise ValueError("success_criteria must be a non-empty string")
        if len(self.success_criteria) > _max_criteria:
            raise ValueError(f"success_criteria exceeds {_max_criteria} characters")

        if self.app_type is not None and len(self.app_type) > _max_app_type:
            raise ValueError(f"app_type exceeds {_max_app_type} characters")
        if self.output_dir is not None and len(self.output_dir) > _max_output_dir:
            raise ValueError(f"output_dir exceeds {_max_output_dir} characters")


# ─────────────────────────────────────────────

# Utilities

# ─────────────────────────────────────────────

import hashlib


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
