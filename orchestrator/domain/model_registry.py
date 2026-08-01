"""
Centralized Model Registry for Multi-LLM Orchestrator
======================================================
Author: Georgios-Chrysovalantis Chatzivantsidis
Date: 2026-04-01

Single source of truth for all model configurations, timeouts, and costs.
This module centralizes model definitions to prevent hardcoded strings scattered
across the codebase.

Usage:
    from orchestrator.model_registry import ModelRegistry, ModelConfig

    # Get model ID
    coder_model = ModelRegistry.QWEN_CODER

    # Get timeout for model
    timeout = ModelRegistry.get_timeout("qwen/qwen3-coder")

    # Get cost info
    cost = ModelRegistry.get_cost("deepseek/deepseek-v4-flash")

    # Check if model is valid
    is_valid = ModelRegistry.is_valid_model("qwen/qwen3-coder")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ModelConfig:
    """Configuration for a single model."""

    model_id: str
    display_name: str
    provider: str
    input_cost_per_1m: float
    output_cost_per_1m: float
    max_tokens: int
    is_reasoning_model: bool
    is_coding_specialist: bool
    description: str
    status: str = "active"  # active, deprecated, unavailable


class ModelRegistry:
    """
    Centralized registry for all LLM models.

    All model IDs, timeouts, and costs should be defined here.
    Other modules should import constants from this module instead of
    hardcoding model strings.

    Note: Updated 2026-04-01 with verified OpenRouter model availability.
    """

    # ═══════════════════════════════════════════════════════
    # VERIFIED AVAILABLE MODEL IDs (Updated 2026-04-01)
    # Verified via direct OpenRouter URL checks
    # ═══════════════════════════════════════════════════════

    # Qwen Models — 2026 lineup
    QWEN_3_6_FLASH = "openai/gpt-4o-mini"  # $0.12/$0.50, coding ⭐ VERIFIED
    QWEN_3_7_FLASH = "qwen/qwen3.7-flash"  # $0.03/$0.13, 1M ctx, vision-language, multimodal agents
    QWEN_3_CODER = "qwen/qwen3-coder"  # $0.20/$0.80, coding specialist
    QWEN_3_CODER_NEXT = "qwen/qwen3-coder-next"  # $0.50/$2.00, next-gen coder
    QWEN_3_5_397B = "qwen/qwen3.5-397b-a17b"  # $1.20/$4.80, 397B MoE
    QWEN_3_235B_THINKING = "qwen/qwen3-235b-a22b-thinking-2507"  # $2.50/$10.00, 235B thinking
    QWEN_3_MAX_THINKING = "qwen/qwen3-max-thinking"  # $3.50/$14.00, max reasoning

    # DeepSeek Models - Best Value
    DEEPSEEK_V4_FLASH = "deepseek/deepseek-v4-flash"  # $0.14/$0.28, 1M context ⭐ VERIFIED
    DEEPSEEK_V4_PRO = "deepseek/deepseek-v4-pro"  # $1.50/$6.00, reasoning specialist

    # Anthropic Claude Models - Balanced Quality
    CLAUDE_SONNET_5 = "anthropic/claude-sonnet-5"  # $3.00/$15.00 ⭐ VERIFIED
    CLAUDE_OPUS_4_8 = "anthropic/claude-opus-4-8"  # $6.00/$30.00, complex analysis
    CLAUDE_HAIKU_4_5 = "anthropic/claude-haiku-4-5"  # $1.00/$5.00, fast
    CLAUDE_OPUS_5 = "anthropic/claude-opus-5"  # $10.00/$50.00, flagship reasoning
    CLAUDE_OPUS_5_FAST = "anthropic/claude-opus-5-fast"  # $20.00/$100.00, fast variant
    CLAUDE_SONNET_4_6 = "anthropic/claude-sonnet-4.6"  # Sonnet 4.6

    # OpenAI Models - Premium Tier
    GPT_5 = "openai/gpt-5"  # $1.25/$10.00, 400K ⭐ VERIFIED
    GPT_5_CODEX = "openai/gpt-5-codex"  # DEPRECATED — retired, use GPT_5_1_CODEX_MAX
    GPT_5_4 = "openai/gpt-5.4"  # $2.50/$15.00, unified
    GPT_5_4_MINI = "openai/gpt-5.4-mini"  # $0.25/$2.00, fast
    GPT_5_4_CODEX = "openai/gpt-5.3-codex"  # $1.75/$14.00, coding specialist
    GPT_5_CODEX = "openai/gpt-5-codex"  # DEPRECATED — retired, use GPT_5_1_CODEX_MAX
    GPT_5_4_PRO = "openai/gpt-5.4-pro"  # $30.00/$180.00, maximum quality
    GPT_5_6_SOL = "openai/gpt-5.6-sol"  # GPT-5.6 Sol
    GPT_5_6_SOL_PRO = "openai/gpt-5.6-sol-pro"  # $5.00/$30.00, premium tier
    GPT_5_6_TERRA = "openai/gpt-5.6-terra"  # $1.00/$6.00, mid tier
    GPT_5_6_TERRA_PRO = "openai/gpt-5.6-terra-pro"  # $1.00/$6.00, mid tier
    GPT_5_6_LUNA = "openai/gpt-5.6-luna"  # $0.10/$0.60, budget tier
    GPT_5_6_LUNA_PRO = "openai/gpt-5.6-luna-pro"  # $0.10/$0.60, budget tier
    GPT_5_1_CODEX_MAX = "openai/gpt-5.1-codex-max"  # $1.25/$10.00, coding specialist
    O1_PRO = "openai/o1-pro"  # o1 Pro
    O3_PRO = "openai/o3-pro"  # o3 Pro
    GPT_4O = "openai/gpt-4o"  # $2.50/$10.00, previous gen
    GPT_4O_MINI = "openai/gpt-4o-mini"  # $0.15/$0.60, 128K ⭐ VERIFIED

    # Google Gemini Models
    GEMINI_FLASH = "google/gemini-3.5-flash"  # DEPRECATED — use GEMINI_3_6_FLASH  # $0.15/$0.60, 1M+ ⭐ VERIFIED
    GEMINI_2_5_PRO = "google/gemini-2.5-pro"  # Gemini 2.5 Pro
    GEMINI_3_1_PRO_PREVIEW = "google/gemini-3.1-pro-preview"  # Gemini 3.1 Pro Preview

    # xAI Grok Models — 2026 lineup
    GROK_4_20 = "x-ai/grok-4.20"  # $2.00/$6.00, 2M context ⭐ VERIFIED
    GROK_4_MINI = "x-ai/grok-4-mini"  # $0.30/$0.60, faster/cheaper variant

    # Moonshot Kimi Models
    KIMI_K2_7_CODE = "moonshotai/kimi-k2.7-code"  # $0.95/$4.00, 256K, MoE 32B/1T
    KIMI_K2 = "moonshotai/kimi-k2"  # $0.57/$2.30, 128K ⭐ VERIFIED
    KIMI_K2_6 = "moonshotai/kimi-k2.6"  # $0.42/$2.20, visual coding SOTA
    KIMI_K3 = "moonshotai/kimi-k3"  # $3/$15, 1M context, 2.8T params, multimodal reasoning ⭐
    QWEN_3_7_MAX = "qwen/qwen3.7-max"  # Qwen 3.7 Max
    QWEN_3_7_PLUS = "qwen/qwen3.7-plus"  # Qwen 3.7 Plus
    DEEPSEEK_V3_2 = "deepseek/deepseek-v3.2"  # DeepSeek V3.2
    DEEPSEEK_R1_0528 = "deepseek/deepseek-r1-0528"  # DeepSeek R1 0528

    # Thinking Machines — Inkling: multimodal MoE, coding/agentic/RAG
    INKLING = "thinkingmachines/inkling"  # $1/$4.05, 1M context, 41B/975B MoE
    INKLING_SMALL = "thinkingmachines/inkling-small"  # $0.50/$1.20, 512K ctx, budget multimodal

    # Meituan — LongCat 2.0: sparse MoE, coding/repo-level/agentic
    LONGCAT_2_0 = "meituan/longcat-2.0"  # $0.30/$1.20, 1M context, 48B/1.6T MoE

    # Meta — Muse Spark 1.1: multimodal reasoning, multi-agent orchestration
    MUSE_SPARK_1_1 = "meta/muse-spark-1.1"  # $1.25/$4.25, 1M context

    # Krea — Krea 2 Large: high-capability image gen, $0.06/image
    KREA_2_LARGE = "krea/krea-2-large"  # $0.06/image, photorealism/artistic
    KREA_2_MEDIUM = "krea/krea-2-medium"  # $0.03/image, illustration/anime/painting
    KREA_2_MEDIUM_TURBO = "krea/krea-2-medium-turbo"  # $0.015/image, speed-focused
    GEMINI_3_6_FLASH = "google/gemini-3.6-flash"  # $1.50/$7.50, 1M ctx, coding/agentic
    GEMINI_3_5_FLASH_LITE = "google/gemini-3.5-flash-lite"  # $0.30/$2.50, 1M ctx, subagents
    LAGUNA_S_2_1 = "poolside/laguna-s-2.1"  # $0.10/$0.20, 1M ctx, coding agent

    # Xiaomi MiMo Models — mimo-v2-{flash,pro} deprecated by OpenRouter (404);
    # repointed to the live v2.5 IDs.
    MIMO_V2_FLASH = "xiaomi/mimo-v2.5"  # was mimo-v2-flash (deprecated), $0.14/$0.28, 1M ctx
    MIMO_V2_PRO = "xiaomi/mimo-v2.5-pro"  # was mimo-v2-pro (deprecated), $0.43/$0.87, 1M ctx

    # StepFun Models - Best Value ⭐ VERIFIED
    STEP_3_5_FLASH = "stepfun/step-3.5-flash"  # $0.10/$0.30, 262K, 196B MoE ⭐ BEST VALUE
    STEP_3_5 = "stepfun/step-3.7-flash"  # $0.15/$0.45

    # Z-AI GLM Models ⭐ VERIFIED
    GLM_5_2 = "z-ai/glm-5.2"  # canonical GLM model
    GLM_5_TURBO = "z-ai/glm-5-turbo"  # fast variant

    # Minimax Models ⭐ VERIFIED
    MINIMAX_M2_7 = "minimax/minimax-m2.7"  # $0.30/$1.20, 205K, multi-agent ⭐

    # Z-AI GLM — latest point release
    GLM_5_1 = "z-ai/glm-5.1"  # $0.97/$3.04, 204K ctx

    # OpenRouter Auto-Router
    OPENROUTER_AUTO = "openrouter/auto"  # Dynamic routing

    # ═══════════════════════════════════════════════════════
    # DEPRECATED/UNAVAILABLE MODELS (DO NOT USE)
    # Verified via OpenRouter URL checks 2026-04-01
    # ═══════════════════════════════════════════════════════

    # Deprecated/removed OpenRouter IDs → live replacements.
    # Keys are the DEAD ids (verified absent from /api/v1/models, 404/400 on a
    # real call); values are live replacements matched by provider + capability
    # + price tier. validate_model_available() redirects these so any persisted
    # state or external reference to an old id keeps resolving.
    # NOTE: anthropic/claude-{opus,sonnet,haiku}-N-M (hyphen) are intentionally
    # NOT listed — OpenRouter normalizes them server-side (verified live call).
    UNAVAILABLE_MODELS = {
        # Anthropic legacy ids (genuine 404)
        "anthropic/claude-3.5-sonnet": "anthropic/claude-sonnet-5",
        "anthropic/claude-3-opus": "anthropic/claude-opus-4-8",
        "anthropic/claude-3-5-haiku": "anthropic/claude-haiku-4-5",
        "anthropic/claude-3.5-haiku": "anthropic/claude-haiku-4.5",
        # Qwen — hyphenated/legacy ids OpenRouter no longer accepts
        "qwen/qwen-3-coder-next": "qwen/qwen3-coder-next",
        "qwen/qwen-3-coder": "qwen/qwen3-coder",
        "qwen/qwen-3.5-397b-a17b": "qwen/qwen3.5-397b-a17b",
        "qwen/qwen-3.5-235b-a22b-thinking-2507": "qwen/qwen3-235b-a22b-thinking-2507",
        "qwen/qwen-3-max-thinking": "qwen/qwen3-max-thinking",
        "qwen/qwen-3-697b-a17b": "qwen/qwen3.5-397b-a17b",
        # NVIDIA — base alias replaced by the sized id
        "nvidia/nemotron-3-super": "nvidia/nemotron-3-super-120b-a12b",
        # AION Labs — provider no longer on OpenRouter
        "aionlabs/aion-2.0": "qwen/qwen3.5-397b-a17b",
        # OpenAI
        "openai/gpt-5.4-codex": "openai/gpt-5.3-codex",
        "openai/o4": "openai/o4-mini",
        # DeepSeek
        "deepseek/deepseek-reasoner": "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-r1": "deepseek/deepseek-v4-pro",
        # Internal test models
        "internal/nano-banana-2": "openai/gpt-4o-mini",
        # StepFun
        "stepfun/step-3.5": "stepfun/step-3.7-flash",
        # OpenAI — gpt-5-codex retired, replaced by the 5.1-codex family
        "openai/gpt-5-codex": "openai/gpt-5.1-codex-max",
        # Mistral — devstral-2512 no longer on OpenRouter under any id
        "mistralai/devstral-2512": "mistralai/codestral-2508",
        # Meta — 405B not on OpenRouter
        "meta-llama/llama-3.1-405b-instruct": "meta-llama/llama-3.3-70b-instruct",
        # xAI — grok-4-mini not on OpenRouter; use grok-4.20
        "x-ai/grok-4-mini": "x-ai/grok-4.20",
        # FLUX 2 models — not on OpenRouter; fall back to gpt-4o-mini
        "black-forest-labs/flux.2-klein-4b": "openai/gpt-4o-mini",
        "black-forest-labs/flux.2-max": "openai/gpt-4o-mini",
        "black-forest-labs/flux.2-flex": "openai/gpt-4o-mini",
        "black-forest-labs/flux.2-pro": "openai/gpt-4o-mini",
        # Recraft — v4/v4.1 ids not on OpenRouter; fall back to gpt-4o-mini
        "recraft/recraft-v3": "openai/gpt-4o-mini",
        "recraft/recraft-v4": "openai/gpt-4o-mini",
        "recraft/recraft-v4-pro": "openai/gpt-4o-mini",
        "recraft/recraft-v4-pro-vector": "openai/gpt-4o-mini",
        "recraft/recraft-v4-vector": "openai/gpt-4o-mini",
        "recraft/recraft-v4.1": "openai/gpt-4o-mini",
        "recraft/recraft-v4.1-pro": "openai/gpt-4o-mini",
        "recraft/recraft-v4.1-utility": "openai/gpt-4o-mini",
        # Sourceful Riverflow — preview ids not on OpenRouter; fall back to gpt-4o-mini
        "sourceful/riverflow-v2-fast": "openai/gpt-4o-mini",
        "sourceful/riverflow-v2-fast-preview": "openai/gpt-4o-mini",
        "sourceful/riverflow-v2-max-preview": "openai/gpt-4o-mini",
        "sourceful/riverflow-v2-pro": "openai/gpt-4o-mini",
        "sourceful/riverflow-v2-standard-preview": "openai/gpt-4o-mini",
        # ByteDance Seedream — not on OpenRouter
        "bytedance-seed/seedream-4.5": "openai/gpt-4o-mini",
        # Krea — not on OpenRouter
        "krea/krea-2-large": "openai/gpt-4o-mini",
        "krea/krea-2-medium": "openai/gpt-4o-mini",
        "krea/krea-2-medium-turbo": "openai/gpt-4o-mini",
    }

    # ═══════════════════════════════════════════════════════
    # TIMEOUT CONFIGURATION (seconds)
    # ═══════════════════════════════════════════════════════

    DEFAULT_TIMEOUT = 60
    MAX_TIMEOUT = 300

    TIMEOUT_CONFIG = {
        # Fast models (60s)
        "qwen/": 60,
        "deepseek/": 60,
        "xiaomi/": 60,
        "nvidia/": 60,
        "moonshotai/": 60,
        # Medium models (90s)
        "anthropic/": 90,
        "google/": 90,
        "z-ai/": 90,
        "x-ai/": 90,
        # Slow reasoning models (120s)
        "openai/gpt-5": 120,
        "openai/o1": 180,
        "openai/o3": 180,
        "openai/o4-mini": 180,
    }

    # Per-model timeout overrides
    MODEL_TIMEOUT_OVERRIDES = {
        DEEPSEEK_V4_PRO: 120,  # Reasoning specialist
        GPT_5_4_PRO: 180,  # Premium reasoning
        GPT_5: 120,  # Complex reasoning
        GROK_4_20: 90,  # 2M context, complex reasoning
    }

    # ═══════════════════════════════════════════════════════
    # COST TABLE (per 1M tokens, USD)
    # Updated 2026-04-01 with verified OpenRouter pricing
    # ═══════════════════════════════════════════════════════

    COST_TABLE: Dict[str, dict] = {  # type: ignore[type-arg]
        # Qwen Models (VERIFIED)
        QWEN_3_6_FLASH: {"input": 0.12, "output": 0.50},
        QWEN_3_7_FLASH: {"input": 0.03, "output": 0.13},
        QWEN_3_CODER: {"input": 0.20, "output": 0.80},
        QWEN_3_CODER_NEXT: {"input": 0.50, "output": 2.00},
        QWEN_3_5_397B: {"input": 1.20, "output": 4.80},
        QWEN_3_235B_THINKING: {"input": 2.50, "output": 10.00},
        QWEN_3_MAX_THINKING: {"input": 3.50, "output": 14.00},
        # DeepSeek Models (VERIFIED)
        DEEPSEEK_V4_FLASH: {"input": 0.14, "output": 0.28},
        DEEPSEEK_V4_PRO: {"input": 1.50, "output": 6.00},
        # Anthropic Models (VERIFIED)
        CLAUDE_SONNET_5: {"input": 3.00, "output": 15.00},
        CLAUDE_OPUS_4_8: {"input": 6.00, "output": 30.00},
        CLAUDE_HAIKU_4_5: {"input": 1.00, "output": 5.00},
        CLAUDE_OPUS_5: {"input": 10.00, "output": 50.00},
        CLAUDE_OPUS_5_FAST: {"input": 20.00, "output": 100.00},
        # OpenAI Models (VERIFIED)
        GPT_5: {"input": 1.25, "output": 10.00},
        GPT_5_CODEX: {"input": 1.25, "output": 10.00},
        GPT_5_4: {"input": 2.50, "output": 15.00},
        GPT_5_4_MINI: {"input": 0.75, "output": 4.50},
        GPT_5_4_CODEX: {"input": 1.75, "output": 14.00},
        GPT_5_4_PRO: {"input": 30.00, "output": 180.00},
        GPT_4O: {"input": 2.50, "output": 10.00},
        GPT_4O_MINI: {"input": 0.15, "output": 0.60},
        # Google Gemini Models (VERIFIED)
        GEMINI_FLASH: {"input": 0.15, "output": 0.60},  # DEPRECATED
        # xAI Grok Models (VERIFIED)
        GROK_4_20: {"input": 2.00, "output": 6.00},
        GROK_4_MINI: {"input": 0.30, "output": 0.60},
        # Moonshot Kimi Models (VERIFIED)
        KIMI_K2_7_CODE: {"input": 1.10, "output": 4.50},
        KIMI_K2: {"input": 0.50, "output": 1.50},
        KIMI_K2_6: {"input": 0.95, "output": 4.00},
        # Kimi K3
        KIMI_K3: {"input": 3.00, "output": 15.00},
        # Inkling
        INKLING: {"input": 1.00, "output": 4.05},
        INKLING_SMALL: {"input": 0.50, "output": 1.20},
        # LongCat 2.0
        LONGCAT_2_0: {"input": 0.30, "output": 1.20},
        # Muse Spark 1.1
        MUSE_SPARK_1_1: {"input": 1.25, "output": 4.25},
        GEMINI_3_6_FLASH: {"input": 1.50, "output": 7.50},
        GEMINI_3_5_FLASH_LITE: {"input": 0.30, "output": 2.50},
        LAGUNA_S_2_1: {"input": 0.10, "output": 0.20},
        # Xiaomi MiMo Models (VERIFIED)
        MIMO_V2_FLASH: {"input": 0.14, "output": 0.28},
        MIMO_V2_PRO: {"input": 0.44, "output": 0.87},
        # StepFun Models (VERIFIED)
        STEP_3_5_FLASH: {"input": 0.10, "output": 0.30},  # ⭐ BEST VALUE
        STEP_3_5: {"input": 0.15, "output": 0.45},  # stepfun/step-3.7-flash
        # Z-AI GLM Models (VERIFIED)
        GLM_5_TURBO: {"input": 1.20, "output": 4.00},  # z-ai/glm-5-turbo
        GLM_5_2: {"input": 0.50, "output": 2.00},  # z-ai/glm-5.2
        # Minimax Models (VERIFIED)
        MINIMAX_M2_7: {"input": 0.30, "output": 1.20},
        GLM_5_1: {"input": 0.97, "output": 3.04},
        # New Models
        GPT_5_6_SOL: {"input": 5.00, "output": 30.00},
        GPT_5_6_SOL_PRO: {"input": 5.00, "output": 30.00},
        GPT_5_6_TERRA: {"input": 1.00, "output": 6.00},
        GPT_5_6_TERRA_PRO: {"input": 1.00, "output": 6.00},
        GPT_5_6_LUNA: {"input": 0.10, "output": 0.60},
        GPT_5_6_LUNA_PRO: {"input": 0.10, "output": 0.60},
        GPT_5_1_CODEX_MAX: {"input": 1.25, "output": 10.00},
        QWEN_3_7_PLUS: {"input": 0.32, "output": 1.28},
        QWEN_3_7_MAX: {"input": 1.48, "output": 4.42},
        GEMINI_3_1_PRO_PREVIEW: {"input": 2.00, "output": 12.00},
        CLAUDE_SONNET_4_6: {"input": 3.00, "output": 15.00},
        DEEPSEEK_V3_2: {"input": 0.27, "output": 0.40},
        GEMINI_2_5_PRO: {"input": 1.25, "output": 10.00},
        O3_PRO: {"input": 20.00, "output": 80.00},
        DEEPSEEK_R1_0528: {"input": 0.50, "output": 2.15},
        O1_PRO: {"input": 150.00, "output": 600.00},
    }

    # ═══════════════════════════════════════════════════════
    # MODEL MAX TOKENS
    # Updated 2026-04-01 with verified context windows
    # ═══════════════════════════════════════════════════════

    MODEL_MAX_TOKENS: Dict[str, int] = {
        # Qwen Models (VERIFIED)
        QWEN_3_6_FLASH: 32768,
        QWEN_3_7_FLASH: 1048576,
        # DeepSeek Models (VERIFIED)
        DEEPSEEK_V4_FLASH: 1048576,
        DEEPSEEK_V4_PRO: 16384,
        # Anthropic Models (VERIFIED)
        CLAUDE_SONNET_5: 200000,
        CLAUDE_OPUS_4_8: 200000,
        CLAUDE_HAIKU_4_5: 200000,
        CLAUDE_OPUS_5: 1000000,
        CLAUDE_OPUS_5_FAST: 1000000,
        # OpenAI Models (VERIFIED)
        GPT_5: 400000,
        GPT_5_CODEX: 400000,
        GPT_5_4: 16384,
        GPT_5_4_MINI: 8192,
        GPT_5_4_CODEX: 16384,
        GPT_5_4_PRO: 32768,
        GPT_4O: 8192,
        GPT_4O_MINI: 128000,
        # Google Gemini Models (VERIFIED)
        GEMINI_FLASH: 1048576,  # 1M+ context
        # xAI Grok Models (VERIFIED)
        GROK_4_20: 2000000,  # 2M context!
        GROK_4_MINI: 131072,
        # Moonshot Kimi Models (VERIFIED)
        KIMI_K2: 131072,
        KIMI_K2_7_CODE: 262144,
        KIMI_K2_6: 131072,
        KIMI_K3: 1048576,  # 1M context
        INKLING: 1048576,  # 1M context
        INKLING_SMALL: 524288,
        LONGCAT_2_0: 1048576,  # 1M context
        MUSE_SPARK_1_1: 1048576,  # 1M context
        GEMINI_3_6_FLASH: 1048576,  # 1M context
        GEMINI_3_5_FLASH_LITE: 1048576,  # 1M context
        LAGUNA_S_2_1: 1048576,  # 1M context
        # Xiaomi MiMo Models (VERIFIED)
        MIMO_V2_FLASH: 262144,
        MIMO_V2_PRO: 1048576,  # 1M+ context
        # StepFun Models (VERIFIED)
        STEP_3_5_FLASH: 262144,
        STEP_3_5: 262144,
        # Z-AI GLM Models (VERIFIED)
        GLM_5_TURBO: 202752,  # z-ai/glm-5-turbo
        GLM_5_2: 202752,  # z-ai/glm-5.2
        # Minimax Models (VERIFIED)
        MINIMAX_M2_7: 204800,
        GPT_5_6_SOL: 1050000,
        GPT_5_6_SOL_PRO: 1050000,
        GPT_5_6_TERRA: 1050000,
        GPT_5_6_TERRA_PRO: 1050000,
        GPT_5_6_LUNA: 1050000,
        GPT_5_6_LUNA_PRO: 1050000,
        GPT_5_1_CODEX_MAX: 400000,
        GLM_5_1: 204800,
        QWEN_3_7_PLUS: 1000000,
        QWEN_3_7_MAX: 1000000,
        GEMINI_3_1_PRO_PREVIEW: 1048576,
        CLAUDE_SONNET_4_6: 1000000,
        DEEPSEEK_V3_2: 163840,
        GEMINI_2_5_PRO: 1048576,
        O3_PRO: 200000,
        DEEPSEEK_R1_0528: 163840,
        O1_PRO: 200000,
        # Qwen extended lineup
        QWEN_3_CODER: 32768,
        QWEN_3_CODER_NEXT: 32768,
        QWEN_3_5_397B: 131072,
        QWEN_3_235B_THINKING: 131072,
        QWEN_3_MAX_THINKING: 131072,
    }

    # ═══════════════════════════════════════════════════════
    # MODEL CATEGORIES
    # Updated 2026-04-01 with verified models
    # ═══════════════════════════════════════════════════════

    # Coding specialists - best for code generation
    CODING_SPECIALISTS = {
        KIMI_K2_7_CODE,
        QWEN_3_6_FLASH,
        QWEN_3_7_FLASH,
        QWEN_3_CODER,
        QWEN_3_CODER_NEXT,
        GPT_5_CODEX,
        GPT_5_4_CODEX,
        MIMO_V2_FLASH,
        KIMI_K2_6,
        MINIMAX_M2_7,
        INKLING,
        LONGCAT_2_0,
        KIMI_K3,
        MUSE_SPARK_1_1,
        INKLING,
    }

    # Reasoning models - best for complex analysis
    REASONING_MODELS = {
        DEEPSEEK_V4_PRO,
        GPT_5,
        GPT_5_4_PRO,
        GROK_4_20,
        STEP_3_5_FLASH,
        QWEN_3_235B_THINKING,
        QWEN_3_MAX_THINKING,
        QWEN_3_5_397B,
        # 2026 reasoning tier (kept in sync with the live OpenRouter catalogue;
        # see docs/REASONING_AND_TEMPERATURE.md). Literal ids: these models are
        # not all declared as constants in this registry.
        "openai/gpt-5.2",
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "x-ai/grok-4.3",
        "x-ai/grok-4.20-multi-agent",
        "minimax/minimax-m3",
        "deepseek/deepseek-v3.2",
        "qwen/qwen3-235b-a22b-thinking-2507",
        "qwen/qwen3-next-80b-a3b-thinking",
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        "anthropic/claude-opus-4-8",  # extended-thinking capable
        "openai/o1",
        "openai/o3-mini",
        "openai/o4-mini",
        KIMI_K3,
        MUSE_SPARK_1_1,
        INKLING,
        GPT_5_1_CODEX_MAX,
    }

    # Budget models - best value
    BUDGET_MODELS = {
        MIMO_V2_FLASH,  # $0.09/$0.29 - Best value! ⭐
        QWEN_3_7_FLASH,  # $0.03/$0.13 - Cheapest 1M-context flash! ⭐
        GLM_5_2,  # $0.50/$2.00 - Canonical GLM
        STEP_3_5_FLASH,  # $0.10/$0.30 - Best value reasoning
        QWEN_3_6_FLASH,  # $0.66/$1.00 - Coding specialist
        GPT_4O_MINI,  # $0.15/$0.60
        GEMINI_3_6_FLASH,  # replaces deprecated GEMINI_FLASH  # $0.15/$0.60
        DEEPSEEK_V4_FLASH,  # $0.32/$0.89
        LONGCAT_2_0,  # $0.30/$1.20 - Great value MoE
        GPT_5_6_LUNA,  # $0.10/$0.60
        GPT_5_6_LUNA_PRO,  # $0.10/$0.60
        INKLING_SMALL,  # $0.50/$1.20
    }

    # Premium models - maximum quality
    PREMIUM_MODELS = {
        GPT_5_4_PRO,
        CLAUDE_OPUS_4_8,
        GROK_4_20,
        QWEN_3_235B_THINKING,
        QWEN_3_MAX_THINKING,
        KIMI_K3,
        INKLING,
        MUSE_SPARK_1_1,
        KREA_2_LARGE,
        CLAUDE_OPUS_5,
        CLAUDE_OPUS_5_FAST,
        GPT_5_6_SOL,
        GPT_5_6_SOL_PRO,
    }

    # Models with 200K+ context capability
    LONG_CONTEXT_MODELS = {
        KIMI_K2_7_CODE,
        GEMINI_3_6_FLASH,  # replaces deprecated GEMINI_FLASH  # 1M+
        MIMO_V2_PRO,  # 1M+
        CLAUDE_SONNET_5,  # 200K
        CLAUDE_OPUS_4_8,  # 200K
        CLAUDE_HAIKU_4_5,  # 200K
        GPT_5,  # 400K
        GPT_5_CODEX,  # 400K
        GROK_4_20,  # 2M!
        STEP_3_5_FLASH,  # 262K
        STEP_3_5,  # 262K
        MIMO_V2_FLASH,  # 256K
        GLM_5_TURBO,  # 202K
        GLM_5_2,  # 202K
        MINIMAX_M2_7,  # 205K
        DEEPSEEK_V4_FLASH,  # 164K
        QWEN_3_7_FLASH,  # 1M
        QWEN_3_5_397B,  # 128K
        QWEN_3_235B_THINKING,  # 128K
        QWEN_3_MAX_THINKING,  # 128K
        INKLING,  # 1M
        LONGCAT_2_0,  # 1M
        KIMI_K3,  # 1M
        MUSE_SPARK_1_1,  # 1M
        GEMINI_3_6_FLASH,  # 1M
        GEMINI_3_5_FLASH_LITE,  # 1M
        LAGUNA_S_2_1,  # 1M
        CLAUDE_OPUS_5,  # 1M
        CLAUDE_OPUS_5_FAST,  # 1M
        GPT_5_6_SOL,  # 1.05M
        GPT_5_6_SOL_PRO,  # 1.05M
        GPT_5_6_TERRA,  # 1.05M
        GPT_5_6_TERRA_PRO,  # 1.05M
        GPT_5_6_LUNA,  # 1.05M
        GPT_5_6_LUNA_PRO,  # 1.05M
        GPT_5_1_CODEX_MAX,  # 400K
        GLM_5_1,  # 205K
        INKLING_SMALL,  # 512K
    }

    # Multimodal models — support image + text input (vision-capable)
    MULTIMODAL_MODELS = {
        KIMI_K2_7_CODE,
        # Google Gemini — natively multimodal
        GEMINI_3_6_FLASH,  # replaces deprecated GEMINI_FLASH
        # Anthropic Claude 3.x+ — all vision-capable
        CLAUDE_SONNET_5,
        CLAUDE_OPUS_4_8,
        CLAUDE_OPUS_5,
        CLAUDE_OPUS_5_FAST,
        CLAUDE_HAIKU_4_5,
        # OpenAI — GPT-4o and GPT-5 family are multimodal
        GPT_4O,
        GPT_4O_MINI,
        GPT_5,
        GPT_5_4,
        GPT_5_6_SOL,
        GPT_5_6_SOL_PRO,
        GPT_5_6_TERRA,
        GPT_5_6_TERRA_PRO,
        GPT_5_6_LUNA,
        GPT_5_6_LUNA_PRO,
        INKLING_SMALL,
        # Moonshot Kimi-K2.5 — described as visual coding SOTA
        KIMI_K3,
        INKLING,
        LONGCAT_2_0,
        MUSE_SPARK_1_1,
        KREA_2_LARGE,
        KREA_2_MEDIUM,
        KREA_2_MEDIUM_TURBO,
        GEMINI_3_6_FLASH,
        GEMINI_3_5_FLASH_LITE,
        CLAUDE_OPUS_5,
        CLAUDE_OPUS_5_FAST,
    }

    # ═══════════════════════════════════════════════════════
    # CLASS METHODS
    # ═══════════════════════════════════════════════════════

    @classmethod
    def get_timeout(cls, model_id: str) -> int:
        """
        Get timeout for a specific model.

        Args:
            model_id: Full model ID (e.g., "qwen/qwen3-coder")

        Returns:
            Timeout in seconds
        """
        # Check for exact model override first
        if model_id in cls.MODEL_TIMEOUT_OVERRIDES:
            return cls.MODEL_TIMEOUT_OVERRIDES[model_id]

        # Check prefix-based configuration
        for prefix, timeout in cls.TIMEOUT_CONFIG.items():
            if model_id.startswith(prefix):
                return timeout

        return cls.DEFAULT_TIMEOUT

    @classmethod
    def get_cost(cls, model_id: str) -> dict:  # type: ignore[type-arg]
        """
        Get cost information for a model.

        Args:
            model_id: Full model ID

        Returns:
            Dictionary with 'input' and 'output' costs per 1M tokens
        """
        return cls.COST_TABLE.get(model_id, {"input": 0.0, "output": 0.0})

    @classmethod
    def get_max_tokens(cls, model_id: str) -> int:
        """
        Get maximum tokens for a model.

        Args:
            model_id: Full model ID

        Returns:
            Maximum tokens allowed
        """
        return cls.MODEL_MAX_TOKENS.get(model_id, 8192)

    @classmethod
    def is_valid_model(cls, model_id: str) -> bool:
        """
        Check if a model ID is valid (not unavailable/deprecated).

        Args:
            model_id: Full model ID

        Returns:
            True if valid, False if unavailable/invalid
        """
        if model_id in cls.UNAVAILABLE_MODELS:
            return False
        return model_id in cls.COST_TABLE

    @classmethod
    def get_replacement_model(cls, deprecated_model_id: str) -> Optional[str]:
        """
        Get replacement model for an unavailable/deprecated model.

        Args:
            deprecated_model_id: Unavailable model ID

        Returns:
            Replacement model ID or None if no replacement
        """
        return cls.UNAVAILABLE_MODELS.get(deprecated_model_id)

    @classmethod
    def is_coding_specialist(cls, model_id: str) -> bool:
        """Check if model is a coding specialist."""
        return model_id in cls.CODING_SPECIALISTS

    @classmethod
    def is_reasoning_model(cls, model_id: str) -> bool:
        """Check if model is a reasoning specialist.

        Strips OpenRouter endpoint-variant suffixes (e.g. ":free") before
        matching, since REASONING_MODELS is keyed by base model id.
        """
        base_id = model_id.split(":", 1)[0]
        return base_id in cls.REASONING_MODELS

    @classmethod
    def is_budget_model(cls, model_id: str) -> bool:
        """Check if model is in budget tier."""
        return model_id in cls.BUDGET_MODELS

    @classmethod
    def is_premium_model(cls, model_id: str) -> bool:
        """Check if model is in premium tier."""
        return model_id in cls.PREMIUM_MODELS

    @classmethod
    def is_long_context(cls, model_id: str) -> bool:
        """Check if model supports 200K+ context."""
        return model_id in cls.LONG_CONTEXT_MODELS

    @classmethod
    def is_multimodal(cls, model_id: str) -> bool:
        """Check if model supports image + text input (vision-capable)."""
        return model_id in cls.MULTIMODAL_MODELS

    @classmethod
    def get_all_valid_models(cls) -> list[str]:
        """Get list of all valid model IDs."""
        return list(cls.COST_TABLE.keys())

    @classmethod
    def get_models_by_provider(cls, provider: str) -> list[str]:
        """
        Get all models for a specific provider.

        Args:
            provider: Provider name (e.g., "qwen", "deepseek", "openai")

        Returns:
            List of model IDs for the provider
        """
        return [m for m in cls.COST_TABLE if m.startswith(f"{provider}/")]

    @classmethod
    def get_cheapest_model(cls) -> str:
        """Get the cheapest model by average cost."""
        cheapest = None
        min_avg_cost = float("inf")

        for model_id, costs in cls.COST_TABLE.items():
            avg_cost = (costs["input"] + costs["output"]) / 2
            if avg_cost < min_avg_cost:
                min_avg_cost = avg_cost
                cheapest = model_id

        return cheapest or cls.MIMO_V2_FLASH

    @classmethod
    def validate_all_models(cls) -> dict:  # type: ignore[type-arg]
        """
        Validate all model IDs against the registry.

        Returns:
            Dictionary with validation results
        """
        results = {  # type: ignore[var-annotated]
            "valid": [],
            "deprecated": [],
            "unknown": [],
        }

        # Check all constants
        for attr_name in dir(cls):
            if attr_name.isupper() and not attr_name.startswith("_"):
                value = getattr(cls, attr_name)
                if isinstance(value, str) and "/" in value:
                    if value in cls.UNAVAILABLE_MODELS:
                        results["deprecated"].append(f"{attr_name}={value}")
                    elif value in cls.COST_TABLE and attr_name not in [
                        "UNAVAILABLE_MODELS",
                        "TIMEOUT_CONFIG",
                        "COST_TABLE",
                        "MODEL_MAX_TOKENS",
                        "CODING_SPECIALISTS",
                        "REASONING_MODELS",
                        "BUDGET_MODELS",
                        "PREMIUM_MODELS",
                        "LONG_CONTEXT_MODELS",
                        "TIMEOUT_CONFIG",
                        "MODEL_TIMEOUT_OVERRIDES",
                    ]:
                        results["valid"].append(f"{attr_name}={value}")

        return results


# ═══════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════


def get_timeout(model_id: str) -> int:
    """Get timeout for a model (convenience function)."""
    return ModelRegistry.get_timeout(model_id)


def get_cost(model_id: str) -> dict:  # type: ignore[type-arg]
    """Get cost for a model (convenience function)."""
    return ModelRegistry.get_cost(model_id)


def get_max_tokens(model_id: str) -> int:
    """Get max tokens for a model (convenience function)."""
    return ModelRegistry.get_max_tokens(model_id)


def is_valid_model(model_id: str) -> bool:
    """Check if model is valid (convenience function)."""
    return ModelRegistry.is_valid_model(model_id)


def get_replacement(deprecated_model: str) -> Optional[str]:
    """Get replacement for deprecated model (convenience function)."""
    return ModelRegistry.get_replacement_model(deprecated_model)


# ═══════════════════════════════════════════════════════
# MIGRATION HELPERS
# ═══════════════════════════════════════════════════════


def migrate_deprecated_models(config: dict) -> dict:  # type: ignore[type-arg]
    """
    Migrate deprecated model IDs in a configuration dictionary.

    Args:
        config: Configuration dictionary with model IDs

    Returns:
        Updated configuration with valid model IDs
    """
    updated = config.copy()

    for key, value in config.items():
        if isinstance(value, str) and value in ModelRegistry.UNAVAILABLE_MODELS:
            replacement = ModelRegistry.UNAVAILABLE_MODELS[value]
            updated[key] = replacement
            print(f"Migrated {key}: {value} → {replacement}")

    return updated


# ═══════════════════════════════════════════════════════
# CLI VALIDATION
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("Model Registry Validation")
    print("=" * 70)

    results = ModelRegistry.validate_all_models()

    print(f"\n✅ Valid models: {len(results['valid'])}")
    for model in results["valid"][:10]:
        print(f"   • {model}")
    if len(results["valid"]) > 10:
        print(f"   ... and {len(results['valid']) - 10} more")

    print(f"\n⚠️  Deprecated models: {len(results['deprecated'])}")
    for model in results["deprecated"]:
        print(f"   • {model}")

    print(f"\n💰 Cheapest model: {ModelRegistry.get_cheapest_model()}")

    print(f"\n⏱️  Default timeout: {ModelRegistry.DEFAULT_TIMEOUT}s")
    print(f"📊 Total models registered: {len(ModelRegistry.get_all_valid_models())}")

    print("\n" + "=" * 70)
