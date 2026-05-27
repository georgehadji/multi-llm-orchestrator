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

    GPT_5_4_MINI = "openai/gpt-5.4-mini"

    GPT_5_4_CODEX = "openai/gpt-5.4-codex"

    O1 = "openai/o1"

    O3_MINI = "openai/o3-mini"

    O4_MINI = "openai/o4-mini"

    # Google Gemini Models

    GEMINI_FLASH = "google/gemini-3.5-flash"  # latest flash

    GEMINI_FLASH_LITE = "google/gemini-3.1-flash-lite"  # cost-effective lite

    # Anthropic Claude Models

    CLAUDE_3_5_SONNET = "anthropic/claude-3.5-sonnet"

    CLAUDE_3_OPUS = "anthropic/claude-3-opus"

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

    LLAMA_3_1_405B = "meta-llama/llama-3.1-405b-instruct"  # 405B

    # Microsoft Phi Models (OpenRouter)

    PHI_4 = "microsoft/phi-4"  # 14B

    PHI_4_REASONING = "openai/o3-mini"  # Use o3-mini for reasoning

    # Google Gemma Models (OpenRouter)

    GEMMA_3_27B = "google/gemma-3-27b-it"  # 27B

    # Nous Research Hermes (OpenRouter)

    HERMES_3_70B = "nousresearch/hermes-3-llama-3.1-70b"  # 70B fine-tuned

    # ═══════════════════════════════════════════════════════

    # XIAOMI MODELS (NEW v3.0) - GAME CHANGERS!

    # ═══════════════════════════════════════════════════════

    XIAOMI_MIMO_V2_FLASH = "xiaomi/mimo-v2-flash"  # $0.09/$0.29, #1 SWE-bench open ⭐

    XIAOMI_MIMO_V2_PRO = "xiaomi/mimo-v2-pro"  # $1.00/$3.00, 1T+ params, 1M+ ctx

    XIAOMI_MIMO_V2_OMNI = "xiaomi/mimo-v2-omni"  # $0.40/$2.00, omni-modal

    # ═══════════════════════════════════════════════════════

    # MOONSHOT KIMI MODELS (NEW v3.0)

    # ═══════════════════════════════════════════════════════

    MOONSHOT_KIMI_K2_6 = "moonshotai/kimi-k2.6"  # $0.95/$4.00, 256K, reasoning SOTA

    MOONSHOT_KIMI_K2 = "moonshotai/kimi-k2"  # $0.50/$1.50

    # Backward compatibility aliases

    KIMI_K2_6 = MOONSHOT_KIMI_K2_6

    KIMI_K2 = MOONSHOT_KIMI_K2

    # ═══════════════════════════════════════════════════════

    # STEPFUN MODELS (NEW v3.0) - BEST VALUE!

    # ═══════════════════════════════════════════════════════

    STEPFUN_STEP_3_5_FLASH = "stepfun/step-3.5-flash"  # $0.10/$0.30, 196B MoE ⭐

    STEPFUN_STEP_3_5 = "stepfun/step-3.5"  # $0.15/$0.45

    # ═══════════════════════════════════════════════════════

    # Z.AI GLM MODELS — two canonical models only

    # ═══════════════════════════════════════════════════════

    ZHIPU_GLM_5_1 = "z-ai/glm-5.1"  # balanced, 202K context

    ZHIPU_GLM_5_TURBO = "z-ai/glm-5-turbo"  # fast variant

    # ═══════════════════════════════════════════════════════

    # XAI GROK MODELS — grok-4.20 only

    # ═══════════════════════════════════════════════════════

    XAI_GROK_4_20 = "x-ai/grok-4.20"  # $2.00/$6.00, 2M context, lowest hallucination

    # ═══════════════════════════════════════════════════════

    # QWEN MODELS — two canonical models only

    # ═══════════════════════════════════════════════════════

    QWEN_3_7_MAX = "qwen/qwen3.7-max"  # flagship reasoning + coding

    QWEN_3_6_FLASH = "qwen/qwen3.6-flash"  # fast + cost-effective

    # ═══════════════════════════════════════════════════════

    # MINIMAX MODELS (NEW v3.0)

    # Note: Verified available 2026-04-01

    # ═══════════════════════════════════════════════════════

    MINIMAX_M2_7 = "minimax/minimax-m2.7"  # $0.30/$1.20, 205K, multi-agent ⭐

    # Backward compatibility alias

    MINIMAX_TEXT_01 = MINIMAX_M2_7

    # ═══════════════════════════════════════════════════════

    # NVIDIA MODELS (redirected via model_registry to fallback)

    # ═══════════════════════════════════════════════════════

    NVIDIA_NEMOTRON_3_SUPER = "nvidia/nemotron-3-super"  # redirects → minimax-m2.7

    # InclusionAI Ring Models

    INCLUSION_RING_2_6_1T = "inclusionai/ring-2.6-1t"  # 1T params, strong reasoning

    # OpenRouter Auto-Router

    OPENROUTER_AUTO = "openrouter/auto"  # Dynamic routing


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

# Cost table (per 1M tokens, USD)

# ─────────────────────────────────────────────


COST_TABLE: dict[Model, dict[str, float]] = {
    # OpenAI Models (via OpenRouter)
    Model.GPT_4O: {"input": 2.50, "output": 10.00},
    Model.GPT_4O_MINI: {"input": 0.15, "output": 0.60},
    Model.GPT_5: {"input": 1.25, "output": 10.00},
    Model.GPT_5_MINI: {"input": 0.25, "output": 2.00},
    Model.GPT_5_NANO: {"input": 0.05, "output": 0.40},
    Model.O1: {"input": 15.00, "output": 60.00},
    Model.O3_MINI: {"input": 1.10, "output": 4.40},
    Model.O4_MINI: {"input": 1.50, "output": 6.00},
    # Google Gemini Models (via OpenRouter)
    Model.GEMINI_FLASH: {"input": 0.15, "output": 0.60},
    Model.GEMINI_FLASH_LITE: {"input": 0.10, "output": 0.40},
    # Anthropic Claude Models (via OpenRouter)
    Model.CLAUDE_3_5_SONNET: {"input": 3.00, "output": 15.00},
    Model.CLAUDE_3_OPUS: {"input": 15.00, "output": 75.00},
    Model.CLAUDE_3_HAIKU: {"input": 0.25, "output": 1.25},
    Model.CLAUDE_SONNET_4_5: {"input": 3.00, "output": 15.00},
    Model.CLAUDE_SONNET_4_6: {"input": 3.00, "output": 15.00},
    Model.CLAUDE_OPUS_4_5: {"input": 5.00, "output": 25.00},
    Model.CLAUDE_OPUS_4_6: {"input": 5.00, "output": 25.00},
    Model.CLAUDE_HAIKU_4_5: {"input": 1.00, "output": 5.00},
    # DeepSeek Models (via OpenRouter)
    Model.DEEPSEEK_V4_PRO: {"input": 1.50, "output": 6.00},  # flagship reasoning
    Model.DEEPSEEK_V4_FLASH: {"input": 0.27, "output": 1.10},  # fast + cost-effective
    # Meta LLaMA Models (OpenRouter)
    Model.LLAMA_4_MAVERICK: {"input": 0.17, "output": 0.17},  # 400B MoE
    Model.LLAMA_4_SCOUT: {"input": 0.11, "output": 0.34},  # 109B MoE
    Model.LLAMA_3_3_70B: {"input": 0.12, "output": 0.30},  # 70B
    Model.LLAMA_3_1_405B: {"input": 2.00, "output": 2.00},  # 405B
    # Microsoft Phi Models (OpenRouter)
    Model.PHI_4: {"input": 0.07, "output": 0.14},  # 14B
    Model.PHI_4_REASONING: {"input": 0.07, "output": 0.35},  # 14B + CoT
    # Google Gemma Models (OpenRouter)
    Model.GEMMA_3_27B: {"input": 0.08, "output": 0.20},  # 27B
    # Nous Research Hermes (OpenRouter)
    Model.HERMES_3_70B: {"input": 0.40, "output": 0.40},  # 70B fine-tuned
    # ═══════════════════════════════════════════════════════
    # XIAOMI MODELS (NEW v3.0) - GAME CHANGERS!
    # ═══════════════════════════════════════════════════════
    Model.XIAOMI_MIMO_V2_FLASH: {"input": 0.09, "output": 0.29},  # #1 SWE-bench open ⭐
    Model.XIAOMI_MIMO_V2_PRO: {"input": 1.00, "output": 3.00},  # 1T+ params
    Model.XIAOMI_MIMO_V2_OMNI: {"input": 0.40, "output": 2.00},  # omni-modal
    # ═══════════════════════════════════════════════════════
    # MOONSHOT KIMI MODELS (NEW v3.0)
    # ═══════════════════════════════════════════════════════
    Model.MOONSHOT_KIMI_K2_6: {"input": 0.95, "output": 4.00},  # reasoning SOTA, 256K
    Model.MOONSHOT_KIMI_K2: {"input": 0.50, "output": 1.50},
    # ═══════════════════════════════════════════════════════
    # STEPFUN MODELS (NEW v3.0) - BEST VALUE!
    # ═══════════════════════════════════════════════════════
    Model.STEPFUN_STEP_3_5_FLASH: {"input": 0.10, "output": 0.30},  # 196B MoE ⭐
    Model.STEPFUN_STEP_3_5: {"input": 0.15, "output": 0.45},
    # ═══════════════════════════════════════════════════════
    # Z.AI GLM MODELS (NEW v3.0)
    # ═══════════════════════════════════════════════════════
    Model.ZHIPU_GLM_5_1: {"input": 0.10, "output": 0.40},  # z-ai/glm-5.1 (balanced)
    Model.ZHIPU_GLM_5_TURBO: {"input": 0.10, "output": 0.40},  # z-ai/glm-5-turbo (fast)
    # ═══════════════════════════════════════════════════════
    # XAI GROK MODELS (NEW v3.0) - LOWEST HALLUCINATION
    # Note: Updated 2026-04-01 - Use grok-4.20 (NOT grok-4.20-beta)
    # ═══════════════════════════════════════════════════════
    Model.XAI_GROK_4_20: {"input": 2.00, "output": 6.00},  # grok-4.20, lowest hallucination
    # ═══════════════════════════════════════════════════════
    # QWEN MODELS (NEW v3.0) - CODING SPECIALISTS
    # Note: Updated 2026-04-01 - Verified available
    # ═══════════════════════════════════════════════════════
    Model.QWEN_3_7_MAX: {"input": 0.78, "output": 3.90},  # flagship reasoning + coding
    Model.QWEN_3_6_FLASH: {"input": 0.12, "output": 0.50},  # fast + cost-effective
    # ═══════════════════════════════════════════════════════
    # MINIMAX MODELS (NEW v3.0)
    # ═══════════════════════════════════════════════════════
    Model.MINIMAX_M2_7: {"input": 0.30, "output": 1.20},  # 205K ⭐
    # ═══════════════════════════════════════════════════════
    # NVIDIA MODELS (redirected via model_registry to fallback)
    # ═══════════════════════════════════════════════════════
    Model.NVIDIA_NEMOTRON_3_SUPER: {"input": 0.10, "output": 0.50},  # redirects → minimax-m2.7
    # GPT-5.4 Models (NEW v3.0)
    Model.GPT_5_4: {"input": 2.50, "output": 15.00},
    Model.GPT_5_4_MINI: {"input": 0.75, "output": 4.50},
    Model.GPT_5_4_CODEX: {"input": 1.75, "output": 14.00},  # SWE-Bench SOTA
    # InclusionAI Ring
    Model.INCLUSION_RING_2_6_1T: {"input": 0.50, "output": 2.00},  # 1T params
    # OpenRouter Auto
    Model.OPENROUTER_AUTO: {"input": 0.00, "output": 0.00},  # Dynamic
}


# ─────────────────────────────────────────────

# Routing table (priority-ordered per task type)

# ═══════════════════════════════════════════════════════════════════════════════

# OPENROUTER ONLY - All models accessible via OpenRouter

# Updated v3.0 with Xiaomi, Moonshot, DeepSeek, GLM models

# ═══════════════════════════════════════════════════════════════════════════════


ROUTING_TABLE: dict[TaskType, list[Model]] = {
    # Optimized May 2026 -- benchlm.ai coding leaderboard evidence
    # Each primary model has 2 fallbacks (next 2 in list)
    # ======================================================================
    # CODE_GEN: DeepSeek V4 Flash leads (83.5 score @ $0.27/M)
    TaskType.CODE_GEN: [
        Model.DEEPSEEK_V4_FLASH,  # 83.5 score, $0.27/$1.10 -- BEST VALUE
        Model.MOONSHOT_KIMI_K2_6,  # 81.5 score, $0.42/$2.20 -- fallback 1
        Model.QWEN_3_7_MAX,  # 92.2 score, $0.78/$3.90 -- fallback 2 (best quality)
        Model.DEEPSEEK_V4_PRO,  # 90.1 score, $1.50/$6.00 -- reasoning premium
        Model.XIAOMI_MIMO_V2_FLASH,  # $0.09/$0.29 -- ultra-cheap backup
        Model.GPT_5_4_CODEX,  # 87.8 score, $1.75/$14.00 -- SWE-Bench specialist
        Model.CLAUDE_SONNET_4_6,  # 82.2 score, $3/$15 -- premium quality
        Model.GEMINI_FLASH,  # 78.3 score, $0.15/$0.60 -- cheap fallback
    ],
    # CODE_REVIEW: DeepSeek V4 Pro (reasoning chains) + Grok 4.20 (low hallucination)
    TaskType.CODE_REVIEW: [
        Model.DEEPSEEK_V4_PRO,  # 90.1 score, reasoning chains -- BEST
        Model.XAI_GROK_4_20,  # lowest hallucination -- fallback 1
        Model.CLAUDE_SONNET_4_6,  # 82.2, best quality prose -- fallback 2
        Model.MOONSHOT_KIMI_K2_6,  # 81.5, strong coding understanding
    ],
    # REASONING: Qwen 3.7 Max leads (92.2 score, #4 globally)
    TaskType.REASONING: [
        Model.QWEN_3_7_MAX,  # 92.2 score, #4 globally -- BEST
        Model.DEEPSEEK_V4_PRO,  # 90.1 score, dedicated reasoner -- fallback 1
        Model.GPT_5_4,  # 87.8 score -- fallback 2
        Model.MOONSHOT_KIMI_K2_6,  # 81.5, moderate cost reasoning
        Model.CLAUDE_OPUS_4_6,  # 86.0, most capable
    ],
    # WRITING: Quality and creativity
    TaskType.WRITING: [
        Model.CLAUDE_SONNET_4_6,  # best prose quality -- BEST
        Model.GPT_5_4,  # excellent writing -- fallback 1
        Model.LLAMA_4_MAVERICK,  # $0.17/$0.17, creative -- fallback 2
        Model.HERMES_3_70B,  # fine-tuned creative
        Model.LLAMA_3_1_405B,  # Meta frontier
    ],
    # DATA_EXTRACT: GLM-5.1 leads (83.4 score @ $0.10 -- beats Claude Sonnet!)
    TaskType.DATA_EXTRACT: [
        Model.ZHIPU_GLM_5_1,  # 83.4 score, $0.10/$0.40 -- BEST VALUE
        Model.DEEPSEEK_V4_FLASH,  # 83.5 score, $0.27/$1.10 -- fallback 1
        Model.ZHIPU_GLM_5_TURBO,  # $0.10/$0.40, fast -- fallback 2
        Model.PHI_4,  # $0.07/$0.14, ultra-cheap
        Model.GEMINI_FLASH,  # 78.3 score, cheap backup
    ],
    # SUMMARIZE: GLM-5.1 leads (same reasoning as data extraction)
    TaskType.SUMMARIZE: [
        Model.ZHIPU_GLM_5_1,  # 83.4 score, $0.10/$0.40 -- BEST VALUE
        Model.DEEPSEEK_V4_FLASH,  # 83.5 score, $0.27/$1.10 -- fallback 1
        Model.GEMINI_FLASH,  # 78.3 score, $0.15/$0.60 -- fallback 2
        Model.PHI_4,  # $0.07/$0.14, ultra-cheap
    ],
    # EVALUATE: Grok 4.20 leads (lowest hallucination = fairest scoring)
    TaskType.EVALUATE: [
        Model.XAI_GROK_4_20,  # lowest hallucination -- BEST
        Model.DEEPSEEK_V4_PRO,  # 90.1 score, nuanced evaluation -- fallback 1
        Model.CLAUDE_SONNET_4_6,  # reliable structured output -- fallback 2
        Model.GPT_5_4,  # 87.8, consistent scoring
        Model.MOONSHOT_KIMI_K2_6,  # 81.5, technical eval
    ],
}


# ─────────────────────────────────────────────

# Fallback chains (always cross-provider)

# ─────────────────────────────────────────────


TASK_PROVIDER_STRATEGIES: dict[TaskType, ProviderStrategy] = {}


FALLBACK_CHAIN: dict[Model, Model] = {
    # OpenRouter fallbacks (cheaper/faster → more capable)
    # OpenAI models fallbacks
    Model.GPT_4O: Model.CLAUDE_SONNET_4_6,  # GPT-4o → Claude Sonnet
    Model.GPT_4O_MINI: Model.LLAMA_3_3_70B,  # GPT-4o-mini → LLaMA 70B
    Model.O1: Model.CLAUDE_OPUS_4_6,  # o1 → Claude Opus
    Model.O3_MINI: Model.PHI_4_REASONING,  # o3-mini → Phi-4 Reasoning
    Model.O4_MINI: Model.CLAUDE_OPUS_4_6,  # o4-mini → Claude Opus
    # Gemini fallbacks
    Model.GEMINI_FLASH: Model.GPT_4O,  # Gemini Pro → GPT-4o
    Model.GEMINI_FLASH: Model.LLAMA_4_SCOUT,  # Gemini Flash → LLaMA Scout
    Model.GEMINI_FLASH_LITE: Model.PHI_4,  # Gemini Flash Lite → Phi-4
    # Claude fallbacks
    Model.CLAUDE_3_5_SONNET: Model.GPT_4O,  # Claude Sonnet → GPT-4o
    Model.CLAUDE_3_OPUS: Model.CLAUDE_OPUS_4_6,  # Claude Opus → Claude Opus 4-6
    Model.CLAUDE_3_HAIKU: Model.LLAMA_3_3_70B,  # Claude Haiku → LLaMA 70B
    Model.CLAUDE_SONNET_4_5: Model.CLAUDE_SONNET_4_6,  # Sonnet 4-5 → Sonnet 4-6
    Model.CLAUDE_SONNET_4_6: Model.GPT_4O,  # Sonnet 4-6 → GPT-4o
    Model.CLAUDE_OPUS_4_5: Model.CLAUDE_OPUS_4_6,  # Opus 4-5 → Opus 4-6
    Model.CLAUDE_OPUS_4_6: Model.GPT_4O,  # Opus 4-6 → GPT-4o
    Model.CLAUDE_HAIKU_4_5: Model.LLAMA_3_3_70B,  # Haiku 4-5 → LLaMA 70B
    # DeepSeek fallbacks
    Model.DEEPSEEK_V4_FLASH: Model.LLAMA_4_SCOUT,  # v4-flash -> LLaMA Scout
    Model.DEEPSEEK_V4_PRO: Model.O3_MINI,  # v4-pro -> o3-mini
    # Meta LLaMA fallbacks
    Model.LLAMA_4_MAVERICK: Model.LLAMA_3_1_405B,  # Maverick → LLaMA 405B
    Model.LLAMA_4_SCOUT: Model.LLAMA_3_3_70B,  # Scout → LLaMA 70B
    Model.LLAMA_3_3_70B: Model.HERMES_3_70B,  # LLaMA 70B → Hermes 70B
    Model.LLAMA_3_1_405B: Model.CLAUDE_SONNET_4_6,  # LLaMA 405B → Claude Sonnet
    # Microsoft Phi fallbacks
    Model.PHI_4: Model.GEMMA_3_27B,  # Phi-4 → Gemma 27B
    Model.PHI_4_REASONING: Model.O3_MINI,  # Phi-4 Reasoning → o3-mini
    # Google Gemma fallbacks
    Model.GEMMA_3_27B: Model.LLAMA_3_3_70B,  # Gemma 27B → LLaMA 70B
    # Nous Hermes fallbacks
    Model.HERMES_3_70B: Model.LLAMA_3_3_70B,  # Hermes → LLaMA 70B
    # OpenRouter Auto fallback
    Model.OPENROUTER_AUTO: Model.LLAMA_3_3_70B,  # Auto → LLaMA 70B safe fallback
    # ── v3.0 models added to ROUTING_TABLE but missing from FALLBACK_CHAIN ──
    # Without these entries, self_consistency.py:FALLBACK_CHAIN.get(model, model)
    # returns the same model as default, making quality retries useless (BUG-006).
    Model.XIAOMI_MIMO_V2_FLASH: Model.DEEPSEEK_V4_FLASH,  # CODE_GEN primary → proven alt
    Model.XAI_GROK_4_20: Model.DEEPSEEK_V4_PRO,  # CODE_REVIEW/EVALUATE primary → reasoning
    Model.STEPFUN_STEP_3_5_FLASH: Model.DEEPSEEK_V4_PRO,  # REASONING primary → reasoning specialist
    Model.ZHIPU_GLM_5_1: Model.PHI_4,  # DATA_EXTRACT/SUMMARIZE primary
    Model.ZHIPU_GLM_5_TURBO: Model.ZHIPU_GLM_5_1,  # turbo -> glm-5.1 → cheap alt
    Model.QWEN_3_7_MAX: Model.DEEPSEEK_V4_PRO,  # REASONING primary fallback
}


# ─────────────────────────────────────────────

# Per-task thresholds & limits

# ─────────────────────────────────────────────


DEFAULT_THRESHOLDS: dict[TaskType, float] = {
    TaskType.DATA_EXTRACT: 0.90,
    TaskType.SUMMARIZE: 0.80,
    TaskType.CODE_GEN: 0.85,
    # CODE_REVIEW: lowered from 0.85 — review quality depends on how much
    # source context the LLM received, which is often partial due to truncation.
    # 0.75 is a realistic target; scores above this reflect genuine analysis.
    TaskType.CODE_REVIEW: 0.75,
    TaskType.REASONING: 0.90,
    TaskType.WRITING: 0.80,
    # EVALUATE: lowered from 0.95 — evaluation outputs are open-ended prose;
    # scoring ≥ 0.95 requires near-perfect structured responses which LLMs
    # rarely produce without a domain-specific rubric.
    TaskType.EVALUATE: 0.80,
}


MAX_OUTPUT_TOKENS: dict[TaskType, int] = {
    TaskType.CODE_GEN: 8192,  # raised: avoid unterminated strings mid-class
    TaskType.CODE_REVIEW: 4096,  # raised: full analysis without truncation
    TaskType.REASONING: 4096,
    TaskType.WRITING: 4096,
    TaskType.DATA_EXTRACT: 2048,
    TaskType.SUMMARIZE: 1024,
    TaskType.EVALUATE: 2048,  # raised: evaluation tasks need more room
}


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
    # Meta LLaMA models
    Model.LLAMA_4_MAVERICK: 8192,
    Model.LLAMA_4_SCOUT: 8192,
    Model.LLAMA_3_3_70B: 8192,
    Model.LLAMA_3_1_405B: 8192,
    # Microsoft Phi models
    Model.PHI_4: 4096,
    Model.PHI_4_REASONING: 4096,
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


def build_default_profiles() -> dict[Model, ModelProfile]:
    """

    Build a ModelProfile for every Model enum value using the static

    COST_TABLE and ROUTING_TABLE as the source of truth.



    Called once at Orchestrator construction time. Telemetry fields

    (quality_score, trust_factor, avg_latency_ms, …) start at their

    defaults and are updated at runtime by TelemetryCollector.



    Lazy-imports ModelProfile from policy to avoid a circular import

    (policy.py → models.py, models.py → policy.py would be circular).

    """

    # Lazy import to avoid circular dependency: policy.py imports models.py

    from .policy import ModelProfile  # noqa: PLC0415

    # Build capability map: {TaskType → priority_rank} for each model

    # Priority rank = index in ROUTING_TABLE list (0 = highest priority)

    capability_map: dict[Model, dict[TaskType, int]] = {m: {} for m in Model}

    for task_type, model_list in ROUTING_TABLE.items():

        for rank, model in enumerate(model_list):

            capability_map[model][task_type] = rank

    profiles: dict[Model, ModelProfile] = {}

    for model in Model:

        costs = COST_TABLE.get(model, {"input": 5.0, "output": 20.0})

        profiles[model] = ModelProfile(
            model=model,
            provider=get_provider(model),
            cost_per_1m_input=costs["input"],
            cost_per_1m_output=costs["output"],
            capable_task_types=capability_map.get(model, {}),
        )

    return profiles


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
