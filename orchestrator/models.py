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

# Re-export Budget so tests can do `from orchestrator.models import Budget`
# budget.py has no imports from models.py, so no circular import risk.
from .budget import Budget as Budget  # noqa: F401

# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────


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
    GEMINI_PRO = "google/gemini-2.5-pro"
    GEMINI_FLASH = "google/gemini-2.5-flash"
    GEMINI_FLASH_LITE = "google/gemini-2.5-flash-lite"

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

    # DeepSeek Models
    DEEPSEEK_CHAT = "deepseek/deepseek-chat"
    DEEPSEEK_REASONER = "deepseek/deepseek-r1"
    DEEPSEEK_R1 = DEEPSEEK_REASONER
    DEEPSEEK_V3 = "deepseek/deepseek-v3"
    DEEPSEEK_V3_2 = "deepseek/deepseek-v3.2"

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
    MOONSHOT_KIMI_K2_5 = "moonshotai/kimi-k2.5"  # $0.42/$2.20, visual coding SOTA
    MOONSHOT_KIMI_K2 = "moonshotai/kimi-k2"  # $0.50/$1.50

    # ═══════════════════════════════════════════════════════
    # STEPFUN MODELS (NEW v3.0) - BEST VALUE!
    # ═══════════════════════════════════════════════════════
    STEPFUN_STEP_3_5_FLASH = "stepfun/step-3.5-flash"  # $0.10/$0.30, 196B MoE ⭐
    STEPFUN_STEP_3_5 = "stepfun/step-3.5"  # $0.15/$0.45

    # ═══════════════════════════════════════════════════════
    # Z.AI GLM MODELS (NEW v3.0)
    # ═══════════════════════════════════════════════════════
    ZHIPU_GLM_4_7_FLASH = "z-ai/glm-4.7-flash"  # $0.06/$0.40, ultra-cheap ⭐
    ZHIPU_GLM_4_7 = "z-ai/glm-4.7"  # $0.39/$1.75, enhanced programming
    ZHIPU_GLM_5 = "z-ai/glm-5"  # $0.72/$2.30, complex systems
    ZHIPU_GLM_5_TURBO = "z-ai/glm-5-turbo"  # $1.20/$4.00, 202K, agents

    # ═══════════════════════════════════════════════════════
    # XAI GROK MODELS (NEW v3.0) - LOWEST HALLUCINATION
    # Note: Updated 2026-04-01 - Use grok-4.20 (NOT grok-4.20-beta)
    # ═══════════════════════════════════════════════════════
    XAI_GROK_4_20 = "x-ai/grok-4.20"  # $2.00/$6.00, 2M context, lowest hallucination ⭐
    XAI_GROK_4_20_BETA = "x-ai/grok-4.20"  # alias — -beta not available, maps to grok-4.20
    XAI_GROK_4_1_FAST = "x-ai/grok-4.1-fast"  # $0.20/$0.50, fast

    # ═══════════════════════════════════════════════════════
    # QWEN MODELS (NEW v3.0) - CODING SPECIALISTS
    # Note: Updated 2026-04-01 - Verified available on OpenRouter
    # ═══════════════════════════════════════════════════════
    QWEN_2_5_CODER_32B = "qwen/qwen-2.5-coder-32b-instruct"  # $0.66/$1.00, 33K coding ⭐
    # Aliases for future models (not yet on OpenRouter — map to valid equivalents)
    QWEN_3_CODER_NEXT = "qwen/qwen-2.5-coder-32b-instruct"  # alias until qwen-3-coder-next ships
    QWEN_3_5_397B_A17B = "qwen/qwen-2.5-coder-32b-instruct"  # alias until qwen-3.5-397b ships

    # ═══════════════════════════════════════════════════════
    # MINIMAX MODELS (NEW v3.0)
    # Note: Verified available 2026-04-01
    # ═══════════════════════════════════════════════════════
    MINIMAX_M2_7 = "minimax/minimax-m2.7"  # $0.30/$1.20, 205K, multi-agent ⭐
    MINIMAX_M2_5 = "minimax/minimax-m2.5"  # $0.30/$1.20

    # ═══════════════════════════════════════════════════════
    # NVIDIA MODELS (redirected via model_registry to fallback)
    # ═══════════════════════════════════════════════════════
    NVIDIA_NEMOTRON_3_SUPER = "nvidia/nemotron-3-super"  # redirects → minimax-m2.7

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
    Model.GEMINI_PRO: {"input": 1.25, "output": 10.00},
    Model.GEMINI_FLASH: {"input": 0.15, "output": 0.60},
    Model.GEMINI_FLASH_LITE: {"input": 0.10, "output": 0.40},  # alias of GEMINI_2_5_FLASH_LITE
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
    Model.DEEPSEEK_CHAT: {"input": 0.28, "output": 0.42},
    Model.DEEPSEEK_REASONER: {"input": 0.28, "output": 0.42},
    Model.DEEPSEEK_V3: {"input": 0.27, "output": 1.10},
    Model.DEEPSEEK_R1: {"input": 0.55, "output": 2.19},
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
    Model.MOONSHOT_KIMI_K2_5: {"input": 0.42, "output": 2.20},  # visual coding SOTA
    Model.MOONSHOT_KIMI_K2: {"input": 0.50, "output": 1.50},
    # ═══════════════════════════════════════════════════════
    # STEPFUN MODELS (NEW v3.0) - BEST VALUE!
    # ═══════════════════════════════════════════════════════
    Model.STEPFUN_STEP_3_5_FLASH: {"input": 0.10, "output": 0.30},  # 196B MoE ⭐
    Model.STEPFUN_STEP_3_5: {"input": 0.15, "output": 0.45},
    # ═══════════════════════════════════════════════════════
    # Z.AI GLM MODELS (NEW v3.0)
    # ═══════════════════════════════════════════════════════
    Model.ZHIPU_GLM_4_7_FLASH: {"input": 0.06, "output": 0.40},  # ultra-cheap ⭐
    Model.ZHIPU_GLM_4_7: {"input": 0.39, "output": 1.75},  # enhanced programming
    Model.ZHIPU_GLM_5: {"input": 0.72, "output": 2.30},  # complex systems
    Model.ZHIPU_GLM_5_TURBO: {"input": 1.20, "output": 4.00},  # 202K, agents
    # ═══════════════════════════════════════════════════════
    # XAI GROK MODELS (NEW v3.0) - LOWEST HALLUCINATION
    # Note: Updated 2026-04-01 - Use grok-4.20 (NOT grok-4.20-beta)
    # ═══════════════════════════════════════════════════════
    Model.XAI_GROK_4_20: {"input": 2.00, "output": 6.00},  # 2M context ⭐
    Model.XAI_GROK_4_20_BETA: {"input": 2.00, "output": 6.00},  # alias for grok-4.20
    Model.XAI_GROK_4_1_FAST: {"input": 0.20, "output": 0.50},  # fast
    # ═══════════════════════════════════════════════════════
    # QWEN MODELS (NEW v3.0) - CODING SPECIALISTS
    # Note: Updated 2026-04-01 - Verified available
    # ═══════════════════════════════════════════════════════
    Model.QWEN_2_5_CODER_32B: {"input": 0.66, "output": 1.00},  # 33K coding ⭐ (QWEN_3_CODER_NEXT aliases this)
    # ═══════════════════════════════════════════════════════
    # MINIMAX MODELS (NEW v3.0)
    # ═══════════════════════════════════════════════════════
    Model.MINIMAX_M2_7: {"input": 0.30, "output": 1.20},  # 205K ⭐
    Model.MINIMAX_M2_5: {"input": 0.30, "output": 1.20},
    # ═══════════════════════════════════════════════════════
    # NVIDIA MODELS (redirected via model_registry to fallback)
    # ═══════════════════════════════════════════════════════
    Model.NVIDIA_NEMOTRON_3_SUPER: {"input": 0.10, "output": 0.50},  # redirects → minimax-m2.7
    # GPT-5.4 Models (NEW v3.0)
    Model.GPT_5_4: {"input": 2.50, "output": 15.00},
    Model.GPT_5_4_MINI: {"input": 0.75, "output": 4.50},
    Model.GPT_5_4_CODEX: {"input": 1.75, "output": 14.00},  # SWE-Bench SOTA
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
    # CODE_GEN: Cheapest capable first, then quality
    # Best: Xiaomi MiMo-V2-Flash (#1 open-source SWE-bench at $0.09/1M!)
    TaskType.CODE_GEN: [
        Model.XIAOMI_MIMO_V2_FLASH,  # $0.09/$0.29, 309B MoE, #1 SWE-bench open ⭐ BEST
        Model.QWEN_2_5_CODER_32B,  # $0.66/$1.00, 33K coding specialist
        Model.DEEPSEEK_V3_2,  # $0.27/$1.10, 1.24T tokens, battle-tested
        Model.MOONSHOT_KIMI_K2_5,  # $0.42/$2.20, visual coding SOTA
        Model.ZHIPU_GLM_4_7,  # $0.39/$1.75, enhanced programming
        Model.MINIMAX_M2_7,  # $0.30/$1.20, multi-agent ⭐
        Model.PHI_4,  # $0.07/$0.14, Microsoft 14B
        Model.GEMMA_3_27B,  # $0.08/$0.20, Google open-weights
        Model.LLAMA_3_3_70B,  # $0.12/$0.30, Meta 70B
        Model.LLAMA_4_SCOUT,  # $0.11/$0.34, Meta 109B MoE
        Model.LLAMA_4_MAVERICK,  # $0.17/$0.17, Meta 400B MoE
        Model.HERMES_3_70B,  # $0.40/$0.40, Nous fine-tuned
        Model.LLAMA_3_1_405B,  # $2.00/$2.00, Meta 405B
        Model.CLAUDE_SONNET_4_6,  # $3/$15, best coding
        Model.GPT_5_4_CODEX,  # $1.75/$14, SWE-Bench Pro SOTA
    ],
    # CODE_REVIEW: Fast and accurate
    # Best: Grok 4.20 (lowest hallucination) + DeepSeek R1 (reasoning)
    TaskType.CODE_REVIEW: [
        Model.XAI_GROK_4_20,  # $2.00/$6.00, 2M context, lowest hallucination ⭐ BEST
        Model.DEEPSEEK_R1,  # $0.55/$2.19, reasoning specialist
        Model.MOONSHOT_KIMI_K2_5,  # $0.42/$2.20, visual coding SOTA
        Model.PHI_4_REASONING,  # $0.07/$0.35, Microsoft CoT
        Model.GEMMA_3_27B,  # $0.08/$0.20, Google
        Model.LLAMA_3_3_70B,  # $0.12/$0.30, Meta 70B
        Model.LLAMA_4_SCOUT,  # $0.11/$0.34, Meta fast
        Model.LLAMA_4_MAVERICK,  # $0.17/$0.17, Meta 400B
        Model.HERMES_3_70B,  # $0.40/$0.40, Nous
        Model.CLAUDE_SONNET_4_6,  # $3/$15, premium
    ],
    # REASONING: Reasoning models prioritized
    # Best: StepFun Step 3.5 Flash (196B MoE at $0.10/1M!)
    TaskType.REASONING: [
        Model.STEPFUN_STEP_3_5_FLASH,  # $0.10/$0.30, 196B MoE ⭐ BEST VALUE
        Model.DEEPSEEK_R1,  # $0.55/$2.19, reasoning specialist
        Model.MOONSHOT_KIMI_K2_5,  # $0.42/$2.20, native multimodal
        Model.ZHIPU_GLM_4_7_FLASH,  # $0.06/$0.40, ultra-cheap 202K
        Model.LLAMA_4_MAVERICK,  # $0.17/$0.17, Meta 400B
        Model.LLAMA_3_1_405B,  # $2.00/$2.00, Meta frontier
        Model.O3_MINI,  # $1.10/$4.40, OpenAI
        Model.O4_MINI,  # $1.50/$6.00, OpenAI
        Model.CLAUDE_OPUS_4_6,  # $5/$25, most capable
    ],
    # WRITING: Quality and creativity
    TaskType.WRITING: [
        Model.LLAMA_4_MAVERICK,  # $0.17/$0.17, Meta 400B
        Model.HERMES_3_70B,  # $0.40/$0.40, Nous
        Model.LLAMA_3_1_405B,  # $2.00/$2.00, Meta frontier
        Model.GPT_5_4,  # $2.50/$10, excellent
        Model.CLAUDE_SONNET_4_6,  # $3/$15, excellent prose
    ],
    # DATA_EXTRACT: Cheapest first
    TaskType.DATA_EXTRACT: [
        Model.ZHIPU_GLM_4_7_FLASH,  # $0.06/$0.40, ultra-cheap ⭐ BEST
        Model.PHI_4,  # $0.07/$0.14, Microsoft
        Model.GEMMA_3_27B,  # $0.08/$0.20, Google
        Model.LLAMA_3_3_70B,  # $0.12/$0.30, Meta 70B
        Model.LLAMA_4_SCOUT,  # $0.11/$0.34, Meta
        Model.LLAMA_4_MAVERICK,  # $0.17/$0.17, Meta 400B
        Model.GPT_5_4_MINI,  # $0.15/$0.60, reliable
    ],
    # SUMMARIZE: Cheap with good context
    TaskType.SUMMARIZE: [
        Model.ZHIPU_GLM_4_7_FLASH,  # $0.06/$0.40, ultra-cheap ⭐ BEST
        Model.PHI_4,  # $0.07/$0.14, fast
        Model.GEMMA_3_27B,  # $0.08/$0.20, concise
        Model.LLAMA_3_3_70B,  # $0.12/$0.30, accurate
        Model.LLAMA_4_SCOUT,  # $0.11/$0.34, fast
    ],
    # EVALUATE: Reliable evaluation
    # Best: Grok 4.20 (lowest hallucination)
    TaskType.EVALUATE: [
        Model.XAI_GROK_4_20,  # $2.00/$6.00, 2M context, lowest hallucination ⭐ BEST
        Model.DEEPSEEK_R1,  # $0.55/$2.19, fair scoring
        Model.MOONSHOT_KIMI_K2_5,  # $0.42/$2.20, technical eval
        Model.LLAMA_4_MAVERICK,  # $0.17/$0.17, Meta 400B
        Model.HERMES_3_70B,  # $0.40/$0.40, Nous
        Model.LLAMA_3_1_405B,  # $2.00/$2.00, Meta
        Model.CLAUDE_SONNET_4_6,  # $3/$15, excellent
        Model.GPT_5_4,  # $2.50/$10, reliable
    ],
}


# ─────────────────────────────────────────────
# Fallback chains (always cross-provider)
# ─────────────────────────────────────────────

FALLBACK_CHAIN: dict[Model, Model] = {
    # OpenRouter fallbacks (cheaper/faster → more capable)
    # OpenAI models fallbacks
    Model.GPT_4O: Model.CLAUDE_SONNET_4_6,  # GPT-4o → Claude Sonnet
    Model.GPT_4O_MINI: Model.LLAMA_3_3_70B,  # GPT-4o-mini → LLaMA 70B
    Model.O1: Model.CLAUDE_OPUS_4_6,  # o1 → Claude Opus
    Model.O3_MINI: Model.PHI_4_REASONING,  # o3-mini → Phi-4 Reasoning
    Model.O4_MINI: Model.CLAUDE_OPUS_4_6,  # o4-mini → Claude Opus
    # Gemini fallbacks
    Model.GEMINI_PRO: Model.GPT_4O,  # Gemini Pro → GPT-4o
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
    Model.DEEPSEEK_CHAT: Model.LLAMA_3_3_70B,  # DeepSeek Chat → LLaMA 70B
    Model.DEEPSEEK_REASONER: Model.PHI_4_REASONING,  # DeepSeek Reasoner → Phi-4 Reasoning
    Model.DEEPSEEK_V3: Model.LLAMA_4_SCOUT,  # DeepSeek V3 → LLaMA Scout
    Model.DEEPSEEK_R1: Model.O3_MINI,  # DeepSeek R1 → o3-mini
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
    Model.GEMINI_PRO: 8192,
    Model.GEMINI_FLASH: 8192,
    Model.GEMINI_FLASH_LITE: 8192,
    # DeepSeek models
    Model.DEEPSEEK_CHAT: 8192,
    Model.DEEPSEEK_REASONER: 8192,
    Model.DEEPSEEK_V3: 8192,
    Model.DEEPSEEK_V3_2: 8192,
    Model.DEEPSEEK_R1: 8192,
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
# Budget partitioning (soft caps)
# ─────────────────────────────────────────────

BUDGET_PARTITIONS: dict[str, float] = {
    "decomposition": 0.05,
    "generation": 0.45,
    "cross_review": 0.25,
    "evaluation": 0.15,
    "reserve": 0.10,
}


# ─────────────────────────────────────────────
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

    def __post_init__(self):
        self.acceptance_threshold = DEFAULT_THRESHOLDS.get(self.type, self.acceptance_threshold)
        self.max_iterations = get_max_iterations(self.type)
        self.max_output_tokens = MAX_OUTPUT_TOKENS.get(self.type, self.max_output_tokens)


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
    budget: object  # Budget type - avoid circular import by using object
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
