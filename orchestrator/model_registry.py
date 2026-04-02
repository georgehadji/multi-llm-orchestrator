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
    timeout = ModelRegistry.get_timeout("qwen/qwen-3-coder")

    # Get cost info
    cost = ModelRegistry.get_cost("deepseek/deepseek-v3.2")

    # Check if model is valid
    is_valid = ModelRegistry.is_valid_model("qwen/qwen-3-coder")
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

    # Qwen Models - Coding Specialists
    QWEN_2_5_CODER_32B = "qwen/qwen-2.5-coder-32b-instruct"  # $0.66/$1.00, 33K coding ⭐ VERIFIED

    # DeepSeek Models - Best Value
    DEEPSEEK_V3_2 = "deepseek/deepseek-v3.2"  # $0.27/$1.10, battle-tested ⭐ VERIFIED
    DEEPSEEK_CHAT = "deepseek/deepseek-chat"  # $0.32/$0.89, 164K context ⭐ VERIFIED
    DEEPSEEK_REASONER = "deepseek/deepseek-reasoner"  # $0.55/$2.19, reasoning specialist

    # Anthropic Claude Models - Balanced Quality
    CLAUDE_SONNET_4_6 = "anthropic/claude-3-5-sonnet"  # $6.00/$30.00, 200K ⭐ VERIFIED
    CLAUDE_OPUS_4_6 = "anthropic/claude-opus-4-6"  # $5.00/$25.00, complex analysis
    CLAUDE_HAIKU_3_5 = "anthropic/claude-3-5-haiku"  # $0.25/$1.25, fast

    # OpenAI Models - Premium Tier
    GPT_5 = "openai/gpt-5"  # $1.25/$10.00, 400K ⭐ VERIFIED
    GPT_5_CODEX = "openai/gpt-5-codex"  # $1.25/$10.00, coding specialist ⭐ VERIFIED
    GPT_5_4 = "openai/gpt-5.4"  # $2.50/$15.00, unified
    GPT_5_4_MINI = "openai/gpt-5.4-mini"  # $0.25/$2.00, fast
    GPT_5_4_CODEX = "openai/gpt-5.4-codex"  # $1.75/$14.00, coding specialist
    GPT_5_4_PRO = "openai/gpt-5.4-pro"  # $30.00/$180.00, maximum quality
    GPT_4O = "openai/gpt-4o"  # $2.50/$10.00, previous gen
    GPT_4O_MINI = "openai/gpt-4o-mini"  # $0.15/$0.60, 128K ⭐ VERIFIED

    # Google Gemini Models
    GEMINI_2_5_FLASH = "google/gemini-2.5-flash"  # $0.30/$2.50, 1M+ ⭐ VERIFIED
    GEMINI_PRO = "google/gemini-pro-1.5"  # $1.25/$10.00
    GEMINI_FLASH = "google/gemini-flash-1.5"  # $0.15/$0.60, fast

    # xAI Grok Models
    GROK_4_20 = "x-ai/grok-4.20"  # $2.00/$6.00, 2M context ⭐ VERIFIED (NOT -beta)
    GROK_4_1_FAST = "x-ai/grok-4.1-fast"  # $0.20/$0.50, fast

    # Moonshot Kimi Models
    KIMI_K2 = "moonshotai/kimi-k2"  # $0.57/$2.30, 128K ⭐ VERIFIED
    KIMI_K2_5 = "moonshotai/kimi-k2.5"  # $0.42/$2.20, visual coding SOTA

    # Xiaomi MiMo Models - New Open-Source Leaders ⭐ VERIFIED
    MIMO_V2_FLASH = "xiaomi/mimo-v2-flash"  # $0.09/$0.29, 256K, #1 SWE-bench ⭐ BEST VALUE
    MIMO_V2_PRO = "xiaomi/mimo-v2-pro"  # $1.00/$3.00, 1T+ params, 1M+ ctx

    # StepFun Models - Best Value ⭐ VERIFIED
    STEP_3_5_FLASH = "stepfun/step-3.5-flash"  # $0.10/$0.30, 262K, 196B MoE ⭐ BEST VALUE
    STEP_3_5 = "stepfun/step-3.5"  # $0.15/$0.45

    # Z-AI GLM Models ⭐ VERIFIED
    GLM_4_7_FLASH = "z-ai/glm-4.7-flash"  # $0.06/$0.40, 202K ultra-cheap ⭐ CHEAPEST
    GLM_4_7 = "z-ai/glm-4.7"  # $0.39/$1.75, enhanced programming
    GLM_5 = "z-ai/glm-5"  # $0.72/$2.30, 80K complex systems
    GLM_5_TURBO = "z-ai/glm-5-turbo"  # $1.20/$4.00, long-horizon

    # Minimax Models ⭐ VERIFIED
    MINIMAX_M2_7 = "minimax/minimax-m2.7"  # $0.30/$1.20, 205K, multi-agent ⭐
    MINIMAX_M2_5 = "minimax/minimax-m2.5"  # $0.30/$1.20

    # OpenRouter Auto-Router
    OPENROUTER_AUTO = "openrouter/auto"  # Dynamic routing

    # ═══════════════════════════════════════════════════════
    # DEPRECATED/UNAVAILABLE MODELS (DO NOT USE)
    # Verified via OpenRouter URL checks 2026-04-01
    # ═══════════════════════════════════════════════════════

    UNAVAILABLE_MODELS = {
        # Qwen models - NOT AVAILABLE
        "qwen/qwen-3-coder-next": "qwen/qwen-2.5-coder-32b-instruct",
        "qwen/qwen-3.5-397b-a17b": "openai/gpt-5",
        "qwen/qwen-3-coder": "qwen/qwen-2.5-coder-32b-instruct",
        "qwen/qwen-3.5-235b-a22b-thinking-2507": "openai/gpt-5",
        # NVIDIA - NOT AVAILABLE
        "nvidia/nemotron-3-super": "minimax/minimax-m2.7",
        # AION Labs - NOT AVAILABLE
        "aionlabs/aion-2.0": "z-ai/glm-5",
        # Google - NOT AVAILABLE
        "google/gemini-3.1-pro": "google/gemini-2.5-flash",
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
        "openai/o4": 180,
    }

    # Per-model timeout overrides
    MODEL_TIMEOUT_OVERRIDES = {
        DEEPSEEK_REASONER: 120,  # Reasoning specialist
        GPT_5_4_PRO: 180,  # Premium reasoning
        GPT_5: 120,  # Complex reasoning
        GROK_4_20: 90,  # 2M context, complex reasoning
    }

    # ═══════════════════════════════════════════════════════
    # COST TABLE (per 1M tokens, USD)
    # Updated 2026-04-01 with verified OpenRouter pricing
    # ═══════════════════════════════════════════════════════

    COST_TABLE: Dict[str, dict] = {
        # Qwen Models (VERIFIED)
        QWEN_2_5_CODER_32B: {"input": 0.66, "output": 1.00},
        # DeepSeek Models (VERIFIED)
        DEEPSEEK_V3_2: {"input": 0.27, "output": 1.10},
        DEEPSEEK_CHAT: {"input": 0.32, "output": 0.89},  # Updated price
        DEEPSEEK_REASONER: {"input": 0.55, "output": 2.19},
        # Anthropic Models (VERIFIED)
        CLAUDE_SONNET_4_6: {"input": 6.00, "output": 30.00},  # Updated price
        CLAUDE_OPUS_4_6: {"input": 5.00, "output": 25.00},
        CLAUDE_HAIKU_3_5: {"input": 0.25, "output": 1.25},
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
        GEMINI_2_5_FLASH: {"input": 0.30, "output": 2.50},
        GEMINI_PRO: {"input": 1.25, "output": 10.00},
        GEMINI_FLASH: {"input": 0.15, "output": 0.60},
        # xAI Grok Models (VERIFIED)
        GROK_4_20: {"input": 2.00, "output": 6.00},
        GROK_4_1_FAST: {"input": 0.20, "output": 0.50},
        # Moonshot Kimi Models (VERIFIED)
        KIMI_K2: {"input": 0.57, "output": 2.30},
        KIMI_K2_5: {"input": 0.42, "output": 2.20},
        # Xiaomi MiMo Models (VERIFIED)
        MIMO_V2_FLASH: {"input": 0.09, "output": 0.29},  # ⭐ BEST VALUE
        MIMO_V2_PRO: {"input": 1.00, "output": 3.00},
        # StepFun Models (VERIFIED)
        STEP_3_5_FLASH: {"input": 0.10, "output": 0.30},  # ⭐ BEST VALUE
        STEP_3_5: {"input": 0.15, "output": 0.45},
        # Z-AI GLM Models (VERIFIED)
        GLM_4_7_FLASH: {"input": 0.06, "output": 0.40},  # ⭐ CHEAPEST
        GLM_4_7: {"input": 0.39, "output": 1.75},
        GLM_5: {"input": 0.72, "output": 2.30},
        GLM_5_TURBO: {"input": 1.20, "output": 4.00},
        # Minimax Models (VERIFIED)
        MINIMAX_M2_7: {"input": 0.30, "output": 1.20},
        MINIMAX_M2_5: {"input": 0.30, "output": 1.20},
    }

    # ═══════════════════════════════════════════════════════
    # MODEL MAX TOKENS
    # Updated 2026-04-01 with verified context windows
    # ═══════════════════════════════════════════════════════

    MODEL_MAX_TOKENS: Dict[str, int] = {
        # Qwen Models (VERIFIED)
        QWEN_2_5_CODER_32B: 32768,
        # DeepSeek Models (VERIFIED)
        DEEPSEEK_V3_2: 163840,
        DEEPSEEK_CHAT: 163840,
        DEEPSEEK_REASONER: 16384,
        # Anthropic Models (VERIFIED)
        CLAUDE_SONNET_4_6: 200000,
        CLAUDE_OPUS_4_6: 200000,
        CLAUDE_HAIKU_3_5: 200000,
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
        GEMINI_2_5_FLASH: 1048576,  # 1M+ context
        GEMINI_PRO: 32768,
        GEMINI_FLASH: 16384,
        # xAI Grok Models (VERIFIED)
        GROK_4_20: 2000000,  # 2M context!
        GROK_4_1_FAST: 131072,
        # Moonshot Kimi Models (VERIFIED)
        KIMI_K2: 131072,
        KIMI_K2_5: 131072,
        # Xiaomi MiMo Models (VERIFIED)
        MIMO_V2_FLASH: 262144,
        MIMO_V2_PRO: 1048576,  # 1M+ context
        # StepFun Models (VERIFIED)
        STEP_3_5_FLASH: 262144,
        STEP_3_5: 262144,
        # Z-AI GLM Models (VERIFIED)
        GLM_4_7_FLASH: 202752,
        GLM_4_7: 202752,
        GLM_5: 80000,
        GLM_5_TURBO: 262144,
        # Minimax Models (VERIFIED)
        MINIMAX_M2_7: 204800,
        MINIMAX_M2_5: 204800,
    }

    # ═══════════════════════════════════════════════════════
    # MODEL CATEGORIES
    # Updated 2026-04-01 with verified models
    # ═══════════════════════════════════════════════════════

    # Coding specialists - best for code generation
    CODING_SPECIALISTS = {
        QWEN_2_5_CODER_32B,
        GPT_5_CODEX,
        GPT_5_4_CODEX,
        MIMO_V2_FLASH,
        KIMI_K2_5,
        MINIMAX_M2_7,
    }

    # Reasoning models - best for complex analysis
    REASONING_MODELS = {
        DEEPSEEK_REASONER,
        GPT_5,
        GPT_5_4_PRO,
        GROK_4_20,
        STEP_3_5_FLASH,
    }

    # Budget models - best value
    BUDGET_MODELS = {
        MIMO_V2_FLASH,  # $0.09/$0.29 - Best value! ⭐
        GLM_4_7_FLASH,  # $0.06/$0.40 - Cheapest!
        STEP_3_5_FLASH,  # $0.10/$0.30 - Best value reasoning
        QWEN_2_5_CODER_32B,  # $0.66/$1.00 - Coding specialist
        GPT_4O_MINI,  # $0.15/$0.60
        GEMINI_FLASH,  # $0.15/$0.60
        DEEPSEEK_CHAT,  # $0.32/$0.89
    }

    # Premium models - maximum quality
    PREMIUM_MODELS = {
        GPT_5_4_PRO,
        CLAUDE_OPUS_4_6,
        GROK_4_20,
    }

    # Models with 200K+ context capability
    LONG_CONTEXT_MODELS = {
        GEMINI_2_5_FLASH,  # 1M+
        MIMO_V2_PRO,  # 1M+
        CLAUDE_SONNET_4_6,  # 200K
        CLAUDE_OPUS_4_6,  # 200K
        CLAUDE_HAIKU_3_5,  # 200K
        GPT_5,  # 400K
        GPT_5_CODEX,  # 400K
        GROK_4_20,  # 2M!
        STEP_3_5_FLASH,  # 262K
        STEP_3_5,  # 262K
        MIMO_V2_FLASH,  # 256K
        GLM_4_7_FLASH,  # 202K
        GLM_4_7,  # 202K
        GLM_5_TURBO,  # 262K
        MINIMAX_M2_7,  # 205K
        MINIMAX_M2_5,  # 205K
        DEEPSEEK_V3_2,  # 164K
        DEEPSEEK_CHAT,  # 164K
    }

    # Multimodal models — support image + text input (vision-capable)
    MULTIMODAL_MODELS = {
        # Google Gemini — natively multimodal
        GEMINI_2_5_FLASH,
        GEMINI_PRO,
        GEMINI_FLASH,
        # Anthropic Claude 3.x+ — all vision-capable
        CLAUDE_SONNET_4_6,
        CLAUDE_OPUS_4_6,
        CLAUDE_HAIKU_3_5,
        # OpenAI — GPT-4o and GPT-5 family are multimodal
        GPT_4O,
        GPT_4O_MINI,
        GPT_5,
        GPT_5_4,
        # Moonshot Kimi-K2.5 — described as visual coding SOTA
        KIMI_K2_5,
    }

    # ═══════════════════════════════════════════════════════
    # CLASS METHODS
    # ═══════════════════════════════════════════════════════

    @classmethod
    def get_timeout(cls, model_id: str) -> int:
        """
        Get timeout for a specific model.

        Args:
            model_id: Full model ID (e.g., "qwen/qwen-3-coder")

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
    def get_cost(cls, model_id: str) -> dict:
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
        """Check if model is a reasoning specialist."""
        return model_id in cls.REASONING_MODELS

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
    def validate_all_models(cls) -> dict:
        """
        Validate all model IDs against the registry.

        Returns:
            Dictionary with validation results
        """
        results = {
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
                    elif value in cls.COST_TABLE or attr_name not in [
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


def get_cost(model_id: str) -> dict:
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


def migrate_deprecated_models(config: dict) -> dict:
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
